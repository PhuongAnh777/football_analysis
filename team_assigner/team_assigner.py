import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """
    Assigns players to teams (1 or 2) based on jersey colour clustering.

    Improvements over the original single-frame approach:
    - Multi-frame calibration: collects jersey colours from up to _CALIB_FRAMES
      frames before fitting the global two-cluster model, giving a much more
      stable colour boundary than relying on a single frame.
    - LAB colour space: perceptually uniform, far less sensitive to broadcast
      lighting changes than raw BGR.
    - Better jersey crop: uses the centre 80 % of the bbox width to exclude
      background pixels on the left/right edges of each bounding box.
    - Majority-vote locking: unchanged from before — still uses _N_VOTE_SAMPLES
      frames before committing a player to a team.
    """

    _N_VOTE_SAMPLES = 10   # frames before locking a player's team
    _CALIB_FRAMES   = 12   # max frames used for global colour calibration
    _MIN_PLAYERS    = 4    # min players per frame for calibration

    def __init__(self):
        self.team_colors: dict = {}        # team_id (1/2) -> BGR array for annotation
        self.player_team_dict: dict = {}   # locked: player_id -> team_id
        self._pending_votes: dict = {}     # player_id -> [team_id, ...]
        self._kmeans: KMeans | None = None  # fitted on LAB jersey colours
        self._team_swapped: bool = False   # True when scoreboard alignment flipped 1↔2

    # ── Crop helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _jersey_crop(frame: np.ndarray, bbox) -> np.ndarray | None:
        """
        Return the top-half, centre-80 % crop of the player bbox.
        This is the part most likely to show the jersey rather than grass/sky.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        h, w = patch.shape[:2]
        top = patch[: max(1, h // 2), max(0, w // 10) : max(1, w - w // 10)]
        return top if top.size > 0 else patch[: max(1, h // 2), :]

    # ── Per-player dominant jersey colour in LAB ───────────────────────────

    def _jersey_lab(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        """
        Extract the dominant jersey colour in CIE-LAB space.
        Uses 2-cluster KMeans inside the crop to separate jersey from
        background, then returns the cluster whose centroid is NOT in the
        image corners (i.e. the player cluster).
        """
        crop = self._jersey_crop(frame, bbox)
        if crop is None:
            return None

        lab = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2LAB)
        flat = lab.reshape(-1, 3).astype(np.float32)

        if flat.shape[0] < 4:
            return flat.mean(axis=0)

        km = KMeans(n_clusters=2, random_state=0, n_init=3, max_iter=50)
        km.fit(flat)

        labels = km.labels_.reshape(crop.shape[:2])
        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        bg_cluster = max(set(corners), key=corners.count)
        jersey_cluster = 1 - bg_cluster

        return km.cluster_centers_[jersey_cluster]   # LAB [L, A, B]

    # ── Global team calibration ────────────────────────────────────────────

    def assign_team_color(self, frames_and_detections: list[tuple]) -> None:
        """
        Fit the two-team colour model from multiple frames.

        Parameters
        ----------
        frames_and_detections : list of (frame_ndarray, player_dict) tuples
            Each tuple is one video frame paired with its player detections.
            Use at least _CALIB_FRAMES frames with _MIN_PLAYERS players each
            for a stable calibration.
        """
        lab_colors: list[np.ndarray] = []
        bgr_colors: list[np.ndarray] = []

        for frame, detections in frames_and_detections:
            for det in detections.values():
                lab = self._jersey_lab(frame, det["bbox"])
                if lab is not None:
                    lab_colors.append(lab)
                    crop = self._jersey_crop(frame, det["bbox"])
                    if crop is not None:
                        bgr_colors.append(crop.reshape(-1, 3).mean(axis=0))
                    else:
                        bgr_colors.append(np.array([128.0, 128.0, 128.0]))

        if len(lab_colors) < 2:
            # Fallback — not enough data; leave defaults
            self.team_colors = {1: np.array([255, 0, 0]), 2: np.array([0, 0, 255])}
            return

        lab_arr = np.array(lab_colors, dtype=np.float32)
        bgr_arr = np.array(bgr_colors, dtype=np.float32)

        self._kmeans = KMeans(
            n_clusters=2, init="k-means++", n_init=10, random_state=0
        )
        labels = self._kmeans.fit_predict(lab_arr)

        # Derive display colours as mean BGR of all pixels in each cluster
        for team_id in (0, 1):
            mask = labels == team_id
            mean_bgr = bgr_arr[mask].mean(axis=0) if mask.any() else np.array([128.0, 128.0, 128.0])
            self.team_colors[team_id + 1] = mean_bgr.astype(np.float32)

    @staticmethod
    def _lab_dist(bgr_a: np.ndarray, bgr_b: np.ndarray) -> float:
        a = cv2.cvtColor(
            np.uint8([[np.clip(bgr_a, 0, 255).astype(np.uint8)]]),
            cv2.COLOR_BGR2LAB,
        )[0, 0].astype(np.float32)
        b = cv2.cvtColor(
            np.uint8([[np.clip(bgr_b, 0, 255).astype(np.uint8)]]),
            cv2.COLOR_BGR2LAB,
        )[0, 0].astype(np.float32)
        return float(np.linalg.norm(a - b))

    def align_to_scoreboard(
        self,
        left_stripe_bgr: np.ndarray,
        right_stripe_bgr: np.ndarray,
    ) -> bool:
        """Match jersey clusters to scoreboard stripes (left = team 1).

        Returns True if team 1 and team 2 were swapped to match the board.
        """
        if self._kmeans is None or not self.team_colors:
            return False

        c1 = self.team_colors.get(1)
        c2 = self.team_colors.get(2)
        if c1 is None or c2 is None:
            return False

        direct = self._lab_dist(c1, left_stripe_bgr) + self._lab_dist(c2, right_stripe_bgr)
        cross  = self._lab_dist(c1, right_stripe_bgr) + self._lab_dist(c2, left_stripe_bgr)

        if cross + 1.0 < direct:
            self.team_colors[1] = c2.copy()
            self.team_colors[2] = c1.copy()
            self._team_swapped = True
            print(
                "[TeamAssigner] Swapped team 1↔2 to match scoreboard stripes "
                f"(direct={direct:.1f}, cross={cross:.1f})",
                flush=True,
            )
            return True

        print(
            "[TeamAssigner] Jersey clusters already match scoreboard stripes "
            f"(direct={direct:.1f}, cross={cross:.1f})",
            flush=True,
        )
        return False

    # ── Per-frame player assignment ────────────────────────────────────────

    def get_player_team(
        self, frame: np.ndarray, player_bbox, player_id: int
    ) -> int:
        # Fast path: already locked
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self._kmeans is None:
            return 1

        lab = self._jersey_lab(frame, player_bbox)
        if lab is None:
            return 1

        team_id = int(self._kmeans.predict(lab.reshape(1, -1))[0]) + 1
        if self._team_swapped:
            team_id = 3 - team_id

        # Accumulate votes
        if player_id not in self._pending_votes:
            self._pending_votes[player_id] = []
        self._pending_votes[player_id].append(team_id)

        votes = self._pending_votes[player_id]
        if len(votes) >= self._N_VOTE_SAMPLES:
            majority = max(set(votes), key=votes.count)
            self.player_team_dict[player_id] = majority
            del self._pending_votes[player_id]
            return majority

        return team_id

    def finalize_pending(self) -> None:
        """
        Lock any player_ids that never accumulated _N_VOTE_SAMPLES frames
        (e.g. players who appeared only briefly).  Call this after the full
        frame loop to ensure every player_id has a stable team assignment.
        """
        for player_id, votes in list(self._pending_votes.items()):
            if votes:
                majority = max(set(votes), key=votes.count)
                self.player_team_dict[player_id] = majority
        self._pending_votes.clear()
