from sklearn.cluster import KMeans
import numpy as np


class TeamAssigner:
    """
    Assigns players to teams (1 or 2) based on jersey colour clustering.

    Improvements over the original sticky-first-frame approach:
    - Collects up to _N_VOTE_SAMPLES colour predictions per player_id before
      locking the team, using majority vote.  This prevents a single noisy
      first-frame assignment from sticking permanently.
    - Once locked, subsequent frames use the cached value (O(1)).
    """

    _N_VOTE_SAMPLES = 7  # frames to collect before locking the team assignment

    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}       # locked: player_id -> team_id
        self._pending_votes: dict = {}   # player_id -> list[team_id] (pre-lock)

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[:image.shape[0] // 2, :]
        kmeans = self.get_clustering_model(top_half_image)

        labels = kmeans.labels_
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1])

        corner_clusters = [
            clustered_image[0, 0], clustered_image[0, -1],
            clustered_image[-1, 0], clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        self.kmeans.fit(player_colors)

        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # Fast path: already locked
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Compute colour prediction for this frame
        player_color = self.get_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1

        # Accumulate votes
        if player_id not in self._pending_votes:
            self._pending_votes[player_id] = []
        self._pending_votes[player_id].append(team_id)

        # Lock via majority vote once enough samples are collected
        votes = self._pending_votes[player_id]
        if len(votes) >= self._N_VOTE_SAMPLES:
            majority = max(set(votes), key=votes.count)
            self.player_team_dict[player_id] = majority
            del self._pending_votes[player_id]
            return majority

        return team_id

    def finalize_pending(self):
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
