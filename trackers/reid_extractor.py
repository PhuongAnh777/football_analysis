"""
Deep appearance feature extractor for player Re-Identification.

Uses MobileNetV3-Small (pretrained ImageNet) as a lightweight backbone.
Produces 576-dim L2-normalised feature vectors — far more discriminative
than colour histograms for players wearing the same team jersey.

Typical usage
-------------
    extractor = DeepAppearanceExtractor()          # init once
    feat = extractor.extract_one(frame_bgr, bbox)  # → (576,) ndarray or None
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class DeepAppearanceExtractor:
    """Extract deep appearance features from player bounding-box crops."""

    # Standard ReID input resolution (H × W)
    _H, _W = 128, 64

    def __init__(self, device: str = "auto"):
        """
        Parameters
        ----------
        device : str
            ``"auto"`` picks CUDA when available, otherwise CPU.
            Pass ``"cpu"`` or ``"cuda"`` to force a device.
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        # Keep only feature extractor + global average pool; discard classifier.
        self.model = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
        )
        self.model.eval().to(device)

        self._transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    # ── internal helpers ──────────────────────────────────────────────────────

    def _crop_jersey(self, frame_bgr: np.ndarray, bbox: list) -> np.ndarray | None:
        """
        Return the upper-55 % of a player crop (jersey region) resized to
        (_H, _W) as an RGB uint8 array, or None when the crop is too small.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h = y2 - y1
        crop = frame_bgr[y1 : y1 + max(1, int(h * 0.55)), x1:x2]
        if crop.size < 100:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (self._W, self._H))

    @staticmethod
    def _l2_normalise(feats: np.ndarray) -> np.ndarray:
        """L2-normalise a 1-D or 2-D (N, D) array row-wise."""
        if feats.ndim == 1:
            n = np.linalg.norm(feats)
            return feats / n if n > 1e-9 else feats
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return feats / norms

    # ── public API ────────────────────────────────────────────────────────────

    def extract_one(self, frame_bgr: np.ndarray, bbox: list) -> np.ndarray | None:
        """
        Extract a 576-dim feature vector for a single player bbox.

        Returns None when the crop is too small to be reliable.
        """
        crop = self._crop_jersey(frame_bgr, bbox)
        if crop is None:
            return None
        tensor = self._transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor).squeeze(0).cpu().numpy()
        return self._l2_normalise(feat)

    def extract_frame(
        self, frame_bgr: np.ndarray, bboxes: list[list]
    ) -> list[np.ndarray | None]:
        """
        Batch-extract features for all bboxes in a single frame.
        More efficient than calling ``extract_one`` in a loop.

        Parameters
        ----------
        frame_bgr : np.ndarray  –  full BGR video frame
        bboxes    : list of [x1, y1, x2, y2] bounding boxes

        Returns
        -------
        List of (576,) float32 arrays, or None for invalid crops.
        Same length and order as ``bboxes``.
        """
        crops = [self._crop_jersey(frame_bgr, bb) for bb in bboxes]

        tensors, valid_idx = [], []
        for i, crop in enumerate(crops):
            if crop is not None:
                tensors.append(self._transform(crop))
                valid_idx.append(i)

        results: list[np.ndarray | None] = [None] * len(bboxes)
        if not tensors:
            return results

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self.model(batch).cpu().numpy()
        feats = self._l2_normalise(feats)

        for j, idx in enumerate(valid_idx):
            results[idx] = feats[j]

        return results
