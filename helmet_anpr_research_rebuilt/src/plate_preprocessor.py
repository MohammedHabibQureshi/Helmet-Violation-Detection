"""Number plate image preprocessing pipelines.

Provides four preprocessing pipelines (A-D) that prepare cropped plate
images for OCR ingestion.
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class PlatePreprocessor:
    """Preprocessing class for number plate crops.

    Provides static methods for four preprocessing pipelines, plus a
    utility for safely extracting a padded crop from a full frame.
    """

    @staticmethod
    def extract_plate_crop(
        frame: np.ndarray,
        bbox: list | tuple,
        padding_frac: float = 0.1,
    ) -> Optional[np.ndarray]:
        """Extract a padded plate crop from a full frame.

        Parameters
        ----------
        frame:
            Full BGR frame.
        bbox:
            Bounding box as [x1, y1, x2, y2] in pixel coordinates.
        padding_frac:
            Fractional padding applied to each side.

        Returns
        -------
        np.ndarray or None
            Cropped BGR image, or None if the crop area is degenerate.
        """
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return None

        pad_x = bw * padding_frac
        pad_y = bh * padding_frac
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))

        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        return crop

    @staticmethod
    def preprocess_pipeline_A(crop: np.ndarray) -> dict:
        """Pipeline A: original crop, no modification.

        Returns
        -------
        dict
            {pipeline_name, image}
        """
        if crop is None or crop.size == 0:
            return {"pipeline_name": "A", "image": None,
                    "description": "original (invalid crop)"}
        return {
            "pipeline_name": "A",
            "image": crop.copy(),
            "description": "original_crop",
        }

    @staticmethod
    def preprocess_pipeline_B(crop: np.ndarray) -> dict:
        """Pipeline B: bicubic 2x enlargement.

        Returns
        -------
        dict
            {pipeline_name, image}
        """
        if crop is None or crop.size == 0:
            return {"pipeline_name": "B", "image": None,
                    "description": "2x_bicubic (invalid crop)"}
        h, w = crop.shape[:2]
        enlarged = cv2.resize(
            crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC
        )
        return {
            "pipeline_name": "B",
            "image": enlarged,
            "description": "bicubic_2x",
        }

    @staticmethod
    def preprocess_pipeline_C(crop: np.ndarray) -> dict:
        """Pipeline C: bicubic 2x + grayscale conversion.

        Returns
        -------
        dict
            {pipeline_name, image}
        """
        if crop is None or crop.size == 0:
            return {"pipeline_name": "C", "image": None,
                    "description": "2x_bicubic_gray (invalid crop)"}
        h, w = crop.shape[:2]
        enlarged = cv2.resize(
            crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC
        )
        if len(enlarged.shape) == 3:
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        else:
            gray = enlarged
        return {
            "pipeline_name": "C",
            "image": gray,
            "description": "bicubic_2x_grayscale",
        }

    @staticmethod
    def preprocess_pipeline_D(crop: np.ndarray) -> dict:
        """Pipeline D: bicubic 2x + grayscale + CLAHE contrast enhancement.

        Returns
        -------
        dict
            {pipeline_name, image}
        """
        if crop is None or crop.size == 0:
            return {"pipeline_name": "D", "image": None,
                    "description": "2x_bicubic_gray_clahe (invalid crop)"}
        h, w = crop.shape[:2]
        enlarged = cv2.resize(
            crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC
        )
        if len(enlarged.shape) == 3:
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        else:
            gray = enlarged
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return {
            "pipeline_name": "D",
            "image": enhanced,
            "description": "bicubic_2x_grayscale_clahe",
        }

    def run_all_pipelines(self, crop: np.ndarray) -> list[dict]:
        """Run all four pipelines on a crop.

        Returns
        -------
        list of dict
            One entry per pipeline (A, B, C, D).
        """
        return [
            self.preprocess_pipeline_A(crop),
            self.preprocess_pipeline_B(crop),
            self.preprocess_pipeline_C(crop),
            self.preprocess_pipeline_D(crop),
        ]

    def run_pipeline(self, crop: np.ndarray, name: str) -> dict:
        """Run a single named pipeline.

        Parameters
        ----------
        crop:
            BGR crop image.
        name:
            Pipeline name: 'A', 'B', 'C', or 'D'.
        """
        name = name.upper()
        if name == "A":
            return self.preprocess_pipeline_A(crop)
        if name == "B":
            return self.preprocess_pipeline_B(crop)
        if name == "C":
            return self.preprocess_pipeline_C(crop)
        if name == "D":
            return self.preprocess_pipeline_D(crop)
        raise ValueError(f"Unknown pipeline name: {name!r}. Must be A, B, C, or D.")


def main() -> int:
    """Smoke-test with a synthetic crop."""
    crop = np.random.randint(0, 256, (40, 120, 3), dtype=np.uint8)
    proc = PlatePreprocessor()
    results = proc.run_all_pipelines(crop)
    for r in results:
        img = r.get("image")
        shape = img.shape if img is not None else None
        print(f"Pipeline {r['pipeline_name']}: {r['description']} "
              f"-> shape={shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
