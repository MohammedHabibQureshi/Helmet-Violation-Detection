"""OCR engine wrappers for the Helmet Violation Detection project.

Provides EasyOCR, PaddleOCR, Tesseract, and a NullOCR fallback, all with
a consistent interface. Text normalization and character correction are
applied uniformly.
"""

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    """Single OCR recognition result."""

    text_raw: str = ""
    text_normalized: str = ""
    text_corrected: str = ""
    confidence: float = 0.0
    latency_ms: float = 0.0
    engine_name: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Strip whitespace and uppercase."""
    return text.strip().upper()


def apply_character_corrections(text: str) -> str:
    """Apply context-aware character corrections for plate text.

    Rules (applied left-to-right on the normalized string):
    - In the numeric portion (digits expected): O→0, I→1, Z→2, S→5, B→8
    - In the alpha portion (letters expected):  0→O, 1→I

    Heuristic: the first two characters of an Indian plate are letters
    (state code), then two digits (district code), then 1–2 letters (series),
    then 4 digits (unique number).  We use a regex-based pattern.
    """
    if not text:
        return text

    # General letter→digit substitutions for positions that look numeric
    # (runs of digits mixed with common misreads)
    corrected = list(text)

    def _looks_numeric_context(chars: list, idx: int) -> bool:
        """Return True if the character at idx is in a mostly-digit run."""
        window = [c for c in chars[max(0, idx - 1): idx + 2] if c != corrected[idx]]
        return sum(1 for c in window if c.isdigit()) >= 1

    for i, ch in enumerate(corrected):
        if ch == "O" and _looks_numeric_context(corrected, i):
            corrected[i] = "0"
        elif ch == "I" and _looks_numeric_context(corrected, i):
            corrected[i] = "1"
        elif ch == "Z" and _looks_numeric_context(corrected, i):
            corrected[i] = "2"
        elif ch == "S" and _looks_numeric_context(corrected, i):
            corrected[i] = "5"
        elif ch == "B" and _looks_numeric_context(corrected, i):
            corrected[i] = "8"
        elif ch == "0" and not _looks_numeric_context(corrected, i):
            corrected[i] = "O"
        elif ch == "1" and not _looks_numeric_context(corrected, i):
            corrected[i] = "I"

    return "".join(corrected)


def make_result(
    text_raw: str,
    confidence: float,
    latency_ms: float,
    engine_name: str,
    error: Optional[str] = None,
) -> OCRResult:
    """Build a fully populated OCRResult."""
    norm = normalize_text(text_raw)
    corrected = apply_character_corrections(norm)
    return OCRResult(
        text_raw=text_raw,
        text_normalized=norm,
        text_corrected=corrected,
        confidence=round(confidence, 4),
        latency_ms=round(latency_ms, 3),
        engine_name=engine_name,
        error=error,
    )


# ---------------------------------------------------------------------------
# Engine wrappers
# ---------------------------------------------------------------------------

class NullOCREngine:
    """Fallback engine — returns an empty result. Always available."""

    name = "null"
    available = True

    def recognize(self, image: np.ndarray) -> OCRResult:
        return make_result("", 0.0, 0.0, self.name, error="null_engine")


class EasyOCREngine:
    """Wraps easyocr.Reader for plate recognition."""

    name = "easyocr"

    def __init__(self, languages: list[str] = None, gpu: bool = False):
        try:
            import easyocr

            self._reader = easyocr.Reader(
                languages or ["en"], gpu=gpu, verbose=False
            )
            self.available = True
        except ImportError:
            self._reader = None
            self.available = False
        except Exception as exc:
            self._reader = None
            self.available = False
            self._init_error = str(exc)

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self.available or self._reader is None:
            return make_result(
                "", 0.0, 0.0, self.name, error="easyocr_unavailable"
            )
        t0 = time.perf_counter()
        try:
            results = self._reader.readtext(image, detail=1)
            latency_ms = (time.perf_counter() - t0) * 1000
            if not results:
                return make_result("", 0.0, latency_ms, self.name)
            # Pick the result with highest confidence
            best = max(results, key=lambda r: r[2])
            text_raw = best[1]
            confidence = float(best[2])
            raw_output = [{"bbox": r[0], "text": r[1], "conf": r[2]}
                          for r in results]
            result = make_result(text_raw, confidence, latency_ms, self.name)
            return result
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return make_result("", 0.0, latency_ms, self.name, error=str(exc))


class PaddleOCREngine:
    """Wraps PaddleOCR for plate recognition."""

    name = "paddleocr"

    def __init__(self, use_gpu: bool = False):
        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=use_gpu,
                show_log=False,
            )
            self.available = True
        except ImportError:
            self._ocr = None
            self.available = False
        except Exception as exc:
            self._ocr = None
            self.available = False
            self._init_error = str(exc)

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self.available or self._ocr is None:
            return make_result(
                "", 0.0, 0.0, self.name, error="paddleocr_unavailable"
            )
        t0 = time.perf_counter()
        try:
            results = self._ocr.ocr(image, cls=True)
            latency_ms = (time.perf_counter() - t0) * 1000
            # results is a list of lists: [[[[x,y],...], (text, conf)], ...]
            if not results or not results[0]:
                return make_result("", 0.0, latency_ms, self.name)
            flat = [item for sublist in results for item in (sublist or [])]
            if not flat:
                return make_result("", 0.0, latency_ms, self.name)
            best = max(flat, key=lambda r: r[1][1] if r and len(r) > 1 else 0)
            text_raw = best[1][0] if best and len(best) > 1 else ""
            confidence = float(best[1][1]) if best and len(best) > 1 else 0.0
            return make_result(text_raw, confidence, latency_ms, self.name)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return make_result("", 0.0, latency_ms, self.name, error=str(exc))


class TesseractEngine:
    """Wraps pytesseract for plate recognition."""

    name = "tesseract"

    def __init__(self, config: str = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
        try:
            import pytesseract

            self._pytesseract = pytesseract
            # Quick sanity check
            pytesseract.get_tesseract_version()
            self.available = True
            self._config = config
        except ImportError:
            self._pytesseract = None
            self.available = False
            self._config = config
        except Exception:
            self._pytesseract = None
            self.available = False
            self._config = config

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self.available or self._pytesseract is None:
            return make_result(
                "", 0.0, 0.0, self.name, error="tesseract_unavailable"
            )
        import cv2 as _cv2
        from PIL import Image as _Image

        t0 = time.perf_counter()
        try:
            # Tesseract expects RGB; convert from BGR/Gray
            if len(image.shape) == 3 and image.shape[2] == 3:
                img_rgb = _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image
            pil_img = _Image.fromarray(img_rgb)
            data = self._pytesseract.image_to_data(
                pil_img, config=self._config,
                output_type=self._pytesseract.Output.DICT,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            # Build best text from highest-confidence word
            texts = []
            confs = []
            for txt, conf in zip(data["text"], data["conf"]):
                txt = txt.strip()
                try:
                    cf = float(conf)
                except ValueError:
                    cf = -1.0
                if txt and cf > 0:
                    texts.append(txt)
                    confs.append(cf)
            if not texts:
                return make_result("", 0.0, latency_ms, self.name)
            text_raw = " ".join(texts)
            confidence = max(confs) / 100.0  # tesseract returns 0-100
            return make_result(text_raw, confidence, latency_ms, self.name)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return make_result("", 0.0, latency_ms, self.name, error=str(exc))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_available_engines(gpu: bool = False) -> dict[str, object]:
    """Return a dict of instantiated, available OCR engines.

    Keys are engine names ('easyocr', 'paddleocr', 'tesseract'). Engines
    that fail to import are omitted from the result.
    """
    candidates = {
        "easyocr": EasyOCREngine(gpu=gpu),
        "paddleocr": PaddleOCREngine(use_gpu=gpu),
        "tesseract": TesseractEngine(),
    }
    return {name: eng for name, eng in candidates.items() if eng.available}


def main() -> int:
    """Smoke-test all available engines on a synthetic plate image."""
    import cv2

    # Create a small synthetic plate-like image with text drawn
    img = np.ones((60, 200, 3), dtype=np.uint8) * 220
    cv2.putText(img, "MH12AB1234", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    engines = get_available_engines(gpu=False)
    if not engines:
        print("No OCR engines available.")
        return 1

    for name, engine in engines.items():
        result = engine.recognize(img)
        print(f"[{name}] raw={result.text_raw!r:20s}  "
              f"corrected={result.text_corrected!r:20s}  "
              f"conf={result.confidence:.3f}  "
              f"latency={result.latency_ms:.1f}ms  "
              f"error={result.error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
