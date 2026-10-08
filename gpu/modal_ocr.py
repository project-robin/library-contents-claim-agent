"""Spine OCR on Modal CPU: PP-OCRv5 server detector + recognizer (PaddleOCR 3.7, Apache-2.0).

Deploy once:  uv run modal deploy gpu/modal_ocr.py
Input: JPEG spine crops. Output per crop: words read at each rotation, and the rotation that read best.
A detector + recognizer cannot invent a title the way a VLM can, so this is the pipeline's independent witness of
what is printed. Chosen by a bake-off on two labelled shelf photos (book_sale/eval_kit/ocr_bakeoff): best title-word
recall of six engines (86% / 49%), ahead of docTR+PARSeq, OpenOCR, EasyOCR, RapidOCR; OvisOCR2 and TeleOCR failed.
"""
import modal

APP_NAME = "library-claim-ocr"
MODELS = dict(text_detection_model_name="PP-OCRv5_server_det", text_recognition_model_name="PP-OCRv5_server_rec",
              use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install("paddlepaddle==3.2.0", "paddleocr==3.7.0", "opencv-python-headless", "numpy<2.3")
    .run_commands(f"python -c \"from paddleocr import PaddleOCR; PaddleOCR(**{MODELS!r})\"")  # bake weights in
)
app = modal.App(APP_NAME)


@app.cls(image=image, cpu=4.0, memory=8192, timeout=900, scaledown_window=300)
class SpineOCR:
    @modal.enter()
    def load(self):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(**MODELS)

    @modal.method()
    def ping(self):  # called at sweep start to hide the cold start
        return "ok"

    def _words(self, img):
        out = []
        for res in self.ocr.predict(img):
            j = res.json.get("res", res.json)
            out += [[t, round(float(s), 3)] for t, s in zip(j["rec_texts"], j["rec_scores"]) if t.strip()]
        return out

    @modal.method()
    def read(self, crops: list[bytes]) -> list[dict]:
        """Spine text runs along the spine, so read both 90-degree turns (and upright for crops wider than tall,
        i.e. books lying flat). Paddle's own line classifier only knows 0/180. Words from every turn are kept, tagged
        with it: picking one turn by confidence chose an upside-down read ('CTIHO MOE') over 'TO… ROW'."""
        import cv2
        import numpy as np
        rot = {90: cv2.ROTATE_90_CLOCKWISE, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        out = []
        for data in crops:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                out.append({"words": [], "rotation": None})
                continue
            h, w = img.shape[:2]
            if min(h, w) < 128:  # a spine is ~75 px wide in a shelf frame: upscale so the letters are big enough
                f = 128 / min(h, w)  # (at native size OCR missed 'EVA' that it reads on the full photo)
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            tries = [(d, cv2.rotate(img, r)) for d, r in rot.items()] + ([(0, img)] if w > h else [])
            reads = [(d, self._words(im)) for d, im in tries]
            best = max(reads, key=lambda x: sum(s * len(t) for t, s in x[1]))[0]
            out.append({"words": [[t, s, d] for d, ws in reads for t, s in ws], "rotation": best})
        return out


def remote():
    return modal.Cls.from_name(APP_NAME, "SpineOCR")()
