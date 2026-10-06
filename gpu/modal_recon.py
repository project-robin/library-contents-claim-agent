"""Multi-view reconstruction on a Modal GPU. The only code that touches torch.

Deploy once:  uv run modal deploy gpu/modal_recon.py
Input: JPEG keyframes already resized (long side 518, both sides multiples of 14).
Output: per-frame camera extrinsic (world->cam, OpenCV), intrinsic, depth and confidence in VGGT units.
Metric scale is NOT taken from this model: pipeline/geometry.py anchors it with the printed marker.
"""
import io

import modal

APP_NAME = "library-claim-recon"
GPU = "L40S"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy<2", "pillow", "huggingface_hub", "einops", "safetensors")
    .pip_install("git+https://github.com/facebookresearch/vggt.git")
    .run_commands("python -c \"from vggt.models.vggt import VGGT; VGGT.from_pretrained('facebook/VGGT-1B')\"")
)
app = modal.App(APP_NAME, image=image)


@app.cls(gpu=GPU, timeout=900, scaledown_window=300)
class Recon:
    @modal.enter()
    def load(self):
        import torch
        from vggt.models.vggt import VGGT
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    @modal.method()
    def ping(self):  # called at sweep start to hide the cold start
        return "ok"

    @modal.method()
    def run(self, jpegs: list[bytes]) -> bytes:
        import numpy as np
        import torch
        from PIL import Image
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        imgs = [np.asarray(Image.open(io.BytesIO(j)).convert("RGB"), dtype=np.float32) / 255 for j in jpegs]
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to("cuda")  # S,3,H,W
        with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype):
            pred = self.model(x)
        extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], x.shape[-2:])
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            extrinsic=extr[0].float().cpu().numpy(),
            intrinsic=intr[0].float().cpu().numpy(),
            depth=pred["depth"][0, ..., 0].float().cpu().numpy().astype(np.float16),
            conf=pred["depth_conf"][0].float().cpu().numpy().astype(np.float16),
        )
        return buf.getvalue()


def remote():
    return modal.Cls.from_name(APP_NAME, "Recon")()
