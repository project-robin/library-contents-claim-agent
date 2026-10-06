"""Multi-view reconstruction on a Modal GPU. The only code that touches torch.

Deploy once:  uv run modal deploy gpu/modal_recon.py
Input: JPEG keyframes already resized (long side 518, both sides multiples of 14).
Output (npz bytes): per-frame extrinsic (world->cam, OpenCV), intrinsic, z-depth, confidence, and `metric`.
Two models, chosen by RECON_MODEL after the geometry spike:
  mapanything (Apache-2.0 weights) - metric output; its own scale becomes an independent cross-check of the marker
  vggt                             - unscaled; the marker alone sets scale
The printed marker anchors scale in both cases.
"""
import io
import os

import modal

APP_NAME = "library-claim-recon"
GPU = os.environ.get("RECON_GPU", "L4")  # L40S/A100 need a card on Modal; L4 (24 GB) works on free credits

ma_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1")
    .pip_install("git+https://github.com/facebookresearch/map-anything.git")
    .run_commands("python -c \"from mapanything.models import MapAnything; "
                  "MapAnything.from_pretrained('facebook/map-anything-apache')\"")
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy<2", "pillow", "huggingface_hub", "einops", "safetensors")
    .pip_install("git+https://github.com/facebookresearch/vggt.git")
    .run_commands("python -c \"from vggt.models.vggt import VGGT; VGGT.from_pretrained('facebook/VGGT-1B')\"")
)
app = modal.App(APP_NAME)


@app.cls(gpu=GPU, image=image, timeout=900, scaledown_window=300)
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
            metric=False,
        )
        return buf.getvalue()


@app.cls(gpu=GPU, image=ma_image, timeout=900, scaledown_window=300)
class ReconMA:
    @modal.enter()
    def load(self):
        from mapanything.models import MapAnything
        self.model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()

    @modal.method()
    def ping(self):
        return "ok"

    @modal.method()
    def run(self, jpegs: list[bytes]) -> bytes:
        import tempfile
        from pathlib import Path

        import numpy as np
        from PIL import Image
        from mapanything.utils.image import load_images

        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, j in enumerate(jpegs):
                paths.append(str(Path(d) / f"{i:05d}.jpg"))
                Path(paths[-1]).write_bytes(j)
            w, h = Image.open(paths[0]).size
            views = load_images(paths, resize_mode="fixed_size", size=(w, h))  # inputs already 14-aligned: no crop
            preds = self.model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                                     apply_mask=True)
        c2w = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds])
        R, t = c2w[:, :3, :3], c2w[:, :3, 3]
        w2c = np.concatenate([R.transpose(0, 2, 1), -np.einsum("nji,nj->ni", R, t)[..., None]], axis=2)
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            extrinsic=w2c.astype(np.float32),
            intrinsic=np.stack([p["intrinsics"][0].float().cpu().numpy() for p in preds]),
            depth=np.stack([p["depth_z"][0, ..., 0].float().cpu().numpy() for p in preds]).astype(np.float16),
            conf=np.stack([p["conf"][0].float().cpu().numpy() * p["mask"][0, ..., 0].float().cpu().numpy()
                           for p in preds]).astype(np.float16),
            metric=True,
        )
        return buf.getvalue()


def remote(model="vggt"):
    return modal.Cls.from_name(APP_NAME, "ReconMA" if model == "mapanything" else "Recon")()
