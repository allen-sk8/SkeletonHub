"""General-purpose image embeddings of person crops (DINOv2, CLIP, ...), for appearance tests beyond OSNet.

New file; existing code untouched. Takes a video plus a list of (frame, box) and returns one L2-normalised vector per box.
Each crop: box padded by --pad on every side, letterboxed onto a mid-grey square (keeps the body shape), resized to the
model's input size, normalised with the model's own mean/std.

Backbones (Hugging Face ids; downloaded to the HF cache on first use):
  dinov2_s / dinov2_b / dinov2_l   facebook/dinov2-{small,base,large}   -> two vectors: CLS token ("cls") and mean of patch tokens ("mean")
  clip_b16 / clip_l14              openai/clip-vit-{base-patch16,large-patch14} -> projected image embedding ("img")
  siglip_b                         google/siglip-base-patch16-224        -> pooled image embedding ("img")

Use as a library:
    from preprocessors.crop_embed import embed_crops
    out = embed_crops("v.mp4", frames, boxes, "dinov2_b")   # {"cls": (N, D), "mean": (N, D)}
or from the command line on enroll_identity.py's sample cache (rows = frame, box of <track-dir>/enroll_feat.npz):
    python preprocessors/crop_embed.py --video V.mp4 --rows T/enroll_feat.npz --backbone dinov2_b clip_l14 --out T/crop_embed.npz
Output npz keys: "<backbone>_<vector>" (float16, normalised) plus "<backbone>_wall_s" (decode + crop + model, without loading). Rows keep the order of the input.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

BACKBONES = {
    "dinov2_s": ("dinov2", "facebook/dinov2-small", 224),
    "dinov2_b": ("dinov2", "facebook/dinov2-base", 224),
    "dinov2_l": ("dinov2", "facebook/dinov2-large", 224),
    "clip_b16": ("clip", "openai/clip-vit-base-patch16", 224),
    "clip_l14": ("clip", "openai/clip-vit-large-patch14", 224),
    "siglip_b": ("siglip", "google/siglip-base-patch16-224", 224),
}


def letterbox_crop(img, box, size, pad=0.1, gray=False):
    """Box padded by `pad` of its size, pasted centred on a grey square, resized to size x size (RGB uint8).
    gray=True: the crop is turned to grey levels first (3 equal channels), to measure what is left without clothing colour."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    pw, ph = pad * (x2 - x1), pad * (y2 - y1)
    x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    x2, y2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
    c = img[y1:max(y2, y1 + 2), x1:max(x2, x1 + 2)]
    if gray:
        c = cv2.cvtColor(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    s = max(c.shape[:2])
    sq = np.full((s, s, 3), 124, np.uint8)
    oy, ox = (s - c.shape[0]) // 2, (s - c.shape[1]) // 2
    sq[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return cv2.cvtColor(cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC),
                        cv2.COLOR_BGR2RGB)


class Embedder:
    def __init__(self, name: str, device: str = "cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.kind, hf, self.size = BACKBONES[name]
        self.torch, self.device = torch, device
        proc = AutoImageProcessor.from_pretrained(hf)
        self.mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(hf, dtype=dtype).to(device).eval()
        self.dtype = dtype

    def __call__(self, crops: list[np.ndarray]) -> dict[str, np.ndarray]:
        torch = self.torch
        x = torch.from_numpy(np.stack(crops)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - self.mean) / self.std).to(self.dtype)
        with torch.no_grad():
            if self.kind == "dinov2":
                h = self.model(pixel_values=x).last_hidden_state
                out = {"cls": h[:, 0], "mean": h[:, 1:].mean(1)}
            else:
                f = self.model.get_image_features(pixel_values=x)
                if not torch.is_tensor(f):              # newer transformers may return a model output
                    f = getattr(f, "image_embeds", None) if getattr(f, "image_embeds", None) is not None else f.pooler_output
                out = {"img": f}
        return {k: torch.nn.functional.normalize(v.float(), dim=1).cpu().numpy().astype(np.float16) for k, v in out.items()}


def embed_crops(video, frames, boxes, backbone: str, device: str = "cuda", pad: float = 0.1, batch: int = 64,
                embedder: Embedder | None = None, gray: bool = False, frame_fn=None) -> dict[str, np.ndarray]:
    """One normalised vector per (frame, box), in input order. Decodes the video once, front to back.
    gray: grey-level crops. frame_fn(img) -> img: applied to each needed frame before cropping
    (e.g. enroll_identity.degrade_frame, to simulate a smaller, blurrier person)."""
    frames = np.asarray(frames).astype(int)
    boxes = np.asarray(boxes, dtype=np.float32)
    emb = embedder or Embedder(backbone, device)
    order = np.argsort(frames, kind="stable")
    res: dict[str, np.ndarray] = {}
    buf, buf_idx = [], []

    def flush():
        if not buf:
            return
        o = emb(buf)
        for k, v in o.items():
            if k not in res:
                res[k] = np.zeros((len(frames), v.shape[1]), np.float16)
            res[k][buf_idx] = v
        buf.clear(); buf_idx.clear()

    cap = cv2.VideoCapture(str(video))
    n, p = 0, 0
    while p < len(order):
        ok, img = cap.read()
        if not ok:
            break
        if frame_fn is not None and p < len(order) and frames[order[p]] == n:
            img = frame_fn(img)
        while p < len(order) and frames[order[p]] == n:
            i = order[p]
            buf.append(letterbox_crop(img, boxes[i], emb.size, pad, gray)); buf_idx.append(i)
            p += 1
            if len(buf) >= batch:
                flush()
        n += 1
    flush()
    cap.release()
    assert p == len(order), f"video ended at frame {n} before all rows were embedded ({p}/{len(order)})"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--rows", required=True, help="npz with 'frame' (N,) and 'box' (N, 4) xyxy, e.g. enroll_feat.npz")
    ap.add_argument("--backbone", nargs="+", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--pad", type=float, default=0.1)
    ap.add_argument("--gray", action="store_true", help="grey-level crops (no clothing colour); keys <backbone>_gray_<vector>")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = np.load(a.rows)
    out = dict(np.load(a.out)) if Path(a.out).exists() else {}
    for b0 in a.backbone:
        b = b0 + "_gray" if a.gray else b0
        if any(k.startswith(b + "_") and not k.endswith("wall_s") and (a.gray or not k.startswith(b0 + "_gray_")) for k in out):
            print(f"{b}: cached in {a.out}")
            continue
        e = Embedder(b0, a.device)                      # model load not counted in the timing
        t0 = time.time()
        r = embed_crops(a.video, rows["frame"], rows["box"], b0, a.device, a.pad, embedder=e, gray=a.gray)
        out.update({f"{b}_{k}": v for k, v in r.items()})
        out[f"{b}_wall_s"] = np.float32(time.time() - t0)
        print(f"{b}: {len(rows['frame'])} crops, {time.time() - t0:.1f}s (incl. decoding)")
        np.savez(a.out, **out)


if __name__ == "__main__":
    main()
