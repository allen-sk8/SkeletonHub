"""Pre-session enrollment ("上場前登錄"): decide, one pass (trip) at a time, whether a tracked person is the enrolled user.

Reads track_person.py output (tracks.txt, boxes.json) plus pose2d_frames.py keypoints; new file, existing code untouched.
Design: hq REPLAY_PLAN.md section 2, "上場前登錄：評估" (how the bank is stored, how it is matched).

  1. Enrollment video (the user turning in front of the camera, in the same camera view): the user = the track(s) chosen by
     track_person.py --target auto (largest, most central; boxes.json "raw_D_ids"). All their crop samples go into the USER bank,
     NOT averaged; each sample keeps its view label (front / side / back, from the shoulder order and width) and upper /
     lower body colour histograms. Everyone else in that video seeds the OTHERS bank (one entry per track).
  2. Test videos, in time order: a trip = one track of tracks.txt (use --tracker botsort-reid). Judged when the track ends
     (only the banks as they were at that moment are used, so it is causal across the session).
  3. Per trip up to 15 samples: confidence >= 0.5, not overlapping another box, not touching the frame edge, spread in time.
     Per sample: sU = mean of the 3 most similar USER samples of the same view (all views if < 5 of that view);
             sO = the most similar OTHER person (mean of that person's 3 most similar samples), 0 if the bank is empty;
             score = sU - sO (+ --color x (colour similarity to user - colour similarity to that other person)).
     Trip score = median. >= --t-yes: "user"; <= --t-no: "other"; else "unsure" (also when < 3 usable samples).
  4. Banks grow: a "user" trip adds its samples to USER; an "other" trip becomes a new person in OTHERS.
  Not implemented (from the design): path cues, "did this trip contain a jump", wave-to-confirm.

Features per track dir are cached in <track-dir>/enroll_feat.npz (GPU, two OSNet weights: sportsmot and dancetrack).
Run in the reidtrack env:

  python preprocessors/enroll_identity.py \
      --enroll-video E.mp4 --enroll-track-dir T/E --enroll-pose T/E/pose2d.npz \
      --video V1.mp4 --track-dir T/V1 --pose T/V1/pose2d.npz  [--video V2.mp4 --track-dir ... --pose ...] \
      --emb sportsmot --out OUT/judge.json

Output json: per video (key "<folder>_<file stem>") a list of trips {tid, start, end, n_samples, score, sU, sO, decision}; enrollment and bank sizes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

EMBS = ("osnet_x1_0_sportsmot", "osnet_x1_0_dancetrack")
TOPK, MAX_SAMPLES, MIN_SAMPLES = 3, 15, 3
SAMPLE_MIN_CONF, OCCL_IOU, EDGE_PX = 0.5, 0.05, 5


# ---------------------------------------------------------------- features
def iou_mat(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def view_of(k):
    """(view, upright): view 0 front / 1 side / 2 back / -1 unknown, from COCO-17 shoulders and hips."""
    ls, rs, lh, rh = k[5], k[6], k[11], k[12]
    if min(ls[2], rs[2], lh[2], rh[2]) < 0.3:
        return -1, False
    sh = (ls[:2] + rs[:2]) / 2; hp = (lh[:2] + rh[:2]) / 2
    torso = np.linalg.norm(sh - hp) + 1e-6
    upright = (hp[1] - sh[1]) / torso > 0.8              # torso within ~37 degrees of vertical
    width = (ls[0] - rs[0]) / torso                       # facing the camera: left shoulder on the image right -> > 0
    if abs(width) < 0.30:
        v = 1
    else:
        v = 0 if width > 0 else 2
        if v == 0 and k[0][2] < 0.3 and min(k[1][2], k[2][2]) < 0.3:   # "front" without a visible face: call it side
            v = 1
    return v, bool(upright)


def hsv_hist(img, box):
    """8 hue x 4 value histogram of a region, normalised."""
    h_, w_ = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w_), min(y2, h_)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return np.zeros(32, np.float32)
    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 2], None, [8, 4], [0, 180, 0, 256]).flatten()
    return (h / (h.sum() + 1e-9)).astype(np.float32)


def part_boxes(k, box):
    """Upper body (shoulders to hips) and lower body (hips to knees) from keypoints; box proportions if no keypoints."""
    if k is None or min(k[5][2], k[6][2], k[11][2], k[12][2]) < 0.3:
        hh = box[3] - box[1]
        return ((box[0], box[1] + 0.15 * hh, box[2], box[1] + 0.5 * hh),
                (box[0], box[1] + 0.5 * hh, box[2], box[1] + 0.8 * hh))
    xs = [k[i][0] for i in (5, 6, 11, 12)]
    pad = 0.15 * (max(xs) - min(xs) + 1)
    sy = min(k[5][1], k[6][1]); hy = (k[11][1] + k[12][1]) / 2
    ky = (k[13][1] + k[14][1]) / 2 if min(k[13][2], k[14][2]) > 0.3 else hy + 0.7 * (box[3] - hy)
    return (min(xs) - pad, sy, max(xs) + pad, hy), (min(xs) - pad, hy, max(xs) + pad, ky)


def extract_features(video: Path, track_dir: Path, pose_npz: Path, device: str, degrade: float = 0.0) -> dict:
    """Per tracked box on every frame that has keypoints: two OSNet embeddings, view, upright, occlusion, colours.
    degrade > 0: each frame is shrunk by this factor and enlarged back first (simulates a smaller, blurrier person)."""
    cache = track_dir / ("enroll_feat.npz" if not degrade else f"enroll_feat_degrade{degrade:g}.npz")
    if cache.exists():
        return dict(np.load(cache))
    import supervision as sv
    import torch
    from reid import ReIDModel

    tracks = np.loadtxt(track_dir / "tracks.txt", delimiter=",").reshape(-1, 7)
    pose = np.load(pose_npz)
    models = {k: ReIDModel.from_pretrained(k, device=device) for k in EMBS}
    pf, pb, pk = pose["frame"], pose["box"], pose["kpt"]
    pose_frames = set(range(0, int(pose["frames"]), int(pose["stride"])))
    by_frame: dict[int, list[int]] = {}
    for i, f in enumerate(tracks[:, 0].astype(int)):
        by_frame.setdefault(f, []).append(i)
    rows = {k: [] for k in ("frame", "tid", "box", "conf", "view", "upright", "occl", "h", "up", "lo")}
    embs = {k: [] for k in EMBS}
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n, t0 = 0, time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if n in pose_frames and n in by_frame:
            if degrade:
                img = cv2.resize(cv2.resize(img, (int(W * degrade), int(H * degrade)), interpolation=cv2.INTER_AREA), (W, H))
            idx = by_frame[n]
            tb = tracks[idx, 2:6]
            im = iou_mat(tb, tb); np.fill_diagonal(im, 0)
            occl = im.max(1) > OCCL_IOU if len(tb) > 1 else np.zeros(len(tb), bool)
            pm = pf == n
            kk, bb = pk[pm], pb[pm]
            match = iou_mat(tb, bb) if len(bb) else np.zeros((len(tb), 0))
            for j, i in enumerate(idx):
                k = kk[int(match[j].argmax())] if match.shape[1] and match[j].max() > 0.5 else None
                v, upr = view_of(k) if k is not None else (-1, False)
                up, lo = part_boxes(k, tb[j])
                rows["frame"].append(n); rows["tid"].append(int(tracks[i, 1])); rows["box"].append(tb[j])
                rows["conf"].append(tracks[i, 6]); rows["view"].append(v); rows["upright"].append(upr)
                rows["occl"].append(bool(occl[j])); rows["h"].append(tb[j][3] - tb[j][1])
                rows["up"].append(hsv_hist(img, up)); rows["lo"].append(hsv_hist(img, lo))
            dets = sv.Detections(xyxy=tb.astype(np.float32))
            for name, m in models.items():
                embs[name].append(m.extract_embeddings(dets, img))
        n += 1
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    out = {k: np.array(v) for k, v in rows.items()}
    for name in EMBS:
        e = np.concatenate(embs[name]) if embs[name] else np.zeros((0, 512), np.float32)
        out["emb_" + name.split("_")[-1]] = (e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
    out.update(wall_s=np.float32(time.time() - t0), frames=np.int32(n), width=np.int32(W), height=np.int32(H))
    np.savez(cache, **out)
    print(f"features {video.name}: {len(out['frame'])} samples, {float(out['wall_s']):.1f}s (incl. decoding, 2 OSNets)")
    return dict(np.load(cache))


# ---------------------------------------------------------------- matching
def usable(d):
    b, W, H = d["box"], int(d["width"]), int(d["height"])
    edge = (b[:, 0] > EDGE_PX) & (b[:, 2] < W - EDGE_PX) & (b[:, 1] > EDGE_PX) & (b[:, 3] < H - EDGE_PX)
    return (d["conf"] >= SAMPLE_MIN_CONF) & (~d["occl"]) & edge


def bhatt(a, b):
    return float(np.sum(np.sqrt(a * b)))


class Bank:
    def __init__(self):
        self.e, self.view, self.up, self.lo, self.key = [], [], [], [], []

    def add(self, d, idx, emb, key):
        self.e += list(d[emb][idx]); self.view += list(d["view"][idx])
        self.up += list(d["up"][idx]); self.lo += list(d["lo"][idx]); self.key += [key] * len(idx)

    def arrays(self):
        return np.array(self.e), np.array(self.view), np.array(self.up), np.array(self.lo), np.array(self.key)


def sample_score(user: Bank, others: Bank, e, v, up, lo, self_key, color_w, use_others):
    ue, uv, uup, ulo, _ = user.arrays()
    sel = (uv == v) if v >= 0 and (uv == v).sum() >= 5 else np.ones(len(ue), bool)
    s = ue[sel] @ e
    top = np.argsort(-s)[:TOPK]
    sU = float(s[top].mean())
    cU = float(np.mean([0.5 * (bhatt(up, uup[sel][i]) + bhatt(lo, ulo[sel][i])) for i in top]))
    sO = cO = 0.0
    if use_others and others.e:
        oe, _, oup, olo, okey = others.arrays()
        s = oe @ e
        best = -1.0
        for k in set(okey.tolist()):
            if k == self_key:
                continue
            mk = np.where(okey == k)[0]
            top = mk[np.argsort(-s[mk])[:TOPK]]
            sk = float(s[top].mean())
            if sk > best:
                best = sk
                cO = float(np.mean([0.5 * (bhatt(up, oup[i]) + bhatt(lo, olo[i])) for i in top]))
        sO = max(best, 0.0)
    return sU - sO + color_w * (cU - cO), sU, sO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enroll-video", required=True)
    ap.add_argument("--enroll-track-dir", required=True)
    ap.add_argument("--enroll-pose", required=True)
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument("--track-dir", action="append", default=[])
    ap.add_argument("--pose", action="append", default=[])
    ap.add_argument("--emb", default="sportsmot", choices=["sportsmot", "dancetrack"])
    ap.add_argument("--color", type=float, default=0.0, help="weight of the upper/lower body colour term (0 = off)")
    ap.add_argument("--t-yes", type=float, default=0.10)
    ap.add_argument("--t-no", type=float, default=0.0)
    ap.add_argument("--no-others", action="store_true", help="ablation: no OTHERS bank (sO = 0)")
    ap.add_argument("--no-grow", action="store_true", help="ablation: banks do not grow after decisions")
    ap.add_argument("--enroll-degrade", type=float, default=0.0,
                    help="experiment: shrink enrollment frames by this factor and enlarge back (0 = off), to imitate the "
                         "size and blur of people farther away when the enrollment was filmed close to the camera")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert len(a.video) == len(a.track_dir) == len(a.pose), "--video, --track-dir, --pose must come in triples"
    emb = "emb_" + a.emb
    t_start = time.time()

    # enrollment
    etd = Path(a.enroll_track_dir)
    d0 = extract_features(Path(a.enroll_video), etd, Path(a.enroll_pose), a.device, a.enroll_degrade)
    ids = json.loads((etd / "boxes.json").read_text()).get("raw_D_ids") or []
    user_ids = sorted({int(i) for i in ids if i is not None})
    g0 = usable(d0)
    user, others = Bank(), Bank()
    mu = np.where(g0 & np.isin(d0["tid"], user_ids))[0]
    user.add(d0, mu, emb, "enroll")
    for t in np.unique(d0["tid"][g0 & ~np.isin(d0["tid"], user_ids)]):
        others.add(d0, np.where(g0 & (d0["tid"] == t))[0], emb, f"enroll:{t}")
    views = d0["view"][mu]
    enroll = dict(user_track_ids=user_ids, user_samples=int(len(mu)),
                  user_views=dict(front=int((views == 0).sum()), side=int((views == 1).sum()),
                                  back=int((views == 2).sum()), unknown=int((views == -1).sum())),
                  others_people=len(set(others.key)), others_samples=len(others.e))

    result = dict(args=vars(a), enroll=enroll, videos={}, timing={})
    for vid, td, pz in zip(a.video, a.track_dir, a.pose):
        name = f"{Path(vid).parent.name}_{Path(vid).stem}"   # e.g. a2_v1 (folder_file), unique across albums
        d = extract_features(Path(vid), Path(td), Path(pz), a.device)
        tj = time.time()
        g = usable(d)
        trips = sorted((int(d["frame"][d["tid"] == t].max()), int(t)) for t in np.unique(d["tid"]))
        out = []
        for end, t in trips:                       # in the order the trips end
            mt = d["tid"] == t
            idx = np.where(mt & g)[0]
            if len(idx) > MAX_SAMPLES:
                idx = idx[np.linspace(0, len(idx) - 1, MAX_SAMPLES).round().astype(int)]
            rec = dict(tid=t, start=int(d["frame"][mt].min()), end=end, n_samples=int(len(idx)),
                       h_med=round(float(np.median(d["h"][mt])), 1))
            if len(idx) < MIN_SAMPLES:
                rec.update(score=None, decision="unsure", reason="few_samples")
            else:
                key = f"{name}:{t}"
                ss = [sample_score(user, others, d[emb][i], int(d["view"][i]), d["up"][i], d["lo"][i], key,
                                   a.color, not a.no_others) for i in idx]
                sc = float(np.median([s[0] for s in ss]))
                rec.update(score=round(sc, 4), sU=round(float(np.median([s[1] for s in ss])), 4),
                           sO=round(float(np.median([s[2] for s in ss])), 4),
                           decision="user" if sc >= a.t_yes else ("other" if sc <= a.t_no else "unsure"))
                if not a.no_grow:
                    if rec["decision"] == "user":
                        user.add(d, idx, emb, key)
                    elif rec["decision"] == "other":
                        others.add(d, idx, emb, key)
            out.append(rec)
        result["videos"][name] = out
        result["timing"][name] = dict(features_s=round(float(d["wall_s"]), 1), judge_s=round(time.time() - tj, 2),
                                      frames=int(d["frames"]))
    result["final_banks"] = dict(user_samples=len(user.e), others_people=len(set(others.key)), others_samples=len(others.e))
    result["total_s"] = round(time.time() - t_start, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(dict(enroll=enroll, final=result["final_banks"],
                          decisions={k: {c: sum(r["decision"] == c for r in v) for c in ("user", "unsure", "other")}
                                     for k, v in result["videos"].items()})))


if __name__ == "__main__":
    main()
