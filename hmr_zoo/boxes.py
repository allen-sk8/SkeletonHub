"""Box post-processing shared by every model in hmr_zoo.

target_rule / repair_boxes / expand_boxes are copied verbatim from finefs_preproc/scripts/stage2_prepare_boxes.py
(Allen's FineFS Stage 2 comparison): choose the main skater's track by area * centrality, repair 1-4 frame
head-clip collapses, interpolate missing frames, then expand uniformly by 1.12x ("padded_112").
"""

from __future__ import annotations

import numpy as np

PADDING_SCALE = 1.12


def target_rule(tracks: np.ndarray, n: int, w: int, h: int) -> tuple[list[list[float] | None], list[int | None]]:
    """Select the primary skater track ID using Stage 1's area * centrality score."""
    if len(tracks) == 0:
        return [None] * n, [None] * n
    half_diag = 0.5 * np.hypot(w, h)
    score: dict[int, float] = {}
    frames: dict[int, set[int]] = {}
    for f, tid, x1, y1, x2, y2, _ in tracks:
        tid_i, f_i = int(tid), int(f)
        area = max(0.0, (x2 - x1) * (y2 - y1)) / max(w * h, 1)
        d = np.hypot((x1 + x2) / 2.0 - w / 2.0, (y1 + y2) / 2.0 - h / 2.0)
        score[tid_i] = score.get(tid_i, 0.0) + area * max(0.0, 1.0 - d / half_diag)
        frames.setdefault(tid_i, set()).add(f_i)

    order = sorted(score, key=lambda k: -score[k])
    top = score[order[0]]
    chosen = [order[0]]
    covered = set(frames[order[0]])
    for tid in order[1:]:
        if score[tid] < 0.1 * top:
            break
        if len(frames[tid] & covered) <= 5:
            chosen.append(tid)
            covered |= frames[tid]

    box: list[list[float] | None] = [None] * n
    ids: list[int | None] = [None] * n
    rank = {t: i for i, t in enumerate(chosen)}
    for f, tid, x1, y1, x2, y2, _ in tracks:
        f_i, tid_i = int(f), int(tid)
        if tid_i in rank and 0 <= f_i < n:
            if ids[f_i] is None or rank[tid_i] < rank[ids[f_i]]:
                box[f_i], ids[f_i] = [float(x1), float(y1), float(x2), float(y2)], tid_i
    return box, ids


def repair_boxes(raw_boxes: list[list[float] | None], w_img: int, h_img: int) -> tuple[list[list[float]], list[int], list[int]]:
    """Repair short top-clip collapses (1-4f) and interpolate missing frames."""
    n = len(raw_boxes)
    boxes = [list(b) if b is not None else None for b in raw_boxes]

    # Pass 1a: detect abrupt top-cliff runs (1-4 frames where y1 drops sharply while y2 stays steady)
    cliff_fixed: set[int] = set()
    i = 1
    while i < n:
        pre = boxes[i - 1]
        cur = boxes[i]
        if pre is None or cur is None:
            i += 1
            continue
        h_pre = pre[3] - pre[1]
        if h_pre > 1.0 and (cur[1] - pre[1]) >= 0.28 * h_pre and abs(cur[3] - pre[3]) <= 0.15 * h_pre:
            j = i
            while j < n and (j - i) < 5:
                bj = boxes[j]
                if bj is None:
                    break
                if (bj[3] - bj[1]) >= 0.78 * h_pre or (bj[1] - cur[1]) <= -0.14 * h_pre or abs(bj[3] - pre[3]) > 0.18 * h_pre:
                    break
                j += 1
            if j < n and boxes[j] is not None and j > i:
                post = boxes[j]
                h_post = post[3] - post[1]
                # Require recovery in height & box top relative to collapsed frames
                if (post[1] <= cur[1] - 0.10 * h_pre) and h_post >= 0.75 * h_pre:
                    for k in range(i, j):
                        alpha = (k - (i - 1)) / float(j - (i - 1))
                        bk = boxes[k]
                        target_h = (1.0 - alpha) * h_pre + alpha * h_post
                        new_y1 = max(0.0, min(bk[1], bk[3] - target_h, (1.0 - alpha) * pre[1] + alpha * post[1]))
                        cx = 0.5 * (bk[0] + bk[2])
                        wi = max(bk[2] - bk[0], 0.85 * min(pre[2] - pre[0], post[2] - post[0]))
                        boxes[k] = [
                            float(np.clip(cx - 0.5 * wi, 0.0, w_img - 1.0)),
                            float(new_y1),
                            float(np.clip(cx + 0.5 * wi, 1.0, float(w_img))),
                            float(np.clip(bk[3], 1.0, float(h_img))),
                        ]
                        cliff_fixed.add(k)
                    i = j
                    continue
        i += 1

    # Pass 1b: detect remaining brief downward height collapses vs surrounding median
    collapsed_set: set[int] = set(cliff_fixed)
    for i in range(n):
        if i in cliff_fixed:
            continue
        b = boxes[i]
        if b is None:
            continue
        nbrs = [boxes[j] for j in range(max(0, i - 6), min(n, i + 7)) if boxes[j] is not None and j != i]
        if len(nbrs) < 3:
            continue
        med_h = float(np.median([x[3] - x[1] for x in nbrs]))
        med_w = float(np.median([x[2] - x[0] for x in nbrs]))
        med_y2 = float(np.median([x[3] for x in nbrs]))
        hi = b[3] - b[1]
        if hi < 0.74 * med_h and abs(b[3] - med_y2) <= 0.18 * med_h:
            new_y1 = max(0.0, b[3] - med_h)
            cx = 0.5 * (b[0] + b[2])
            wi = max(b[2] - b[0], 0.85 * med_w)
            boxes[i] = [
                float(np.clip(cx - 0.5 * wi, 0.0, w_img - 1.0)),
                float(new_y1),
                float(np.clip(cx + 0.5 * wi, 1.0, float(w_img))),
                float(np.clip(b[3], 1.0, float(h_img))),
            ]
            collapsed_set.add(i)
    collapsed_idx = sorted(collapsed_set)

    # Pass 2: interpolate internal gaps, clamp leading/trailing gaps
    valid_idx = [i for i, b in enumerate(boxes) if b is not None]
    if not valid_idx:
        raise RuntimeError("No valid bounding boxes found in clip")
    gap_idx = [i for i, b in enumerate(boxes) if b is None]

    out: list[list[float]] = []
    for i in range(n):
        if boxes[i] is not None:
            out.append([float(v) for v in boxes[i]])
            continue
        prevs = [j for j in valid_idx if j < i]
        nexts = [j for j in valid_idx if j > i]
        if prevs and nexts:
            a, b_idx = prevs[-1], nexts[0]
            alpha = (i - a) / float(b_idx - a)
            interp = [(1.0 - alpha) * boxes[a][k] + alpha * boxes[b_idx][k] for k in range(4)]
            out.append(interp)
        elif nexts:
            out.append([float(v) for v in boxes[nexts[0]]])
        else:
            out.append([float(v) for v in boxes[prevs[-1]]])

    return out, collapsed_idx, gap_idx


def expand_boxes(boxes: list[list[float]], scale: float, w_img: int, h_img: int) -> list[list[float]]:
    """Expand xyxy boxes around center by `scale`, clamped within image bounds."""
    out = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        nw, nh = (x2 - x1) * scale, (y2 - y1) * scale
        nx1 = max(0.0, cx - 0.5 * nw)
        ny1 = max(0.0, cy - 0.5 * nh)
        nx2 = min(float(w_img), cx + 0.5 * nw)
        ny2 = min(float(h_img), cy + 0.5 * nh)
        out.append([round(nx1, 2), round(ny1, 2), round(nx2, 2), round(ny2, 2)])
    return out
