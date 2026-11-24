# FineFS v2 stage 6: skater on the ice (spec, confirmed with Allen 2026-11-24)

Adds to the GVHMR 3D body (3_gvhmr) what GVHMR gets wrong for broadcast figure skating:
the body put back on the ice, jump height, and travel over the ice. CPU only.

## What changes and what does not

- Body pose (joint angles, body shape, orientation) stays exactly GVHMR's.
- Only the whole-body position changes, frame by frame. So every variant is stored as a per-frame
  offset (T, 3) in metres: corrected joints = 3_gvhmr joints24_world + offset; corrected SMPL transl =
  smpl_world_transl + offset. Nothing from 3_gvhmr is copied.
- Coordinates: GVHMR world of each segment (Y up, metres), segments still separate. After correction the ice is
  at y = 0 in every segment. The horizontal path starts at GVHMR's pelvis position at the segment start.

## Variants (both kept, chosen at use time; also for ablation)

| name | vertical | horizontal |
|---|---|---|
| `rotation` | feet on the ice + flight from body rotation, parabola h = g T^2 / 8 (typical flight time by revolutions when rotation is unclear) | travel measured from the image |
| `image` | feet on the ice + flight and height measured in the image (ankle lift above the ice line); jumps where the image measurement fails use the `rotation` flight | travel measured from the image |

Both variants: no foot below the ice. Travel: background homographies, pixels to metres with GVHMR depth,
camera elevation fixed at 20 degrees, speed capped at 12 m/s.

## Files per video, `6_world/<id>/`

- `world.npz`
  - `offset_rotation`, `offset_image` (T, 3) float32; NaN where 3_gvhmr has no 3D
  - `ground_dy` (T,) the re-grounding part alone; `travel_xz` (T, 2) the horizontal path alone
  - `speed` (T,) m/s over the ice; `speed_capped` (T,) bool
  - `homography` (T, 3, 3) background motion from frame t-1 to t (full-resolution pixels), NaN where it failed;
    `homography_inliers` (T,) - camera pan / tilt / zoom, reusable
  - `meta_json`: settings (elevation, caps, thresholds), code version
- `jumps.json`: one entry per jump part (a combination like 3Lz+3T has two): element, revolutions, GOE, annotated time;
  rotation method (flight frames, measured or typical); image method (take-off, landing, flight time, apex height,
  height from flight time, ok or why not, and whether height and flight time agree within 15 cm).

## Rules

- Jumps come from the FineFS annotation (element names and times). Jumps without annotation are not found.
- A measurement that fails is recorded as failed with the reason; never silently replaced. The `image` variant's
  fallback to `rotation` is marked per jump in jumps.json.
- Runs as a separate CPU job next to the GPU batch, on videos whose stages 1-5 are finished; resumable; can be rerun
  with new settings without touching stages 1-5.
- Preview videos: not for every video (about 1 min of CPU per 6 s clip).

## Decisions (Allen, 2026-11-24)

1. Image method fails on a jump: the `image` variant uses the rotation flight for that jump, marked in jumps.json.
2. Stage 6 runs in parallel with the GPU batch (each video as soon as its stages 1-5 are done), about 12 CPU cores.
3. Preview videos: a sample, one jump from every 20th video, kept in SkeletonHub (outputs/finefs_v2_stage6_previews),
   not in the dataset.
4. The GPU batch resumes with the earlier settings (neverdie card 5 if gpufree says it is free, no stop when a
   colleague's process appears, move to an empty card on out-of-memory).
