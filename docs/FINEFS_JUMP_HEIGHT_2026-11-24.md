# Image-based jump height, first test (2026-11-24, 40 finished FineFS videos, 174 jump parts)

Script: scripts/finefs_jump_image.py (method "line" is the default; "contact" was tried and is worse).
Files: jumps_40_line.jsonl, jumps_40_contact.jsonl (one JSON line per jump part; ok = measurement passed the checks).

| | rotation method (finefs_jump_height.py) | image method "line" | image method "contact" |
|---|---|---|---|
| measured (not the typical value) | 59% | 86% | 82% |
| height and flight time agree within 15 cm | - | 59% of all | 26% of all |
| triple jumps: height / flight | 50 cm (typical) / 0.62 s | 51 ± 19 cm / 0.63 s | 58 ± 53 cm / 0.77 s |
| correlation with GOE, within the same revolution count | flight +0.10 (triples) | height +0.02, flight +0.03 | height -0.04, flight -0.02 |

None of the measurements correlates with GOE on these 40 videos. Not known yet whether that is measurement noise
(25 fps, motion blur on the ankles, homography chaining) or because GOE depends little on height; no ground truth.
