# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Run 7 -- clean three-fold sequence (minimal motion).
#
# Modeled on the original scripted demo (example_cloth_franka.py), where each
# fold is ONE grasp followed by a short drag path to the target, then a release
# -- no parks, no reactive retries, no rake-outs. Three folds, in order:
#
#   1. LEFT half  -> center
#   2. RIGHT half -> center   (now a narrow vertical strip)
#   3. BOTTOM (near edge) -> TOP (far edge)  (folds the strip in half)
#
# Each grabbed edge is dragged OVER the fabric at low height (carry_z ~ 28), so
# the cloth stays flat on the table and never hangs from the gripper. Driven by
# the smooth damped-least-squares controller (peak joint speed ~1.3 rad/s).
#
# Null-search result: fx 57->~20, fy 61->~36, footprint 3475 -> ~700 (~80%),
# cloth on the table, ~1085 control steps (vs ~6000 for the multi-grab Run 6).
#
# Outputs (this directory): final.mp4, debug.mp4, metrics.json, final_frame.png

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agent/cloth_folding

from fold_lib import CLEAN_FOLD_WAYPOINTS, SMOOTH_CONTROL, run_clean_fold

if __name__ == "__main__":
    summary = run_clean_fold(
        run_dir=Path(__file__).resolve().parent,
        waypoints=CLEAN_FOLD_WAYPOINTS,
        view="gl",
        settle_steps=60,
        control=SMOOTH_CONTROL,
        width=1280,
        height=720,
        final_fps=30,
        capture_stride=2,
        snap_stride=5,
    )
    print("RUN7 SUMMARY")
    for k, v in summary.items():
        if k in ("metrics_ts", "op_log"):
            continue
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
