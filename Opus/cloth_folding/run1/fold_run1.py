# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Run 1 -- faithful baseline.
#
# Replays the proven cloth-folding trajectory (FOLD_WAYPOINTS) through the new
# end-effector command API Example.step(ee_pos, ee_rot, gripper), at the
# original control rate (30 control steps per second of pose duration).
#
# Strategy: the shirt lies on the table; the arm grasps each of the four
# corners/edges in turn (top-left, bottom-left, top-right, bottom-right) and
# folds it toward the center, then folds the bottom edge up. Each waypoint is
# an absolute end-effector pose + gripper activation; resolved-rate IK drives
# the arm there.
#
# Outputs (this directory): final.mp4 (GL 3D), debug.mp4 (top-down/side +
# foldedness metric), metrics.json, final_frame.png.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agent/cloth_folding

from fold_lib import FOLD_WAYPOINTS, run_fold

if __name__ == "__main__":
    summary = run_fold(
        run_dir=Path(__file__).resolve().parent,
        waypoints=FOLD_WAYPOINTS,
        steps_per_sec=30.0,
        settle_steps=30,
        view="gl",
        width=1280,
        height=720,
        final_fps=30,
        capture_stride=2,
        snap_stride=5,
        debug_fps=20,
    )
    print("RUN1 SUMMARY")
    for k, v in summary.items():
        if k == "metrics_ts":
            continue
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
