# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Run 2 -- working fold via convergence-based pick-drag-release primitives.
#
# Run 1 (faithful replay of the scripted waypoints) failed: the resolved-rate
# IK lags, so the gripper closed ~8 cm above the cloth and never grasped, and
# targeting the cloth-top height (z=20) only grazed the fabric.
#
# Run 2 fixes this with two changes:
#   1. A "goto" controller that drives the EE to each target until convergence
#      (error < tol) instead of holding a fixed-duration target -- so the arm
#      actually reaches grasp poses before the gripper acts.
#   2. Grasps descend ~10 cm BELOW the cloth top (z~=14) so the fingers
#      straddle the fabric (verified: cloth lifts only when EE z < cloth top).
#
# Strategy: fold both X sides in toward the center, then fold the far Y half
# across the center line (y=-50) onto the near half (two grabs for width).
#
# Outputs (this directory): final.mp4, debug.mp4, metrics.json, final_frame.png

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agent/cloth_folding

from fold_lib import FOLD_OPS, run_primitive_fold

if __name__ == "__main__":
    summary = run_primitive_fold(
        run_dir=Path(__file__).resolve().parent,
        ops=FOLD_OPS,
        view="gl",
        settle_steps=60,
        grasp_z=14.0,
        width=1280,
        height=720,
        final_fps=30,
        capture_stride=2,
        snap_stride=5,
    )
    print("RUN2 SUMMARY")
    for k, v in summary.items():
        if k in ("metrics_ts",):
            continue
        if k == "op_log":
            print("  op_log:")
            for op in v:
                print(
                    f"    {op['label']:18s} grasp_ee_z={op['grasp_ee_z']:5.1f} "
                    f"footprint={op['footprint']:5.0f} fx={op['fx']:5.1f} fy={op['fy']:5.1f}"
                )
            continue
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
