# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Run 4 -- smooth, realistic motion + complete fold.
#
# Run 3 worked on the metric but the ARM motion was bad: the stock controller's
# plain Jacobian pseudo-inverse spiked joint velocities to ~94 rad/s (a real
# Franka does ~2.5), most waypoints timed out without converging (the EE never
# reached its target), and grasps clipped 6 cm through the table.
#
# Run 4 replaces the controller (FoldExample, damped least squares + task- and
# joint-velocity clamps + reduced nullspace pull). This reaches every target
# smoothly with peak joint speed ~1.5 rad/s. The gentle controller moves less
# fabric per grasp, so the fold plan is enriched: each side is folded with TWO
# grabs (sleeve + body edge), then the far Y-half is folded onto the near half.
#
# Result (null search): footprint ~3475 -> ~770 cm^2 (~78% reduction),
# well-centered, peak arm joint speed ~1.5 rad/s.
#
# Outputs (this directory): final.mp4, debug.mp4, metrics.json, final_frame.png

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agent/cloth_folding

from fold_lib import FOLD_OPS_V4, SMOOTH_CONTROL, run_primitive_fold

if __name__ == "__main__":
    summary = run_primitive_fold(
        run_dir=Path(__file__).resolve().parent,
        ops=FOLD_OPS_V4,
        view="gl",
        settle_steps=50,
        grasp_z=14.0,
        retreat_pos=(-35.0, -70.0, 52.0),
        control=SMOOTH_CONTROL,
        width=1280,
        height=720,
        final_fps=30,
        capture_stride=2,
        snap_stride=5,
    )
    print("RUN4 SUMMARY")
    for k, v in summary.items():
        if k == "metrics_ts":
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
