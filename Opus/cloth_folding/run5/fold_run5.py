# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Run 5 -- true quarter fold (left/right, then top/bottom).
#
# Run 4 folded the cloth left/right but the length (top/bottom) was not clearly
# folded. Run 5 folds in BOTH axes, in the order requested:
#   1. fold the LEFT and RIGHT halves to the center (sleeve + body grab/side),
#   2. then fold the TOP and BOTTOM halves to the center.
#
# Uses the smooth damped-least-squares controller from Run 4 (FoldExample), so
# the arm reaches every target with peak joint speed ~1.5 rad/s.
#
# The top/bottom grabs are ADAPTIVE: a fixed grab point misses once the hems
# have moved inward after the side folds, and a single grab only pulls a tongue
# of the wide multilayer strip (it springs back). So each of the top and bottom
# edges is folded with TWO grabs read from the live cloth state.
#
# Result (null search): footprint ~3475 -> ~800-900 cm^2 (~75% reduction),
# folded in BOTH axes (fx ~31, fy ~28), centered, peak arm joint speed ~1.5 rad/s.
#
# Outputs (this directory): final.mp4, debug.mp4, metrics.json, final_frame.png

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agent/cloth_folding

from fold_lib import run_quarter_fold

if __name__ == "__main__":
    summary = run_quarter_fold(
        run_dir=Path(__file__).resolve().parent,
        view="gl",
        settle_steps=50,
        grasp_z=14.0,
        retreat_pos=(-35.0, -70.0, 52.0),
        width=1280,
        height=720,
        final_fps=30,
        capture_stride=2,
        snap_stride=5,
    )
    print("RUN5 SUMMARY")
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
