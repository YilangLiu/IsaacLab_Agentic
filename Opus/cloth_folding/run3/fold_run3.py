# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Run 3 -- polished, tighter fold.
#
# Same convergence-based pick-drag-release strategy as Run 2, refined after
# null-viewer search over placements:
#   * Side folds place the grabbed edge just PAST the center line (x=-2 / +2),
#     so the two halves overlap more -> smaller, squarer footprint.
#   * Far-Y folds carry the far hem across the center (y=-50) with a grasp deep
#     enough (z=14) to actually catch the hem.
#   * Clean retreat to a high pose clear of the folded cloth.
#
# Best null-viewer result for this plan: footprint ~800 cm^2 (vs ~3475 settled,
# a ~77% reduction) and radius-of-gyration ~10 cm (vs ~24). Note the VBD
# self-contact solver is mildly non-deterministic, so the rendered run lands
# within a few % of that.
#
# Outputs (this directory): final.mp4, debug.mp4, metrics.json, final_frame.png

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agent/cloth_folding

from fold_lib import FOLD_OPS_TIGHT, run_primitive_fold

if __name__ == "__main__":
    summary = run_primitive_fold(
        run_dir=Path(__file__).resolve().parent,
        ops=FOLD_OPS_TIGHT,
        view="gl",
        settle_steps=60,
        grasp_z=14.0,
        retreat_pos=(-35.0, -70.0, 52.0),
        width=1280,
        height=720,
        final_fps=30,
        capture_stride=3,
        snap_stride=5,
    )
    print("RUN3 SUMMARY")
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
