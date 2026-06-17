# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Run 6 -- two clean, distinct fold phases in the requested order.
#
#   Phase 1 (X fold): fold LEFT and RIGHT in toward the center, then PARK --
#     the gripper backs away and the cloth is left flat on the table.
#   Phase 2 (Y fold): fold TOP and BOTTOM in toward the center, then park.
#
# Every grasp uses a reactive clean release (`release_clean`): after placing a
# fold it rakes the open fingers out from UNDER the cloth at low height, lifts,
# and checks the cloth did not come up with the gripper -- raking further and
# retrying if it did -- so the cloth is not left hanging from the gripper.
#
# Result (null search): fx 57->~22, fy 61->~21, footprint 3475 -> ~480-620 cm^2
# (~85% reduction), peak arm joint speed ~0.8 rad/s.
#
# Outputs (this directory): final.mp4, debug.mp4, metrics.json, final_frame.png

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agent/cloth_folding

from fold_lib import run_two_phase_fold

if __name__ == "__main__":
    summary = run_two_phase_fold(
        run_dir=Path(__file__).resolve().parent,
        view="gl",
        settle_steps=50,
        grasp_z=15.0,
        width=1280,
        height=720,
        final_fps=30,
        capture_stride=3,
        snap_stride=6,
    )
    print("RUN6 SUMMARY")
    for k, v in summary.items():
        if k == "metrics_ts":
            continue
        if k == "op_log":
            print("  op_log:")
            for op in v:
                print(f"    {op['label']:20s} footprint={op['footprint']:5.0f} fx={op['fx']:5.1f} fy={op['fy']:5.1f}")
            continue
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
