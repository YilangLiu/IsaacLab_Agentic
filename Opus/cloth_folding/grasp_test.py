# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Diagnostic: nail a single reliable grasp primitive before composing a fold.
# Uses a "goto" controller that drives the EE to a target until convergence,
# then closes and lifts, measuring whether the cloth is actually grasped
# (cloth ztop should rise when lifted).

import sys
from pathlib import Path

import numpy as np

# fold_lib lives next to this script and puts the Newton working tree on the
# path when imported (see fold_lib's NEWTON_REPO resolver).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fold_lib import CLOSE, OPEN, Example, _ee_pose, foldedness

import newton.viewer

# Orientation used for the +x,-y shoulder grasp in the original demo.
ROT_A = [0.8536, -0.3536, 0.3536, -0.1464]


def goto(ex, pos, rot, grip, max_steps=120, tol=1.5, hold=0):
    pos = np.asarray(pos, float)
    last = None
    for k in range(max_steps):
        ex.step(pos, rot, grip)
        ee = _ee_pose(ex)
        err = float(np.linalg.norm(ee[:3] - pos))
        last = (k, ee, err)
        if err < tol:
            break
    for _ in range(hold):
        ex.step(pos, rot, grip)
    return last


def cloth_stats(ex):
    q = ex.state_0.particle_q.numpy()
    m = foldedness(q)
    return m["ztop"], m["footprint"], m["zmean"]


def main():
    v = newton.viewer.ViewerNull(num_frames=0)
    ex = Example(v, None)
    # settle
    ip = np.array([ex.target_ee_pos[i] for i in range(3)])
    ir = np.array([ex.target_ee_rot[i] for i in range(4)])
    for _ in range(40):
        ex.step(ip, ir, OPEN)
    z0, f0, zm0 = cloth_stats(ex)
    print(f"settled: cloth ztop={z0:.1f} footprint={f0:.0f} zmean={zm0:.1f}")

    corner = np.array([31.0, -60.0])  # +x shoulder region
    for grasp_z in (20.0, 16.0, 12.0, 8.0):
        v2 = newton.viewer.ViewerNull(num_frames=0)
        ex = Example(v2, None)
        for _ in range(40):
            ex.step(ip, ir, OPEN)
        # approach above
        goto(ex, [corner[0], corner[1], 40.0], ROT_A, OPEN, max_steps=120, tol=2.0)
        # descend open to straddle the cloth
        kd = goto(ex, [corner[0], corner[1], grasp_z], ROT_A, OPEN, max_steps=200, tol=1.0)
        zee_low = kd[1][2]
        # close on the fabric
        goto(ex, [corner[0], corner[1], grasp_z], ROT_A, CLOSE, max_steps=60, tol=0.5, hold=30)
        # lift
        goto(ex, [corner[0], corner[1], 40.0], ROT_A, CLOSE, max_steps=150, tol=2.0, hold=10)
        z1, f1, zm1 = cloth_stats(ex)
        print(
            f"grasp_z={grasp_z:4.1f}: EE reached z={zee_low:5.1f} (conv@{kd[0]:3d}, err{kd[2]:.1f}) "
            f"-> after lift cloth ztop={z1:5.1f} (+{z1 - z0:+.1f}) footprint={f1:5.0f} zmean={zm1:.1f} "
            f"{'<<< GRASP OK' if z1 - z0 > 6 else ''}"
        )


if __name__ == "__main__":
    main()
