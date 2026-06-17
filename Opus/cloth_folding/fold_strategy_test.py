# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Prototype a full fold from convergence-based pick-drag-release primitives,
# measuring footprint after each fold op (fast null-viewer iteration).

import sys
from pathlib import Path

import numpy as np

# fold_lib lives next to this script and puts the Newton working tree on the
# path when imported (see fold_lib's NEWTON_REPO resolver).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fold_lib import CLOSE, OPEN, Example, _ee_pose, foldedness

import newton.viewer

ROT = [0.8536, -0.3536, 0.3536, -0.1464]  # gripper-down


def goto(ex, pos, rot, grip, max_steps=160, tol=1.2, hold=0):
    pos = np.asarray(pos, float)
    err = 99.0
    for _ in range(max_steps):
        ex.step(pos, rot, grip)
        err = float(np.linalg.norm(_ee_pose(ex)[:3] - pos))
        if err < tol:
            break
    for _ in range(hold):
        ex.step(pos, rot, grip)
    return _ee_pose(ex)[:3], err


def fold_op(ex, pick_xy, place_xy, rot=ROT, grasp_z=14.0, lift_z=38.0, place_z=26.0):
    goto(ex, [pick_xy[0], pick_xy[1], lift_z], rot, OPEN, max_steps=140, tol=2.0)
    ee, _ = goto(ex, [pick_xy[0], pick_xy[1], grasp_z], rot, OPEN, max_steps=200, tol=1.0)
    goto(ex, [pick_xy[0], pick_xy[1], grasp_z], rot, CLOSE, max_steps=60, tol=0.5, hold=25)
    goto(ex, [pick_xy[0], pick_xy[1], lift_z], rot, CLOSE, max_steps=160, tol=2.0)
    goto(ex, [place_xy[0], place_xy[1], lift_z], rot, CLOSE, max_steps=200, tol=2.0)
    goto(ex, [place_xy[0], place_xy[1], place_z], rot, CLOSE, max_steps=160, tol=1.5)
    goto(ex, [place_xy[0], place_xy[1], place_z], rot, OPEN, max_steps=40, tol=1.0, hold=15)
    goto(ex, [place_xy[0], place_xy[1], lift_z], rot, OPEN, max_steps=120, tol=2.0)
    return ee[2]


def fp(ex):
    return foldedness(ex.state_0.particle_q.numpy())


def main():
    v = newton.viewer.ViewerNull(num_frames=0)
    ex = Example(v, None)
    ip = np.array([ex.target_ee_pos[i] for i in range(3)])
    ir = np.array([ex.target_ee_rot[i] for i in range(4)])
    for _ in range(60):
        ex.step(ip, ir, OPEN)
    m = fp(ex)
    print(f"settled: footprint={m['footprint']:.0f} fx={m['fx']:.0f} fy={m['fy']:.0f} rgxy={m['rgxy']:.1f}")

    ops = [
        ("fold +X side in", (28.0, -58.0), (-8.0, -58.0)),
        ("fold -X side in", (-28.0, -58.0), (8.0, -58.0)),
        # fold far half across the center (y=-50) onto the near half
        ("fold far-Y (L)", (-7.0, -76.0), (-7.0, -26.0)),
        ("fold far-Y (R)", (7.0, -76.0), (7.0, -26.0)),
    ]
    for name, pick, place in ops:
        zee = fold_op(ex, pick, place)
        m = fp(ex)
        print(
            f"{name:18s} pick{pick} ee_z={zee:5.1f} -> footprint={m['footprint']:5.0f} "
            f"fx={m['fx']:5.1f} fy={m['fy']:5.1f} rgxy={m['rgxy']:5.1f}"
        )


if __name__ == "__main__":
    main()
