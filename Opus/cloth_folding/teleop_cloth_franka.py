# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keyboard teleoperation for the cloth-Franka example.

A small standalone driver to manually verify that
``Example.step(ee_pos, ee_rot, gripper)`` in
``newton/examples/cloth/example_cloth_franka_no_target.py`` works.

It opens the GL viewer, lets you move the Franka end effector with the
keyboard, and steps the coupled robot-cloth simulation each frame.

Run from the repository root:

    uv run --extra examples python agent/cloth_folding/teleop_cloth_franka.py

(or with an explicit viewer: ``... agent/cloth_folding/teleop_cloth_franka.py --viewer gl``)

Controls (all positions in cm, the example's working scale):

    Translation (hold):
        I / K   +Y / -Y   (away from / toward the robot base)
        J / L   -X / +X   (left / right)
        U / O   +Z / -Z   (up / down)

    Rotation (hold, world axes):
        T / G   rotate about X (+/-)
        R / Y   rotate about Y (+/-)
        B / N   rotate about Z (+/-)

    Gripper (hold):
        C       close (decrease activation)
        V       open  (increase activation)

    Misc:
        P       reset end-effector command to the initial pose
        SPACE   pause / resume (viewer)
        .       step one frame while paused (viewer)
        ESC     quit

Note: W/A/S/D, Q/E and the arrow keys drive the viewer camera, so teleop
deliberately avoids them.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# This teleop driver controls the Newton example
# newton/examples/cloth/example_cloth_franka_no_target.py, so it needs the
# Newton working tree importable as ``newton``. Point NEWTON_REPO at your
# Newton checkout to override the default below.
_NEWTON_REPO = os.environ.get("NEWTON_REPO", "/home/yilang/research/newton")
if _NEWTON_REPO not in sys.path:
    sys.path.insert(0, _NEWTON_REPO)

import newton.examples  # noqa: E402
from newton.examples.cloth.example_cloth_franka_no_target import Example  # noqa: E402

# Per-frame command increments.
TRANS_STEP = 0.5  # cm
ROT_STEP = 0.03  # rad
GRIP_STEP = 0.03  # activation units in [0, 1]


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``(x, y, z, w)`` quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quat_from_axis_angle(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Return the ``(x, y, z, w)`` quaternion for a rotation about ``axis``."""
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    a = a / n
    s = np.sin(0.5 * angle)
    return np.array([a[0] * s, a[1] * s, a[2] * s, np.cos(0.5 * angle)])


def main():
    parser = newton.examples.create_parser()
    # num_frames is unused by this manual loop, but init() reads it.
    parser.set_defaults(num_frames=0)
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    if not hasattr(viewer, "is_key_down"):
        raise SystemExit("The selected viewer does not support keyboard input. Run with the GL viewer: --viewer gl")

    # Initial command = the example's initial EE pose / gripper activation.
    init_pos = np.array(
        [example.target_ee_pos[0], example.target_ee_pos[1], example.target_ee_pos[2]],
        dtype=np.float64,
    )
    init_rot = np.array(
        [
            example.target_ee_rot[0],
            example.target_ee_rot[1],
            example.target_ee_rot[2],
            example.target_ee_rot[3],
        ],
        dtype=np.float64,
    )
    init_grip = float(example.gripper_activation)

    pos = init_pos.copy()
    rot = init_rot.copy()
    grip = init_grip

    print(__doc__)

    def key(k: str) -> bool:
        return bool(viewer.is_key_down(k))

    while viewer.is_running():
        # Only update the command and step when the viewer is not paused, so a
        # paused viewer freezes both the arm and the command (no drift).
        if viewer.should_step():
            # translation (cm)
            dx = (TRANS_STEP if key("l") else 0.0) - (TRANS_STEP if key("j") else 0.0)
            dy = (TRANS_STEP if key("i") else 0.0) - (TRANS_STEP if key("k") else 0.0)
            dz = (TRANS_STEP if key("u") else 0.0) - (TRANS_STEP if key("o") else 0.0)
            pos = pos + np.array([dx, dy, dz], dtype=np.float64)

            # rotation about world axes (rad)
            rx = (ROT_STEP if key("t") else 0.0) - (ROT_STEP if key("g") else 0.0)
            ry = (ROT_STEP if key("r") else 0.0) - (ROT_STEP if key("y") else 0.0)
            rz = (ROT_STEP if key("b") else 0.0) - (ROT_STEP if key("n") else 0.0)
            if rx or ry or rz:
                dq = quat_from_axis_angle((1.0, 0.0, 0.0), rx)
                dq = quat_mul(quat_from_axis_angle((0.0, 1.0, 0.0), ry), dq)
                dq = quat_mul(quat_from_axis_angle((0.0, 0.0, 1.0), rz), dq)
                rot = quat_mul(dq, rot)
                rot = rot / np.linalg.norm(rot)

            # gripper activation in [0, 1]
            if key("c"):
                grip = max(0.0, grip - GRIP_STEP)
            if key("v"):
                grip = min(1.0, grip + GRIP_STEP)

            # reset to the initial command
            if key("p"):
                pos = init_pos.copy()
                rot = init_rot.copy()
                grip = init_grip

            example.step(pos, rot, grip)

        example.render()

    viewer.close()


if __name__ == "__main__":
    main()
