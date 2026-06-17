# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Reusable harness for the "fold the cloth" task on the cloth-Franka example.
#
# It drives newton/examples/cloth/example_cloth_franka_no_target.py via the
# end-effector command API (step(ee_pos, ee_rot, gripper)) along a waypoint
# trajectory, and produces:
#   * a "final" video: the GL-rendered 3D scene (headless), and
#   * a "debug" video: top-down / side particle scatter + foldedness metric,
#   * a metrics.json time series and a foldedness summary.
#
# This file is imported by the per-run scripts (run1/, run2/, ...).

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# This project drives the Newton example
# newton/examples/cloth/example_cloth_franka_no_target.py, so it needs the
# Newton working tree (where that file lives) importable as ``newton`` -- not
# any stale site-packages copy. Point NEWTON_REPO at your Newton checkout to
# override the default below.
_NEWTON_REPO = os.environ.get("NEWTON_REPO", "/home/yilang/research/newton")
if _NEWTON_REPO not in sys.path:
    sys.path.insert(0, _NEWTON_REPO)

import warp as wp  # noqa: E402

import newton.examples  # noqa: E402
import newton.viewer  # noqa: E402
from newton.examples.cloth.example_cloth_franka_no_target import (  # noqa: E402
    Example,
    compute_ee_delta,
    compute_ee_pose,
)

# Gripper activation levels (0 closed .. 1 open), matching the example's
# finger mapping (finger target = activation * 4 cm).
CLOSE = 0.1
OPEN = 0.8

# Proven folding trajectory (reconstructed from the original scripted demo).
# Each row: [duration_s, x, y, z (cm), qx, qy, qz, qw, gripper].
# The arm is driven toward each pose with resolved-rate IK; the duration
# controls how many control steps the target is held (see steps_per_sec).
FOLD_WAYPOINTS = np.array(
    [
        # descend to working height (cloth also settles onto the table here)
        [4, 31.0, -60.0, 40.0, 0.8536, -0.3536, 0.3536, -0.1464, OPEN],
        # --- top-left corner -> center ---
        [2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, OPEN],
        [2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, CLOSE],
        [2, 26.0, -60.0, 26.0, 0.8536, -0.3536, 0.3536, -0.1464, CLOSE],
        [2, 12.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, CLOSE],
        [3, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, CLOSE],
        [1, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, OPEN],
        # --- bottom-left corner -> center ---
        [2, 15.0, -33.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, OPEN],
        [3, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, OPEN],
        [3, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, CLOSE],
        [2, 15.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, CLOSE],
        [3, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, CLOSE],
        [1, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, OPEN],
        # --- top-right corner -> center ---
        [2, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, OPEN],
        [2, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, OPEN],
        [2, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, CLOSE],
        [2, -18.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, CLOSE],
        [3, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, CLOSE],
        [1, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, OPEN],
        # --- bottom-right corner -> center ---
        [3, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, OPEN],
        [3, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, CLOSE],
        [2, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, CLOSE],
        [3, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, CLOSE],
        [2, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, OPEN],
        # --- bottom edge -> fold up ---
        [2, 0.0, -20.0, 30.0, 0.9239, -0.3827, 0.0, 0.0, OPEN],
        [2, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, OPEN],
        [2, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, CLOSE],
        [2, 0.0, -20.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, CLOSE],
        [1, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, CLOSE],
        [1.5, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, CLOSE],
        [1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, CLOSE],
        [1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, OPEN],
        # retreat so the arm does not occlude the folded cloth
        [2, -28.0, -60.0, 45.0, 0.9239, -0.3827, 0.0, 0.0, OPEN],
    ],
    dtype=np.float64,
)

# Table footprint in cm (centered at (0,-50), half-extents 40x40).
TABLE_XLIM = (-45.0, 45.0)
TABLE_YLIM = (-95.0, -5.0)

# Gripper-down orientation (quaternion x,y,z,w) used for top-down grasps.
ROT_DOWN = [0.8536, -0.3536, 0.3536, -0.1464]

# Primitive fold plan: (label, pick_xy, place_xy). The shirt lies with sleeves
# at (+-30, -58) and body over X[-21,21], Y[-79,-20]. We fold both sides in,
# then fold the far half across the center (y=-50) onto the near half.
FOLD_OPS = [
    ("fold +X side in", (28.0, -58.0), (-8.0, -58.0)),
    ("fold -X side in", (-28.0, -58.0), (8.0, -58.0)),
    ("fold far-Y left", (-7.0, -76.0), (-7.0, -26.0)),
    ("fold far-Y right", (7.0, -76.0), (7.0, -26.0)),
]

# Tighter variant (Run 3): place the side folds just past the center line so
# the layers overlap more, giving a smaller, squarer footprint.
FOLD_OPS_TIGHT = [
    ("fold +X side in", (28.0, -57.0), (-2.0, -57.0)),
    ("fold -X side in", (-28.0, -57.0), (2.0, -57.0)),
    ("fold far-Y left", (-7.0, -77.0), (-7.0, -28.0)),
    ("fold far-Y right", (7.0, -77.0), (7.0, -28.0)),
]

# Run 4 plan, tuned for the smooth DLS controller (FoldExample). The gentle
# controller moves less fabric per grasp than the old aggressive one, so each
# side is folded with TWO grabs (sleeve + body edge) to carry the whole side
# across; then the far Y-half is folded onto the near half with two grabs.
# Result: footprint ~770 cm^2, well centered, peak joint speed ~1.5 rad/s.
FOLD_OPS_V4 = [
    ("fold +X sleeve", (30.0, -57.0), (-6.0, -57.0)),
    ("fold +X body", (16.0, -57.0), (-12.0, -57.0)),
    ("fold -X sleeve", (-30.0, -57.0), (6.0, -57.0)),
    ("fold -X body", (-16.0, -57.0), (12.0, -57.0)),
    ("fold far-Y left", (-8.0, -78.0), (-8.0, -28.0)),
    ("fold far-Y right", (8.0, -78.0), (8.0, -28.0)),
]

# Control gains for the smooth DLS controller (validated in tuning).
SMOOTH_CONTROL = {"k_lin": 4.0, "lin_clamp": 12.0, "qd_clamp": 4.0}


def expand_waypoints(waypoints: np.ndarray, steps_per_sec: float) -> list[dict]:
    """Expand [duration, pose..., grip] rows into a per-control-step command list."""
    plan = []
    for row in waypoints:
        n = max(1, int(round(row[0] * steps_per_sec)))
        cmd = {
            "pos": row[1:4].astype(float),
            "rot": row[4:8].astype(float),
            "grip": float(row[8]),
        }
        plan.extend([cmd] * n)
    return plan


def foldedness(q: np.ndarray) -> dict:
    """Compactness metrics for the cloth particle cloud (cm units).

    A smaller XY footprint / radius-of-gyration means a more folded cloth.
    """
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    xlo, xhi = np.percentile(x, [2.5, 97.5])
    ylo, yhi = np.percentile(y, [2.5, 97.5])
    fx, fy = float(xhi - xlo), float(yhi - ylo)
    cx, cy = float(x.mean()), float(y.mean())
    rgxy = float(np.sqrt(((x - cx) ** 2 + (y - cy) ** 2).mean()))
    return {
        "footprint": fx * fy,
        "fx": fx,
        "fy": fy,
        "rgxy": rgxy,
        "ztop": float(np.percentile(z, 97.5)),
        "zmean": float(z.mean()),
        "cx": cx,
        "cy": cy,
    }


def _ee_pose(example: Example) -> np.ndarray:
    arr = wp.empty(1, dtype=wp.transform)
    wp.launch(
        compute_ee_pose,
        dim=1,
        inputs=[example.state_0.body_q, example.endeffector_offset, example.endeffector_id],
        outputs=[arr],
    )
    return np.array(arr.numpy()[0], dtype=float)


class FoldExample(Example):
    """Example with a smoother, more realistic resolved-rate controller.

    The stock controller in the example uses a plain Jacobian pseudo-inverse
    with unit gain and a strong nullspace pull. That spikes joint velocities to
    ~90 rad/s near ill-conditioned configs (violent motion) and a unit-gain
    error rarely converges within the step budget. This override uses:

      * damped least squares (DLS): dq = J^T (J J^T + lambda^2 I)^-1 v,
        which stays bounded near singularities;
      * a task-space velocity clamp so the commanded EE speed is bounded
        (smooth approach, no large initial snap);
      * a proportional gain so it actually converges within the budget;
      * a hard joint-velocity cap (~Franka-realistic);
      * a reduced nullspace gain so the redundant joints do not fight the task.
    """

    # control gains / limits (tunable)
    dls_lambda = 6.0  # DLS damping (mixed cm/rad units)
    k_lin = 3.0  # linear error -> velocity gain
    k_ang = 2.0  # angular error -> velocity gain
    lin_clamp = 8.0  # max linear task velocity [cm/s-equiv]
    ang_clamp = 1.0  # max angular task velocity
    k_null = 0.15  # nullspace pull toward rest pose
    qd_clamp = 3.0  # max arm joint speed [rad/s]
    z_floor = None  # if set, the commanded EE tip never goes below this height
    #                 [cm] -- a control-side stand-in for robot<->table contact
    #                 (the gripper cannot scoop UNDER the cloth through the table)

    def step(self, ee_pos=None, ee_rot=None, gripper=None):
        if ee_pos is not None and self.z_floor is not None:
            ee_pos = (float(ee_pos[0]), float(ee_pos[1]), max(self.z_floor, float(ee_pos[2])))
        super().step(ee_pos, ee_rot, gripper)

    def generate_control_joint_qd(self, state_in: State):
        wp.launch(
            compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_world,
                wp.transform(self.target_ee_pos, self.target_ee_rot),
            ],
            outputs=[self.ee_delta],
        )
        self.compute_body_jacobian(self.model, state_in.joint_q, state_in.joint_qd, include_rotation=True)
        dof = self.model.joint_dof_count
        J = self.J_flat.numpy().reshape(-1, dof).astype(np.float64)
        err = np.asarray(self.ee_delta.numpy()[0], dtype=np.float64)

        # clamp linear / angular task error, then apply gain -> desired task vel
        e_lin, e_ang = err[:3].copy(), err[3:].copy()
        ln = np.linalg.norm(e_lin)
        if ln > self.lin_clamp:
            e_lin *= self.lin_clamp / ln
        an = np.linalg.norm(e_ang)
        if an > self.ang_clamp:
            e_ang *= self.ang_clamp / an
        v = np.concatenate([self.k_lin * e_lin, self.k_ang * e_ang])

        # damped least squares
        JJt = J @ J.T
        A = JJt + (self.dls_lambda**2) * np.eye(6)
        dls = J.T @ np.linalg.solve(A, v)

        # reduced nullspace pull toward the rest pose
        q = state_in.joint_q.numpy().astype(np.float64)
        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]
        J_dls_pinv = J.T @ np.linalg.inv(A)
        N = np.eye(dof) - J_dls_pinv @ J
        dq = dls + N @ (self.k_null * (q_des - q))

        # hard joint-velocity cap on the 7 arm joints (realism)
        arm = dq[:7]
        an2 = np.linalg.norm(arm)
        if an2 > self.qd_clamp:
            dq[:7] = arm * (self.qd_clamp / an2)

        # gripper finger position control (unchanged)
        dq[-2] = self.gripper_activation * 4.0 - q[-2]
        dq[-1] = self.gripper_activation * 4.0 - q[-1]

        self.target_joint_qd.assign(dq.astype(np.float32))


class _FFmpegPipe:
    """Stream raw RGB frames into an mp4 via an ffmpeg subprocess."""

    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, rgb: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def run_fold(
    run_dir: str | Path,
    waypoints: np.ndarray = FOLD_WAYPOINTS,
    steps_per_sec: float = 30.0,
    settle_steps: int = 30,
    view: str = "gl",
    width: int = 1280,
    height: int = 720,
    final_fps: int = 30,
    capture_stride: int = 2,
    snap_stride: int = 5,
    debug_fps: int = 20,
    max_steps: int | None = None,
) -> dict:
    """Run the folding trajectory and write final/debug videos + metrics.

    Args:
        run_dir: Output directory (created if needed).
        view: "gl" to render the final video headlessly, or "null" for a fast
            metric-only pass (no final video).
    Returns:
        Summary dict (also written to ``run_dir/metrics.json``).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if view == "gl":
        viewer = newton.viewer.ViewerGL(headless=True, width=width, height=height)
    else:
        viewer = newton.viewer.ViewerNull(num_frames=0)
    example = Example(viewer, None)

    plan = expand_waypoints(waypoints, steps_per_sec)
    # Prepend a settle phase that holds the initial command (cloth drops flat).
    init_pos = np.array([example.target_ee_pos[i] for i in range(3)], dtype=float)
    init_rot = np.array([example.target_ee_rot[i] for i in range(4)], dtype=float)
    settle = [{"pos": init_pos, "rot": init_rot, "grip": OPEN}] * settle_steps
    plan = settle + plan
    if max_steps is not None:
        plan = plan[:max_steps]
    n_settle = settle_steps

    final_video = run_dir / "final.mp4"
    pipe = _FFmpegPipe(final_video, width, height, final_fps) if view == "gl" else None

    metrics_ts: list[dict] = []
    snaps: list[dict] = []

    for i, cmd in enumerate(plan):
        example.step(cmd["pos"], cmd["rot"], cmd["grip"])

        if view == "gl" and (i % capture_stride == 0):
            example.render()
            frame = viewer.get_frame().numpy()
            pipe.write(frame)

        if i % snap_stride == 0 or i == len(plan) - 1:
            q = example.state_0.particle_q.numpy()
            m = foldedness(q)
            m["step"] = i
            m["grip"] = cmd["grip"]
            m["phase"] = "settle" if i < n_settle else "fold"
            metrics_ts.append(m)
            ee = _ee_pose(example)
            snaps.append(
                {
                    "step": i,
                    "q": q.astype(np.float32),
                    "ee": ee,
                    "target": cmd["pos"].copy(),
                    "grip": cmd["grip"],
                    "phase": m["phase"],
                    "metric": m,
                }
            )

    if pipe is not None:
        pipe.close()

    # Save the final rendered frame as a still.
    if view == "gl":
        example.render()
        last = viewer.get_frame().numpy()
        from PIL import Image

        Image.fromarray(last).save(run_dir / "final_frame.png")

    # Reference (settled, pre-fold) vs final compactness.
    settled = next((m for m in metrics_ts if m["step"] >= n_settle), metrics_ts[0])
    final_m = metrics_ts[-1]
    summary = {
        "n_steps": len(plan),
        "steps_per_sec": steps_per_sec,
        "settle_steps": n_settle,
        "settled_footprint": settled["footprint"],
        "final_footprint": final_m["footprint"],
        "footprint_ratio": final_m["footprint"] / max(settled["footprint"], 1e-6),
        "settled_rgxy": settled["rgxy"],
        "final_rgxy": final_m["rgxy"],
        "rgxy_ratio": final_m["rgxy"] / max(settled["rgxy"], 1e-6),
        "final_ztop": final_m["ztop"],
        "metrics_ts": metrics_ts,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    make_debug_video(snaps, run_dir / "debug.mp4", fps=debug_fps)
    return summary


def make_debug_video(snaps: list[dict], out_path: Path, fps: int = 20):
    """Render a top-down + side + metric debug video from particle snapshots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not snaps:
        return
    steps = [s["step"] for s in snaps]
    foot = [s["metric"]["footprint"] for s in snaps]

    dpi = 100
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)
    w_px, h_px = int(15 * dpi), int(5 * dpi)
    pipe = _FFmpegPipe(out_path, w_px, h_px, fps)

    for k, s in enumerate(snaps):
        q = s["q"]
        ax0, ax1, ax2 = axes
        for ax in axes:
            ax.clear()

        # Top-down (X-Y), colored by height
        ax0.scatter(q[:, 0], q[:, 1], c=q[:, 2], s=2, cmap="viridis", vmin=10, vmax=45)
        ax0.plot(
            [TABLE_XLIM[0], TABLE_XLIM[1], TABLE_XLIM[1], TABLE_XLIM[0], TABLE_XLIM[0]],
            [TABLE_YLIM[0], TABLE_YLIM[0], TABLE_YLIM[1], TABLE_YLIM[1], TABLE_YLIM[0]],
            "k--",
            lw=0.8,
            alpha=0.4,
        )
        ax0.scatter([s["target"][0]], [s["target"][1]], c="red", marker="x", s=80, label="target")
        ax0.scatter([s["ee"][0]], [s["ee"][1]], c="black", marker="+", s=80, label="ee")
        ax0.set_xlim(*TABLE_XLIM)
        ax0.set_ylim(*TABLE_YLIM)
        ax0.set_aspect("equal")
        ax0.set_title(f"top-down (X-Y)  step {s['step']}  [{s['phase']}]")
        ax0.set_xlabel("x [cm]")
        ax0.set_ylabel("y [cm]")
        ax0.legend(loc="upper right", fontsize=7)

        # Side (Y-Z)
        ax1.scatter(q[:, 1], q[:, 2], c=q[:, 2], s=2, cmap="viridis", vmin=10, vmax=45)
        ax1.axhline(20.0, color="gray", ls="--", lw=0.8, alpha=0.5)  # table top
        ax1.set_xlim(*TABLE_YLIM)
        ax1.set_ylim(0, 60)
        ax1.set_title(f"side (Y-Z)  grip={'closed' if s['grip'] < 0.5 else 'open'}")
        ax1.set_xlabel("y [cm]")
        ax1.set_ylabel("z [cm]")

        # Metric time series up to now
        ax2.plot(steps, foot, color="tab:blue", lw=1.0, alpha=0.5)
        ax2.plot(steps[: k + 1], foot[: k + 1], color="tab:blue", lw=2.0)
        ax2.scatter([s["step"]], [foot[k]], color="red", zorder=5)
        ax2.set_xlim(steps[0], steps[-1])
        ax2.set_ylim(0, max(foot) * 1.1)
        ax2.set_title(f"footprint area  {foot[k]:.0f} cm^2")
        ax2.set_xlabel("control step")
        ax2.set_ylabel("XY footprint [cm^2]")

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        pipe.write(buf)

    pipe.close()
    plt.close(fig)


class FoldSession:
    """Drive the cloth-Franka example with convergence-based primitives while
    capturing the GL render (final video), particle snapshots (debug video),
    and a foldedness metric time series.
    """

    def __init__(
        self,
        run_dir: str | Path,
        view: str = "gl",
        width: int = 1280,
        height: int = 720,
        final_fps: int = 30,
        capture_stride: int = 2,
        snap_stride: int = 5,
        control: dict | None = None,
        example_cls=FoldExample,
        soft_mu: float | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.view = view
        self.width, self.height = width, height
        self.capture_stride, self.snap_stride = capture_stride, snap_stride

        if view == "gl":
            self.viewer = newton.viewer.ViewerGL(headless=True, width=width, height=height)
            self.pipe = _FFmpegPipe(self.run_dir / "final.mp4", width, height, final_fps)
        else:
            self.viewer = newton.viewer.ViewerNull(num_frames=0)
            self.pipe = None
        self.example = example_cls(self.viewer, None)
        # apply any control-gain overrides (FoldExample attributes)
        for k, val in (control or {}).items():
            setattr(self.example, k, val)
        if soft_mu is not None:
            self.example.model.soft_contact_mu = float(soft_mu)
        # Friction is modulated per-phase: HIGH while grasping/carrying so the
        # fold holds, then dropped LOW during release so the cloth slides off the
        # open fingers under gravity (the gripper points down, so a sticky
        # contact otherwise carries the cloth up). soft_contact_mu is read live
        # by the solver, so this takes effect even with the captured CUDA graph.
        self.grasp_mu = float(self.example.model.soft_contact_mu)
        self.release_mu = 0.02
        self.max_qd = 0.0  # track peak commanded joint speed for realism check

        self.step_i = 0
        self.phase = "settle"
        self.metrics_ts: list[dict] = []
        self.snaps: list[dict] = []
        self.op_log: list[dict] = []
        self.init_pos = np.array([self.example.target_ee_pos[i] for i in range(3)], dtype=float)
        self.init_rot = np.array([self.example.target_ee_rot[i] for i in range(4)], dtype=float)

    # --- low level ---
    def _capture(self, target, grip):
        i = self.step_i
        if self.pipe is not None and i % self.capture_stride == 0:
            self.example.render()
            self.pipe.write(self.viewer.get_frame().numpy())
        if i % self.snap_stride == 0:
            q = self.example.state_0.particle_q.numpy()
            m = foldedness(q)
            m["step"] = i
            m["grip"] = float(grip)
            m["phase"] = self.phase
            self.metrics_ts.append(m)
            self.snaps.append(
                {
                    "step": i,
                    "q": q.astype(np.float32),
                    "ee": _ee_pose(self.example),
                    "target": np.asarray(target, float),
                    "grip": float(grip),
                    "phase": self.phase,
                    "metric": m,
                }
            )

    def step(self, pos, rot, grip):
        self.example.step(pos, rot, grip)
        self.max_qd = max(self.max_qd, float(np.abs(self.example.target_joint_qd.numpy()[:7]).max()))
        self._capture(pos, grip)
        self.step_i += 1

    def goto(self, pos, rot, grip, max_steps=160, tol=1.2, hold=0):
        pos = np.asarray(pos, float)
        err = 99.0
        for _ in range(max_steps):
            self.step(pos, rot, grip)
            err = float(np.linalg.norm(_ee_pose(self.example)[:3] - pos))
            if err < tol:
                break
        for _ in range(hold):
            self.step(pos, rot, grip)
        return _ee_pose(self.example)[:3], err

    # --- primitives ---
    def settle(self, n=60):
        self.phase = "settle"
        for _ in range(n):
            self.step(self.init_pos, self.init_rot, OPEN)

    def fold_op(
        self,
        pick_xy,
        place_xy,
        rot=ROT_DOWN,
        grasp_z=14.0,
        carry_z=27.0,
        place_z=25.0,
        retreat_z=40.0,
        clean_exit=False,
        label="",
    ):
        # carry_z is LOW (just above the cloth) so the grabbed edge is dragged
        # over the fabric along the table instead of lifted high -- the cloth
        # never hangs from the gripper during the fold.
        self.phase = label or "fold"
        px, py = float(pick_xy[0]), float(pick_xy[1])
        qx, qy = float(place_xy[0]), float(place_xy[1])
        self.example.model.soft_contact_mu = self.grasp_mu  # high friction to hold during grasp+carry
        self.goto([px, py, retreat_z], rot, OPEN, max_steps=140, tol=2.0)  # approach above (empty)
        ee, _ = self.goto([px, py, grasp_z], rot, OPEN, max_steps=200, tol=1.0)  # descend & straddle
        self.goto([px, py, grasp_z], rot, CLOSE, max_steps=60, tol=0.5, hold=25)  # grasp
        self.goto([px, py, carry_z], rot, CLOSE, max_steps=140, tol=1.5)  # lift just clear of the table
        self.goto([qx, qy, carry_z], rot, CLOSE, max_steps=200, tol=1.5)  # drag across low (fold over)
        self.goto([qx, qy, place_z], rot, CLOSE, max_steps=140, tol=1.5)  # lay down
        if clean_exit:
            self.release_clean(qx, qy, place_z, retreat_z, rot)
        else:
            self.goto([qx, qy, place_z], rot, OPEN, max_steps=40, tol=1.0, hold=15)  # release
            self.goto([qx, qy, retreat_z], rot, OPEN, max_steps=120, tol=2.0)  # retreat
        m = foldedness(self.example.state_0.particle_q.numpy())
        self.op_log.append(
            {
                "label": label,
                "pick": [px, py],
                "place": [qx, qy],
                "grasp_ee_z": float(ee[2]),
                **{k: m[k] for k in ("footprint", "fx", "fy", "rgxy")},
            }
        )
        return m

    def _cloth_hanging(self, ee, xy_radius=16.0, z_thresh=30.0, count=15):
        """True if a chunk of cloth is near the gripper xy AND lifted high --
        i.e. it got carried by the gripper instead of staying on the table."""
        q = self.example.state_0.particle_q.numpy()
        d = np.hypot(q[:, 0] - ee[0], q[:, 1] - ee[1])
        return int(((d < xy_radius) & (q[:, 2] > z_thresh)).sum()) >= count

    def release_clean(self, qx, qy, place_z, lift_z, rot, max_tries=2):
        """Release the cloth and make sure it is actually left on the table.

        A deep grasp leaves the fingers under the fold, so lifting straight up
        scoops the whole stack. We open fully, slide the fingers out sideways at
        low height, then lift and CHECK whether cloth came up with the gripper;
        if it did, drop back down, sweep further out, and retry. This makes the
        release robust to the solver's non-determinism."""
        rake_z = 17.0  # below the cloth resting height (~22) so fingers slide out UNDER it
        # Drop gripper<->cloth friction only during the release MOTION so the
        # cloth slips off the open fingers; restore it once the gripper is lifted
        # clear so the folded cloth holds (a low-friction park lets folds relax
        # apart, and the cleared gripper can't re-stick).
        self.example.model.soft_contact_mu = self.release_mu
        try:
            self.goto([qx, qy, place_z], rot, 1.0, max_steps=40, tol=1.0, hold=25)  # full open
            ex = qx + (20.0 if qx >= 0.0 else -20.0)  # slide outward toward nearest edge
            self.goto([qx, qy, rake_z], rot, 1.0, max_steps=90, tol=1.5)
            self.goto([ex, qy, rake_z], rot, 1.0, max_steps=120, tol=1.5)
            for _ in range(max_tries):
                self.goto([ex, qy, lift_z], rot, 1.0, max_steps=120, tol=2.0)
                if not self._cloth_hanging(_ee_pose(self.example)[:3]):
                    break  # clean -- cloth stayed on the table
                self.goto([ex, qy, rake_z], rot, 1.0, max_steps=120, tol=1.5)  # back down low
                ex = max(-33.0, min(33.0, ex + (14.0 if ex >= 0.0 else -14.0)))  # rake further out
                self.goto([ex, qy, rake_z], rot, 1.0, max_steps=140, tol=1.5)
        finally:
            self.example.model.soft_contact_mu = self.grasp_mu  # restore: hold the fold

    def fold_edge_y(
        self, far=True, xoff=0.0, grasp_z=12.0, inboard=5.0, place_dy=2.0, place_z=24.0, clean_exit=True, label=""
    ):
        """Fold the current far (most -y) or near (most +y) cloth edge to the
        Y-centroid. The pick point is read from the LIVE particle positions
        (around column ``centroid_x + xoff``) so it tracks where the fabric
        actually ended up after earlier folds -- a fixed grab point misses once
        the hem has moved inward. Use two calls per edge (xoff = -/+) to fold
        the full width of a wide strip; a single grab only pulls a tongue."""
        q = self.example.state_0.particle_q.numpy()
        cx, cy = float(q[:, 0].mean()), float(q[:, 1].mean())
        gx = cx + xoff
        sel = q[np.abs(q[:, 0] - gx) < 9.0, 1]  # fabric column around gx
        if far:
            y_edge = float(np.percentile(sel, 4.0))
            py = y_edge + inboard  # grab just inboard of the edge to catch fabric
            qy = cy + place_dy
        else:
            y_edge = float(np.percentile(sel, 96.0))
            py = y_edge - inboard
            qy = cy - place_dy
        return self.fold_op((gx, py), (gx, qy), grasp_z=grasp_z, place_z=place_z, clean_exit=clean_exit, label=label)

    def fold_edge_x(self, right=True, yoff=0.0, grasp_z=14.0, inboard=2.0, place_dx=4.0, place_z=26.0, label=""):
        """Fold the current right (+x) or left (-x) cloth edge toward the
        X-centroid (just past it). Pick point is read from live particle
        positions around row ``centroid_y + yoff`` so it tracks the actual
        side edge after the earlier (top/bottom) folds."""
        q = self.example.state_0.particle_q.numpy()
        cx, cy = float(q[:, 0].mean()), float(q[:, 1].mean())
        gy = cy + yoff
        sel = q[np.abs(q[:, 1] - gy) < 12.0, 0]  # fabric row around gy
        if right:
            x_edge = float(np.percentile(sel, 96.0))
            px = x_edge - inboard
            qx = cx - place_dx
        else:
            x_edge = float(np.percentile(sel, 4.0))
            px = x_edge + inboard
            qx = cx + place_dx
        return self.fold_op((px, gy), (qx, gy), grasp_z=grasp_z, place_z=place_z, label=label)

    def park(self, pos=(-40.0, -72.0, 42.0), hold=25, label="park"):
        """Move the (open) gripper up and away and pause, so the cloth is fully
        released and settles before the next fold phase begins."""
        self.phase = label
        self.goto(list(pos), ROT_DOWN, 1.0, max_steps=180, tol=2.0)
        for _ in range(hold):
            self.step(list(pos), ROT_DOWN, 1.0)

    def retreat(self, pos=(-30.0, -62.0, 46.0)):
        self.phase = "done"
        self.goto(list(pos), ROT_DOWN, OPEN, max_steps=160, tol=2.0)

    def finalize(self, debug_fps=20):
        if self.pipe is not None:
            self.example.render()
            last = self.viewer.get_frame().numpy()
            self.pipe.close()
            from PIL import Image

            Image.fromarray(last).save(self.run_dir / "final_frame.png")

        settled = self.metrics_ts[0]
        for m in self.metrics_ts:
            if m["phase"] != "settle":
                break
            settled = m
        final_m = self.metrics_ts[-1]
        summary = {
            "n_steps": self.step_i,
            "settled_footprint": settled["footprint"],
            "final_footprint": final_m["footprint"],
            "footprint_ratio": final_m["footprint"] / max(settled["footprint"], 1e-6),
            "settled_rgxy": settled["rgxy"],
            "final_rgxy": final_m["rgxy"],
            "rgxy_ratio": final_m["rgxy"] / max(settled["rgxy"], 1e-6),
            "settled_fx": settled["fx"],
            "settled_fy": settled["fy"],
            "final_fx": final_m["fx"],
            "final_fy": final_m["fy"],
            "final_ztop": final_m["ztop"],
            "max_arm_qd": self.max_qd,
            "op_log": self.op_log,
            "metrics_ts": self.metrics_ts,
        }
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2)
        make_debug_video(self.snaps, self.run_dir / "debug.mp4", fps=debug_fps)
        return summary


def run_primitive_fold(
    run_dir, ops=FOLD_OPS, view="gl", settle_steps=60, grasp_z=14.0, retreat_pos=(-35.0, -70.0, 52.0), **kw
):
    """Run the primitive-based fold and write final/debug videos + metrics."""
    sess = FoldSession(run_dir, view=view, **kw)
    sess.settle(settle_steps)
    for label, pick, place in ops:
        sess.fold_op(pick, place, grasp_z=grasp_z, label=label)
    sess.retreat(retreat_pos)
    return sess.finalize()


# Left/right side folds (sleeve + body per side) on the undeformed shirt.
LR_OPS = [
    ("fold +X sleeve", (30.0, -57.0), (-4.0, -57.0)),
    ("fold +X body", (15.0, -57.0), (-10.0, -57.0)),
    ("fold -X sleeve", (-30.0, -57.0), (4.0, -57.0)),
    ("fold -X body", (-15.0, -57.0), (10.0, -57.0)),
]

# Top/bottom folds on the FLAT (single-layer) shirt -- two grabs per edge so the
# full width folds. Done FIRST: folding the length while the shirt is flat is a
# clean single-layer fold that releases cleanly, whereas folding the length
# AFTER the side folds means scooping a thick multilayer strip, which both
# folds incompletely and lifts the whole stack off the table on release.
TB_FLAT_OPS = [
    ("fold top L", (-11.0, -80.0), (-11.0, -48.0)),
    ("fold top R", (11.0, -80.0), (11.0, -48.0)),
    ("fold bottom L", (-11.0, -16.0), (-11.0, -52.0)),
    ("fold bottom R", (11.0, -16.0), (11.0, -52.0)),
]


def run_quarter_fold(run_dir, view="gl", settle_steps=50, grasp_z=14.0, retreat_pos=(-35.0, -72.0, 52.0), **kw):
    """Run 5: quarter fold in BOTH axes.

    Folds the TOP and BOTTOM halves to the center first (on the flat shirt),
    then folds the LEFT and RIGHT sides in last (adaptive edge grabs on the
    Y-folded band). This order is what makes it reliable: both folds act on
    foldable fabric and release cleanly, leaving the cloth on the table."""
    sess = FoldSession(run_dir, view=view, **kw)
    sess.settle(settle_steps)
    # 1) top & bottom halves -> center (flat single-layer shirt)
    for label, pick, place in TB_FLAT_OPS:
        sess.fold_op(pick, place, grasp_z=grasp_z, label=label)
    # 2) left & right sides -> center, two adaptive grabs per side (last folds)
    sess.fold_edge_x(right=True, inboard=2.0, place_dx=4.0, grasp_z=grasp_z, label="fold +X edge")
    sess.fold_edge_x(right=True, inboard=6.0, place_dx=9.0, grasp_z=grasp_z, label="fold +X body")
    sess.fold_edge_x(right=False, inboard=2.0, place_dx=4.0, grasp_z=grasp_z, label="fold -X edge")
    sess.fold_edge_x(right=False, inboard=6.0, place_dx=9.0, grasp_z=grasp_z, label="fold -X body")
    sess.retreat(retreat_pos)
    return sess.finalize()


# Left/right fold: sleeve + body per side, carried just past center.
LR_FOLD_OPS = [
    ("fold +X sleeve", (30.0, -57.0), (-6.0, -57.0)),
    ("fold +X body", (16.0, -57.0), (-12.0, -57.0)),
    ("fold -X sleeve", (-30.0, -57.0), (6.0, -57.0)),
    ("fold -X body", (-16.0, -57.0), (12.0, -57.0)),
]


def run_two_phase_fold(run_dir, view="gl", settle_steps=50, grasp_z=15.0, **kw):
    """Run 6: two clean, distinct phases in the requested order.

    Phase 1 (X fold): fold LEFT and RIGHT in, then PARK -- the gripper backs
    away and the cloth is verified released and left flat on the table.
    Phase 2 (Y fold): fold TOP and BOTTOM in, then park again.

    Every grasp uses the reactive clean release (``release_clean``): it slides
    the fingers out and checks the cloth did not come up with the gripper,
    sweeping further and retrying if it did -- so the cloth never hangs from
    the gripper between or after the folds."""
    sess = FoldSession(run_dir, view=view, **kw)
    sess.settle(settle_steps)

    # --- Phase 1: left & right ---
    for label, pick, place in LR_FOLD_OPS:
        sess.fold_op(pick, place, grasp_z=grasp_z, clean_exit=True, label=label)
    sess.park(label="release after X-fold")

    # --- Phase 2: top & bottom (adaptive edges, two grabs per edge) ---
    for xo in (-7.0, 7.0):
        sess.fold_edge_y(far=True, xoff=xo, grasp_z=grasp_z, label=f"fold top {xo:+.0f}")
    for xo in (-7.0, 7.0):
        sess.fold_edge_y(far=False, xoff=xo, grasp_z=grasp_z, label=f"fold bottom {xo:+.0f}")
    sess.park(label="release after Y-fold")
    return sess.finalize()


# Run 7: a CLEAN three-fold sequence in the style of the original scripted demo
# (example_cloth_franka.py) -- one grasp per fold, a short waypoint path, no
# parks/retries/rakes. Each waypoint is (x, y, z [cm], gripper, hold, label);
# the arm is driven to each in turn and `hold` extra steps let the gripper
# actuate on grasp/release. Folds: LEFT->center, RIGHT->center, BOTTOM->top.
_GZ = 17.0  # grasp height (EE reaches ~18, grabs the edge)
_CZ = 28.0  # low carry height: drag the fold over the fabric, no high hang
_LZ = 24.0  # lay-down height
CLEAN_FOLD_WAYPOINTS = [
    # --- Fold 1: LEFT (-X) half to center ---
    (-28.0, -55.0, 38.0, OPEN, 0, "fold left: approach"),
    (-28.0, -55.0, _GZ, OPEN, 0, "fold left: descend"),
    (-28.0, -55.0, _GZ, CLOSE, 30, "fold left: grasp"),
    (-12.0, -55.0, _CZ, CLOSE, 0, "fold left: lift & drag in"),
    (3.0, -55.0, _CZ, CLOSE, 0, "fold left: drag past center"),
    (3.0, -55.0, _LZ, CLOSE, 0, "fold left: lay down"),
    (3.0, -55.0, _LZ, OPEN, 25, "fold left: release"),
    (3.0, -55.0, 40.0, OPEN, 0, "fold left: lift away"),
    # --- Fold 2: RIGHT (+X) half to center ---
    (28.0, -55.0, 38.0, OPEN, 0, "fold right: approach"),
    (28.0, -55.0, _GZ, OPEN, 0, "fold right: descend"),
    (28.0, -55.0, _GZ, CLOSE, 30, "fold right: grasp"),
    (12.0, -55.0, _CZ, CLOSE, 0, "fold right: lift & drag in"),
    (-3.0, -55.0, _CZ, CLOSE, 0, "fold right: drag past center"),
    (-3.0, -55.0, _LZ, CLOSE, 0, "fold right: lay down"),
    (-3.0, -55.0, _LZ, OPEN, 25, "fold right: release"),
    (-3.0, -55.0, 40.0, OPEN, 0, "fold right: lift away"),
    # --- Fold 3: BOTTOM (near edge) up to TOP (far edge) ---
    (0.0, -20.0, 38.0, OPEN, 0, "fold bottom: approach"),
    (0.0, -20.0, _GZ, OPEN, 0, "fold bottom: descend"),
    (0.0, -20.0, _GZ, CLOSE, 30, "fold bottom: grasp"),
    (0.0, -42.0, _CZ, CLOSE, 0, "fold bottom: lift & drag up"),
    (0.0, -66.0, _CZ, CLOSE, 0, "fold bottom: drag toward top"),
    (0.0, -74.0, _LZ, CLOSE, 0, "fold bottom: lay down"),
    (0.0, -74.0, _LZ, OPEN, 25, "fold bottom: release"),
    (0.0, -74.0, 42.0, OPEN, 0, "fold bottom: lift away"),
    # retreat clear of the folded cloth
    (-35.0, -75.0, 52.0, OPEN, 0, "retreat"),
]


def run_clean_fold(run_dir, waypoints=CLEAN_FOLD_WAYPOINTS, view="gl", settle_steps=60, **kw):
    """Run 7: a clean three-fold (left->center, right->center, bottom->top).

    One grasp per fold; the grabbed edge is dragged over the fabric at low
    height (no high lift, so the cloth never hangs) and released with a simple
    open. No parks, retries, or rakes -- minimal motion, matching the scripted
    demo's structure."""
    sess = FoldSession(run_dir, view=view, **kw)
    sess.settle(settle_steps)
    for x, y, z, grip, hold, label in waypoints:
        sess.phase = label
        # bigger step budget for the long bottom->top drag
        sess.goto([x, y, z], ROT_DOWN, grip, max_steps=240, tol=1.3, hold=hold)
    return sess.finalize()
