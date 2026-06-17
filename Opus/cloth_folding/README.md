# Fold the cloth — Franka + VBD cloth

Task: drive the Franka in the Newton example
`newton/examples/cloth/example_cloth_franka_no_target.py` to **fold the T-shirt
on the table**, using only the end-effector command API
`Example.step(ee_pos, ee_rot, gripper)`. Each run saves a debug video, a final
video, a metrics file, and the code that produced it.

**Dependency:** this project drives a Newton example, so it imports the Newton
working tree. The scripts default to `/home/yilang/research/newton`; set the
`NEWTON_REPO` environment variable to point at your Newton checkout if it lives
elsewhere. Run with the `newton` conda env (which has warp/pxr/pyglet).

**Run 7 is the latest** — a clean three-fold (left→center, right→center,
bottom→top), minimal motion. From this project directory:

```
NEWTON_REPO=/home/yilang/research/newton \
  /home/yilang/miniconda3/envs/newton/bin/python run7/fold_run7.py
```

## Shared harness — `fold_lib.py`

- `FoldSession` drives the example and, while stepping, captures the headless
  GL render (`final.mp4`), particle snapshots for a top-down/side + metric
  `debug.mp4`, and a foldedness time series (`metrics.json`).
- `FoldExample(Example)` overrides the controller with **damped least squares**
  IK + task/joint velocity clamps + reduced nullspace gain (Run 4). The stock
  controller spiked joint velocities to ~94 rad/s and rarely converged; this one
  reaches every target at ~1.5 rad/s.
- `goto(pos, rot, grip, ...)` drives the EE to a target **until convergence**
  (error < tol), instead of holding a target for a fixed time.
- `fold_op(pick_xy, place_xy)` = approach → descend open (straddle) → close →
  lift → carry across → lower → release → retreat.
- Foldedness metric: XY **footprint** (2.5–97.5 pct bbox area) and XY
  **radius of gyration**. `metrics.json` also records `max_arm_qd` (peak arm
  joint speed) as a motion-realism check.

## Runs

| run | strategy | footprint (cm²) | reduction | rgxy (cm) | peak arm speed | result |
|-----|----------|-----------------|-----------|-----------|----------------|--------|
| 1 | faithful replay of the original scripted waypoints | 3493 → 3500 | **0%** | 23.7 → 23.8 | ~94 rad/s | FAIL — no grasp |
| 2 | convergence `goto` + grasp below cloth top; fold sides in + far-Y across center | 3475 → 1006 | **71%** | 23.6 → 11.8 | ~94 rad/s | working but violent motion |
| 3 | as run 2, side folds placed just past center (tighter) | 3476 → 748 | **78.5%** | 23.6 → 9.95 | ~94 rad/s | tight fold, **unrealistic motion** |
| 4 | **DLS controller** + 2-grab-per-side plan (folds left/right only) | 3473 → 857 | **75%** | 23.6 → 12.4 | **1.5 rad/s** | tight fold, smooth — but length not folded |
| 5 | **quarter fold** — top/bottom then left/right (`run_quarter_fold`) | 3475 → 626 | **82%** | 23.6 → ~9 | ~1 rad/s | folded in **both axes** (fx 57→33, fy 61→19), on table |
| 6 | **two phases + low-carry** — left/right → release → top/bottom (`run_two_phase_fold`) | 3475 → 478 | **86%** | 23.6 → ~9 | ~0.9 rad/s | fx 57→20, fy 61→23; cloth **dragged flat along the table** (no hanging during folds), distinct phases, flat on table at end |
| 7 | **clean 3-fold** — left→center, right→center, bottom→top (`run_clean_fold`) | 3475 → 672 | **81%** | 23.6 → ~10 | ~1.4 rad/s | one grasp per fold, **minimal motion** (1085 steps / 18 s vs 6000 / 64 s), low-carry (no hanging), flat on table |

### Run 1 — why the naive replay fails
Replaying the original demo's key poses through the new API does **not** fold the
cloth. The resolved-rate IK lags ~5–15 cm, so at the "close gripper" waypoint the
EE is still ~8 cm above the fabric — the gripper closes in mid-air and never
grabs. Targeting the cloth-top height (z=20) only reached z≈23 (grazing the top).
Footprint stays flat the entire run.

### The grasp fix (`grasp_test.py`)
A grasp succeeds only when the EE descends **below** the cloth top so the fingers
straddle the fabric: targeting z=16 reaches z≈17 and the cloth lifts +17 cm;
z=20 grazes and lifts nothing. Combined with a `goto` that converges before the
gripper acts, grasping became reliable.

### The fold strategy (`fold_strategy_test.py`)
The shirt lies with sleeves at (±30, −58) and body over X[−21,21], Y[−79,−20].
Fold both X sides in toward the center, then fold the far (top) half **across**
the center line (y=−50) onto the near half (two grabs for full width). Carrying
the far hem only partway (to y=−46) just pulls a tongue; carrying it across the
center (to y=−26) folds the length in half.

### Run 4 — fixing the motion (the real problem in runs 2–3)
Runs 2–3 folded well on the *metric* but the **arm motion was bad**, which the
diagnostic (`diagnose`-style sweep) pinned to three causes:
1. the stock plain-pseudo-inverse controller spiked joint velocities to
   **94 rad/s** near ill-conditioned configs (a Franka does ~2.5) → violent,
   unrealistic motion;
2. most waypoints **timed out without converging** → the EE never reached its
   target and the arm perpetually chased;
3. grasps clipped ~6 cm through the table.

`FoldExample` replaces the controller with **damped least squares**
(`dq = Jᵀ(JJᵀ+λ²I)⁻¹ v`) + a task-velocity clamp + a hard joint-speed cap +
a reduced nullspace pull. Result: **all targets converge** and the **peak arm
joint speed drops to 1.5 rad/s** — smooth, realistic motion.

The gentle controller moves less fabric per grasp than the old aggressive one,
so the fold plan is enriched (`FOLD_OPS_V4`): each side is folded with **two
grabs** (sleeve + body edge), then the far Y-half is folded onto the near half.
This recovers a tight, well-centered fold (~75–78% footprint reduction) *and*
realistic motion.

Remaining caveat: the grasp still descends ~5 cm below the table top, which the
solver lets pass because **robot↔table contact is disabled during the robot
substep** in the example (`shape_contact_pair_count = 0`). That is a property of
the environment, not the controller, so it can't be removed from the control
side alone.

### Run 5 — quarter fold (both axes)
Run 4 only folded the cloth left/right; the length was never folded. Run 5
(`run_quarter_fold`) folds in **both** axes: fx 57→33, fy 61→19, footprint
3475→626 (~82%), cloth left flat on the table, peak joint speed ~1 rad/s.

Two non-obvious things were needed:
1. **Adaptive edge grabs** (`fold_edge_x` / `fold_edge_y`): a fixed grab point
   misses once a hem has moved inward after an earlier fold, and grabs the bare
   table instead of the fabric. The top/bottom and side grabs read the live
   cloth edge each time. A single grab also only pulls a *tongue* of a wide
   strip, so each edge is folded with two grabs.
2. **Fold the length FIRST (top/bottom), sides LAST.** Folding the length
   *after* the side folds means scooping a thick multilayer strip: the fingers
   end up under the stack and lifting scoops the whole fold ~30 cm off the
   table on release. Folding the length while the shirt is still flat
   (single layer) is a clean fold that releases cleanly, and the side folds —
   which always released reliably — come last. So Run 5 folds top/bottom then
   left/right, which is the reverse of a "left/right then top/bottom" reading
   but is what reliably leaves a flat fold on the table. (A `clean_exit` lateral
   withdraw is also available on `fold_op` for deep grasps.)

### Run 6 — two clean, distinct phases (left/right, release, then top/bottom)
Run 6 (`run_two_phase_fold`) does the fold in the requested order as two clearly
separated phases, with the gripper releasing the cloth onto the table between
them:
1. **X fold**: fold left and right in (sleeve + body per side), then PARK — the
   arm backs away and the cloth is left as a flat folded strip on the table.
2. **Y fold**: fold top and bottom in (adaptive two-grab edges), then park.

Result: fx 57→20, fy 61→32, footprint 3475→645 (~81%), cloth flat on the table,
peak joint speed ~1.5 rad/s.

**Low-carry folds (no hanging during the fold).** Each fold drags the grabbed
edge *over the fabric at low height* (`carry_z ≈ 27`, just above the stack)
instead of lifting it ~16 cm into the air, so the cloth stays flat on the table
throughout the fold and never hangs from the gripper while being carried. (This
fixed the "cloth sticks to the gripper the whole time" issue.)

**The release is the hard part** (the cloth being carried up on the way out). To
grasp cloth lying flat, the fingers must reach *under* it — robot↔table contact
is disabled in the example (`shape_contact_pair_count = 0`), so the gripper
passes through the table — and a downward-pointing gripper then tends to carry
the cloth up when it leaves. Three things help:
- **Per-phase friction modulation**: `soft_contact_mu` is high (0.25) while
  grasping/carrying so the fold holds, then dropped to 0.02 *during the release
  motion* so the cloth slides off the open fingers, then restored so the folded
  cloth doesn't relax apart while the arm parks.
- **Rake-out below the cloth**: open fully, then slide the fingers out sideways
  at a height *below* the cloth so they exit from under it.
- **Reactive verify + retry** (`release_clean`): after lifting, check whether
  cloth came up near the gripper; if so, drop back and rake further out.

Honest caveat: this is **not 100% reliable**. The VBD self-contact solver is
non-deterministic, so ~60–75% of runs leave the cloth perfectly flat (this
render did, ztop 25 cm) and the rest lift it partially on the final release.
A fully reliable fix needs an *environment* change (enable robot↔table contact,
or exclude the gripper from the cloth contact set during release) — not solvable
from the control side alone.

### Run 7 — clean three-fold, minimal motion
Run 7 (`run_clean_fold`) strips the process down to the structure of the
original scripted demo: **one grasp per fold**, a short drag path to the target,
then a simple release — no parks, no reactive retries, no rake-outs. Three folds
in order:
1. **LEFT** half → center
2. **RIGHT** half → center (now a narrow strip)
3. **BOTTOM** (near edge) → **TOP** (far edge) — folds the strip in half

The trajectory is just a flat list of waypoints (`CLEAN_FOLD_WAYPOINTS`, each
`(x, y, z, gripper, hold, label)`) driven through the smooth DLS controller, and
every grabbed edge is dragged **over the fabric at low height** (`carry_z ≈ 28`)
so the cloth stays flat and never hangs. Result: fx 57→19, fy 61→36, footprint
3475→672 (~81%), cloth on the table, **1085 control steps / 18 s** versus
~6000 / 64 s for the multi-grab Run 6 — much cleaner, fewer motions.

(The final-release non-determinism noted for Run 6 still applies, but the
low-carry drag keeps the cloth flat throughout the folds either way.)

## Notes / limitations
- The folded bundle consistently settles ~15–20 cm to −X. Recentering it (lift
  or low slide) tends to unfold it, so it's left as-is — it stays well within the
  table and the footprint reduction is unaffected.
- The VBD self-contact solver is mildly **non-deterministic** (~±5% footprint
  run-to-run), so a rendered run lands within a few % of the null-viewer search.
- `final.mp4` is the GL 3D view; `debug.mp4` is top-down + side scatter (colored
  by height) with the live footprint curve.

## Files
- `fold_lib.py` — shared harness (`FoldExample` controller, `FoldSession`,
  primitives, video + metrics, fold plans)
- `grasp_test.py`, `fold_strategy_test.py` — diagnostics used to derive the fold
- `run1/` … `run7/` — per-run script + `final.mp4` + `debug.mp4` +
  `metrics.json` + `final_frame.png` + `run.log`
- `teleop_cloth_franka.py` — keyboard teleop driver (earlier task)
