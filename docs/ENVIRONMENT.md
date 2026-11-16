# Environment: Isaac Lab's Allegro cube repose in mjlab

The environment re-implements Isaac Lab's `Isaac-Repose-Cube-Allegro-Direct-v0` (Isaac Lab v2.3.2:
`isaaclab_tasks/direct/allegro_hand/allegro_hand_env_cfg.py`, `isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py`,
`isaaclab_assets/robots/allegro.py`) on mjlab 1.5.3 (MuJoCo Warp). This page lists every setting
and every place where the port differs from Isaac Lab, with the reason.

## Hand

| | Isaac Lab | here |
|---|---|---|
| model | `allegro_hand_instanceable.usd` (Isaac Nucleus asset) | MuJoCo Menagerie `wonik_allegro/right_hand.xml` (Allegro Hand V3), unmodified on disk |
| joint / body names | `index_joint_0` ... `thumb_joint_3`, `index_link_0` ... | same (renamed in memory: `ffj0` -> `index_joint_0`, `ff_base` -> `index_link_0`, `palm` -> `palm_link`) |
| joint limits | from the USD asset | the same limits, in degrees in `configs/task/allegro_repose.yaml` (the Menagerie model's own limits are narrower) |
| actuators | implicit PD, stiffness 3.0, damping 0.1, effort limit 0.5 N m, joint friction 0.01 | position actuators kp 3.0, kv 0.1, force range +/-0.5 N m; joint friction loss 0.01; no other joint damping |
| gravity | disabled on the hand (`disable_gravity=True`) | gravity compensation on every hand body |
| self-collision | enabled | enabled (Menagerie excludes only palm / finger-base pairs) |
| default joint pos | 0, `thumb_joint_0` = 0.28 | same |
| mounting | root at (0, 0, 0.5), rotation (0.257551, 0.283045, 0.683330, -0.621782) | same root pose; see below |
| **armature** | 0 | **3e-4 kg m^2** |
| **integrator** | PhysX (implicit drives) | **implicitfast**, 1/120 s |

**Why the armature.** PhysX integrates its joint drives implicitly. MuJoCo integrates the
actuator damping implicitly (`implicitfast`) but the PD stiffness explicitly, and the Menagerie
distal links weigh 7 g with inertias down to 4e-7 kg m^2. With kp = 3 at 120 Hz this diverges
(the hand explodes within 0.13 s). A rotor inertia of 3e-4 kg m^2 per joint keeps it stable; a
0.8 rad step is then reached within 3 control steps (0.1 s), about as fast as a stiff drive allows
at 30 Hz.

**Mounting.** Isaac Lab's asset places its palm frame (`palm_link`) with a fixed joint inside
the asset; the Menagerie palm frame has different axes. The two frames are matched by their axes
(fingers along Menagerie +z = Isaac `palm_link` +x, finger row along +y / -y, palm normal along
+x / +z) and their origins at the middle finger's base joint. Index and ring finger bases then
agree within 2 mm. The Isaac Lab asset's thumb base sits 4.4 cm closer to the wrist than on the
Allegro V3 model (a different hand revision); this cannot be matched without changing the hand.
With Isaac Lab's root pose the palm faces up and the fingers point along world -y, 5 deg below
horizontal.

**Why not the Isaac Lab asset itself.** The USD hand and cube are NVIDIA Isaac Sim assets whose
license does not allow redistribution in this repository. They were read only for numbers (joint
limits, palm frame, cube size). The Menagerie model is BSD-2 and derived from the same public
Allegro URDF.

## Cube

| | Isaac Lab | here |
|---|---|---|
| shape | DexCube (6 cm) at scale 1.2 = 7.2 cm | box, half-size 3.6 cm |
| mass | density 400 -> 0.149 kg | same (inertia of a solid box) |
| friction | static = dynamic = 1.0 (scene material) | 1.0 sliding; torsional 0.005, rolling 1e-4 (`condim 4`; the DexCube collider has a torsional patch) |
| spawn | (0, -0.17, 0.56) | same |

## MDP

All formulas are Isaac Lab's (`src/allegro_repose/task/core.py`).

* **Control:** 30 Hz (physics 1/120 s, decimation 4). Episode 10 s; time-out at step 299, as in Isaac Lab.
* **Action:** 16 absolute joint targets in [-1, 1], mapped linearly onto each joint's limits
  (`act_moving_average` = 1, i.e. no smoothing). Order: index 0-3, middle 0-3, ring 0-3, thumb 0-3
  (the MJCF order; Isaac Lab's actuation order is a permutation of it).
* **Observation (124-D, "full"):** joint positions scaled to [-1, 1]; 0.2 x joint velocities; cube
  position, orientation, linear velocity, 0.2 x angular velocity; `in_hand_pos`; goal orientation;
  orientation error (cube x goal^-1); fingertip positions, orientations and velocities (4 x 3,
  4 x 4, 4 x 6); last action. The critic sees the same vector (`asymmetric_obs = False`).
  Fingertip poses and velocities are computed by forward kinematics from the joint state
  (`task/kinematics.py`, checked against MuJoCo in `tests/test_kinematics.py`); Isaac Lab reads the
  same quantities from the simulator (`body_pos_w`, `body_quat_w`, `body_vel_w`, velocity at the
  link's centre of mass).
* **Reward:** -10 |p - in_hand_pos| + 1 / (rot_dist + 0.1) - 0.0002 |a|^2 + 250 on success; no
  fall penalty. `rot_dist = 2 asin(|vec(q g^-1)|)`; `in_hand_pos` = spawn point - 4 cm in z.
* **Success:** rot_dist <= 0.2 rad. A new goal (Rx(pi u0) Ry(pi u1), u ~ U(-1, 1)) is sampled
  immediately; episodes do not end on success (`max_consecutive_success = 0`).
* **Termination:** |p - in_hand_pos| >= 0.24 m (the cube fell).
* **Reset:** cube at the spawn point + U(-1, 1) cm per axis, random orientation (same
  distribution as goals), zero velocity, dropped onto the hand; joints at the default positions
  plus Isaac Lab's noise `0.2 * (delta_min + (delta_max - delta_min) * 0.5 * U(-1, 1))`, clamped to
  the limits; zero joint velocity.
* **No domain randomization** (the Isaac Lab Direct task has none).

## Solver

Newton solver, 100 iterations, 50 line-search iterations, elliptic friction cone, impratio 1,
`njmax` 400, `nconmax` 200 per env.

## Evaluation

`src/allegro_repose/common/evaluate.py`: deterministic actions, 1024 (training) or 4096 (final)
independent episodes of 10 s, fixed seed. **Goals per episode** (consecutive successes) is the
main metric; also fall rate, mean rotation distance and return.

## Hold variant (`configs/task/allegro_repose_hold.yaml`)

The Isaac Lab policy reaches goals but does not stay on them: a new goal is drawn the step the cube
first enters the tolerance, so nothing rewards holding a pose, and the 1 / (rot_dist + 0.1) term keeps
pushing for the last fraction of a degree. The hold variant keeps the hand, cube, physics, actions,
resets and terminations, and changes:

* **Dwell:** a goal counts, and is replaced, only after the cube has stayed within the 0.2 rad
  tolerance for 30 consecutive steps (1 s). Leaving the tolerance resets the count.
* **Reach bonus per step:** the 250 bonus is paid on every step within tolerance (as before), so with
  the dwell it pays for every step of holding instead of for a single touch.
* **Flat orientation reward near the goal:** 1 / (max(0, rot_dist - 0.31) + 0.1): unchanged at large
  errors, constant within 0.31 rad (18 deg), so there is no incentive to keep adjusting once there.
* **Action-rate penalty:** -0.05 * sum_j relu(|a_j - a_prev,j| - 0.05)^2: small per-step changes of
  the joint targets are free, large jumps cost quadratically.
* **Observation (asymmetric).** Actor, 89-D, only what a real hand with a cube tracker measures:
  joint positions scaled to [-1, 1], fingertip positions, fingertip orientations, cube orientation,
  goal orientation and their difference, the in-hand target position, and the last action; all
  orientations in rotation-6D (the first two columns of the rotation matrix), which, unlike a
  quaternion, has no sign ambiguity. Critic, 138-D: the actor's input plus 0.2 x joint velocities,
  fingertip linear (link frame) and angular velocities, cube position, cube linear velocity and
  0.2 x cube angular velocity.

Evaluation counts a goal when it is completed (held for the dwell and replaced), and also reports the
fraction of time the cube is within tolerance (`on_goal_frac`).
