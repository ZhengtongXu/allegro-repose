# Allegro Hand V3 (right), MJCF and meshes

Copied byte-for-byte from MuJoCo Menagerie, `wonik_allegro/`
(https://github.com/google-deepmind/mujoco_menagerie, commit 98a70c569c4c1b6da467a20012f766ee1f8dc521):
`right_hand.xml`, `LICENSE` and the meshes it references. `SHA256SUMS` lists the copied files; a
test checks them.

License: BSD-2-Clause (Copyright (c) 2016, SimLab), see `LICENSE`. The model is derived from the
public Allegro Hand URDF of Wonik Robotics / SimLab.

The files are never edited. The Isaac Lab Allegro setup (joint and body names, joint limits, PD
gains, effort limits, joint friction, disabled gravity, mounting pose) is applied in memory when
the scene is built, see `src/allegro_repose/assets.py`.
