# GELLO dynamics asset

`yam_active_gello.urdf` comes from
[`wuphilipp/gello_software`](https://github.com/wuphilipp/gello_software/tree/main/gello/factr/urdf/yam_active_gello)
and is used by its FACTR gravity-compensation implementation.

The `link_N_visual.stl` / `link_N_collision.stl` meshes referenced by the URDF
were never published — the upstream folder contains only `robot.urdf` and
`config.json`. That is fine here: the gravity model reads only the
`<inertial>` tags (mass + COM); `<visual>`/`<collision>` are ignored.

## Link → physical part mapping

Derived from the joint chain and link lengths. Masses include the motors
housed in each link (XL330 ≈ 18 g). This is also the base-to-tip order of
`gravity_link_masses_kg`.

| Slot | URDF link | Physical part | URDF mass |
| --- | --- | --- | --- |
| 0 | `link_base` | table pedestal below the pan axis (never moves — its mass has **no** effect on gravity torque) | 142 g |
| 1 | `link_1` | shoulder yoke between pan and lift axes | 92 g |
| 2 | `link_2` | upper arm, ~198 mm from shoulder lift to elbow | 121 g |
| 3 | `link_3` | forearm, ~184 mm from elbow to wrist-flex | 63 g |
| 4 | `link_4` | wrist-flex → wrist-roll bracket | 36 g |
| 5 | `link_5` | small roll → yaw coupler | 24 g |
| 6 | `link_6` | everything past the last axis: EEF handle, trigger mechanism, and the trigger motor | 103 g |

`link_6` is distal to every joint, so an underestimated handle mass
under-assists all joints, worst at extended poses.

Copyright (c) 2023 Philipp Wu. Distributed under the MIT License; see the
top-level `LICENSE-GELLO-SOFTWARE` file in this package.
