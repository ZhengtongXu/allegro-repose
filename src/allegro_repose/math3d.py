"""Rotation helpers used by the task (quaternions are wxyz, batched on the last axis; plain PyTorch)."""

from __future__ import annotations

import torch

_EPS = 1e-8


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    r = torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    )
    return r.reshape(*q.shape[:-1], 3, 3)




def rot6d_from_matrix(r: torch.Tensor) -> torch.Tensor:
    """First two columns of R, flattened (Zhou et al. 2019 continuous 6D)."""
    return torch.cat((r[..., :, 0], r[..., :, 1]), dim=-1)


def rot6d_from_quat(q: torch.Tensor) -> torch.Tensor:
    return rot6d_from_matrix(quat_to_matrix(q))




def quat_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    """so(3) vector -> quaternion; smooth at zero."""
    theta_sq = (axis_angle * axis_angle).sum(-1, keepdim=True)
    theta = torch.sqrt(theta_sq + _EPS * _EPS)
    half = 0.5 * theta
    # sin(half)/theta with a Taylor fallback near zero.
    small = theta_sq < 1e-6
    k = torch.where(small, 0.5 - theta_sq / 48.0, torch.sin(half) / theta)
    return torch.cat((torch.cos(half), axis_angle * k), dim=-1)






def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (quat_to_matrix(q) @ v.unsqueeze(-1)).squeeze(-1)
