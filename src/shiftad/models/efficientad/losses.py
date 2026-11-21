"""Loss functions for EfficientAD training.

Implements the local (student-teacher) and global (autoencoder) loss
components, combined into a configurable total loss. The local loss
measures the discrepancy between student and teacher feature maps, while
the global loss measures the autoencoder reconstruction error.
"""

from __future__ import annotations

import torch
from torch import Tensor


def local_loss(student_match: Tensor, teacher_output: Tensor) -> Tensor:
    """Compute the local student-teacher feature-matching loss.

    Uses mean squared error between the student's teacher-matching channels
    and the frozen teacher output. Both tensors must have the same shape.

    Args:
        student_match: Student output channels matching the teacher,
            shaped ``[B, C, H, W]``.
        teacher_output: Frozen teacher output, shaped ``[B, C, H, W]``.

    Returns:
        A scalar loss tensor.

    Raises:
        RuntimeError: If tensor shapes do not match.
    """

    if student_match.shape != teacher_output.shape:
        raise RuntimeError(
            f"Local loss shape mismatch: student {tuple(student_match.shape)} "
            f"vs teacher {tuple(teacher_output.shape)}."
        )
    return torch.mean((student_match - teacher_output) ** 2)


def global_loss(ae_reconstruction: Tensor, ae_target: Tensor) -> Tensor:
    """Compute the global autoencoder reconstruction loss.

    Uses mean squared error between the autoencoder output and the student's
    AE branch output (detached to prevent gradients flowing back through
    the student via the AE loss).

    Args:
        ae_reconstruction: Autoencoder output, shaped ``[B, C, H, W]``.
        ae_target: Student AE branch output (will be detached),
            shaped ``[B, C, H, W]``.

    Returns:
        A scalar loss tensor.

    Raises:
        RuntimeError: If tensor shapes do not match.
    """

    if ae_reconstruction.shape != ae_target.shape:
        raise RuntimeError(
            f"Global loss shape mismatch: reconstruction {tuple(ae_reconstruction.shape)} "
            f"vs target {tuple(ae_target.shape)}."
        )
    return torch.mean((ae_reconstruction - ae_target.detach()) ** 2)


class EfficientADLoss:
    """Combine local and global losses with configurable weights.

    Args:
        lambda_global: Weight for the global (autoencoder) loss term.
            The total loss is ``local + lambda_global * global``.

    Notes:
        The local loss trains the student to predict teacher features.
        The global loss trains the autoencoder to reconstruct student
        AE-channel output. Gradients from the global loss do not flow
        back through the student (the AE target is detached).
    """

    def __init__(self, *, lambda_global: float = 1.0) -> None:
        if lambda_global < 0:
            raise ValueError("lambda_global must be non-negative.")
        self._lambda_global = lambda_global

    @property
    def lambda_global(self) -> float:
        """Return the global loss weight."""

        return self._lambda_global

    def __call__(
        self,
        student_match: Tensor,
        teacher_output: Tensor,
        ae_reconstruction: Tensor,
        ae_target: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the combined EfficientAD loss.

        Args:
            student_match: Student's teacher-matching channels.
            teacher_output: Frozen teacher output.
            ae_reconstruction: Autoencoder reconstruction.
            ae_target: Student's AE channels (target for reconstruction).

        Returns:
            A dictionary with keys ``total``, ``local``, and ``global``,
            each mapping to a scalar tensor.
        """

        l_local = local_loss(student_match, teacher_output)
        l_global = global_loss(ae_reconstruction, ae_target)
        total = l_local + self._lambda_global * l_global
        return {"total": total, "local": l_local, "global": l_global}
