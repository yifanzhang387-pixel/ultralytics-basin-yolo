"""WIoU loss patch for Basin-YOLO experiments.

This module replaces the default BboxLoss.forward implementation with a
WIoU-v1 style weighted IoU loss for axis-aligned detection/segmentation boxes.
It keeps the original DFL branch unchanged.
"""

from __future__ import annotations

import torch

from ultralytics.utils import loss as loss_module


def _wiou_weight(pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Return WIoU-v1 distance attention weight for xyxy boxes."""
    px1, py1, px2, py2 = pred_bboxes.chunk(4, dim=-1)
    tx1, ty1, tx2, ty2 = target_bboxes.chunk(4, dim=-1)

    p_cx, p_cy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
    t_cx, t_cy = (tx1 + tx2) * 0.5, (ty1 + ty2) * 0.5
    center_dist = (p_cx - t_cx).pow(2) + (p_cy - t_cy).pow(2)

    cw = torch.maximum(px2, tx2) - torch.minimum(px1, tx1)
    ch = torch.maximum(py2, ty2) - torch.minimum(py1, ty1)
    enclosing_diag = cw.pow(2) + ch.pow(2) + eps

    # Detach the geometry weight to keep WIoU stable and avoid over-amplifying gradients.
    return torch.exp((center_dist / enclosing_diag).detach())


def _bbox_forward_wiou(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
    imgsz: torch.Tensor,
    stride: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute WIoU-v1 box loss and the original DFL loss."""
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    pred_fg = pred_bboxes[fg_mask]
    target_fg = target_bboxes[fg_mask]

    iou = loss_module.bbox_iou(pred_fg, target_fg, xywh=False, CIoU=False)
    loss_iou = ((1.0 - iou) * _wiou_weight(pred_fg, target_fg) * weight).sum() / target_scores_sum

    # Keep the original DFL branch unchanged.
    if self.dfl_loss:
        target_ltrb = loss_module.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        target_ltrb = loss_module.bbox2dist(anchor_points, target_bboxes)
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        loss_dfl = (
            loss_module.F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True)
            * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    return loss_iou, loss_dfl


def apply_wiou_patch() -> None:
    """Patch Ultralytics BboxLoss with WIoU once per Python process."""
    if getattr(loss_module.BboxLoss, "_basin_wiou_enabled", False):
        return
    loss_module.BboxLoss.forward = _bbox_forward_wiou
    loss_module.BboxLoss._basin_wiou_enabled = True
    loss_module.BboxLoss._basin_loss_name = "WIoU-v1"
