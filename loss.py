import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, delta=2):
        super(Loss, self).__init__()
        self.delta = delta

    @staticmethod
    def structure_loss(self, outputs, gt):
        h_o = 2 * outputs[:, :, 1:-1, 1:-1] - outputs[:, :, 1:-1, : -2] - outputs[:, :, 1:-1, 2:]
        v_o = 2 * outputs[:, :, 1:-1, 1:-1] - outputs[:, :, :-2, 1: -1] - outputs[:, :, 2:, 1:-1]

        h_gt = 2 * gt[:, :, 1:-1, 1:-1] - gt[:, :, 1:-1, : -2] - gt[:, :, 1:-1, 2:]
        v_gt = 2 * gt[:, :, 1:-1, 1:-1] - gt[:, :, :-2, 1: -1] - gt[:, :, 2:, 1:-1]

        structure_o = (h_o.abs() + v_o.abs()).float().mean()
        structure_gt = (h_gt.abs() + v_gt.abs()).float().mean()
        return (structure_gt - structure_o).abs()

    def forward(self, outputs, gt, mask):
        s_loss = self.structure_loss(self, outputs, gt)

        mask_gt = (gt >= 0.0001).detach()
        outputs = outputs[mask_gt]
        gt = gt[mask_gt]

        diff = torch.abs(outputs - gt)
        squared_err = 0.5 * diff ** 2
        linear_err = diff - 0.5 * self.delta
        loss = torch.mean(torch.where(diff < self.delta, squared_err, linear_err))

        return loss * 0.8 + s_loss * 0.2
