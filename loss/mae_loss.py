import torch
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction


class MAELoss(_Loss):
    def __init__(self,reduction=LossReduction.MEAN.value):
        super().__init__()
        self.reduction=reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_=torch.norm(input-target,p=1)
        loss=loss_
        # loss = loss_*loss_
        if self.reduction == LossReduction.MEAN.value:
            loss = loss/input.shape[0]
        elif self.reduction == LossReduction.SUM.value:
            pass
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum"].')
        return loss