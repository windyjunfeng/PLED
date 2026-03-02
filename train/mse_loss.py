import torch
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction


class MSELoss(_Loss):
    def __init__(self,reduction=LossReduction.MEAN.value):
        super().__init__()
        self.reduction=reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # loss_=torch.norm(input-target,p='fro')  # 同p=2的结果
        # loss=loss_
        # loss = loss_*loss_
        loss_ = torch.sqrt(torch.sum((input - target) ** 2, dim=(1, 2, 3, 4)))
        if self.reduction == LossReduction.MEAN.value:
            # loss = loss/(input.shape[0]*input.shape[1])
            loss = torch.mean(loss_)
        elif self.reduction == LossReduction.SUM.value:
            # pass
            loss = torch.sum(loss_)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum"].')
        return loss