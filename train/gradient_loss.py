import torch
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction
from monai.losses import DiceCELoss, DiceLoss


class Gradient_Loss(_Loss):
    def __init__(self,reduction=LossReduction.MEAN.value):
        super().__init__()
        self.reduction=reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        true_x_shifted_right = target[:, :, 1:, :, :]  # batch_size, channel, width, height, depth
        true_x_shifted_left = target[:, :, :-1, :, :]
        true_x_gradient = torch.abs(true_x_shifted_left - true_x_shifted_right)  # 差分

        generated_x_shift_right = input[:, :, 1:, :, :]
        generated_x_shift_left = input[:, :, :-1, :, :]
        generated_x_griednt = torch.abs(generated_x_shift_left - generated_x_shift_right)

        difference_x = true_x_gradient - generated_x_griednt

        loss_x_gradient = torch.sqrt(torch.sum(difference_x**2, dim=[2,3,4]))

        true_y_shifted_right = target[:, :, :, 1:, :]
        true_y_shifted_left = target[:, :, :, :-1, :]
        true_y_gradient = torch.abs(true_y_shifted_left - true_y_shifted_right)

        generated_y_shift_right = input[:, :, :, 1:, :]
        generated_y_shift_left = input[:, :, :, :-1, :]
        generated_y_griednt = torch.abs(generated_y_shift_left - generated_y_shift_right)

        difference_y = true_y_gradient - generated_y_griednt
        loss_y_gradient = torch.sqrt(torch.sum(difference_y**2,dim=[2,3,4]))

        true_z_shifted_right = target[:, :, :, :, 1:]
        true_z_shifted_left = target[:, :, :, :, :-1]
        true_z_gradient = torch.abs(true_z_shifted_left - true_z_shifted_right)

        generated_z_shift_right = input[:, :, :, :, 1:]
        generated_z_shift_left = input[:, :, :, :, :-1]
        generated_z_griednt = torch.abs(generated_z_shift_left - generated_z_shift_right)

        difference_z = true_z_gradient - generated_z_griednt
        loss_z_gradient = torch.sqrt(torch.sum(difference_z**2,dim=[2,3,4]))

        loss = loss_x_gradient + loss_y_gradient + loss_z_gradient

        if self.reduction == LossReduction.MEAN.value:
            loss = torch.mean(loss)
        elif self.reduction == LossReduction.SUM.value:
            loss = torch.sum(loss)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum"].')
        return loss


class DiceGradient_Loss(_Loss):
    def __init__(self,reduction=LossReduction.MEAN.value,factor=1e-3):
        super().__init__()
        self.factor=factor
        self.gradient_loss = Gradient_Loss(reduction=reduction)
        self.dice_loss = DiceLoss(reduction=reduction, to_onehot_y=True, softmax=True)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss=self.dice_loss(input,target)
        gradient_loss=self.gradient_loss(input, target)
        total_loss=dice_loss+self.factor*gradient_loss
        return total_loss