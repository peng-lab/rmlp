import torch
import torch.nn.functional as F
from torch import nn

"""
DINO loss in DINOv2
"""


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center = torch.zeros((1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.update_center(teacher_output)
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    def forward(self, student_output_list, teacher_out_list, student_temp, teacher_temp, ibot_masks=None):
        if ibot_masks is None:
            teacher_out_softmaxed_centered_list = self.softmax_center_teacher(teacher_out_list, teacher_temp)
            total_loss = torch.tensor(0.).to(student_output_list.device)
            for ns, s in enumerate(student_output_list):
                lsm = F.log_softmax(s / student_temp, dim=-1)
                loss = torch.sum(teacher_out_softmaxed_centered_list[ns].unsqueeze(1) * lsm.unsqueeze(0), dim=-1)
                total_loss -= loss.mean()
            total_loss /= len(student_output_list)
            self.update_center(teacher_out_list)
            return total_loss
        else:
            teacher_out_softmaxed_centered_list = self.softmax_center_teacher(teacher_out_list, teacher_temp)
            total_loss = torch.tensor(0.).to(student_output_list.device)
            for ns, s in enumerate(student_output_list):
                lsm = F.log_softmax(s / student_temp, dim=-1)
                loss = torch.sum(teacher_out_softmaxed_centered_list[ns] * lsm, dim=-1)
                loss = torch.where(ibot_masks[ns, :, 0], loss, 0.)
                total_loss -= loss.sum()/(1+ibot_masks[ns, :, 0].sum())
            total_loss /= len(student_output_list)
            self.update_center(teacher_out_list)
            return total_loss

    def update_center(self, teacher_output):

        if self.center.shape != teacher_output.detach().mean(dim=0).shape:
            self.center = teacher_output.detach().mean(dim=0)

        c = self.center_momentum * self.center.to(teacher_output.device)
        c += (1 - self.center_momentum) * teacher_output.detach().mean(dim=0)
        self.center = c
