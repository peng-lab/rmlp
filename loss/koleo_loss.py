import torch
import torch.nn as nn
import torch.nn.functional as F


class KoLeoLoss(nn.Module):
    """
    KoLeo regularizer
    """
    def __init__(self):
        super().__init__()

    def forward(self, student_output, memory_bank, eps=1e-8):
        student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        memory_bank = F.normalize(memory_bank, eps=eps, p=2, dim=-1)
        dists = torch.cdist(student_output.type(torch.float32), memory_bank.type(torch.float32))
        dists_diag = torch.zeros_like(dists)
        for _ in range(min(len(student_output),len(memory_bank))):
            dists_diag[_,_] += torch.tensor(1.).to(dists_diag.device)
        dists = dists + dists_diag.to(dists.device)
        loss = -torch.log(torch.amin(dists, dim=-1) + eps).mean()
        return loss

