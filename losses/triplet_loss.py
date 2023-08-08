import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        # L2-normalize inputs
        inputs = F.normalize(inputs, p=2, dim=1)
        dist_matrix = torch.cdist(inputs, inputs, p=2)

        # Create mask for positive pairs
        mask_pos = torch.eq(targets.unsqueeze(1), targets.unsqueeze(0)).float().cuda()
        mask_neg = 1 - mask_pos

        # Find hardest positive and negative examples
        dist_ap = torch.max((dist_matrix - mask_neg * 99999999.), dim=1)[0]
        dist_an = torch.min((dist_matrix + mask_pos * 99999999.), dim=1)[0]

        # Compute triplet loss
        loss = torch.clamp(dist_ap - dist_an + self.margin, min=0.0).mean()

        return loss
