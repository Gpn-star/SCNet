from torch import nn
from losses.cross_entropy_loss_with_label_smooth import CrossEntropyWithLabelSmooth
from losses.triplet_loss import TripletLoss
from losses.contrastive_loss import ContrastiveLoss
from losses.arcface_loss import ArcFaceLoss
from losses.cosface_loss import CosFaceLoss, PairwiseCosFaceLoss
from losses.circle_loss import CircleLoss, PairwiseCircleLoss
from losses.adasp_loss import AdaSPLoss

def build_losses(config, num_train_clothes):

    # Build identity classification loss
    if config.LOSS.CLA_LOSS == 'crossentropy':
        criterion_cla = nn.CrossEntropyLoss()
    elif config.LOSS.CLA_LOSS == 'crossentropylabelsmooth':
        criterion_cla = CrossEntropyWithLabelSmooth()
    elif config.LOSS.CLA_LOSS == 'arcface':
        criterion_cla = ArcFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'cosface':
        criterion_cla = CosFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'circle':
        criterion_cla = CircleLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    else:
        raise KeyError("Invalid classification loss: '{}'".format(config.LOSS.CLA_LOSS))

    # Build pairwise loss
    if config.LOSS.PAIR_LOSS == 'triplet':
        criterion_pair = TripletLoss(margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'contrastive':
        criterion_pair = ContrastiveLoss(scale=config.LOSS.PAIR_S)
    elif config.LOSS.PAIR_LOSS == 'cosface':
        criterion_pair = PairwiseCosFaceLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'circle':
        criterion_pair = PairwiseCircleLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'adasp':
        criterion_pair = AdaSPLoss()
    else:
        raise KeyError("Invalid pairwise loss: '{}'".format(config.LOSS.PAIR_LOSS))


    return criterion_cla, criterion_pair
