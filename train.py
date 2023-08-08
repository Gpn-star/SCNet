import time
import datetime
import logging
import torch
from torch import nn
from tools.utils import AverageMeter
from torchvision import transforms as T
from losses.part_based_matching_loss import match_loss
from torch.nn import MSELoss

torch.set_printoptions(profile="full")

def train_model(config, epoch, model, attention, gap_classifier, gap_classifier_h, gap_classifier_b, criterion_cla, criterion_pair, optimizer, trainloader):
    logger = logging.getLogger('reid.train')
    batch_cla_loss_h = AverageMeter()
    batch_cla_loss_f = AverageMeter()
    batch_cla_loss_b = AverageMeter()
    batch_pair_loss_h = AverageMeter()
    batch_pair_loss_f = AverageMeter()
    batch_pair_loss_b = AverageMeter()
    batch_part_loss = AverageMeter()
    batch_vc_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mse_loss = MSELoss()
    model.train()
    attention.train()
    gap_classifier.train()
    gap_classifier_h.train()
    gap_classifier_b.train()
    Resize = T.Resize((24, 12))
    avgpool = nn.AdaptiveAvgPool2d(1)
    end = time.time()

    for batch_idx, (parsing_results, imgs, imgs_b, pids, clothes_ids) in enumerate(trainloader):

        parsing_results, imgs, imgs_b, pids, clothes_ids = parsing_results.cuda(), imgs.cuda(), imgs_b.cuda(), pids.cuda(), clothes_ids.cuda()
        data_time.update(time.time() - end)
        features, features_b = model(torch.cat((imgs, imgs_b), dim=0)).split(imgs.size(0), dim=0)
        attention_maps = attention(features)
        parsing_results = Resize(parsing_results)
        part_loss = match_loss(attention_maps, parsing_results)
        head_features = attention_maps[:, 0, :, :].unsqueeze(dim=1) * features

        features_pool = features.mean(dim=1)
        head_features_pool = head_features.mean(dim=1)
        features_b_pool = features_b.mean(dim=1)

        features_avg = avgpool(features).view(features.size(0), -1)
        head_features_avg = avgpool(head_features).view(head_features.size(0), -1)
        features_b_avg = avgpool(features_b).view(features_b.size(0), -1)

        features = gap_classifier.module.bn(features)
        features = gap_classifier.module.conv(features)
        features_g = features[torch.arange(features.shape[0]), pids, :, :]

        features_b = gap_classifier_b.module.bn(features_b)
        features_b = gap_classifier_b.module.conv(features_b)
        features_b_g = features_b[torch.arange(features_b.shape[0]), pids, :, :]

        head_features = gap_classifier_h.module.bn(head_features)
        head_features = gap_classifier_h.module.conv(head_features)
        head_features_g = head_features[torch.arange(head_features.shape[0]), pids, :, :]

        features_att = torch.max(torch.stack((features_g, features_b_g, head_features_g), dim=1), dim=1)[0]

        vc_loss = mse_loss(features_att, features_pool) + mse_loss(features_att, head_features_pool) \
                  + mse_loss(features_att, features_b_pool)

        features_out = gap_classifier.module.globalpooling(features).view(features.size(0), -1)
        features_b_out = gap_classifier_b.module.globalpooling(features_b).view(features_b.size(0), -1)
        head_features_out = gap_classifier_h.module.globalpooling(head_features).view(head_features.size(0), -1)

        pair_loss_f = criterion_pair(features_avg, pids)
        pair_loss_h = criterion_pair(head_features_avg, pids)
        pair_loss_b = criterion_pair(features_b_avg, pids)
        cla_loss_f = criterion_cla(features_out, pids)
        cla_loss_h = criterion_cla(head_features_out, pids)
        cla_loss_b = criterion_cla(features_b_out, pids)

        _, preds = torch.max(features_out.data, 1)

        loss = 0.01 * (part_loss + vc_loss) + pair_loss_f + pair_loss_h + pair_loss_b + cla_loss_f + cla_loss_h + cla_loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float() / pids.size(0), pids.size(0))
        batch_part_loss.update(part_loss.item(), pids.size(0))
        batch_vc_loss.update(vc_loss.item(), pids.size(0))
        batch_pair_loss_f.update(pair_loss_f.item(), pids.size(0))
        batch_pair_loss_h.update(pair_loss_h.item(), pids.size(0))
        batch_pair_loss_b.update(pair_loss_b.item(), pids.size(0))
        batch_cla_loss_f.update(cla_loss_f.item(), pids.size(0))
        batch_cla_loss_h.update(cla_loss_h.item(), pids.size(0))
        batch_cla_loss_b.update(cla_loss_b.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'PartLoss:{part_loss.avg:.4f} '
                'VcLoss:{vc_loss.avg:.4f} '
                'PairLoss_F:{pair_loss_f.avg:.4f} '
                'PairLoss_H:{pair_loss_h.avg:.4f} '
                'PairLoss_B:{pair_loss_b.avg:.4f} '
                'ClaLoss_F:{cla_loss_f.avg:.4f} '
                'ClaLoss_H:{cla_loss_h.avg:.4f} '
                'ClaLoss_b:{cla_loss_b.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
        epoch + 1, batch_time=batch_time, data_time=data_time, part_loss=batch_part_loss, vc_loss=batch_vc_loss,
        cla_loss_f=batch_cla_loss_f, cla_loss_h=batch_cla_loss_h, cla_loss_b=batch_cla_loss_b, pair_loss_f=batch_pair_loss_f,
        pair_loss_h=batch_pair_loss_h, pair_loss_b=batch_pair_loss_b, acc=corrects))