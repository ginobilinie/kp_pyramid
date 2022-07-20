import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbOhemCrossEntropy4PointCloud(nn.Module):
    def __init__(self,
                 ignore_label,
                 reduction='mean',
                 thresh=0.6,
                 min_kept=256,
                 down_ratio=1,
                 use_weight=False):
        super(ProbOhemCrossEntropy4PointCloud, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor([
                0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                1.0865, 1.1529, 1.0507
            ])
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction=reduction, weight=weight, ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction=reduction, ignore_index=ignore_label)

    def forward(self, pred, target):
        N, c = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=-1)  # [N, c]
        prob = (prob.transpose(0, -1)).reshape(c, -1)  # [N,c]->[c,N]

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target,
                             torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long(
                )  # these are thought to be hard examples
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_label)

        return self.criterion(pred, target)


class DiceLoss4PointCloud:
    def __init__(self, class_weights=None):
        super(DiceLoss4PointCloud, self).__init__()
        self.class_weights = class_weights

    def __call__(self, outputs, targets):
        loss_dice = 0
        smooth = 1.
        N, c = outputs.size()
        outputs = F.softmax(outputs, dim=-1)
        for cls in range(c):
            jaccard_target = (targets == cls).float()  # N, [0,1]
            jaccard_output = outputs[:, cls]  # N, (0,1)
            intersection = (jaccard_output * jaccard_target).sum()
            if self.class_weights is not None:
                w = self.class_weights[cls]
            else:
                w = 1.
            union = jaccard_output.sum() + jaccard_target.sum()
            loss_dice += w * (1 - (2. * intersection + smooth) /
                              (union + smooth))
        return loss_dice / c


class FocalLoss(nn.Module):
    def __init__(self,
                 weight=None,
                 gamma=2.,
                 reduction='none',
                 ignore_indx=-100):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_indx = ignore_indx

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        fl = F.nll_loss(
            ((1 - prob)**self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            ignore_index=self.ignore_indx,  # newly added
            reduction=self.reduction)
        return fl


class CombineLoss(nn.Module):
    def __init__(self,
                 mylambda_ohem=1,
                 mylambda_dice=1,
                 ignore_label=255,
                 thresh=0.7,
                 class_weights=None,
                 min_kept=1024 * 1024 // 16):
        super(CombineLoss, self).__init__()
        self.class_weights = class_weights
        self.dice_loss = DiceLoss4PointCloud(class_weights=class_weights)

        self.ohem_loss = ProbOhemCrossEntropy4PointCloud(ignore_label=255,
                                                         thresh=thresh,
                                                         min_kept=min_kept,
                                                         use_weight=False)

        self.mylambda_dice = mylambda_dice
        self.mylambda_ohem = mylambda_ohem

    def __call__(self, outputs, targets):
        loss_dice = self.dice_loss(outputs, targets) * self.mylambda_dice
        loss_ohem = self.ohem_loss(outputs, targets) * self.mylambda_ohem
        # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return loss_dice + loss_ohem, loss_dice, loss_ohem


class Combine_Ohem_Dice_Loss(nn.Module):
    def __init__(self,
                 mylambda_ohem=1,
                 mylambda_dice=1,
                 ignore_label=255,
                 thresh=0.7,
                 class_weights=None,
                 min_kept=4096 * 5 * 12 // 16):
        super(Combine_Ohem_Dice_Loss, self).__init__()
        self.class_weights = class_weights
        self.dice_loss = DiceLoss4PointCloud(class_weights=class_weights)

        self.ohem_loss = ProbOhemCrossEntropy4PointCloud(ignore_label=255,
                                                         thresh=thresh,
                                                         min_kept=min_kept,
                                                         use_weight=False)

        self.mylambda_dice = mylambda_dice
        self.mylambda_ohem = mylambda_ohem

    def __call__(self, outputs, targets):
        loss_dice = self.dice_loss(outputs, targets) * self.mylambda_dice
        loss_ohem = self.ohem_loss(outputs, targets) * self.mylambda_ohem
        # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return loss_dice + loss_ohem, loss_dice, loss_ohem


class Combine_CE_Dice_Loss(nn.Module):
    def __init__(self, mylambda=1, ignore_label=255, class_weights=None):
        super(Combine_CE_Dice_Loss, self).__init__()
        self.class_weights = class_weights
        self.dice_loss = DiceLoss4PointCloud(class_weights=class_weights)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_label)

        self.mylambda = mylambda

    def __call__(self, outputs, targets):
        loss_dice = self.dice_loss(outputs, targets) * self.mylambda
        loss_ce = self.ce_loss(outputs, targets)
        return loss_dice + loss_ce, loss_dice, loss_ce


class Combine_Dice_Focal_Loss(nn.Module):
    def __init__(self,
                 mylambda_dice=1,
                 mylambda_focal=1,
                 ignore_label=255,
                 class_weights=None):
        super(Combine_Dice_Focal_Loss, self).__init__()
        self.class_weights = class_weights
        self.dice_loss = DiceLoss4PointCloud(class_weights=class_weights)

        self.focal_loss = FocalLoss(weight=class_weights,
                                    reduction='sum',
                                    ignore_indx=ignore_label)
        self.mylambda_dice = mylambda_dice
        self.mylambda_focal = mylambda_focal

    def __call__(self, outputs, targets):
        outputs = outputs.squeeze(0).transpose(0, 1)
        targets = targets.squeeze(0)
        loss_dice = self.dice_loss(outputs, targets) * self.mylambda_dice
        loss_focal = self.focal_loss(
            outputs, targets) / len(targets) * self.mylambda_focal
        return loss_dice + loss_focal, loss_dice, loss_focal
