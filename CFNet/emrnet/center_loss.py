import torch
import torch.nn as nn
from torch.autograd.function import Function

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        #distmat.addmm_(1, -2, x, self.centers.t())
        distmat.addmm_(x, self.centers.t(),beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class CenterLossA(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLossA, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).cuda())
        self.centerlossfunc = CenterlossFuncA.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss

class CenterLossB(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLossB, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).cuda())
        self.centerlossfunc = CenterlossFuncB.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label, wei):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, wei, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None

class CenterlossFuncA(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        labelex1 = centers.new_ones(label.size())
        labelex2 = centers.new_ones(label.size())
        labelex3 = centers.new_ones(label.size())
        labelex4 = centers.new_ones(label.size())
        labelex5 = centers.new_ones(label.size())
        for i in range(len(label)):
            if label[i] == 0:
                labelnu = torch.tensor([1,2]).cuda()
            elif label[i] == 1:
                labelnu = torch.tensor([0,2]).cuda()
            elif label[i] == 2:
                labelnu = torch.tensor([0,1]).cuda()
            else:
                print("error")
            labelex1[i] = labelnu[0]
            labelex2[i] = labelnu[1]
            labelex3[i] = torch.tensor(0).cuda()
            labelex4[i] = torch.tensor(1).cuda()
            labelex5[i] = torch.tensor(2).cuda()
        centers_batch = centers.index_select(0, label.long())
        centers_batch1 = centers.index_select(0, labelex1.long())
        centers_batch2 = centers.index_select(0, labelex2.long())
        centers = torch.mean(feature, 0, True)
        centers = centers.expand_as(feature)
        distocen = (feature - centers_batch1).pow(2).sum() + (feature - centers_batch2).pow(2).sum()
        #distocen1 = distocen/distocen.mean()
        #intracen = (feature - centers_batch).pow(2)
        #cendis = (centers_batch - centers_batch1).pow(2) + (centers_batch - centers_batch2).pow(2)
        #cenloss = intracen.sum() + (intracen.sum()/distocen.sum())
        #return cenloss / batch_size
        return ((feature - centers_batch).pow(2).sum()*(1+1/distocen)) / 2.0 / batch_size


    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None

class CenterlossFuncB(Function):
    @staticmethod
    def forward(ctx, feature, label, wei, centers, batch_size):
        ctx.save_for_backward(feature, label, wei, centers, batch_size)
        labelex1 = centers.new_ones(label.size())
        labelex2 = centers.new_ones(label.size())
        labelex3 = centers.new_ones(label.size())
        labelex4 = centers.new_ones(label.size())
        labelex5 = centers.new_ones(label.size())
        for i in range(len(label)):
            if label[i] == 0:
                labelnu = torch.tensor([1,2]).cuda()
            elif label[i] == 1:
                labelnu = torch.tensor([0,2]).cuda()
            elif label[i] == 2:
                labelnu = torch.tensor([0,1]).cuda()
            else:
                print("error")
            labelex1[i] = labelnu[0]
            labelex2[i] = labelnu[1]
            labelex3[i] = torch.tensor(0).cuda()
            labelex4[i] = torch.tensor(1).cuda()
            labelex5[i] = torch.tensor(2).cuda()
        centers_batch = centers.index_select(0, label.long())
        centers_batch1 = centers.index_select(0, labelex1.long())
        centers_batch2 = centers.index_select(0, labelex2.long())
        centers = torch.mean(feature, 0, True)
        centers = centers.expand_as(feature)
        wei = wei.reshape(len(label),1)
        wei = wei.expand_as(feature)
        distocen = (wei*(feature - centers_batch1).pow(2)).sum() + (wei*(feature - centers_batch2).pow(2)).sum()
        #distocen1 = distocen/distocen.mean()
        #cendis = (centers_batch - centers_batch1).pow(2) + (centers_batch - centers_batch2).pow(2)
        #cenloss = intracen.sum() + (intracen.sum()/distocen.sum())
        #return cenloss / batch_sizeprint
        return ((wei*(feature - centers_batch).pow(2)).sum()*(1+1/distocen)) / 2.0 / batch_size


    @staticmethod
    def backward(ctx, grad_output):
        feature, label, wei, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, None, grad_centers / batch_size, None