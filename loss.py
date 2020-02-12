from torch import nn
import torch.nn.functional as F


# loss function: seven probability map --- 6 scale + 1 fuse
class Loss(nn.Module):
    def __init__(self, weight=[1.0] * 7):
        super(Loss, self).__init__()
        self.weight = weight

    def forward(self, x_list, label):
        loss = self.weight[0] * nn.BCEWithLogitsLoss()(x_list[0], label)
        for i, x in enumerate(x_list[1:]):
            loss += self.weight[i + 1] * nn.BCEWithLogitsLoss()(x, label)
        return loss
    # def forward(self, x_list, label):
    #     loss = self.weight[0] * F.binary_cross_entropy(x_list[0], label)
    #     for i, x in enumerate(x_list[1:]):
    #         loss += self.weight[i + 1] * F.binary_cross_entropy(x, label)
    #     return loss

# class Loss(nn.Module):
#     def __init__(self, weight=[1.0] * 1):
#         super(Loss, self).__init__()
#         self.weight = weight

#     def forward(self, x_list, label):
#         loss = self.weight[0] * nn.BCEWithLogitsLoss()(x_list, label)
#         return loss

def _iouloss(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:] * pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:]) - Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IoULoss(nn.Module):
    def __init__(self, size_average = True):
        super(IoULoss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iouloss(pred, target, self.size_average)