import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utilsa import *
from data_loader import  dataprtrraf, dataprteraf, dataprterafou, dataprterafpo
import torch.optim as optim
import torchvision.transforms as transforms
from options import Options
import torchvision.models as models
from model import MBSNET
from PIL import Image

parser = argparse.ArgumentParser(description='zero')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = MBSNET().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
milestones = [10,20,30]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
center_loss = CenterLossA(num_classes=3, feat_dim=512).to(device)
optimizer_centloss = optim.SGD(center_loss.parameters(), lr=0.5)

# define dataset
args = Options().initialize()
data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32)
            ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))
        ])


data_transforms_val = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

train_list = r'/media/zx/My Passport/Data/FED/rafdb/'
 
dataset_source = dataprtrraf(
    data_list=train_list,
    transform=data_transforms
)

trainloader = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=args.train_batch_size,
    shuffle=True,
    pin_memory = True)

lengthtr = len(trainloader)

test_list = r'/media/zx/My Passport/Data/FED/rafdb/'

dataset_target = dataprteraf(
    data_list=test_list,
    transform=data_transforms_val
)

testloader = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=args.test_batch_size,
    shuffle=False,
    pin_memory = True)

lengthte = len(testloader)
print('Train set size:', dataset_source.__len__())
print('Validation set size:', dataset_target.__len__())

# Train and evaluate multi-task network
if __name__ == '__main__':
    multi_task_trainer(trainloader,
                   testloader,
                   net,
                   device,
                   optimizer,
                   scheduler,
                   opt,
                   lengthtr,
                   lengthte,
                   center_loss,
                   optimizer_centloss,
                   70)