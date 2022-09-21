import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import numbers
from center_loss import CenterLoss, CenterLossA, CenterLossB
from torchvision.transforms import functional as F1
from sklearn.metrics import confusion_matrix


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_fit(x_pred, x_output, brach):
    device = x_pred.device#6,4,5/0,1/2,3
    lab = torch.tensor([[6,6,6,6],
    [0,1,3,3],
    [2,4,5,5]])
    lab = lab.to(device)
    loss = 0.0

    criterion = nn.CrossEntropyLoss()
    #print(x_pred.size())
    j = 0
    for i in range(len(x_output)):
        x_p = torch.unsqueeze(x_pred[i],0)
        x_o = x_output[i]
        x_o = torch.tensor([x_o]).to(device)
        if x_o in lab[brach]:
            loss = loss +criterion(x_p, x_o)
            j=j+1
    if j==0:
        loss = loss + criterion(x_p, x_o)*0
    else:
        loss = loss/j
    return loss

def model_fit1(x_pred, x_output, x_output1):
    device = x_pred.device
    loss = 0.0
    criterion = nn.CrossEntropyLoss()
    #loss = criterion(x_pred, x_output)
    j = 0
    for i in range(len(x_output)):
        x_p = torch.unsqueeze(x_pred[i],0)
        x_o = x_output[i]
        x_o = torch.tensor([x_o]).to(device)
        x_o1 = x_output1[i]
        x_o1 = torch.tensor([x_o1]).to(device)
        loss = loss +criterion(x_p, x_o)
        j=j+1
    if j==0:
        loss = loss +criterion(x_p, x_o)*0
    else:
        loss = loss/j
    return loss


def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, opt, btr, bte, center_loss,
                   optimizer_centloss, total_epoch=200):
    train_batch = btr
    test_batch = bte
    T = opt.temp
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    lambda_weight = np.ones([4, total_epoch])
    ones = torch.sparse.torch.eye(6)
    ones = ones.to(device)
    bestac = 50.0
    with open("acc.txt", "w") as f:
        for index in range(total_epoch):
            print('\nEpoch: %d' % (index + 1))
            cost = np.zeros(24, dtype=np.float32)
            correctg = 0.0
            correcti = 0.0
            correcti1 = 0.0
            correcti2 = 0.0
            totalg = 0.0
            totali = 0.0
            alpha = 0.02
            # apply Dynamic Weight Average
            if opt.weight == 'dwa':
                if index == 0 or index == 1:
                    lambda_weight[:, index] = 1.0
                else:
                    w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
                    w_2 = avg_cost[index - 1, 1] / avg_cost[index - 2, 1]
                    w_3 = avg_cost[index - 1, 2] / avg_cost[index - 2, 2]
                    lambda_weight[0, index] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                    lambda_weight[1, index] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                    lambda_weight[2, index] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

            # iteration for all batches
            multi_task_model.train()
            train_dataset = iter(train_loader)
            i=1;
            for k in range(train_batch):
                #alpha = 0.03
                train_data, train_label = train_dataset.next()
                train_data, train_label = train_data.to(device), train_label.to(device)

                feature,train_pred = multi_task_model(train_data)
                si = len(train_label)
                train_label1 = torch.ones_like(train_label)
                for tr in range(len(train_label)):
                    if train_label[tr] in [6]:
                        train_label1[tr] = 0
                    elif train_label[tr] in [0,1,3]:
                        train_label1[tr] = 1
                    else:
                        train_label1[tr] = 2
                train_label1 = train_label1.to(device)


                train_loss = [model_fit(train_pred[0], train_label,0),
                            model_fit(train_pred[1], train_label,1),
                            model_fit(train_pred[2], train_label,2),
                            model_fit1(train_pred[3], train_label1, train_label)]
                train_loss[3] = center_loss(feature, train_label1) * alpha + train_loss[3]

                if opt.weight == 'equal' or opt.weight == 'dwa':
                    loss = sum([lambda_weight[i, index] *train_loss[i] for i in range(4)])
                optimizer.zero_grad()
                optimizer_centloss.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_centloss.step()
                _, predictedg = torch.max(train_pred[0].data, 1)
                totalg += train_label.size(0)
                correctg += predictedg.eq(train_label.data).cpu().sum()
                _, predictedi1 = torch.max(train_pred[1].data, 1)
                correcti1 += predictedi1.eq(train_label.data).cpu().sum()
                _, predictedi2 = torch.max(train_pred[2].data, 1)
                correcti2 += predictedi2.eq(train_label.data).cpu().sum()
                _, predictedi = torch.max(train_pred[3].data, 1)
                correcti += predictedi.eq(train_label1.data).cpu().sum()
                cost[0] = train_loss[0].item()
                cost[1] = train_loss[1].item()
                cost[2] = train_loss[2].item()
                cost[3] = train_loss[3].item()
                avg_cost[index, :4] += cost[:4] / train_batch
                print('Batch: {:04d} | TRAINLOSS: {:.4f} {:.4f} | TRAINACC: {:.4f} {:.4f} {:.4f} {:.4f}|'.format(i, avg_cost[index, 0], avg_cost[index, 3], 100. * correctg / totalg, 100. * correcti1 / totalg, 100. * correcti2 / totalg, 100. * correcti / totalg))
                i=i+1
            scheduler.step()
            # evaluating test data
            multi_task_model.eval()
            with torch.no_grad():  # operations inside don't track history
                test_dataset = iter(test_loader)
                correctg = 0.0
                correcti = 0.0
                correcti1 = 0.0
                correcti2 = 0.0
                totalg = 0.0
                totali = 0.0
                labelte = []
                outte = []
                pre = []
                for k in range(test_batch):
                    test_data, test_label = test_dataset.next()
                    test_data, test_label = test_data.to(device), test_label.to(device)
                    test_label1 = torch.ones_like(test_label)
                    for te in range(len(test_label)):
                        if test_label[te] in [4,6]:
                            test_label1[te] = 0
                        elif test_label[te] in [3,1,0]:
                            test_label1[te] = 1
                        else:
                            test_label1[te] = 2
                    test_label1 = test_label1.to(device)

                    feature,test_pred = multi_task_model(test_data)
                    test_loss = [model_fit(test_pred[0], test_label,0),
                            model_fit(test_pred[1], test_label,1),
                            model_fit(test_pred[2], test_label,2),
                            #model_fit(test_pred[3], test_label,3),
                            model_fit1(test_pred[3], test_label1, test_label)]

                    cost[2] = test_loss[0].item()
                    cost[3] = test_loss[1].item()

                    avg_cost[index, 2:] += cost[2:] / test_batch

                    _, predictedg = torch.max(test_pred[0].data, 1)
                    totalg += test_label.size(0)
                    correctg += predictedg.eq(test_label.data).cpu().sum()
                    _, predictedi1 = torch.max(test_pred[1].data, 1)
                    correcti1 += predictedi1.eq(test_label.data).cpu().sum()
                    _, predictedi2 = torch.max(test_pred[2].data, 1)
                    correcti2 += predictedi2.eq(test_label.data).cpu().sum()
                    pos, predictedi = torch.max(test_pred[3].data, 1)
                    for te in range(len(test_label)):
                        pos1, predictedx = torch.max(test_pred[0].data, 1)
                        pos2, predictedy = torch.max(test_pred[1].data, 1)
                        pos3, predictedz = torch.max(test_pred[2].data, 1)
                        if predictedi[te] == 0:
                            predictedi[te] = predictedx[te]
                        elif predictedi[te] == 1:
                            predictedi[te] = predictedy[te]
                        else:
                            predictedi[te] = predictedz[te]
                    totali += test_label.size(0)
                    correcti += predictedi.eq(test_label.data).cpu().sum()
                    labelte = labelte+test_label.cpu().numpy().tolist()
                    pre = pre+predictedi.cpu().numpy().tolist()
            print('Epoch: {:04d} | TESTLOSS: {:.4f} {:.4f}| TESTACC: {:.4f} {:.4f} {:.4f} {:.4f} '
                .format(index+1,  avg_cost[index, 2], avg_cost[index, 3], 100. * correctg / totalg, 100. * correcti1 / totali, 100. * correcti2 / totali, 100. * correcti / totali))
            f.write("EPOCH=%d,accges= %.3f%%,accges1= %.3f%%,accges2= %.3f%%,accid= %.3f%%" % (index + 1, 100. * correctg / totalg, 100. * correcti1 / totali, 100. * correcti2 / totali, 100. * correcti / totali))
            f.write('\n')
            f.flush()
            if (100. * correcti / totali)>bestac:
                bestac = 100. * correcti / totali
                cm = confusion_matrix(labelte, pre)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print('Saving Model')
                torch.save(multi_task_model.state_dict(),r'model/bestmodel.pth')
                f3 = open("bestacc.txt", "w")
                f3.write(str(bestac))
                f3.write(str(cm_normalized))
                f3.close()