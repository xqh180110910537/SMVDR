import os
import random
import numpy as np
import torch
import cv2
import timm
import pandas as pd
import torchvision.transforms as transform
from sklearn import model_selection
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from dataset import CustomDataset, MultiviewImgDataset_mask, SingleimgDataset
import models.create_models as create


def test_model2(netname, model, test_loader, dataset_size, criterion, device):
    n_class = 5
    print('dataset_size', dataset_size)
    since = time.time()
    # roc matrix
    roc_matrix = pd.DataFrame(columns=['label', '0', '1', '2', '3', '4'])

    # fusion matrix
    total_outs = torch.tensor([])
    total_labels = torch.tensor([])
    FM = np.zeros((n_class, n_class))
    tp = [0] * n_class
    tn = [0] * n_class
    fp = [0] * n_class
    fn = [0] * n_class
    precision = [0] * n_class
    recall = [0] * n_class
    specificity = [0] * n_class
    f1 = [0] * n_class
    valid_epoch_loss = 0.0
    valid_epoch_acc = 0.0

    running_loss = 0.0
    running_corrects = 0
    model.eval()
    valid_bar = tqdm(test_loader)
    for i, (img, label) in enumerate(valid_bar):
        img = img.to(device)
        # labels = label.to(device)
        B, V, C, H, W = img.size()
        img = img.view(-1, C, H, W)
        img = img.to(device, non_blocking=True)

        # b,v = label.shape
        # label=label.view(b*v)

        with torch.no_grad():
            evidences, expert_probabilities, selecter = model(img)
        outputs = evidences.cpu()
        # outputs = output[1].cpu()
        # _, pred = torch.max(outputs.data, 1)
        loss = criterion(outputs, label)

        total_labels = torch.concat((total_labels, label), dim=0) if total_labels.size(0) > 0 else label
        total_outs = torch.concat((total_outs, outputs.detach()), dim=0) if total_outs.size(0) > 0 else outputs.detach()

        running_loss += loss.item()
    preds = torch.argmax(total_outs, dim=1)
    outputs_softmax = torch.softmax(total_outs, 1)
    labels = total_labels
    # print(classification_report(labels,preds))
    cls_statis = [torch.sum(labels == 0), torch.sum(labels == 1), sum(labels == 2), sum(labels == 3), sum(labels == 4)]

    running_corrects += torch.sum(preds == labels)
    # running_corrects+=(output1.argmax(dim=1) == label).sum()

    for batch_i in range(len(labels)):

        # save roc matrix
        roc_df = {'label': labels[batch_i].numpy(),
                  '0': outputs_softmax[batch_i][0].numpy(),
                  '1': outputs_softmax[batch_i][1].numpy(),
                  '2': outputs_softmax[batch_i][2].numpy(),
                  '3': outputs_softmax[batch_i][3].numpy(),
                  '4': outputs_softmax[batch_i][4].numpy()
                  }
        roc_matrix = roc_matrix.append(roc_df, ignore_index=True)

        # fusion matrix
        predict_label = preds[batch_i]
        true_label = labels[batch_i]
        # FM[predict_label][true_label] = FM[predict_label][true_label] + 1
        FM[true_label][predict_label] = FM[true_label][predict_label] + 1

        for label in range(n_class):
            p_or_n_from_pred = (label == preds[batch_i])
            p_or_n_from_label = (label == labels[batch_i])

            if p_or_n_from_pred == 1 and p_or_n_from_label == 1:
                tp[label] += 1
            if p_or_n_from_pred == 0 and p_or_n_from_label == 0:
                tn[label] += 1
            if p_or_n_from_pred == 1 and p_or_n_from_label == 0:
                fp[label] += 1
            if p_or_n_from_pred == 0 and p_or_n_from_label == 1:
                fn[label] += 1

    # each class test results
    for label in range(n_class):
        precision[label] = tp[label] / (tp[label] + fp[label] + 1e-8)
        recall[label] = tp[label] / (tp[label] + fn[label] + 1e-8)
        specificity[label] = tn[label] / (tn[label] + fp[label] + 1e-8)
        f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label] + 1e-8)

        print('Class {}: \t Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, F1:{:.4f}'.format(
            label, precision[label], recall[label], specificity[label], f1[label]))
        fileHandle = open(netname + '_test_result.txt', 'a')
        fileHandle.write('Class {}: \t Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, F1:{:.4f}\n'.format(
            label, precision[label], recall[label], specificity[label], f1[label]))
        fileHandle.close()

        # save Fusion Matric
    print('\nFusion Matrix:')
    print(FM)
    fileHandle = open(netname + '_test_result.txt', 'a')
    fileHandle.write('\nFusion Matrix:\n')
    for f_i in FM:
        fileHandle.write(str(f_i) + '\r\n')
    fileHandle.close()

    # save roc data
    roc_matrix.to_csv(netname + '_roc_data.csv', encoding='gbk')

    # calculate the Kappa
    pe0 = (tp[0] + fn[0]) * (tp[0] + fp[0])
    pe1 = (tp[1] + fn[1]) * (tp[1] + fp[1])
    pe2 = (tp[2] + fn[2]) * (tp[2] + fp[2])
    pe3 = (tp[3] + fn[3]) * (tp[3] + fp[3])
    pe4 = (tp[4] + fn[4]) * (tp[4] + fp[4])
    pe = (pe0 + pe1 + pe2 + pe3 + pe4) / (dataset_size * dataset_size)
    pa = (tp[0] + tp[1] + tp[2] + tp[3] + tp[4]) / dataset_size
    kappa = (pa - pe) / (1 - pe)

    # overall test results
    test_epoch_loss = running_loss / dataset_size
    test_epoch_acc = running_corrects / dataset_size
    overall_precision = sum([cls_statis[i] * p for i, p in enumerate(precision)]) / sum(cls_statis)
    overall_recall = sum([cls_statis[i] * r for i, r in enumerate(recall)]) / sum(cls_statis)

    overall_specificity = sum([cls_statis[i] * s for i, s in enumerate(specificity)]) / sum(cls_statis)
    overall_f1 = sum([cls_statis[i] * f for i, f in enumerate(f1)]) / sum(cls_statis)

    elapsed_time = time.time() - since
    print(
        'Test loss: {:.4f}, acc: {:.4f}, kappa: {:.4f}, avg_precision: {:.4f},avg_recall: {:.4f},avg_specificity: {:.4f},avg_f1: {:.4f},Total elapsed time: {:.4f} '.format(
            test_epoch_loss, test_epoch_acc, kappa, overall_precision, overall_recall, overall_specificity, overall_f1,
            elapsed_time))
    fileHandle = open(netname + '_test_result.txt', 'a')
    fileHandle.write(
        'Test loss: {:.4f}, acc: {:.4f}, kappa: {:.4f}, all_precision: {:.4f},all_recall: {:.4f}, all_specificity: {:.4f}, all_f1: {:.4f}, Total elapsed time: {:.4f} \n'.format(
            test_epoch_loss, test_epoch_acc, kappa, overall_precision, overall_recall, overall_specificity, overall_f1,
            elapsed_time))
    fileHandle.close()
    return (test_epoch_loss, test_epoch_acc)


if __name__ == '__main__':
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    transform_test = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((224, 224)),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    DATA_PATH = "../../EYData_BaseEye_newdata/"
    TEST_PATH = "../../EYData_BaseEye_newdata/test_process"
    MASK_PATH2 = "../../EYData_BaseEye_newdata/test_mask"
    BATCH_SIZE = 1
    MODELPATH = "./weights/best_smvdr.pth"
    checkpoint = torch.load(MODELPATH, map_location=device)
    #model = checkpoint
    model = create.my_SMVDR(
            in_chans=4,
            pretrained=True,
            num_classes=5,
            pre_Path=MODELPATH,
            depth=12,
            num_heads=9,
            drop_rate=0.0,
            drop_path_rate=0.0,
            cuda=device
        )
    test_csv_path = os.path.join(DATA_PATH, 'test_rgb_label_newname.csv')
    test_df = pd.read_csv(test_csv_path)
    test_dataset = MultiviewImgDataset_mask(TEST_PATH, MASK_PATH2, test_df, transform=transform_test, Single=False)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=24, pin_memory=False)

    # device = torch.device('cpu')
    model.to(device)

    test_model2('final', model, test_loader, len(test_dataset),
                nn.CrossEntropyLoss(reduction='sum'), device)
