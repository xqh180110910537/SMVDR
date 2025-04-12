import os
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import random
import numpy as np
import torch
import cv2
import timm
import pandas as pd
import copy
import matplotlib.pyplot as plt
import torchvision.transforms as transform
from sklearn import model_selection
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
import models.create_models as create
from tqdm import tqdm
from dataset import CustomDataset, MultiviewImgDataset_mask, MultiviewImgDataset_no_lesion, SingleimgDataset
from models.FLoss import FocalLoss
from models.mamba import _ratio_loss
from models.moe import entropy_regularization_loss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(model_name, N_EPOCHS=100, LR=0.0001, depth=12, head=9, num_classes=5):
    print(model_name, 1)
    model = create.my_SMVDR(
        in_chans=4,
        pretrained=False,
        num_classes=5,
        pre_Path='weights/new_epoch_0.840074360370636.pth',
        depth=depth,
        num_heads=head,
        # embed_dim=768,
        drop_rate=0.1,
        drop_path_rate=0.1,
        cuda=device,
    )
    # for name, parameter in model.named_parameters():
    #     p = name.split('.')
    #     print(p)
    #     if p[0] == 'trans_cls_head' or p[0] == 'trans_cls_head_dan' or p[0] == 'key_select' or p[0] == 'fusion':
    #         parameter.requires_grad = True
    #     else:
    #         parameter.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    criterion = FocalLoss(gamma=2)

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    inter_val = 1

    model.to(device)
    best_acc = 0
    for epoch in range(N_EPOCHS):

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0

        model.train()
        train_bar = tqdm(train_loader)

        for i, (img, label) in enumerate(train_bar):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            B, V, C, H, W = img.size()

            imgs_mix = img.view(-1, C, H, W)
            evidences, expert_probabilities, selector = model(imgs_mix)
            loss = criterion(evidences, label) + 0.1 * _ratio_loss(True, selector,
                                                                   ratio=0.7) + 0.1 * entropy_regularization_loss(
                expert_probabilities)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_acc += (evidences.argmax(dim=1) == label).sum()
            train_epoch_loss += loss.item()
            scheduler.step(epoch + i / len(train_loader))
            optimizer.step()

        train_loss_mean = train_epoch_loss / len(train_loader)
        train_acc_mean = train_epoch_acc / (len(train_dataset) * NUM_VIEW)

        train_loss.append(train_loss_mean)
        train_acc.append(train_acc_mean.cpu())

        print('{} train loss: {:.3f} train acc: {:.3f}  lr:{}'.format(epoch, train_loss_mean,
                                                                      train_acc_mean,
                                                                      optimizer.param_groups[-1]['lr']))
        if (epoch + 1) % inter_val == 0:
            val_acc_mean, val_loss_mean = val_model(model, test_loader, criterion, device)

            if val_acc_mean > best_acc:
                model.cpu()
                best_model = copy.deepcopy(model.state_dict())
                model.to(device)
                best_acc = val_acc_mean
                # torch.save(model, f'weights/0.4_0.4_epoch_{best_acc}.pth')
            valid_loss.append(val_loss_mean)
            valid_acc.append(val_acc_mean.cpu())
        print('now_best:{:.6f}'.format(best_acc))
        # test_acc_mean = testModel(model,test_loader,len(test_dataset),device)
        # if test_acc_mean>best_test:
        #     best_test = test_acc_mean

    print("best val acc:", best_acc)
    # if best_test != 0:
    #     print("best test acc:", best_test)
    # torch.save(best_model, os.path.join(SAVE_PT_DIR, '{}-{:.4f}.pth'.format(model_name, best_acc)))
    torch.save(model.state_dict(), 'best_smvdr.pth')  # 保存修改后的模型权重
    print("model saved at weights")


def val_model(model, valid_loader, criterion, device):
    valid_epoch_loss = 0.0
    valid_epoch_acc = 0.0

    model.eval()
    valid_bar = tqdm(valid_loader)
    gamma = 1
    for i, (img, label) in enumerate(valid_bar):
        img = img.to(device)
        label = label.to(device)
        num_classes = 5
        B, V, C, H, W = img.size()
        img = img.view(-1, C, H, W)
        img = img.to(device, non_blocking=True)
        # b,v = label.shape
        # label=label.view(b*v)

        with torch.no_grad():
            evidences, expert_probabilities, selector = model(img)
        # loss = criterion(evidences, label)+0.1*_ratio_loss(True,selector,ratio=0.7)+0.1*entropy_regularization_loss(expert_probabilities)
        # loss = criterion(evidences, label) + 0.1 * entropy_regularization_loss(
        #     expert_probabilities) + 0.2 * entropy_regularization_loss(expert_probabilities)
        # loss = criterion(evidences, label) + 0.5 * _ratio_loss(True, selector, ratio=0.7)+ 0.1 * entropy_regularization_loss(expert_probabilities)
        loss = criterion(evidences, label)
        # output = output[1]
        # output = output[0]
        # with torch.no_grad():
        #     output,_= model(img)
        # loss = criterion(evidences, label)

        # loss = get_loss(evidences, evidence_a, label, i, num_classes, 50, gamma, device, use_KL=False)
        valid_epoch_loss += loss.item()
        valid_epoch_acc += (evidences.argmax(dim=1) == label).sum()

    val_acc_mean = valid_epoch_acc / (len(valid_dataset) * NUM_VIEW)
    val_loss_mean = valid_epoch_loss / len(valid_loader)

    print('valid loss: {:.3f} valid acc: {:.3f}'.format(val_loss_mean, val_acc_mean))

    return val_acc_mean, val_loss_mean


if __name__ == '__main__':
    seed_everything(1001)
    # general global variables
    DATA_PATH = "../../EYData_BaseEye_newdata/"
    TRAIN_PATH = "../../EYData_BaseEye_newdata/train_process"
    MASK_PATH = "../../EYData_BaseEye_newdata/train_mask"
    TEST_PATH = "../../EYData_BaseEye_newdata/test_process"
    MASK_PATH2 = "../../EYData_BaseEye_newdata/test_mask"
    # DATA_PATH = "/disk2/xuqihao/MVCN/EYData_BaseEye_newdata/"
    # TRAIN_PATH = "/disk2/xuqihao/MVCN/EYData_BaseEye_newdata/train_process"
    # MASK_PATH = "/disk2/xuqihao/MVCN/EYData_BaseEye_newdata/train_mask"
    # TEST_PATH = "/disk2/xuqihao/MVCN/EYData_BaseEye_newdata/test_process"
    # MASK_PATH2 = "/disk2/xuqihao/MVCN/EYData_BaseEye_newdata/test_mask"
    SAVE_IMG_DIR = 'imgs'
    SAVE_PT_DIR = 'weights'
    NUM_VIEW = 1
    IMAGE_SIZE = 224
    LR = 0.00001
    N_EPOCHS = 100
    DEPTH = 12
    HEAD = 9
    BATCH_SIZE = 4
    train_csv_path = os.path.join(DATA_PATH, 'train_rgb_label_newname.csv')
    assert os.path.exists(train_csv_path), '{} path is not exists...'.format(train_csv_path)
    test_csv_path = os.path.join(DATA_PATH, 'test_rgb_label_newname.csv')
    test_df = pd.read_csv(test_csv_path)

    all_data = pd.read_csv(train_csv_path)
    all_data.head()

    transform_train = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.RandomHorizontalFlip(p=0.0),
        transform.RandomVerticalFlip(p=0.0),
        transform.RandomResizedCrop(IMAGE_SIZE),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_valid = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((224, 224)),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = MultiviewImgDataset_mask(TRAIN_PATH, MASK_PATH, all_data, transform=transform_train, Single=False,
                                             no_mask=False)

    valid_dataset = MultiviewImgDataset_mask(TEST_PATH, MASK_PATH2, test_df, transform=transform_test, Single=False,
                                             no_mask=False)
    test_dataset = MultiviewImgDataset_mask(TEST_PATH, MASK_PATH2, test_df, transform=transform_test, Single=False,
                                            no_mask=False)
    # train_dataset = MultiviewImgDataset_no_lesion(TRAIN_PATH, all_data, transform=transform_train)
    # valid_dataset = MultiviewImgDataset_no_lesion(TEST_PATH, test_df, transform=transform_test)
    # test_dataset = MultiviewImgDataset_no_lesion(TEST_PATH, test_df, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    main(model_name=f'main_{LR}_{N_EPOCHS}_d{DEPTH}_h{HEAD}', N_EPOCHS=N_EPOCHS, LR=LR, depth=DEPTH, head=HEAD)
