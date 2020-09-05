import os
import numpy as np
import torch
import torch.nn as nn
import torch.cuda
import torch.utils.data
import logging
import copy
import skimage.transform
import cv2
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm
from dataset.dataset_VQA_ODV import DS_VQA_ODV, VQA_ODV_Transform
from s2cnn import s2_near_identity_grid, S2Convolution, SO3Convolution, so3_near_identity_grid
from s2cnn import so3_integrate
from torch.backends import cudnn
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

NUM_EPOCHS = 10

LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
def train(log_dir,train_interval, train_start_frame):
    bandwidth = 128

    train_set = DS_VQA_ODV(root=os.path.join(log_dir, "VQA_ODV"), dataset_type='train', tr_te_file='tr_te_VQA_ODV.txt',
                          ds_list_file='VQA_ODV.txt', test_interval=train_interval, test_start_frame=train_start_frame,
                          transform=VQA_ODV_Transform(bandwidth=bandwidth, down_resolution=(1024, 2048)))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=0, drop_last=True)

    network = S2Model()
    network = network.to(DEVICE)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    ctriterion1 = nn.SmoothL1Loss()
    ctriterion2 = nn.KLDivLoss()
    ctriterion1 = ctriterion1.to(DEVICE)
    ctriterion2 = ctriterion1.to(DEVICE)

    for epoch in range(NUM_EPOCHS):
        for batch_idx, img_tuple in enumerate(train_loader):
            network.train()
            img_s2, img_ori, _, target = img_tuple
            # x = cv2.cvtColor(img_s2[0].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
            # x_ori = cv2.cvtColor(img_ori[0].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
            # fig = plt.figure()
            # plt.imshow(x.astype(np.int))
            # fig2 = plt.figure()
            # plt.imshow(x_ori.astype(np.int))
            img_s2 = img_s2.to(DEVICE)
            target = target.to(DEVICE)

            pred = network(img_s2)
            loss = ctriterion1(pred, target) + ctriterion2(pred, target)
            loss.backward()

            optimizer.step()
            print('\rEpoch [{0}/{1}], Iter [{2}] Loss: {3:.4f}'.format(
                epoch + 1, NUM_EPOCHS, batch_idx + 1, loss.item()), end="")
        print("")
    torch.save(network.state_dict(), './s2model_param_v8_cat.pkl')
    print("done")
        # print("")
        # preds = []
        # targets = []
        # for timg_tuple in test_loader:
        #
        #     network.eval()
        #     with torch.no_grad():
        #         timg_s2, timg_ori, _, gt = timg_tuple
        #         timg_s2 = timg_s2.to(DEVICE)
        #         gt = gt.to(DEVICE)
        #
        #         output = network(timg_s2)
        #         preds.append(float(output))
        #         targets.append(gt.numpy())
        #     # i += 1
        #     # if i == 5:
        #     #     break
        # preds = np.array(preds)
        # targets = np.array(targets)
        # srocc, _ = scipy.stats.spearmanr(preds, targets)
        # # cc, _ = scipy.stats.pearsonr(preds, targets)
        # # kl = scipy.stats.entropy(preds, targets)
        # print(srocc)
        # # print(cc)
        # # print(kl)
def test(log_dir, test_interval, test_start_frame):
    global batch_idx
    bandwidth = 128

    test_set = DS_VQA_ODV(root=os.path.join(log_dir, "VQA_ODV"), dataset_type='test', tr_te_file='tr_te_VQA_ODV.txt',
                          ds_list_file='VQA_ODV.txt', test_interval=test_interval, test_start_frame=test_start_frame,
                          transform=VQA_ODV_Transform(bandwidth=bandwidth, down_resolution=(1024, 2048)))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0, drop_last=True)

    preds = []
    targets = []
    network = S2Model()
    network = network.to(DEVICE)
    network.load_state_dict(torch.load('./s2model_param_v8_cat.pkl'))
    network.eval()

    length = len(test_set)


    for batch_idx, img_tuple in enumerate(test_loader):

        with torch.no_grad():
            img_s2, timg_ori, _, target = img_tuple
            img_s2 = img_s2.to(DEVICE)
            # target = target.to(DEVICE)

            pred = network(img_s2)

            preds.append(float(pred))
            targets.append(target.numpy())


    preds = np.array(preds)
    targets = np.concatenate(targets, 0)
    video_cnt = len(test_set.cum_frame_num)
    pred = [preds[test_set.cum_frame_num_prev[i]:test_set.cum_frame_num[i]].mean() for i in range(video_cnt)]   # the average score of each video
    targets = [targets[test_set.cum_frame_num_prev[i]:test_set.cum_frame_num[i]].mean() for i in range(video_cnt)]
    np.savetxt(os.path.join(log_dir, 'test_pred_scores_v8.txt'), np.array(pred))
    np.savetxt(os.path.join(log_dir, 'test_targets.txt'), np.array(targets))
    srocc, _ = scipy.stats.spearmanr(pred, targets)
    print(srocc)

class S2Model(nn.Module):
    def __init__(self):
        super(S2Model, self).__init__()

        self.leaky_alpha = 0.1

        grid_s2 = s2_near_identity_grid(max_beta=np.pi / 64, n_alpha=4, n_beta=2)
        self.layer0 = nn.Sequential(
            S2Convolution(3, 16, 128, 64, grid_s2),
            nn.GroupNorm(1, 16),
            nn.LeakyReLU(self.leaky_alpha, inplace=True),
        )

        grid_so3 = so3_near_identity_grid(max_beta=np.pi / 32, max_gamma=0, n_alpha=4, n_beta=2, n_gamma=1)
        self.layer1 = nn.Sequential(
            SO3Convolution(16, 16, 64, 32, grid_so3),
            nn.GroupNorm(1, 16),
            nn.LeakyReLU(self.leaky_alpha, inplace=True),
            SO3Convolution(16, 32, 32, 32, grid_so3),
            nn.GroupNorm(2, 32),
            nn.LeakyReLU(self.leaky_alpha, inplace=True)

        )
        grid_so3 = so3_near_identity_grid(max_beta=np.pi / 16, max_gamma=0, n_alpha=4, n_beta=2, n_gamma=1)
        self.layer2 = nn.Sequential(
            SO3Convolution(48, 48, 32, 16, grid_so3),
            nn.GroupNorm(2, 48),
            nn.LeakyReLU(self.leaky_alpha, inplace=True),
            SO3Convolution(48, 64, 16, 16, grid_so3),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(self.leaky_alpha, inplace=True),

        )
        grid_so3 = so3_near_identity_grid(max_beta=np.pi / 8, max_gamma=0, n_alpha=4, n_beta=2, n_gamma=1)
        self.layer3 = nn.Sequential(
            SO3Convolution(96, 96, 16, 8, grid_so3),
            nn.GroupNorm(4, 96),
            nn.LeakyReLU(self.leaky_alpha, inplace=True),
            SO3Convolution(96, 128, 8, 8, grid_so3),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(self.leaky_alpha, inplace=True),
        )
        grid_so3 = so3_near_identity_grid(max_beta=np.pi / 16, max_gamma=0, n_alpha=4, n_beta=2, n_gamma=1)
        self.layer4 = nn.Sequential(
            SO3Convolution(144, 144, 8, 8, grid_so3),
            nn.GroupNorm(8, 144),
            nn.LeakyReLU(self.leaky_alpha, inplace=True)
        )

        # self.score_layers = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.MaxPool2d(2, 2),
        #     nn.LeakyReLU(self.leaky_alpha, inplace=True),
        #     nn.Conv2d(64, 32, 3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.MaxPool2d(2, 2),
        #     nn.AdaptiveAvgPool2d(1)
        # )

        self.fc = nn.Sequential(
            nn.Linear(589824, 32, bias=False),
            nn.GroupNorm(1, 32),
            nn.Linear(32, 1)

        )


    def forward(self, x):
        x0 = self.layer0(x)  # 1 16 64 64 64
        x1 = self.layer1(x0)  # 1 32 32 32 32
        x0 = nn.functional.interpolate(x0, scale_factor=0.5)  # 1 16 32 32 32
        x2 = self.layer2(torch.cat((x0, x1), 1))  # 1 64 16 16 16

        x1 = nn.functional.interpolate(x1, scale_factor=0.5)  # 1 32 16 16 16
        x3 = self.layer3(torch.cat((x2, x1), 1))  # 1 128 8 8 8

        x0 = nn.functional.interpolate(x0, scale_factor=0.5**2)  # 1 16 8 8 8
        x4 = self.layer4(torch.cat((x3, x0), 1))
        # x = x.mean(-1)
        # x = self.score_layers(x)
        # x = x.view(batch_size, -1)
        # y = self.fc(x)
        # for layer in (self.layer0, self.layer1, self.layer2, self.layer3, self.layer4):
        #     x = layer(x)
        # x = x.mean(-1)
        # x = self.score_layers(x)
        x = x4.view(batch_size, -1)
        y = self.fc(x)
        return y

if __name__=="__main__":
    # train(log_dir='./log', train_interval=2, train_start_frame=10)
    test(log_dir='./log', test_interval=10, test_start_frame=1)