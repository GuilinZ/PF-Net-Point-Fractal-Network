#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import numpy as np
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from utils import PointLoss
from utils import distance_squre
import data_utils as d_utils
import ModelNet40Loader
import shapenet_part_loader
from model_PFNet import _netlocalD,_netG
from test_debugged.test import pointnet2_cls_ssg as pointnet2


parser = argparse.ArgumentParser()
#parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10|ModelNet40|ShapeNet')
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int,default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=512,help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='Checkpoint/point_netG.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
opt = parser.parse_args()
print(opt)
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
def rs(point, npoint):
    sample_idx = random.sample(range(len(point)),npoint)
    new_point = point[sample_idx]
    return new_point
def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 

transforms = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
    ]
)



test_dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=False, transforms=transforms, download = False)
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                        shuffle=False,num_workers = int(opt.workers))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


experiment_dir = './test_debugged/test/log/classification/pointnet2_ssg_wo_normals'
classifier = pointnet2.get_model()
classifier = classifier.cuda()
checkpoint = torch.load(str(experiment_dir) + '/checkpoints/ckpt.pt')
classifier.load_state_dict(checkpoint)
classifier = classifier.eval().eval()

n = 0
crop_acc = []
complete_acc = []
for i, data in enumerate(test_dataloader, 0):
    n = n + 1
    print(i)
    real_point, target = data      
    np_crop = np.loadtxt('./test_example/crop_txt_l'+str(target.item())+'_'+str(n)+'.txt', delimiter=',')
    np_fake = np.loadtxt('./test_example/fake_txt_l'+str(target.item())+'_'+str(n)+'.txt', delimiter=',')
    np_real = np.loadtxt('./test_example/real_txt_l'+str(target.item())+'_'+str(n)+'.txt', delimiter=',')
    # np.savetxt('test_example/crop_txt_l'+str(target.item())+'_'+str(n)+'.txt', np_crop, fmt = "%f,%f,%f")
    # np.savetxt('test_example/fake_txt_l'+str(target.item())+'_'+str(n)+'.txt', np_fake, fmt = "%f,%f,%f")
    # np.savetxt('test_example/real_txt_l'+str(target.item())+'_'+str(n)+'.txt', np_real, fmt = "%f,%f,%f")
    np_crop = np.array(np_crop)
    np_completed = np.vstack((np_crop,np_fake))
    # points = farthest_point_sample(np.array(np_crop), 1024)
    # points = rs(np_crop, 1024)
    points = rs((np_crop), 1024)
    points = torch.Tensor(points).cuda().unsqueeze(0)
    points = points.transpose(2, 1)
    # print(points.shape)  # 1x3xn
    pred, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    if target.item() == pred_choice.item():
        crop_acc.append(1)
    else:
        crop_acc.append(0)

    points = rs((np_completed), 1024)
    points = torch.Tensor(points).cuda().unsqueeze(0)
    points = points.transpose(2, 1)
    # print(points.shape)  # 1x3xn
    pred, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    if target.item() == pred_choice.item():
        complete_acc.append(1)
    else:
        complete_acc.append(0)
    # print('target: ', target.item(), 'p++ prediction: ', pred_choice.item())
print('crop acc: ', sum(crop_acc)/len(crop_acc))
print('complete acc: ', sum(complete_acc)/len(complete_acc))