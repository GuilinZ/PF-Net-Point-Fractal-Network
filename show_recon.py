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
parser.add_argument('--workers', type=int,default=8, help='number of data loading workers')
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


# dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=True, transforms=transforms, download = False)
# assert dset
# dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
#                                         shuffle=True,num_workers = int(opt.workers))


test_dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=False, transforms=transforms, download = False)
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                        shuffle=False,num_workers = int(opt.workers))

# test_dset = shapenet_part_loader.PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice='Airplane', npoints=opt.pnum, split='test')
# test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
#                                          shuffle=False,num_workers = int(opt.workers))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num) 
# point_netG = torch.nn.DataParallel(point_netG)
point_netG.to(device)
point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])   
point_netG.eval()

experiment_dir = './test_debugged/test/log/classification/pointnet2_ssg_wo_normals'
classifier = pointnet2.get_model()
classifier = classifier.cuda()
checkpoint = torch.load(str(experiment_dir) + '/checkpoints/ckpt.pt')
classifier.load_state_dict(checkpoint)
classifier = classifier.eval().eval()

input_cropped1 = torch.FloatTensor(opt.batchSize, 1, opt.pnum, 3)
criterion_PointLoss = PointLoss().to(device)
errG_min = 100
n = 0
acc = []
for i, data in enumerate(test_dataloader, 0):
    print(i)
    real_point, target = data      
    
    real_point = torch.unsqueeze(real_point, 1)
    batch_size = real_point.size()[0]
    real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
         
    input_cropped1.resize_(real_point.size()).copy_(real_point)
    p_origin = [0,0,0]
    
    choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
    index = random.sample(choice,1)
    distance_list = []
    p_center = index[0]
    center_id = 9
    # p_center = choice[center_id]
    for num in range(opt.pnum):
        distance_list.append(distance_squre(real_point[0,0,num],p_center))
    distance_order = sorted(enumerate(distance_list), key = lambda x:x[1])
    
    for sp in range(opt.crop_point_num):
        input_cropped1.data[0,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])
        real_center.data[0,0,sp] = real_point[0,0,distance_order[sp][0]]
    
    real_center.to(device) 
    real_center = torch.squeeze(real_center,1)
    
    input_cropped1 = torch.squeeze(input_cropped1,1)
    input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
    input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
    input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
    input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
    input_cropped1 = input_cropped1.to(device)
    input_cropped2 = input_cropped2.to(device)
    input_cropped3 = input_cropped3.to(device)      
    input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
    

    fake_center1,fake_center2,fake=point_netG(input_cropped)
    fake = fake.cuda()
    real_center = real_center.cuda()
    real_center =real_center.cuda()
    errG = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))#+0.1*criterion_PointLoss(torch.squeeze(fake_part,1),torch.squeeze(real_center,1))
    errG = errG.cpu()
    a = random.randint(4,8)
    b = 6
#     if errG.detach().numpy()>errG_min:
# #    if a!=b:
#         pass
    
    # else:
    errG_min = errG.detach().numpy()
    # print(errG_min)
    fake =fake.cpu()
    np_fake = fake[0].detach().numpy()  #256
    real_center = real_center.cpu()
    np_real = real_center.data[0].detach().numpy() #256
    input_cropped1 = input_cropped1.cpu()
    np_inco = input_cropped1[0].detach().numpy() #1024
    np_crop = []
    n=n+1
    k = 0
    for m in range(opt.pnum):
        if distance_squre1(np_inco[m],p_origin)==0.00000 and k<opt.crop_point_num:
            k += 1
            pass
        else:
            np_crop.append(np_inco[m])

    # np.savetxt('test_example/crop_l'+str(target.item())+'_'+str(n)+'.csv', np_crop, fmt = "%f,%f,%f")
    # np.savetxt('test_example/fake_l'+str(target.item())+'_'+str(n)+'.csv', np_fake, fmt = "%f,%f,%f")
    # np.savetxt('test_example/real_l'+str(target.item())+'_'+str(n)+'.csv', np_real, fmt = "%f,%f,%f")
    np.savetxt('test_example/'+'%02d/'%(center_id)+str(n)+'_'+'crop_label'+str(target.item())+'.txt', np_crop, fmt = "%f;%f;%f")
    np.savetxt('test_example/'+'%02d/'%(center_id)+str(n)+'_'+'fake_label'+str(target.item())+'.txt', np_fake, fmt = "%f;%f;%f")
    np.savetxt('test_example/'+'%02d/'%(center_id)+str(n)+'_'+'real_label'+str(target.item())+'.txt', np_real, fmt = "%f;%f;%f")
    # np_crop = np.array(np_crop)
    # np_completed = np.vstack((np_crop,np_fake))
    # # points = farthest_point_sample(np.array(np_crop), 1024)
    # # points = rs(np_crop, 1024)
    # points = rs((np_completed), 1024)
    # points = torch.Tensor(points).cuda().unsqueeze(0)
    # points = points.transpose(2, 1)
    # # print(points.shape)  # 1x3xn
    # pred, _ = classifier(points)
    # pred_choice = pred.data.max(1)[1]
    # if target.item() == pred_choice.item():
    #     acc.append(1)
    # else:
    #     acc.append(0)
    # print('target: ', target.item(), 'p++ prediction: ', pred_choice.item())
# print('done: ', sum(acc)/len(acc))
print('done')