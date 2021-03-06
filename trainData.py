from data import *
from layers.img_utils import show_loss
from resnet_ssd import build_resnet_ssd
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_ENABLE_DEVICES'] = '0'

class TrainData:
    def __init__(self, iter_num, dataset='VOC', dataset_root=VOC_ROOT,
                 basenet='vgg16_reducedfc.pth', batch_size=4, cuda=False,
                 lr=1e-4, momentum=0.9, weight_decay=5e-4, gamma=0.1,
                 save_folder="E:\Code\Examples\Gao\ssd.pytorch-master\weights/"):
        self.dataset_name = dataset
        self.dataset_root = dataset_root
        self.basenet = basenet
        self.iter_num = iter_num
        self.batch_size = batch_size
        self.cuda = cuda
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.save_folder = save_folder

    def set_equipment(self):
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            if self.cuda:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if not self.cuda:
                print("WARNING: It looks like you have a CUDA device, but aren't " +
                      "using CUDA.\nRun with --cuda for optimal training speed.")
                torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

    def set_dataset(self):
        global cfg, dataset
        if self.dataset_name == 'COCO':
            if self.dataset_root == VOC_ROOT:
                print("WARNING: Using default COCO dataset_root because " +
                      "--dataset_root was not specified.")
                self.dataset_root = COCO_ROOT
            cfg = coco
            dataset = COCODetection(root=self.dataset_root,
                                    transform=SSDAugmentation(cfg['min_dim'],
                                                              MEANS))
        elif self.dataset_name == 'VOC':
            cfg = voc
            dataset = VOCDetection(root=self.dataset_root,
                                   transform=SSDAugmentation(cfg['min_dim'],
                                                             MEANS))

        return cfg, dataset


    def set_net(self):
        global ssd_net, net
        if self.basenet == 'vgg16_reducedfc.pth':
            print('Loading base network vgg16...')
            ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
            net = ssd_net

            if self.cuda:
                net = torch.nn.DataParallel(ssd_net)
                cudnn.benchmark = False

            vgg_weights = torch.load(self.save_folder + self.basenet)
            ssd_net.vgg.load_state_dict(vgg_weights)

        elif self.basenet == 'resnet50-19c8e357.pth':
            print('Loading base network resnet50...')
            ssd_net = build_resnet_ssd('train', 300, num_classes=21)
            net = ssd_net

            # 为resnet_ssd加载预训练权重文件
            # 1、加载模型
            resnet_pretrained_weights = torch.load(self.save_folder + self.basenet)
            # 2、初始化网络
            resnet_dict = ssd_net.state_dict()
            # 3、剔除掉网络中没有的键
            pretrained_dict_l = {k: v for k, v in resnet_pretrained_weights.items() if k in resnet_dict}
            # 4、用预训练的权重文件，更新网络权重
            resnet_dict.update(pretrained_dict_l)
            # 5、将更新了的参数放入网络中
            ssd_net.load_state_dict(resnet_dict)

        return ssd_net, net

    def train(self):
        self.set_equipment()
        cfg, dataset = self.set_dataset()
        ssd_net, net = self.set_net()

        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                 False, self.cuda)

        net.train()
        # loss counters
        loc_loss = 0
        conf_loss = 0
        epoch = 0
        loss_ass = []
        print('Loading the dataset...')

        epoch_size = len(dataset) // self.batch_size
        print('Training SSD on:', dataset.name)

        step_index = 0

        data_loader = data.DataLoader(dataset, self.batch_size,
                                      shuffle=True, collate_fn=detection_collate,
                                      pin_memory=True)
        # create batch iterator
        batch_iterator = iter(data_loader)
        for iteration in range(self.iter_num):

            if iteration in cfg['lr_steps']:
                step_index += 1
                self.adjust_learning_rate(optimizer, self.gamma, step_index)

            # load train data
            try:
                images, targets = next(batch_iterator)
            except StopIteration as e:
                batch_iterator = iter(data_loader)
                images, targets = next(batch_iterator)

            if self.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            if iteration % 2 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')
            loss_ass = loss_ass + [loss.item()]

            if iteration != 0 and iteration % 5 == 0:
                print('Saving state, iter:', iteration)
                # torch.save(ssd_net.state_dict(), 'E:\Code\Examples\Gao\ssd.pytorch-master\weights/ssd300_VOC_' +
                #            repr(iteration) + '.pth')

        torch.save(ssd_net.state_dict(), self.save_folder + '' + self.dataset_name + '.pth')
        return loss_ass


    def adjust_learning_rate(self, optimizer, gamma, step):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        lr = self.lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def xavier(param):
        init.xavier_uniform(param)


    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            m.bias.data.zero_()
