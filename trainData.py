import visdom

from data import *
from resnet_ssd import build_ssd
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
# from vgg import build_ssd
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

viz = visdom.Visdom()

os.environ['CUDA_ENABLE_DEVICES'] = '0'

class TrainData:
    def __init__(self, iter_num, dataset='VOC', dataset_root=VOC_ROOT, model="vgg",
                 basenet='vgg16_reducedfc.pth', batch_size=4, cuda=False,
                 lr=1e-4, momentum=0.9, weight_decay=5e-4, gamma=0.1,
                 save_folder="D:\WorkSpace\PyCharmSpace\SSD\ssd.pytorch-master\weights\\"):

        self.basenet = basenet
        self.batch_size = batch_size
        self.cuda = cuda
        self.dataset = dataset
        self.dataset_root = dataset_root
        self.gamma = gamma
        self.iter_num = iter_num
        self.lr = lr
        self.momentum = momentum
        self.model = model
        self.save_folder = save_folder
        self.weight_decay = weight_decay

        self.visdom = True

    def set_equipment(self):
        if torch.cuda.is_available():
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


    def train(self):
        self.set_equipment()
        if self.dataset == 'COCO':
            if self.dataset_root == VOC_ROOT:
                # if not os.path.exists(COCO_ROOT):
                #     parser.error('Must specify dataset_root if specifying dataset')
                print("WARNING: Using default COCO dataset_root because " +
                      "--dataset_root was not specified.")
                self.dataset_root = COCO_ROOT
            cfg = coco
            dataset = COCODetection(root=self.dataset_root,
                                    transform=SSDAugmentation(cfg['min_dim'],
                                                              MEANS))
        elif self.dataset == 'VOC':
            # if self.dataset_root == COCO_ROOT:
            #     parser.error('Must specify dataset if specifying dataset_root')
            cfg = voc
            dataset = VOCDetection(root=self.dataset_root,
                                   transform=SSDAugmentation(cfg['min_dim'],
                                                             MEANS))

        # if args.visdom:
        #     viz = visdom.Visdom()

        ssd_net = build_ssd('train', self.model, cfg['min_dim'], cfg['num_classes'])
        net = ssd_net

        if self.cuda:
            net = torch.nn.DataParallel(ssd_net)
            cudnn.benchmark = True

        print('Loading base network...')
        if self.model=="vgg":
            base_weights = torch.load(self.save_folder +"vgg\\"+ self.basenet)
        elif self.model=="resnet":
            base_weights = torch.load(self.save_folder + "resnet\\" + self.basenet)
        ssd_net.base.load_state_dict(base_weights)

        if self.cuda:
            net = net.cuda()

        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(self.weights_init)
        ssd_net.loc.apply(self.weights_init)
        ssd_net.conf.apply(self.weights_init)

        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                 False, self.cuda)

        net.train()
        # loss counters
        loc_loss = 0
        conf_loss = 0

        loss_ass = []
        conf_loss_ass = []
        loc_loss_ass = []

        print('Loading the dataset...', 'Training SSD on:', self.dataset)
        step_index = 0

        if self.visdom:
            vis_title = 'SSD.PyTorch on ' + dataset.name + '-backbone ' + self.model
            vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
            iter_plot = self.create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)

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

            iteration = iteration + 1
            if iteration % 2 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

            if self.visdom:
                self.update_vis_plot(iteration, loss_l.item(), loss_c.item(), iter_plot, None, 'append')

            loc_loss_ass = loc_loss_ass + [loss_l.item()]
            conf_loss_ass = conf_loss_ass + [loss_c.item()]
            loss_ass = loss_ass + [loss.item()]

        return loc_loss_ass, conf_loss_ass, loss_ass

    def adjust_learning_rate(self, optimizer, gamma, step):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        lr = self.lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # xavier(m.weight.data)
            # m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def create_vis_plot(self, _xlabel, _ylabel, _title, _legend):
        return viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel=_xlabel,
                ylabel=_ylabel,
                title=_title,
                legend=_legend
            )
        )

    def update_vis_plot(self, iteration, loc, conf, window1, window2, update_type,
                        epoch_size=1):
        viz.line(
            X=torch.ones((1, 3)).cpu() * iteration,
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
            win=window1,
            update=update_type
        )
        # initialize epoch plot on first iteration
        if iteration == 0:
            viz.line(
                X=torch.zeros((1, 3)).cpu(),
                Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
                win=window2,
                update=True
            )

