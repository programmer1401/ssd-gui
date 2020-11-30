from __future__ import print_function
from torch.autograd import Variable
from data import VOC_CLASSES
from layers.img_utils import show_box
from ssd import build_ssd

import os
import cv2
import torch
import torch.backends.cudnn as cudnn

import warnings

warnings.filterwarnings("ignore")

def test_net(net, cuda, img, transform):
    # print(torch.cuda.get_device_name(0))

    # dump predictions and assoc. ground truth to text file for now
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))

    if cuda:
        x = x.cuda()

    y = net(x)  # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    pred_annotation = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            j += 1

            pt = [pt[0], pt[1], pt[2], pt[3], i - 1, float(score)]
            pred_annotation = pred_annotation + [pt]

        # predicted box
        show_box(img, pred_annotation, (0, 0, 255), True)
        # cv2.moveWindow("camera", 1, 30)
        # cv2.imshow("camera-capture", img, )


def test_voc(trained_model_path, cuda):
    if cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # load net
    num_classes = len(VOC_CLASSES) + 1  # +1 background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(trained_model_path))
    net.eval()
    print('Finished loading model!')

    if cuda:
        net = net.cuda()
        cudnn.benchmark = False

    return net


if __name__ == '__main__':
    test_voc()
