from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap

from layers.img_utils import show_PrecisionRecall
from ssd import build_ssd

import sys
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


class EvalData:
    def __init__(self, trained_model="D:\WorkSpace\PyCharmSpace\SSD\ssd.pytorch-master\weights\VOC.pth",
                 save_folder="D:\WorkSpace\PyCharmSpace\SSD\ssd.pytorch-master\gui\eval\\",
                 cuda=False, voc_root=VOC_ROOT):
        self.trained_model = trained_model
        self.save_folder = save_folder
        self.cuda = cuda
        self.voc_root = voc_root

        self.annopath = 'D:\WorkSpace\PyCharmSpace\SSD\ssd.pytorch-master\data\VOCdevkit\VOC2007\Annotations\%s.xml'
        self.imgpath = 'D:\WorkSpace\PyCharmSpace\SSD\ssd.pytorch-master\data\VOCdevkit\VOC2007\JPEGImages\%s.jpg'
        self.imgsetpath = 'D:\WorkSpace\PyCharmSpace\SSD\ssd.pytorch-master\data\VOCdevkit\VOC2007\ImageSets\Main\{:s}.txt'
        self.YEAR = '2007'
        self.devkit_path = self.save_folder + 'VOC' + self.YEAR
        self.dataset_mean = (104, 117, 123)
        self.set_type = 'test'

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

    def start_eval(self):
        # load net
        num_classes = len(labelmap) + 1  # +1 for background
        net = build_ssd('test', 300, num_classes)  # initialize SSD
        net.load_state_dict(torch.load(self.trained_model))
        net.eval()
        print('Finished loading model!')
        # load data
        dataset = VOCDetection(self.voc_root, [('2007', self.set_type)],
                               BaseTransform(300, self.dataset_mean),
                               VOCAnnotationTransform())
        if self.cuda:
            net = net.cuda()
            cudnn.benchmark = False
        # evaluation
        self.test_net(net, self.cuda, dataset)

    def get_voc_results_file_template(self, image_set, cls):
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'det_' + image_set + '_%s.txt' % (cls)
        filedir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def write_voc_results_file(self, all_boxes, dataset):
        for cls_ind, cls in enumerate(labelmap):
            print('Writing {:s} VOC results file'.format(cls))
            filename = self.get_voc_results_file_template(self.set_type, cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(dataset.ids):
                    dets = all_boxes[cls_ind + 1][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def voc_ap(self, rec, prec, use_07_metric=True):
        """
        ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def voc_eval(self, detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=True):
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                           (BBGT[:, 2] - BBGT[:, 0]) *
                           (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.

        show_PrecisionRecall(self.save_folder+"PRcurve\\", classname, prec, rec, ap)
        return rec, prec, ap

    def do_python_eval(self, use_07=True):
        self.aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        for i, cls in enumerate(labelmap):
            filename = self.get_voc_results_file_template(self.set_type, cls)
            rec, prec, ap = self.voc_eval(
                filename, self.annopath, self.imgsetpath.format(self.set_type), cls,
                ovthresh=0.5, use_07_metric=use_07_metric)
            self.aps += [ap]

        # print('~~~~~~~~')
        # print('Results:')
        # for ap in aps:
        #     print('{:.3f}'.format(ap))
        # print('Mean AP = {:.4f}'.format(np.mean(aps)))

    def evaluate_detections(self, box_list, dataset):
        self.write_voc_results_file(box_list, dataset)
        self.do_python_eval()

    def test_net(self, net, cuda, dataset):
        num_images = 3
        # num_images = len(dataset)
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in range(9963)]
                     for _ in range(len(labelmap) + 1)]

        for i in range(num_images):
            im, gt, h, w = dataset.pull_item(i)

            x = Variable(im.unsqueeze(0))
            if cuda:
                x = x.cuda()
            detections = net(x).data

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[j][i] = cls_dets

        print('Evaluating detections')
        self.evaluate_detections(all_boxes, dataset)


if __name__ == '__main__':
    a = EvalData()
    a.start_eval()