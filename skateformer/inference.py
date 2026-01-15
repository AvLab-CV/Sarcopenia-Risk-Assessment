from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import pandas as pd

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction

# LR Scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

torch.multiprocessing.set_sharing_strategy('file_system')

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_parser():
    parser = argparse.ArgumentParser(description='SkateFormer: Skeletal-Temporal Trnasformer for Human Action Recognition')
    parser.add_argument('--work-dir', default='./work_dir', help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument('--config', default='./config', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=30, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=4, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for test')
    parser.add_argument('--val-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for validation')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
    parser.add_argument('--freeze_all', type=str2bool, default=False, help='if the weights are going to be frozen for training')
    parser.add_argument('--unfreeze', type=str, default=[], nargs='+', help='the name of weights which will be unfreezed for training')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--min-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--warmup-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--warmup_prefix', type=bool, default=False)
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--grad-clip', type=bool, default=False)
    parser.add_argument('--grad-max', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='AdamW', help='type of optimizer')
    parser.add_argument('--lr-scheduler', default='cosine', help='type of learning rate scheduler')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr-ratio', type=float, default=0.001, help='decay rate for learning rate')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--loss-type', type=str, default='CE')
    return parser


class Processor():
    model: nn.Module

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.global_step = 0
        self.load_model()
        self.load_data()
        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        if self.arg.loss_type == 'CE':
            self.loss = nn.CrossEntropyLoss().cuda(output_device)
        else:
            self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(output_device)

        if self.arg.weights:
            #self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

            if self.arg.freeze_all:
                self.print_log("Freezing all parameters")

                for name, param in self.model.named_parameters():
                    unfreeze = name in self.arg.unfreeze
                    param.requires_grad_(unfreeze)

                    if unfreeze:
                        self.print_log(f"Unfreezing {name}")
                        

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def inference_sliding_window(self, epoch, save_score, loader_name=['test']):
        self.model.eval()
        self.print_log('Sliding window: {}'.format(epoch + 1))
        for ln in loader_name:
            WINDOW_SIZE = 64
            STRIDE = 8
            process = tqdm(self.data_loader[ln])
            ds = []

            for batch_idx, (data, data_len, label, index) in enumerate(process):
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    Y = []

                    data = data[:, :, :data_len, :, :]
                    for i in range(0, data.shape[2] -  WINDOW_SIZE, STRIDE):
                        t_indices = torch.arange(i, i + WINDOW_SIZE, device=self.output_device)
                        index_t = 2 * t_indices / WINDOW_SIZE - 1
                        index_t = index_t.unsqueeze(0).repeat(1, 1)  # shape [B, WINDOW_SIZE]
                        #print(f"wind1 = {i}:{i+WINDOW_SIZE}")
                        data_window = data[:, :, i:i+WINDOW_SIZE, :, :]
                        pred = self.model(data_window, index_t)
                        pred_prob = F.softmax(pred, dim=1)
                        Y.append(pred_prob[0, 1].item())
                    
                        
                    Y = np.array(Y)
                    ds.append({
                                     "index": index.item(),
                                     "seq_len": data_len.item(),
                                     "subject_has_sarcopenia": label.item(),
                                     # 0 = stable, 1 = unstable
                                     # but Y is actually the probability of unstable after softmax
                                     "predictions": Y,
                                 })




            df = pd.DataFrame(ds)
            df.to_csv(f"{self.arg.work_dir}/sliding_window_windowsize{WINDOW_SIZE}_stride{STRIDE}.csv")

    def start(self):
        if self.arg.weights is None:
            raise ValueError('Please appoint --weights.')
        self.arg.print_log = False
        self.print_log('Model:   {}.'.format(self.arg.model))
        self.print_log('Weights: {}.'.format(self.arg.weights))
        self.inference_sliding_window(epoch=0, save_score=False, loader_name=['test'])
        self.print_log('Done.\n')


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()

