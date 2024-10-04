import os
import sys
import json
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from importlib import import_module
from collections import defaultdict
from multiprocessing import Process

logger = logging.getLogger(__name__)



#########################
# Argparser Restriction #
#########################
class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def __eq__(self, other):
        return self.start <= other <= self.end
    
    def __str__(self):
        return f'Specificed Range: [{self.start:.2f}, {self.end:.2f}]'

########
# Seed #
########
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f'[SEED] ...seed is set: {seed}!')
    
###############
# TensorBaord #
###############
class TensorBoardRunner:
    def __init__(self, path, host, port):
        logger.info('[TENSORBOARD] Start TensorBoard process!')
        self.server = TensorboardServer(path, host, port)
        self.server.start()
        self.daemon = True
         
    def finalize(self):
        if self.server.is_alive():    
            self.server.terminate()
            self.server.join()
        self.server.pkill()
        logger.info('[TENSORBOARD] ...finished TensorBoard process!')
        
    def interrupt(self):
        self.server.pkill()
        if self.server.is_alive():    
            self.server.terminate()
            self.server.join()
        logger.info('[TENSORBOARD] ...interrupted; killed all TensorBoard processes!')

class TensorboardServer(Process):
    def __init__(self, path, host, port):
        super().__init__()
        self.os_name = os.name
        self.path = str(path)
        self.host = host
        self.port = port
        self.daemon = True

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.path}" --host {self.host} --reuse_port=true --port {self.port} 2> NUL')
        elif self.os_name == 'posix':  # Linux
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.path}" --host {self.host} --reuse_port=true --port {self.port} >/dev/null 2>&1')
        else:
            err = f'Current OS ({self.os_name}) is not supported!'
            logger.exception(err)
            raise Exception(err)
    
    def pkill(self):
        if self.os_name == 'nt':
            os.system(f'taskkill /IM "tensorboard.exe" /F')
        elif self.os_name == 'posix':
            os.system('pgrep -f tensorboard | xargs kill -9')

###############
# tqdm add-on #
###############
class TqdmToLogger(tqdm):
    def __init__(self, *args, logger=None, 
    mininterval=0.1, 
    bar_format='{desc:<}{percentage:3.0f}% |{bar:20}| [{n_fmt:6s}/{total_fmt}]', 
    desc=None, 
    **kwargs
    ):
        self._logger = logger
        super().__init__(*args, mininterval=mininterval, bar_format=bar_format, desc=desc, **kwargs)

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        return logger

    def display(self, msg=None, pos=None):
        if not self.n:
            return
        if not msg:
            msg = self.__str__()
        self.logger.info('%s', msg.strip('\r\n\t '))

#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Initialize network weights.

    Args:
        model (torch.nn.Module): network to be initialized
        init_type (string): the name of an initialization method: normal | xavier | xavier_uniform | kaiming | orthogonal | none
        init_gain (float): scaling factor for normal, xavier and orthogonal

    Returns:
        model (torch.nn.Module): initialized model with `init_type` and `init_gain`
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, mean=1.0, std=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Linear') == 0 or classname.find('Conv') == 0):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, mean=0., std=init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'truncnorm':
                torch.nn.init.trunc_normal_(m.weight.data, mean=0., std=init_gain)
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(f'[ERROR] Initialization method {init_type} is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

####################
# Stratified Split #
####################
def stratified_split(raw_dataset, test_size):
    indices_per_label = defaultdict(list)
    for index, label in enumerate(np.array(raw_dataset.dataset.targets)[raw_dataset.indices]):
        indices_per_label[label.item()].append(index)
    
    train_indices, test_indices = [], []
    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * test_size)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        test_indices.extend(random_indices_sample)
        train_indices.extend(set(indices) - set(random_indices_sample))
    return torch.utils.data.Subset(raw_dataset, train_indices), torch.utils.data.Subset(raw_dataset, test_indices)

#####################
# Arguments checker #
#####################
def check_args(args):
    # check device
    if 'cuda' in args.device:
        assert torch.cuda.is_available(), 'Please check if your GPU is available now!' 

    # check optimizer
    if args.optimizer not in torch.optim.__dict__.keys():
        err = f'`{args.optimizer}` is not a submodule of `torch.optim`... please check!'
        logger.exception(err)
        raise AssertionError(err)
    
    # check criterion
    if args.criterion not in torch.nn.__dict__.keys():
        err = f'`{args.criterion}` is not a submodule of `torch.nn`... please check!'
        logger.exception(err)
        raise AssertionError(err)

    # check dataset
    if args.dataset in ['SpeechCommands']:
        args.need_embedding = False
        
    # check algorithm
    if args.algorithm == 'fedsgd':
        args.E = 1
    elif args.algorithm in ['fedavgm', 'fedadam', 'fedyogi', 'fedadagrad']:
        if (args.beta1 < 0) and (args.algorithm in ['fedavgm', 'fedadam', 'fedyogi', 'fedadagrad']):
            err = f'server momentum factor (i.e., `beta1`) should be positive... please check!'
            logger.exception(err)
            raise AssertionError(err)
        if (args.beta2 < 0) and (args.algorithm in ['fedadam', 'fedyogi']):
            err = f'server momentum factor (i.e., `beta1`) should be positive... please check!'
            logger.exception(err)
            raise AssertionError(err)
        
    # check model
    if args.use_pt_model:
        assert args.model_name in ['EfficientNetPT', 'DistilBert', 'SqueezeBert', 'MobileBert', 'Bert']
        
    if args.use_model_tokenizer:
        assert args.model_name in ['DistilBert', 'SqueezeBert', 'MobileBert', 'Bert']

    if args.model_name == 'Sent140LSTM':
        with open(os.path.join(args.data_path, 'sent140', 'vocab', 'glove.6B.300d.json'), 'r') as file:
            emb_weights = torch.tensor(json.load(file))
        args.glove_emb = emb_weights
    else:
        args.glove_emb = None

    # check train only mode
    if args.test_size == 0:
        args.train_only = True
    else:
        args.train_only = False

    # check compatibility of evaluation metrics
    if hasattr(args, 'num_classes'):
        if args.num_classes > 2:
            if ('auprc' or 'youdenj') in args.eval_metrics:
                err = f'some metrics (`auprc`, `youdenj`) are not compatible with multi-class setting... please check!'
                logger.exception(err)
                raise AssertionError(err)
        else:
            if 'acc5' in args.eval_metrics:
                err = f'Top5 accruacy (`acc5`) is not compatible with binary-class setting... please check!'
                logger.exception(err)
                raise AssertionError(err)

        if ('mse' or 'mae' or 'mape' or 'rmse' or 'r2' or 'd2') in args.eval_metrics:
            err = f'selected dataset (`{args.dataset}`) is for a classification task... please check evaluation metrics!'
            logger.exception(err)
            raise AssertionError(err)
    else:
        if ('acc1' or 'acc5' or 'auroc' or 'auprc' or 'youdenj' or 'f1' or 'precision' or 'recall' or 'seqacc') in args.eval_metrics:
            err = f'selected dataset (`{args.dataset}`) is for a regression task... please check evaluation metrics!'
            logger.exception(err)
            raise AssertionError(err)

    # adjust the number of classes in a binary classification task (EXCEPT segementation!)
    if (args.num_classes == 2) and (args.model_name not in ['UNet3D']):
        args.num_classes = 1
        args.criterion = 'BCEWithLogitsLoss'

    # check task
    if args.criterion == 'Seq2SeqLoss':
        args.is_seq2seq = True
    else:
        args.is_seq2seq = False
        
    # print welcome message
    logger.info('[CONFIG] List up configurations...')
    for arg in vars(args):
        if 'glove_emb' in str(arg):
            if getattr(args, arg) is not None:
                logger.info(f'[CONFIG] - {str(arg).upper()}: USE!')
            else:
                logger.info(f'[CONFIG] - {str(arg).upper()}: NOT USE!')
            continue
        logger.info(f'[CONFIG] - {str(arg).upper()}: {getattr(args, arg)}')
    else:
        print('')
    return args

#####################
# BCEWithLogitsLoss #
#####################
class PainlessBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """Native `torch.nn.BCEWithLogitsLoss` requires squeezed logits shape and targets with float dtype.
    """
    def __init__(self, **kwargs):
        super(PainlessBCEWithLogitsLoss, self).__init__(**kwargs)

    def forward(self, inputs, targets):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            torch.atleast_1d(inputs.squeeze()), 
            torch.atleast_1d(targets).float()
        )

torch.nn.BCEWithLogitsLoss = PainlessBCEWithLogitsLoss

################
# Seq2Seq Loss #
################
class Seq2SeqLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super(Seq2SeqLoss, self).__init__(**kwargs)

    def forward(self, inputs, targets, ignore_indices=torch.tensor([0, 1, 2, 3])):
        num_classes = inputs.size(-1)
        inputs, targets = inputs.view(-1, num_classes), targets.view(-1)
        targets[torch.isin(targets, ignore_indices.to(targets.device))] = -1
        if targets.float().mean() == -1.: # if all targets are special tokens
            return inputs.mul(torch.zeros_like(inputs).float()).sum()
        loss = torch.nn.functional.cross_entropy(inputs, targets, ignore_index=-1)
        if loss.isnan():
            return torch.nan_to_num(loss)
        return loss

torch.nn.Seq2SeqLoss = Seq2SeqLoss

#############
# Dice Loss #
#############
class DiceLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def get_dice_score(self, inputs, targets, epsilon=1e-6):
        SPATIAL_DIMENSIONS = 2, 3, 4
        p0 = inputs
        g0 = targets
        p1 = 1 - p0
        g1 = 1 - g0
        tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
        fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
        fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
        num = 2 * tp
        denom = 2 * tp + fp + fn + epsilon
        dice_score = num / denom
        return torch.nan_to_num(dice_score, 0.)

    def forward(self, inputs, targets):
        return torch.mean(1 - self.get_dice_score(inputs, targets))

torch.nn.DiceLoss = DiceLoss

##############
# Focal Loss #
##############
class FocalLoss(torch.nn.modules.loss._Loss):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        if self.alpha is None:
            num_classes = inputs.shape[-1]
            self.alpha = torch.ones(num_classes).div(num_classes).float().to(inputs.device)

        targets = targets.view(-1, 1).type_as(inputs)
        logpt = inputs.log_softmax(dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        
        pt = logpt.exp()
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt.mul_(at)
        loss = logpt.mul((1 - pt).pow(self.gamma)).mul(-1)
        return loss.mean()

torch.nn.FocalLoss = FocalLoss

##################
# Metric manager #
##################
class MetricManager:
    """Managing metrics to be used.
    """
    def __init__(self, eval_metrics):
        self.metric_funcs = {
            name: import_module(f'.metrics', package=__package__).__dict__[name.title()]()
            for name in eval_metrics
            }
        self.figures = defaultdict(int) 
        self._results = dict()

        # use optimal threshold (i.e., Youden's J or not)
        if 'youdenj' in self.metric_funcs:
            for func in self.metric_funcs.values():
                if hasattr(func, '_use_youdenj'):
                    setattr(func, '_use_youdenj', True)

    def track(self, loss, pred, true):
        # update running loss
        self.figures['loss'] += loss * len(pred)

        # update running metrics
        for module in self.metric_funcs.values():
            module.collect(pred, true)

    def aggregate(self, total_len, curr_step=None):
        running_figures = {name: module.summarize() for name, module in self.metric_funcs.items()}
        running_figures['loss'] = self.figures['loss'] / total_len
        if curr_step is not None:
            self._results[curr_step] = {
                'loss': running_figures['loss'], 
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
                }
        else:
            self._results = {
                'loss': running_figures['loss'], 
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
                }
        self.figures = defaultdict(int)

    @property
    def results(self):
        return self._results
