import torch
import logging
import pdb
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from config import cfg
from data import make_data_loader
from solver import build_optimizer, build_scheduler

from utils.check_point import DetectronCheckpointer
from engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from utils import comm
from utils.backup_files import sync_root

from engine.trainer import do_train
from engine.test_net import run_test

from model.detector import KeypointDetector
from data import build_test_loader
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.backends.cudnn.enabled = True # enable cudnn and uncertainty imported
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True # enable cudnn to search the best algorithm

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    para_size = (param_size) / 1024 / 1024
    print('模型参数大小为：{:.3f}MB'.format(para_size))
    buffer_size = 0
    buffer_sum = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

def train(cfg, model, device, distributed):
    data_loader = make_data_loader(cfg, is_train=True)  # 加载数据
    data_loaders_val = build_test_loader(cfg, is_train=False)

    total_iters_each_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
    # use epoch rather than iterations for saving checkpoint and validation
    if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
        cfg.SOLVER.MAX_ITERATION = cfg.SOLVER.MAX_EPOCHS * total_iters_each_epoch
        cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL = total_iters_each_epoch * cfg.SOLVER.SAVE_CHECKPOINT_EPOCH_INTERVAL
        cfg.SOLVER.EVAL_INTERVAL = total_iters_each_epoch * cfg.SOLVER.EVAL_EPOCH_INTERVAL
        cfg.SOLVER.STEPS = [total_iters_each_epoch * x for x in cfg.SOLVER.DECAY_EPOCH_STEPS]
        cfg.SOLVER.WARMUP_STEPS = cfg.SOLVER.WARMUP_EPOCH * total_iters_each_epoch
    
    cfg.freeze()

    optimizer = build_optimizer(model, cfg) # 选择优化器
    scheduler, warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=total_iters_each_epoch, 
        optim_cfg=cfg.SOLVER,
    ) 

    arguments = {}
    arguments["iteration"] = 0
    arguments["iter_per_epoch"] = total_iters_each_epoch

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = comm.get_rank() == 0

    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    if len(cfg.MODEL.WEIGHT) > 0:
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, use_latest=False)
        arguments.update(extra_checkpoint_data)

    # 主训练函数入口
    do_train(
        cfg,
        distributed,
        model,
        data_loader,
        data_loaders_val,
        optimizer,
        scheduler,
        warmup_scheduler,
        checkpointer,
        device,
        arguments,
    )

def setup(args):
    cfg.merge_from_file(args.config_file) # args.config_file ='runs/baseline_v0.yaml'
    cfg.merge_from_list(args.opts)
    
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_work
    cfg.TEST.EVAL_DIS_IOUS = args.eval_iou
    cfg.TEST.EVAL_DEPTH = args.eval_depth 
    
    if args.vis_thre > 0:
        cfg.TEST.VISUALIZE_THRESHOLD = args.vis_thre 
    
    if args.output is not None:
        cfg.OUTPUT_DIR = args.output

    if args.test:
        cfg.DATASETS.TEST_SPLIT = 'test'
        cfg.DATASETS.TEST = ("kitti_test",)

    cfg.START_TIME = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%H-%M-%S')

    default_setup(cfg, args)

    return cfg

def main(args):
    cfg = setup(args)

    distributed = comm.get_world_size() > 1
    
    if not distributed: cfg.MODEL.USE_SYNC_BN = False

    model = KeypointDetector(cfg) # 创建模型
    device = torch.device(cfg.MODEL.DEVICE) # cpu or gpu
    print('model parameters')
    getModelSize(model)
    model.to(device)

    if args.eval_only:
        checkpointer = DetectronCheckpointer(
            cfg, model, save_dir=cfg.OUTPUT_DIR
        )
        ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
        _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

        return run_test(cfg, checkpointer.model, vis=args.vis, eval_score_iou=args.eval_score_iou, eval_all_depths=args.eval_all_depths, test_flag=args.test)

    if distributed:
        # convert BN to SyncBN
        if cfg.MODEL.USE_SYNC_BN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True,
        )
    
    if args.ckpt:
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        ckpt = args.ckpt
        _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
		
        train(cfg, checkpointer.model, device, distributed)
    else:
        train(cfg, model, device, distributed)

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    
    print("Command Line Args:", args)
    
    # backup all python files when training
    if not args.eval_only and args.output is not None:
        sync_root('.', os.path.join(args.output, 'backup'))
        import shutil
        shutil.copy2(args.config_file, os.path.join(args.output, 'backup', os.path.basename(args.config_file)))

        print("Finish backup all files")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    ) # 启动分布式
