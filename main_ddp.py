import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import torch.distributed as dist

from dataset import TransDataset
from model.GRIT import GRIT as SRModel
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

from sklearn.metrics import recall_score, confusion_matrix
from utils.eval_utils import compute_map, save_model, edge_loss
from utils.logger import logger
from utils.args import build_args
import logging

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def build_scheduler(optimizer, args):
    scheduler = create_scheduler(args, optimizer)
    return scheduler


def build_optimizer(args, model):
    opt = create_optimizer(args, model)
    return opt


def tensor_all_gather(tensor):
    all_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensor, tensor)
    all_tensor = torch.cat(all_tensor, dim=0)

    return all_tensor


def varsize_tensor_all_gather(tensor: torch.Tensor):
    tensor = tensor.contiguous()

    cuda_device = f'cuda:{dist.get_rank()}'
    size_tens = torch.tensor(
        [tensor.shape[0]], dtype=torch.int64, device=cuda_device)

    size_tens = tensor_all_gather(size_tens).cpu()

    max_size = size_tens.max()

    padded = torch.empty(max_size, *tensor.shape[1:],
                         dtype=tensor.dtype,
                         device=cuda_device)
    padded[:tensor.shape[0]] = tensor

    ag = tensor_all_gather(padded)

    slices = []
    for i, sz in enumerate(size_tens):
        start_idx = i * max_size
        end_idx = start_idx + sz.item()

        if end_idx > start_idx:
            slices.append(ag[start_idx:end_idx])

    ret = torch.cat(slices, dim=0)

    return ret.to(tensor)


def train_and_test_ddp(args, model, trainloader, testloader, train_sampler, num_class, local_rank, logger):
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    device = f"cuda:{local_rank}"
    # optimizer = build_optimizer(args, model)
    # scheduler, _ = build_scheduler(optimizer, args)  # nepochs
    optimizer = torch.optim.Adam([{'params': model.backbone.parameters(), 'lr': args.lr_finetune},
                                  {'params': [p for n, p in model.named_parameters() if
                                              'backbone' not in n and p.requires_grad]}],
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)

    # model = torch.nn.DataParallel(model).to(device)
    # criterion = nn.CrossEntropyLoss()

    # TODO: used in DDP mode
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    loss_fn = edge_loss(num_class)

    total_step = 0
    best_test_result = {'acc': 0, 'recall': 0, 'mAP': 0, 'epoch': -1}

    for epoch in range(args.epochs):
        if local_rank == 0:
            logger.info("==> running epoch {}".format(epoch))
        scheduler.step()
        train_sampler.set_epoch(epoch)

        model.train()
        true_labels, pred_labels, confs = [], [], []
        for batch_i, batch in tqdm(enumerate(trainloader), total=len(trainloader)):
            total_step += 1
            model_input = {k: v.to(device) for k, v in batch.items()}

            logits = model(model_input)
            loss, score, label = loss_fn(
                logits, model_input['relations_id'], model_input['relation_mask'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if local_rank == 0:
                writer.add_scalar('train/loss', loss.cpu().detach().numpy(), total_step)

            true_labels.append(label.data.cpu().long().numpy())
            logits_np = F.softmax(score, dim=-1).data.cpu().numpy()
            pred_labels.append(np.argmax(logits_np, axis=-1))
            confs.append(logits_np)

        true_labels = np.concatenate(true_labels).reshape(-1)
        pred_labels = np.concatenate(pred_labels).reshape(-1)
        confs = np.concatenate(confs)

        acc = np.mean(pred_labels == true_labels)
        recalls = recall_score(true_labels, pred_labels, average=None)
        mAP = compute_map(confs, true_labels)
        if local_rank == 0:
            writer.add_scalar('train/acc', acc, epoch)
            writer.add_scalar('train/mAP', mAP, epoch)
            writer.add_scalar('train/lr', optimizer.param_groups[-1]['lr'], epoch)
            writer.add_scalar('train/lr_group_0', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('train/lr_group_1', optimizer.param_groups[-1]['lr'], epoch)
            logger.info("Train epoch {} mAP {:.4f} acc {:.4f} recall {}".format(epoch, mAP, acc, recalls))
            # logger.info("Train epoch {} mAP {:.4f} acc {:.4f}".format(epoch, mAP, acc))

        # start_testing
        true_labels, pred_labels, confs = [], [], []

        model.eval()
        with torch.no_grad():
            for batch_i, batch in tqdm(enumerate(testloader), total=len(testloader)):
                model_input = {k: v.to(device) for k, v in batch.items()}
                logits = model(model_input)
                loss, score, label = loss_fn(
                    logits, model_input['relations_id'], model_input['relation_mask'])

                true_labels.append(label.data.cpu().long())
                prob = F.softmax(score, dim=1).data.cpu()
                pred_labels.append(torch.argmax(score, dim=1))
                confs.append(prob)

        # true_labels = np.concatenate(true_labels).reshape(-1)
        # pred_labels = np.concatenate(pred_labels).reshape(-1)
        # confs = np.concatenate(confs)
        true_labels = torch.cat(true_labels, dim=0)
        pred_labels = torch.cat(pred_labels, dim=0)
        confs = torch.cat(confs, dim=0)

        total_labels = varsize_tensor_all_gather(true_labels)
        total_pred_labels = varsize_tensor_all_gather(pred_labels)
        total_confs = varsize_tensor_all_gather(confs)
        # print(total_labels.shape, total_pred_labels.shape, total_confs.shape)
        # raise

        if local_rank == 0:
            true_labels = total_labels.cpu().numpy()
            pred_labels = total_pred_labels.cpu().numpy()
            confs = total_confs.cpu().numpy()

            acc = np.mean(pred_labels == true_labels)
            recalls = recall_score(true_labels, pred_labels, average=None)
            mAP = compute_map(confs, true_labels)
            writer.add_scalar('test/acc', acc, epoch)
            writer.add_scalar('test/mAP', mAP, epoch)
            # logger.info("Test epoch {} mAP {:.4f} acc {:.4f} recall {} lr {}".format(epoch, mAP, acc, recalls,
            logger.info(" ==> Test epoch {}: mAP {:.4f}  |  acc {:.4f}  |  lr {}".format(epoch, mAP, acc,
                                                                                         optimizer.param_groups[0]['lr']))

            if mAP >= best_test_result['mAP']:
                best_test_result['mAP'] = mAP
                best_test_result['acc'] = acc
                best_test_result['recall'] = recalls
                best_test_result['epoch'] = epoch
                save_model(model, mAP, args, logger)

    if local_rank == 0:
        logger.info("Dataset {} Best-test-result: epoch {} mAP {:.4f} acc {:.4f} recall {}".format(args.dataset,
                                                                                                   best_test_result['epoch'],
                                                                                                   best_test_result['mAP'],
                                                                                                   best_test_result['acc'],
                                                                                                   best_test_result['recall']))


if __name__ == '__main__':

    args = build_args()

    if args.manualSeed is None or args.manualSeed < 0:
        args.manualSeed = random.randint(1, 10000)
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    setup_seed(args.manualSeed)

    # args.output_dir = args.output_dir + "/{}/".format(args.dataset)  # datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir + "/{}/".format(args.dataset))

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        file_handler = logging.FileHandler(args.output_dir+"/log_{}_{}.txt".format(args.dataset, args.manualSeed))
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s|%(filename)s[%(lineno)d]|%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # set up dist env first
    # logger.info("local rank: {}".format(local_rank))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    if local_rank == 0:
        logger.info(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # prepare dataset
    train_dataset = TransDataset(args, args.dataset, 'train')
    test_dataset = TransDataset(args, args.dataset, 'test')
    train_sampler = None
    test_smapler = None
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                                   shuffle=(train_sampler is None))

    test_sampler = torch.utils.data.DistributedSampler(test_dataset, shuffle=False)  # shuffle false
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, sampler=test_sampler,
                                                  shuffle=(test_sampler is None))

    # construct model
    model = SRModel(args, test_dataset.num_classes, test_dataset.max_person)

    if local_rank == 0:
        logger.info("Start training ...")
        # logger.info(model)

    train_and_test_ddp(
        args,
        model,
        train_dataloader,
        test_dataloader,
        train_sampler,
        test_dataset.num_classes,
        local_rank,
        logger
    )


