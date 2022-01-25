import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        # print(module)
        module.eval()

def train_and_test(args, trainloader, testloader, model, writer, num_class):
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam([{'params': model.backbone.parameters(), 'lr': args.lr_finetune},
                                  {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]}],
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
    # model = torch.nn.DataParallel(model).to(device)
    # criterion = nn.CrossEntropyLoss()
    loss_fn = edge_loss(num_class)

    total_step = 0
    best_test_result = {'acc': 0, 'recall': 0, 'mAP': 0, 'epoch': -1}
    for epoch in range(args.epochs):
        # print("epoch {}".format(epoch))
        model.train()
        model.apply(set_bn_eval)
        true_labels, pred_labels, confs = [], [], []
        for batch in tqdm(trainloader):
            total_step += 1
            model_input = {k: v.to(device) for k, v in batch.items()}

            logits = model(model_input)
            loss, score, label = loss_fn(logits, model_input['relations_id'], model_input['relation_mask'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        writer.add_scalar('train/acc', acc, epoch)
        writer.add_scalar('train/mAP', mAP, epoch)
        writer.add_scalar('train/lr_group_0', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train/lr_group_1', optimizer.param_groups[1]['lr'], epoch)
        logger.info("Train epoch {} mAP {:.4f} acc {:.4f} recall {}".format(epoch, mAP, acc, recalls))

        scheduler.step()

        conf_matrix = confusion_matrix(true_labels, pred_labels)
        logger.info('Confusion Matrix')
        logger.info(conf_matrix)

        # start_testing
        model.eval()
        true_labels, pred_labels, confs = [], [], []
        with torch.no_grad():
            for batch in tqdm(testloader):
                model_input = {k: v.to(device) for k, v in batch.items()}
                logits = model(model_input)

                loss, score, label = loss_fn(logits, model_input['relations_id'], model_input['relation_mask'])

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
        writer.add_scalar('test/acc', acc, epoch)
        writer.add_scalar('test/mAP', mAP, epoch)
        logger.info("Test epoch {} mAP {:.4f} acc {:.4f} recall {}".format(epoch, mAP, acc, recalls))
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        logger.info('Confusion Matrix')
        logger.info(conf_matrix)

        if mAP >= best_test_result['mAP']:
            best_test_result['mAP'] = mAP
            best_test_result['acc'] = acc
            best_test_result['recall'] = recalls
            best_test_result['epoch'] = epoch
            save_model(model, mAP, args, logger)

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

    file_handler = logging.FileHandler(args.output_dir+"/log_{}_{}.txt".format(args.dataset, args.manualSeed))
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s|%(filename)s[%(lineno)d]|%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(args)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # prepare dataset
    train_dataset = TransDataset(args, args.dataset, 'train')
    test_dataset = TransDataset(args, args.dataset, 'test')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(args.manualSeed))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=np.random.seed(args.manualSeed))

    # construct model
    model = SRModel(args, test_dataset.num_classes, test_dataset.max_person)

    total_param, backbone_param, trans_param = 0, 0, 0
    for param in model.parameters():
        if param.requires_grad:
            total_param += np.prod(list(param.data.size()))
    for param in model.backbone.parameters():
        if param.requires_grad:
            backbone_param += np.prod(list(param.data.size()))
    # for param in model.transformer.parameters():
    #     if param.requires_grad:
    #         trans_param += np.prod(list(param.data.size()))
    logger.info("Model total parameters in Model is {}, backbone param {}, transformer param {}".format(total_param, backbone_param, trans_param))
    model.to(device)
    # model = torch.nn.DataParallel(model).to(device)

    # training
    logger.info("Start training ...")
    logger.info(model)
    train_and_test(args, train_dataloader, test_dataloader, model, writer, test_dataset.num_classes)

