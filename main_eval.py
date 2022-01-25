import numpy as np
import argparse
import torch
import torch.nn.functional as F

from dataset import TransDataset
from model.GRIT import GRIT as SRModel
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

from sklearn.metrics import recall_score, confusion_matrix
from utils.eval_utils import compute_map, load_model, edge_loss
from utils.logger import logger
import logging


def test(args, trainloader, testloader, model, num_class):
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    # criterion = nn.CrossEntropyLoss()
    loss_fn = edge_loss(num_class)

    # total_step = 0
    # # print("epoch {}".format(epoch))
    # true_labels, pred_labels, confs = [], [], []
    # with torch.no_grad():
    #     for batch in tqdm(trainloader):
    #         total_step += 1
    #         model_input = {k: v.to(device) for k, v in batch.items()}
    #
    #         logits = model(model_input)
    #         loss, score, label = loss_fn(logits, model_input['relations_id'], model_input['relation_mask'])
    #
    #         true_labels.append(label.data.cpu().long().numpy())
    #         logits_np = F.softmax(score, dim=-1).data.cpu().numpy()
    #         pred_labels.append(np.argmax(logits_np, axis=-1))
    #         confs.append(logits_np)
    #
    # true_labels = np.concatenate(true_labels).reshape(-1)
    # pred_labels = np.concatenate(pred_labels).reshape(-1)
    # confs = np.concatenate(confs)
    #
    # acc = np.mean(pred_labels == true_labels)
    # recalls = recall_score(true_labels, pred_labels, average=None)
    # mAP = compute_map(confs, true_labels)
    # logger.info("Train mAP {:.4f} acc {:.4f} recall {}".format(mAP, acc, recalls))
    # conf_matrix = confusion_matrix(true_labels, pred_labels)
    # logger.info('Confusion Matrix')
    # logger.info(conf_matrix)

    # start_testing
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
    logger.info("Test mAP {:.4f} acc {:.4f} recall {}".format(mAP, acc, recalls))
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    logger.info('Confusion Matrix')
    logger.info(conf_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer for social relation inference")
    parser.add_argument(
        "--dataset",
        default="pisc_fine",
        type=str,
        help="name of dataset: pisc_fine pisc_coarse pipa_fine pipa_coarse",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='Transformer_output/exp_resnet101_roi/',
    )
    parser.add_argument('--manualSeed', type=int, default=-1, help='manual seed')
    parser.add_argument('--use_gcn', action='store_true', default=False, help='use gcn')

    parser.add_argument('--gcn_hidden_dim', type=int, default=1024, help='hidden dimension')
    parser.add_argument('--img_size', type=int, default=224, help='hidden dimension')
    parser.add_argument('--backbone', type=str, default='swin_transformer',
                        choices=('swin_transformer', 'resnet101', 'pvt', 'twins'))
    # transformer settings
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--dropout', type=int, default=0.1, help='dropout')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nheads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='dimension of FFN')
    parser.add_argument('--enc_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=3, help='number of decoder layers')
    parser.add_argument('--pre_norm', action='store_true', default=True, help='layer norm before of after')
    parser.add_argument('--concate_after', action='store_true', default=False,
                        help='concate bbox feature before or after transformer')
    parser.add_argument('--raw_bbox', action='store_true', default=False, help='use raw bbox as input')
    parser.add_argument('--remove_transformer', action='store_true', default=False, help='gcn only')
    parser.add_argument('--remove_gcn', action='store_true', default=False, help='transformer only')
    args = parser.parse_args()

    if args.manualSeed is None or args.manualSeed < 0:
        args.manualSeed = random.randint(1, 10000)
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    setup_seed(args.manualSeed)

    file_handler = logging.FileHandler("Transformer_output/log_test_{}.txt".format(args.dataset))
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s|%(filename)s[%(lineno)d]|%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # prepare dataset
    train_dataset = TransDataset(args, args.dataset, 'train')
    test_dataset = TransDataset(args, args.dataset, 'test')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(args.manualSeed))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=np.random.seed(args.manualSeed))

    # construct model
    model = SRModel(args, test_dataset.num_classes, test_dataset.max_person)
    model = load_model(args, model, logger)

    # model.to(device)
    # model = torch.nn.DataParallel(model).to(device)

    # training
    logger.info("Start training ...")
    logger.info(model)
    test(args, train_dataloader, test_dataloader, model, test_dataset.num_classes)

