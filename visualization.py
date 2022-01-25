import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image,  ImageDraw
import cv2

from dataset import TransDataset
# from model.TransformerModel import TransformerModel as SRModel
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

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

FEATUREMAP_SIZE = 7

def visualization(dataset, loader, model, num_samples=100, visul_path='./visualization/SRTransformer'):
    model = model.to(device)
    model.eval()
    max_person = dataset.max_person  # max number of person

    # start_testing
    # invalid_imgs = np.load("./seg_output/pisc_fine_test_v2/invalid_image.npy")
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            model_input = {k: v.to(device) for k, v in batch.items()}
            image_name = dataset.image_names[model_input['img_index']]
            bbox_idx = [[i, j] for i in range(max_person) for j in range(max_person) if
                        model_input['relation_mask'][0, i, j] == 1]
            # if len(bbox_idx) not in [3,4]:
            #     continue
            # if image_name not in ['05642.jpg', '00072.jpg', '00174.jpg', '01591.jpg', '72157633387947248_8694900935.jpg', '72157631721749974_8066831089.jpg']:
            #     continue
            # else:
            #     print(image_name)
            conv_features, enc_attn_weights, dec_attn_weights = [], [], []
            hooks = [
                # model.resnet.layer4.register_forward_hook(
                #     lambda self, input, output: conv_features.append(output)
                # ),
                # model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                #     lambda self, input, output: enc_attn_weights.append(output[1])
                # ),
                model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                    lambda self, input, output: dec_attn_weights.append(output[1])
                ),
            ]

            model_input = {k: v.to(device) for k, v in batch.items()}
            logits = model(model_input)

            trues = model_input['relations_id'].data.cpu().long().numpy()
            logits_np = F.softmax(logits, dim=-1).data.cpu().numpy()
            preds = np.argmax(logits_np, axis=-1).reshape(-1, max_person, max_person)
            # confs.append(logits_np)

            for hook in hooks:
                hook.remove()

            # don't need the list anymore
            # conv_features = conv_features[0]  # [1, 2048, 14, 14]
            # enc_attn_weights = enc_attn_weights[0]
            dec_attn_weights = dec_attn_weights[0]  # [1, max_person*max_person, 196]

            atten = dec_attn_weights[0].view(-1, max_person, max_person, FEATUREMAP_SIZE*FEATUREMAP_SIZE)

            image_name = dataset.image_names[model_input['img_index']]
            raw_image = Image.open(os.path.join(dataset.images_root, image_name)).convert('RGB')
            rgb_img = np.array(raw_image) / 255
            raw_bbox = dataset.image_bboxes[model_input['img_index']]
            bbox_idx = [[i, j] for i in range(max_person) for j in range(max_person) if
                        model_input['relation_mask'][0, i, j] == 1]
            if not os.path.exists(visul_path):
                os.mkdir(visul_path)
            for idxs in bbox_idx:
                # atten_max = torch.max(atten[0, 0, 1], atten[0, 0, 2])
                # atten_raw = atten_max - atten[0, 0, 2]
                # atten_raw = (atten_raw/atten_raw.max()).view(FEATUREMAP_SIZE, FEATUREMAP_SIZE).cpu().detach().numpy()
                atten_raw = atten[0, idxs[0], idxs[1]].view(FEATUREMAP_SIZE, FEATUREMAP_SIZE).cpu().detach().numpy()
                # normalize the heatmap
                # atten_img = atten_img - np.mean(atten_img)
                # atten_img = atten_img / (np.std(atten_img) + 1e-5)
                atten_img = deprocess_image(atten_raw)
                atten_img = cv2.resize(atten_img, rgb_img.shape[:2][::-1])

                atten_img = atten_img - np.min(atten_img)
                atten_img = atten_img / (np.max(atten_img) + 1e-6)
                atten_img = np.float32(atten_img)

                visualization = show_cam_on_image(rgb_img, atten_img, use_rgb=True)
                # cv2.imwrite('{}_cam.jpg'.format(image_name.split('.')[0]), visualization)
                new_img = Image.fromarray(visualization)
                draw = ImageDraw.Draw(new_img)
                draw.rectangle(raw_bbox[idxs[0]], outline='white', width=2)
                draw.rectangle(raw_bbox[idxs[1]], outline='white', width=2)
                true_label = trues[0, idxs[0], idxs[1]]
                pred_label = preds[0, idxs[0], idxs[1]]
                # if true_label != pred_label:
                # draw.text(raw_bbox[idxs[0]][:2], 'True: {}'.format(true_label), (255, 255, 255))
                # draw.text(raw_bbox[idxs[1]][:2], 'Pred: {}'.format(pred_label), (255, 255, 255))
                new_img.save(visul_path + '/{}_{}_{}_true_{}_pred_{}.jpg'.format(image_name.split('.')[0], idxs[0], idxs[1], true_label, pred_label))

                # atten_img = Image.fromarray(np.uint8(255 * atten_raw))
                # atten_img.save(visul_path + '/{}_{}_{}_atten_raw.jpg'.format(image_name.split('.')[0], idxs[0], idxs[1]))

            count += 1
            if count > num_samples:
                break

    # true_labels = np.concatenate(true_labels).reshape(-1)
    # pred_labels = np.concatenate(pred_labels).reshape(-1)
    # confs = np.concatenate(confs)
    #
    # acc = np.mean(pred_labels == true_labels)
    # recalls = recall_score(true_labels, pred_labels, average=None)
    # mAP = compute_map(confs, true_labels)
    # logger.info("Test mAP {:.4f} acc {:.4f} recall {}".format(mAP, acc, recalls))
    # conf_matrix = confusion_matrix(true_labels, pred_labels)
    # logger.info('Confusion Matrix')
    # logger.info(conf_matrix)


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
        default=1,
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # prepare dataset
    train_dataset = TransDataset(args, args.dataset, 'train')
    test_dataset = TransDataset(args, args.dataset, 'test')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=np.random.seed(args.manualSeed))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=np.random.seed(args.manualSeed))

    # construct model
    model = SRModel(args, test_dataset.num_classes, test_dataset.max_person)
    model = load_model(args, model, logger)

    # model.to(device)
    # model = torch.nn.DataParallel(model).to(device)

    if not os.path.exists('./visualization/{}_{}_SRTransformer'.format(args.dataset, args.img_size)):
        os.mkdir('./visualization/{}_{}_SRTransformer'.format(args.dataset, args.img_size))
    # training
    test_visul_path = './visualization/{}_{}_SRTransformer/test'.format(args.dataset, args.img_size)
    visualization(test_dataset, test_dataloader, model, num_samples=len(test_dataloader), visul_path=test_visul_path)
    train_visul_path = './visualization/{}_{}_SRTransformer/train'.format(args.dataset, args.img_size)
    visualization(train_dataset, train_dataloader, model, num_samples=len(train_dataloader), visul_path=train_visul_path)

