import argparse


def build_args():
    parser = argparse.ArgumentParser(description="Transformer for social relation inference")

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument("--dataset", default="pisc_fine", type=str,
                        help="name of dataset: pisc_fine pisc_coarse pipa_fine pipa_coarse")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate",)
    parser.add_argument("--lr_finetune", type=float, default=1e-5, help="learning rate for finetune")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--output_dir", type=str, default='Transformer_output')
    parser.add_argument('--manualSeed', type=int, default=-1, help='manual seed')

    parser.add_argument('--img_size', type=int, default=224, help='hidden dimension')
    parser.add_argument('--backbone', type=str, default='swin_transformer', choices=('swin_transformer', 'resnet101', 'pvt', 'twins'))
    # transformer settings
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--nheads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='dimension of FFN')
    parser.add_argument('--enc_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=3, help='number of decoder layers')
    parser.add_argument('--pre_norm', action='store_true', default=True, help='layer norm before of after')

    parser.add_argument('--remove_transformer', action='store_true', default=False, help='gcn only')
    parser.add_argument('--remove_gcn', action='store_true', default=False, help='transformer only')

    args = parser.parse_args()
    return args