import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import argparse
from train_multiview import train_multiview
from test_3DVAEGAN_MULTIVIEW import test_3DVAEGAN_MULTIVIEW

import torch

def main(args):
    if args.test == False:          
        train_multiview(args)
    else:       
        test_3DVAEGAN_MULTIVIEW(args)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Parmeters
    parser.add_argument('--n_epochs', type=float, default=100,
                        help='max epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='each batch size')
    parser.add_argument('--g_lr', type=float, default=0.0025,
                        help='generator learning rate')
    parser.add_argument('--e_lr', type=float, default=1e-4,
                        help='encoder learning rate')
    parser.add_argument('--d_lr', type=float, default=0.001,
                        help='discriminator learning rate')
    parser.add_argument('--beta', type=tuple, default=(0.5, 0.5),
                        help='beta for adam')
    parser.add_argument('--d_thresh', type=float, default=0.8,
                        help='for balance dsicriminator and generator')
    parser.add_argument('--z_size', type=float, default=200,
                        help='latent space size')
    parser.add_argument('--z_dis', type=str, default="norm", choices=["norm", "uni"],
                        help='uniform: uni, normal: norm')
    parser.add_argument('--bias', type=str2bool, default=False,
                        help='using cnn bias')
    parser.add_argument('--leak_value', type=float, default=0.2,
                        help='leakeay relu')
    parser.add_argument('--cube_len', type=float, default=32,
                        help='cube length')
    parser.add_argument('--image_size', type=float, default=137,
                        help='cube length')
    parser.add_argument('--obj', type=str, default="airplane",
                        help='tranining dataset object category')
    parser.add_argument('--soft_label', type=str2bool, default=True,
                        help='using soft_label')    

    # dir parameters
    parser.add_argument('--output_dir', type=str, default="output",
                        help='output path')
    parser.add_argument('--input_dir', type=str, default='input',
                        help='input path')
    parser.add_argument('--pickle_dir', type=str, default='/pickle/',
                        help='input path')
    parser.add_argument('--log_dir', type=str, default='/log/',
                        help='for tensorboard log path save in output_dir + log_dir')
    parser.add_argument('--image_dir', type=str, default='/image/',
                        help='for output image path save in output_dir + image_dir')
    parser.add_argument('--data_dir', type=str, default='/airplane/',
                        help='dataset load path')

    # step parameter
    parser.add_argument('--pickle_step', type=int, default=10,
                        help='pickle save at pickle_step epoch')
    parser.add_argument('--log_step', type=int, default=1,
                        help='tensorboard log save at log_step epoch')

    parser.add_argument('--combine_type', type=str, default='mean',
                        help='for test')
    parser.add_argument('--num_views', type=int, default=1,
                        help='for test')

    parser.add_argument('--model_name', type=str, default="airplane_MV3DVAEGAN",
                        help='this model name for save pickle, logs, output image path and if model_name contain V2 modelV2 excute')
    parser.add_argument('--use_tensorboard', type=str2bool, default=True,
                        help='using tensorboard logging')    
    parser.add_argument('--test', type=str2bool, default=False,
                        help='for test')
    
    # Reconstruction Loss Selection
    parser.add_argument('--recon_loss', type=str, default='mse', 
                    choices=['mse', 'bce', 'focal', 'dice', 'iou', 'awif'],
                    help='reconstruction loss type: mse, bce, focal, dice, iou, awif (our innovative loss)')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='gamma parameter for focal loss')
    parser.add_argument('--dice_smooth', type=float, default=1e-6,
                        help='smoothing parameter for dice loss')

    args = parser.parse_args()
    main(args)

