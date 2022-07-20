#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from data_config import (NPM3DConfig, S3DISConfig, Semantic3DConfig,
                         XMap3DConfig)
from models.architectures_test import (KPCNN, KPFCNN, KP_Pyramid_V1,
                                       KP_Pyramid_V2)
from utils.tester_test import ModelTester


def parse_args():
    parser = argparse.ArgumentParser(description='Train kp_pyramid')
    parser.add_argument(
        'model_path',
        help='the path to the log and checkpoint file to load for inference')
    parser.add_argument(
        '--chkp-idx',
        type=int,
        default=500,
        help='the specific checkpoint epoch number for the model')
    parser.add_argument(
        '--network',
        type=str,
        default='KPConv',
        help='the network architecture (kpconv, kpconv_deform, Pyramid_v1, '
        'pyramid_v2, pyramid_v1_deform, pyramid_v2_deform) to use')
    parser.add_argument(
        '--dataset',
        type=str,
        default='S3DIS',
        help='the Dataset (S3DIS, XMap3D, NPM3D, Semantic3D) to use')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--gpu-id',
                        default='3',
                        type=str,
                        help='the gpu to use')
    parser.add_argument('--test-area-idx',
                        default=5,
                        type=int,
                        help='the test area index')
    parser.add_argument('--is-on-val',
                        default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='test on validation or test dataset?')
    parser.add_argument('--num-votes',
                        type=float,
                        default=100,
                        help='number of votes')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--saving-path',
                        default=None,
                        type=str,
                        help='saving path for predictions')

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


args = parse_args()

if args.dataset.lower() == 's3dis':
    from datasets.S3DIS import S3DISDataset as myDataset
    from datasets.S3DIS import S3DISSampler as mySampler
    from datasets.S3DIS import S3DISCollate as myCollate
    myConfig = S3DISConfig
elif args.dataset.lower() == 'npm3d':
    from datasets.NPM3D_test import NPM3DDataset as myDataset
    from datasets.NPM3D_test import NPM3DSampler as mySampler
    from datasets.NPM3D_test import NPM3DCollate as myCollate
    myConfig = NPM3DConfig
elif args.dataset.lower() == 'semantic3d':
    from datasets.Semantic3D_test import Semantic3DDataset as myDataset
    from datasets.Semantic3D_test import Semantic3DSampler as mySampler
    from datasets.Semantic3D_test import Semantic3DCollate as myCollate
    myConfig = Semantic3DConfig
elif args.dataset.lower() == 'xmap3d_kp':
    from datasets.XMap3D_KP import XMap3DDataset as myDataset
    from datasets.XMap3D_KP import XMap3DSampler as mySampler
    from datasets.XMap3D_KP import XMap3DCollate as myCollate
    myConfig = XMap3DConfig
elif args.dataset.lower() == 'xmap3d':
    from datasets.XMap3D import XMap3DDataset as myDataset
    from datasets.XMap3D import XMap3DSampler as mySampler
    from datasets.XMap3D import XMap3DCollate as myCollate
    myConfig = XMap3DConfig
else:
    print('This Dataset has not been implemented')

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    chosen_log = args.model_path

    # Choose the index of the checkpoint to load OR None if you want to load
    # the current checkpoint
    chkp_idx = args.chkp_idx

    chosen_chkp = 'chkp_%04d.tar' % chkp_idx
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)
    print('used chkp', chosen_chkp)
    config = myConfig(args.network)
    config.load(chosen_log)
    config.saving_path = args.saving_path
    config.saving = False if config.saving_path is None else True
    config.validation_size = 200
    config.input_threads = 10

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if args.is_on_val:
        set = 'validation'
    else:
        set = 'test'

    if config.dataset == 'NPM3D':
        test_dataset = myDataset(config,
                                 set=set,
                                 use_potentials=True,
                                 on_val=args.is_on_val)
    elif config.dataset == 'S3DIS':
        test_dataset = myDataset(config,
                                 set=set,
                                 use_potentials=True,
                                 validation_split=args.test_area_idx)
    else:
        test_dataset = myDataset(config, set=set, use_potentials=True)

    test_sampler = mySampler(test_dataset)
    collate_fn = myCollate

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    if config.dataset_task == 'classification':
        net = KPCNN(config)
    elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        if args.network.lower() == 'pyramid_v1' or args.network.lower(
        ) == 'pyramid_v1_deform':  # vertical
            net = KP_Pyramid_V1(config,
                                test_dataset.label_values,
                                test_dataset.ignored_labels,
                                use_multi_layer=config.use_multi_layer)
        elif args.network.lower() == 'pyramid_v2' or args.network.lower(
        ) == 'pyramid_v2_deform':  # horizontal
            net = KP_Pyramid_V2(config,
                                test_dataset.label_values,
                                test_dataset.ignored_labels,
                                use_multi_layer=config.use_multi_layer)
        else:
            net = KPFCNN(config, test_dataset.label_values,
                         test_dataset.ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' +
                         config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Training
    if config.dataset_task == 'classification':
        a = 1 / 0
    elif config.dataset_task == 'cloud_segmentation':
        with torch.no_grad():
            tester.cloud_segmentation_test(net,
                                           test_loader,
                                           config,
                                           num_votes=args.num_votes)
    elif config.dataset_task == 'slam_segmentation':
        tester.slam_segmentation_test(net, test_loader, config)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' +
                         config.dataset_task)
    print('All test finished in {:.1f}s\n for num_votes {}'.format(
        time.time() - t1, args.num_votes))
