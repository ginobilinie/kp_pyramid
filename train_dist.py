#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on S3DIS dataset
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
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_config import (NPM3DConfig, S3DISConfig, Semantic3DConfig,
                         XMap3DConfig)
from models.arch_dist import KPFCNN, KP_Pyramid_V1, KP_Pyramid_V2
from utils.dist_dataloader_wrapper import DistributedSamplerWrapper

# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


def parse_args():
    parser = argparse.ArgumentParser(description='Train randla_pyramid')
    parser.add_argument(
        '--network',
        type=str,
        default='KPConv',
        help='the network architecture (kpconv, kpconv_deform, Pyramid_v1,'
        ' pyramid_v2, pyramid_v1_deform, pyramid_v2_deform) to use')
    parser.add_argument('--dataset',
                        type=str,
                        default='S3DIS',
                        help='the Dataset (S3DIS, XMap3D, NPM3D, '
                        'Semantic3D, SensatUrban) to use')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--gpu-list',
                        default='0,1,2,3',
                        type=str,
                        help='the list of gpus to use')
    parser.add_argument('--resume-from',
                        type=str,
                        help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--test-area-idx',
                        default=5,
                        type=int,
                        help='the test area index')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument(
        '--log-dir',
        default=None,
        type=str,
        help='saving filename for detected boxes in text and json format')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


args = parse_args()

# Dataset
if args.dataset.lower() == 's3dis':
    from datasets.S3DIS import S3DISDataset as myDataset
    from datasets.S3DIS import S3DISSampler as mySampler
    from datasets.S3DIS import S3DISCollate as myCollate
    from utils.trainer_dist import ModelTrainer
    myConfig = S3DISConfig
elif args.dataset.lower() == 'npm3d':
    from datasets.NPM3D import NPM3DDataset as myDataset
    from datasets.NPM3D import NPM3DSampler as mySampler
    from datasets.NPM3D import NPM3DCollate as myCollate
    from utils.trainer_dist import ModelTrainer
    myConfig = NPM3DConfig
elif args.dataset.lower() == 'semantic3d':
    from datasets.Semantic3D import Semantic3DDataset as myDataset
    from datasets.Semantic3D import Semantic3DSampler as mySampler
    from datasets.Semantic3D import Semantic3DCollate as myCollate
    from utils.trainer_dist import ModelTrainer
    myConfig = Semantic3DConfig
elif args.dataset.lower() == 'xmap3d_kp':
    from datasets.XMap3D_KP import XMap3DDataset as myDataset
    from datasets.XMap3D_KP import XMap3DSampler as mySampler
    from datasets.XMap3D_KP import XMap3DCollate as myCollate
    from utils.trainer_dist_xmap3d import ModelTrainer
    myConfig = XMap3DConfig
elif args.dataset.lower() == 'xmap3d':
    from datasets.XMap3D import XMap3DDataset as myDataset
    from datasets.XMap3D import XMap3DSampler as mySampler
    from datasets.XMap3D import XMap3DCollate as myCollate
    from utils.trainer_dist_xmap3d import ModelTrainer
    myConfig = XMap3DConfig
else:
    print('This Dataset has not been implemented')

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ############################
    # Initialize the environment
    ############################

    # Initialize configuration class
    config = myConfig(args.network.lower())

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    config.world_size = len(args.gpu_list.split(','))
    config.learning_rate = config.base_learning_rate / config.world_size * config.world_size
    chkp_idx = None
    if args.resume_from:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join(args.resume_from, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join(args.resume_from, 'checkpoints',
                                   chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # config.architecture = get_architectures(args.network.lower())
    if args.log_dir is not None:
        config.saving_path = args.log_dir
    elif args.dataset.lower() == 's3dis':
        config.saving_path = 'result/{}_lr0p01'.format(args.network)
    else:
        config.saving_path = 'result/{}_lr0p01'.format(args.network)

    if args.resume_from:
        config.load(os.path.join(args.resume_from))
        # config.saving_path = None

    # step1. initialization
    torch.distributed.init_process_group(backend='nccl')

    # step2. config gpu(s) for each processor
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    attrs = vars(config)
    print(', '.join('%s: %s' % item for item in attrs.items()))

    # step3. appy DistributedSampler over the custom sampler and dist the dataloader
    # Initialize datasets
    if args.dataset == 's3dis':
        training_dataset = myDataset(config,
                                     set='training',
                                     use_potentials=config.use_potential,
                                     validation_split=args.test_area_idx)
        test_dataset = myDataset(config,
                                 set='validation',
                                 use_potentials=True,
                                 validation_split=args.test_area_idx)
    else:
        training_dataset = myDataset(config,
                                     set='training',
                                     use_potentials=config.use_potential)
        test_dataset = myDataset(config, set='validation', use_potentials=True)

    # Initialize samplers
    training_sampler = mySampler(training_dataset)
    test_sampler = mySampler(test_dataset)

    if training_sampler is not None:
        training_dist_sampler = DistributedSamplerWrapper(
            sampler=training_sampler)
    else:
        training_dist_sampler = DistributedSampler(training_dataset)

    if test_sampler is not None:
        test_dist_sampler = DistributedSamplerWrapper(sampler=test_sampler)
    else:
        test_dist_sampler = DistributedSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_dist_sampler,
                                 collate_fn=myCollate,
                                 num_workers=config.input_threads)
    # pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_dist_sampler,
                             collate_fn=myCollate,
                             num_workers=config.input_threads)
    # pin_memory=True)

    # Calibrate samplers
    '''Here is the issue, how to deal with this part is a still a problem'''
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    # Optional debug functions
    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_upsampling(training_dataset, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()

    # step4. wrap the model to DDP
    if args.network.lower() == 'pyramid_v1' or args.network.lower(
    ) == 'pyramid_v1_deform':  # vertical
        net = KP_Pyramid_V1(config,
                            training_dataset.label_values,
                            training_dataset.ignored_labels,
                            use_multi_layer=config.use_multi_layer)
    elif args.network.lower() == 'pyramid_v2' or args.network.lower(
    ) == 'pyramid_v2_deform':  # horizontal
        net = KP_Pyramid_V2(config,
                            training_dataset.label_values,
                            training_dataset.ignored_labels,
                            use_multi_layer=config.use_multi_layer)
    else:
        net = KPFCNN(config, training_dataset.label_values,
                     training_dataset.ignored_labels)
    net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)
    net.train()
    if torch.distributed.get_rank() == 0:
        print('config is', config)
        print('config is', config.__dict__)
        print('net arch is', net)
    debug = False
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print('Model size %i' %
              sum(param.numel()
                  for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp, device=device)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    trainer.train(net, training_loader, test_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)
