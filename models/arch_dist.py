#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------


from pickle import NONE
from models.blocks import *
import numpy as np
from models.myloss import FocalLoss, Combine_Dice_Focal_Loss, Combine_Ohem_Dice_Loss, Combine_CE_Dice_Loss

def p2p_fitting_regularizer(net):
    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


class KPCNN(nn.Module):
    """
    Class defining KPCNN
    """

    def __init__(self, config):
        super(KPCNN, self).__init__()

        #####################
        # Network opperations
        #####################
        print('Welcome to KPCNN')
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block,
                                                r,
                                                in_dim,
                                                out_dim,
                                                layer,
                                                config))

            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0, no_relu=True)

        ################
        # Network Losses
        ################

        self.criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Save all block operations in a list of modules
        x = batch.features.clone().detach()
        print('x.shape', x.shape)
        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()
        print('Welcome to KPFCNN')
        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        self.config = config

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0, no_relu=True)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        class_w = None
        if len(config.class_w) > 0:
            print('class_w is set', config.class_w)
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))

        if config.loss_type == 'focal_loss':
            criterion = FocalLoss(weight=class_w, reduction='sum', ignore_indx=-1)
        if config.loss_type == 'Dice_Ohem':
            # criterion = Combine_Ohem_Dice_Loss(mylambda_dice=1, ignore_label=cfg.ignored_label_mapping, thresh=0.7, min_kept=int(
            criterion = Combine_Ohem_Dice_Loss(mylambda_dice=1, mylambda_ohem=0.7,
                                               ignore_label=-1, thresh=0.7, min_kept=int(
                    config.batch_size * config.num_points // 16))
            # output_loss, loss_dice, loss_ohem = criterion(logits, labels)
        elif config.loss_type == 'Dice_CE':
            criterion = Combine_CE_Dice_Loss(mylambda=0.7, ignore_label=-1)
        elif config.loss_type == 'Dice_Focal':
            # criterion = Combine_Dice_Focal_Loss(mylambda_dice=0.7, mylambda_focal=1.3, ignore_label=cfg.ignored_label_mapping)
            criterion = Combine_Dice_Focal_Loss(mylambda_dice=0.7, mylambda_focal=1.3,
                                                ignore_label=-1, class_weights=class_w)
        else:  # CE
            if len(config.class_w) > 0:
                self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
            else:
                self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):
        # torch.cuda.empty_cache()
        # Get input features
        x = batch.features.clone().detach()
        # print('x.shape',x.shape)
        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        # torch.cuda.empty_cache()

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        labels = None
        if batch.labels is not None:
            labels = batch.labels
        if labels is None:
            return x
        else:
            return x, self.loss(x, labels), self.accuracy(x, labels)

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)
        if self.config.loss_type in ['Dice_Focal','Dice_Ohem','Dice_CE']:
            self.output_loss = 0.5*self.output_loss

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total


'''
    KP_Pyramid_V1, originally KPFCNN_plus
'''
class KP_Pyramid_V1(nn.Module): # vertical
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls, use_multi_layer=False, use_resnetb=False):
        super(KP_Pyramid_V1, self).__init__()
        print('Welcome to KP_Pyramid_V1, vertical')
        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        self.use_multi_layer = use_multi_layer
        self.use_resnetb = use_resnetb
        self.config = config
        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []
        self.encoder_skip_r = []
        self.encoder_skip_layer = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[0]):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)
                self.encoder_skip_r.append(r)
                self.encoder_skip_layer.append(layer)

            # Detect upsampling block to stop
            # if 'upsample' in block:
            #     break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
        # self.encoder_skips.append(block_i) can't have
        self.encoder_skip_dims.append(in_dim)
        self.encoder_skip_r.append(r)
        self.encoder_skip_layer.append(layer)
        #####################
        # List Decoder blocks
        #####################

        # Num of maxpool. Note: num of upsample = num of maxpool + 1
        self.num = len(self.encoder_skip_dims) - 2

        # Save all pooling operations
        self.maxpool_blocks = nn.ModuleList()
        self.feats_att = nn.ModuleList()
        self.feats_att_down = nn.ModuleList()
        for i in range(self.num):
            self.maxpool_blocks.append(block_decider('max_pool',
                                                     self.encoder_skip_r[i + 1],
                                                     self.encoder_skip_dims[i + 1],
                                                     self.encoder_skip_dims[i + 2],
                                                     self.encoder_skip_layer[i],
                                                     config))
            self.feats_att.append(block_decider('unary',
                                                self.encoder_skip_r[i+1],
                                                self.encoder_skip_dims[i+1],
                                                1,
                                                self.encoder_skip_layer[i],
                                                config))
            if i < self.num - 1:
                self.feats_att_down.append(block_decider('unary',
                                                         self.encoder_skip_r[i+1],
                                                         self.encoder_skip_dims[i+1],
                                                         1,
                                                         self.encoder_skip_layer[i],
                                                         config))

        # Save all upsample operations
        self.upsample_blocks = nn.ModuleList()
        self.feats_att_up = nn.ModuleList()
        for i in range(self.num):
            self.upsample_blocks.append(block_decider('nearest_upsample',
                                                      self.encoder_skip_r[i + 2],
                                                      self.encoder_skip_dims[i + 2],
                                                      self.encoder_skip_dims[i + 1],
                                                      self.encoder_skip_layer[i + 2],
                                                      config))
            self.feats_att_up.append(block_decider('unary',
                                                   self.encoder_skip_r[i + 2],
                                                   self.encoder_skip_dims[i + 2],
                                                   1,
                                                   self.encoder_skip_layer[i + 2],
                                                   config))
        self.upsample_blocks.append(block_decider('nearest_upsample',
                                                  r,
                                                  out_dim,
                                                  self.encoder_skip_dims[-1],
                                                  layer,
                                                  config))

        # Save all block operations in a list of modules
        self.lateral_blocks = nn.ModuleList()
        for arch_i in config.architecture[1:]:
            self.lateral_blocks.append(self.build_block(arch_i,
                                                        self.encoder_skip_r,
                                                        self.encoder_skip_dims,
                                                        self.encoder_skip_layer,
                                                        config))
        # use resnetb before cat in first row of the architecture
        if self.use_resnetb:
            self.resnetb_blocks = nn.ModuleList()
            for arch_i in config.architecture[1:]:
                self.resnetb_blocks.append(block_decider('resnetb',
                                                         self.encoder_skip_r[1],
                                                         self.encoder_skip_dims[1],
                                                         self.encoder_skip_dims[1],
                                                         self.encoder_skip_layer[1],
                                                         config))

        self.last_upsample = block_decider('nearest_upsample',
                                           self.encoder_skip_r[1],
                                           self.encoder_skip_dims[1],
                                           self.encoder_skip_dims[0],
                                           self.encoder_skip_layer[1],
                                           config)
        if self.use_multi_layer == False:
            self.last_block = block_decider('resnetb',
                                            self.encoder_skip_r[0],
                                            self.encoder_skip_dims[0] + self.encoder_skip_dims[1],
                                            self.encoder_skip_dims[0],
                                            self.encoder_skip_layer[0],
                                            config)
        else:
            self.last_block = block_decider('resnetb',
                                            self.encoder_skip_r[0],
                                            self.encoder_skip_dims[0] + self.encoder_skip_dims[1] +
                                            self.encoder_skip_dims[2] + self.encoder_skip_dims[3],
                                            self.encoder_skip_dims[0],
                                            self.encoder_skip_layer[0],
                                            config)

        # # Find first upsampling block
        # start_i = 0
        # for block_i, block in enumerate(config.architecture):
        #     if 'upsample' in block:
        #         start_i = block_i
        #         break

        # # Loop over consecutive blocks
        # for block_i, block in enumerate(config.architecture[start_i:]):

        #     # Add dimension of skip connection concat
        #     if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
        #         in_dim += self.encoder_skip_dims[layer]
        #         self.decoder_concats.append(block_i)

        #     # Apply the good block function defining tf ops
        #     self.decoder_blocks.append(block_decider(block,
        #                                             r,
        #                                             in_dim,
        #                                             out_dim,
        #                                             layer,
        #                                             config))

        #     # Update dimension of input from output
        #     in_dim = out_dim

        #     # Detect change to a subsampled layer
        #     if 'upsample' in block:
        #         # Update radius and feature dimension for next layer
        #         layer -= 1
        #         r *= 0.5
        #         out_dim = out_dim // 2

        # self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_mlp = UnaryBlock(self.encoder_skip_dims[0], config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0, no_relu=True)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        class_w = None
        if len(config.class_w) > 0:
            print('class_w is set', config.class_w)
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))

        if config.loss_type == 'focal_loss':
            criterion = FocalLoss(weight=class_w, reduction='sum', ignore_indx=-1)
        if config.loss_type == 'Dice_Ohem':
            # criterion = Combine_Ohem_Dice_Loss(mylambda_dice=1, ignore_label=cfg.ignored_label_mapping, thresh=0.7, min_kept=int(
            criterion = Combine_Ohem_Dice_Loss(mylambda_dice=1, mylambda_ohem=0.7,
                                               ignore_label=-1, thresh=0.7, min_kept=int(
                    config.batch_size * config.num_points // 16))
            # output_loss, loss_dice, loss_ohem = criterion(logits, labels)
        elif config.loss_type == 'Dice_CE':
            criterion = Combine_CE_Dice_Loss(mylambda=0.7, ignore_label=-1)
        elif config.loss_type == 'Dice_Focal':
            # criterion = Combine_Dice_Focal_Loss(mylambda_dice=0.7, mylambda_focal=1.3, ignore_label=cfg.ignored_label_mapping)
            criterion = Combine_Dice_Focal_Loss(mylambda_dice=0.7, mylambda_focal=1.3,
                                                ignore_label=-1, class_weights=class_w)
        else:  # CE
            if len(config.class_w) > 0:
                self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
            else:
                self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def attention_func1(self, master_feats, small_feats, large_feats, small_ind, large_ind):
        _small_feats = self.feats_att_up[small_ind](small_feats)
        _large_feats = self.feats_att_down[large_ind](large_feats)
        att_map = (_small_feats + _large_feats) + (_small_feats * _large_feats)
        att_map = torch.sigmoid(att_map)

        feats = torch.cat([small_feats, master_feats, large_feats], dim=1)
        feats = feats * att_map
        return feats

    def attention_func2(self, master_feats, small_feats, large_feats, small_ind, large_ind):
        _small_feats = self.feats_att_up[small_ind](small_feats)
        _large_feats = self.feats_att_down[large_ind](large_feats)
        att_map = torch.abs(_small_feats - _large_feats)
        att_map = torch.sigmoid(att_map)

        feats = torch.cat([small_feats, master_feats, large_feats], dim=1)
        feats = feats * att_map
        return feats

    '''
        * for sem, + for resolution
    '''
    def attention_func3(self, master_feats, small_feats, cur_ind, large_feats=None):
        _feats = self.feats_att[cur_ind](master_feats)
        _small_feats = self.feats_att_up[cur_ind](small_feats)
        att_map = torch.sigmoid(_feats * _small_feats)
        #import pdb; pdb.set_trace()
        small_feats = small_feats * att_map[:,None]
        feats = torch.cat([small_feats, master_feats], dim=1)

        if large_feats is not None:
            _large_feats = self.feats_att_down[cur_ind-1](large_feats)
            att_map = torch.sigmoid(_feats + _large_feats)
            large_feats = large_feats * att_map[:,None]
            feats = torch.cat([feats, large_feats], dim=1)
        return feats

    '''
        + for sem, * for resolution
    '''
    def attention_func4(self, master_feats, small_feats, cur_ind, large_feats=None):
        _feats = self.feats_att[cur_ind](master_feats)
        _small_feats = self.feats_att_up[cur_ind](small_feats)
        att_map = torch.sigmoid(_feats + _small_feats)
        #import pdb; pdb.set_trace()
        small_feats = small_feats * att_map[:,None]
        feats = torch.cat([small_feats, master_feats], dim=1)

        if large_feats is not None:
            _large_feats = self.feats_att_down[cur_ind-1](large_feats)
            att_map = torch.sigmoid(_feats * _large_feats)
            large_feats = large_feats * att_map[:,None]
            feats = torch.cat([feats, large_feats], dim=1)
        return feats

    '''
        all + to produce att map
    '''
    def attention_func5(self, master_feats, small_feats, cur_ind, large_feats=None):
        _feats = self.feats_att[cur_ind](master_feats)
        _small_feats = self.feats_att_up[cur_ind](small_feats)
        att_map = torch.sigmoid(_feats + _small_feats)
        #import pdb; pdb.set_trace()
        small_feats = small_feats * att_map[:,None]
        feats = torch.cat([small_feats, master_feats], dim=1)

        if large_feats is not None:
            _large_feats = self.feats_att_down[cur_ind-1](large_feats)
            att_map = torch.sigmoid(_feats + _large_feats)
            large_feats = large_feats * att_map[:,None]
            feats = torch.cat([feats, large_feats], dim=1)
        return feats

    '''
        all * to produce att map
    '''
    def attention_func6(self, master_feats, small_feats, cur_ind, large_feats=None):
        _feats = self.feats_att[cur_ind](master_feats)
        _small_feats = self.feats_att_up[cur_ind](small_feats)
        att_map = torch.sigmoid(_feats * _small_feats)
        #import pdb; pdb.set_trace()
        small_feats = small_feats * att_map[:,None]
        feats = torch.cat([small_feats, master_feats], dim=1)

        if large_feats is not None:
            _large_feats = self.feats_att_down[cur_ind-1](large_feats)
            att_map = torch.sigmoid(_feats * _large_feats)
            large_feats = large_feats * att_map[:,None]
            feats = torch.cat([feats, large_feats], dim=1)
        return feats

    '''
        * for sem, + for resolution, average for base scale
    '''
    def attention_func7(self, master_feats, small_feats, cur_ind, large_feats=None):
        _feats = self.feats_att[cur_ind](master_feats)
        _small_feats = self.feats_att_up[cur_ind](small_feats)
        att_map = torch.sigmoid(_feats * _small_feats)
        #import pdb; pdb.set_trace()
        small_feats = small_feats * att_map[:,None]

        _master_feats = master_feats*(1-att_map[:,None])
        # feats = torch.cat([small_feats, master_feats], dim=1)

        if large_feats is not None:
            _large_feats = self.feats_att_down[cur_ind-1](large_feats)
            att_map = torch.sigmoid(_feats + _large_feats)
            large_feats = large_feats * att_map[:,None]

            _master_feats = (master_feats*(1-att_map[:,None]) + _master_feats)/2

            feats = torch.cat([small_feats, _master_feats, large_feats], dim=1)
        else:
            feats = torch.cat([small_feats, _master_feats], dim=1)
        return feats

    '''
        * for sem (+0.5), + for resolution (+0.5), keep base scale
    '''
    def attention_func8(self, master_feats, small_feats, cur_ind, large_feats=None):
        _feats = self.feats_att[cur_ind](master_feats)
        _small_feats = self.feats_att_up[cur_ind](small_feats)
        att_map = (torch.sigmoid(_feats * _small_feats) + 0.5)
        #import pdb; pdb.set_trace()
        small_feats = small_feats * att_map[:,None]
        feats = torch.cat([small_feats, master_feats], dim=1)

        if large_feats is not None:
            _large_feats = self.feats_att_down[cur_ind-1](large_feats)
            att_map = (torch.sigmoid(_feats + _large_feats) + 0.5)
            large_feats = large_feats * att_map[:,None]
            feats = torch.cat([feats, large_feats], dim=1)
        return feats

    '''
        directly concat
    '''
    def attention_func0(self, master_feats, small_feats, cur_ind, large_feats=None):
        feats = torch.cat([small_feats, master_feats], dim=1)

        if large_feats is not None:
            feats = torch.cat([feats, large_feats], dim=1)
        return feats

    def build_block(self, architecture, rs, dims, layers, config):
        decoder_blocks = nn.ModuleList()

        for block_i, block in enumerate(architecture):
            # Apply the good block function defining tf ops
            if block_i == 0:
                in_dim = dims[block_i + 1] + dims[block_i + 2]
            else:
                in_dim = dims[block_i] + dims[block_i + 1] + dims[block_i + 2]
            decoder_blocks.append(block_decider(block,
                                                rs[block_i + 1],
                                                in_dim,
                                                dims[block_i + 1],
                                                layers[block_i + 1],
                                                config))
        return decoder_blocks

    def forward(self, batch, config):
        # torch.cuda.empty_cache()
        # Get input features
        x = batch.features.clone().detach()
        # print('x.shape',x.shape)
        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
        skip_x.append(x)
        # for block_i, block_op in enumerate(self.decoder_blocks):
        #     if block_i in self.decoder_concats:
        #         x = torch.cat([x, skip_x.pop()], dim=1)
        #     x = block_op(x, batch)
        # torch.cuda.empty_cache()
        for blocks_i, blocks in enumerate(self.lateral_blocks):
            for block_i, block_op in enumerate(blocks):
                if block_i == 0:
                    if self.use_resnetb:
                        res_op = self.resnetb_blocks[blocks_i]
                        skip_x[block_i + 1] = res_op(skip_x[block_i + 1], batch)
                    up_op = self.upsample_blocks[block_i]
                    # old method
                    #feat = torch.cat([skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch)], dim=1)
                    # method 1
                    #feat = self.attention_func1(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), skip_x[block_i + 1], block_i, block_i)
                    # method 2
                    #feat = self.attention_func2(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), skip_x[block_i + 1], block_i, block_i)
                    # method 3
                    feat = self.attention_func3(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i)
                    # method 4
                    #feat = self.attention_func4(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i)
                    # method 5
                    #feat = self.attention_func5(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i)
                    # method 6
                    #feat = self.attention_func6(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i)
                    # method 7
                    #feat = self.attention_func7(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i)
                    # method 8
                    #feat = self.attention_func8(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i)
                    # method 0 
                    #feat = self.attention_func0(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i)
                else:
                    up_op = self.upsample_blocks[block_i]
                    pool_op = self.maxpool_blocks[block_i - 1]
                    # old method
                    #feat = torch.cat([pool_op(skip_x[block_i], batch), skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch)], dim=1)
                    # method1
                    # feat = self.attention_func1(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch),
                    #                 pool_op(skip_x[block_i], batch), block_i, block_i-1)
                    # method2
                    # feat = self.attention_func1(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch),
                    #                 pool_op(skip_x[block_i], batch), block_i, block_i-1)
                    # method3
                    feat = self.attention_func3(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i, pool_op(skip_x[block_i], batch))
                    # method4
                    #feat = self.attention_func4(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i, pool_op(skip_x[block_i], batch))
                    # method5
                    #feat = self.attention_func5(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i, pool_op(skip_x[block_i], batch))
                    # method6
                    #feat = self.attention_func6(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i, pool_op(skip_x[block_i], batch))
                    # method7
                    #feat = self.attention_func7(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i, pool_op(skip_x[block_i], batch))
                    # method8
                    #feat = self.attention_func8(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i, pool_op(skip_x[block_i], batch))
                    # method0
                    #feat = self.attention_func0(skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch), block_i, pool_op(skip_x[block_i], batch))
                skip_x[block_i + 1] = block_op(feat, batch)

        if self.use_multi_layer == False:
            feat = torch.cat([skip_x[0], self.last_upsample(skip_x[1], batch)], dim=1)
        else:
            up_op_2to1 = self.upsample_blocks[0]
            up_op_3to2 = self.upsample_blocks[1]
            feat = torch.cat([skip_x[0],
                              self.last_upsample(skip_x[1], batch),
                              self.last_upsample(up_op_2to1(skip_x[2], batch), batch),
                              self.last_upsample(up_op_2to1(up_op_3to2(skip_x[3], batch), batch), batch)],
                             dim=1)
        x = self.last_block(feat, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        labels = None
        if batch.labels is not None:
            labels = batch.labels
        if labels is None:
            return x
        else:
            return x, self.loss(x, labels), self.accuracy(x, labels)

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)
        if self.config.loss_type in ['Dice_Focal','Dice_Ohem','Dice_CE']:
            self.output_loss = 0.5*self.output_loss

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total



'''
    KPFCNN+BP+Scale+local_agg 
    Correction for last_block and upsampling
    scale for layer0 is switched from layer 1
    KP_Pyramid_V2, originally KPFCNN_plus_nie
'''
class KP_Pyramid_V2(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls, use_multi_layer=False, use_resnetb=False):
        super(KP_Pyramid_V2, self).__init__()

        ############
        # Parameters
        ############
        print('welcome to KP_Pyramid_v2, horizontal')
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        #print('in_dim',in_dim)
        #if in_dim in [4,5]: # use color info
        #    in_dim = 3
        #else: #only use point info
        #in_dim = 1
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        self.use_multi_layer = use_multi_layer
        self.use_resnetb = use_resnetb
        self.config = config
        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []
        self.encoder_skip_r = []
        self.encoder_skip_layer = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[0]):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)
                self.encoder_skip_r.append(r)
                self.encoder_skip_layer.append(layer)

            # Detect upsampling block to stop
            # if 'upsample' in block:
            #     break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
        # self.encoder_skips.append(block_i) can't have
        self.encoder_skip_dims.append(in_dim)
        self.encoder_skip_r.append(r)
        self.encoder_skip_layer.append(layer)
        #####################
        # List Decoder blocks
        #####################

        # Num of maxpool. Note: num of upsample = num of maxpool + 1
        self.num = len(self.encoder_skip_dims) - 2

        # Save all pooling operations
        self.maxpool_blocks = nn.ModuleList()
        for i in range(self.num):
            self.maxpool_blocks.append(block_decider('max_pool',
                                                     self.encoder_skip_r[i + 1],
                                                     self.encoder_skip_dims[i + 1],
                                                     self.encoder_skip_dims[i + 2],
                                                     self.encoder_skip_layer[i],
                                                     config))

        # Save all upsample operations
        self.upsample_blocks = nn.ModuleList()
        for i in range(self.num):
            self.upsample_blocks.append(block_decider('nearest_upsample',
                                                      self.encoder_skip_r[i + 2],
                                                      self.encoder_skip_dims[i + 2],
                                                      self.encoder_skip_dims[i + 1],
                                                      self.encoder_skip_layer[i + 2],
                                                      config))
        self.upsample_blocks.append(block_decider('nearest_upsample',
                                                  r,
                                                  out_dim,
                                                  self.encoder_skip_dims[-1],
                                                  layer,
                                                  config))

        # Save all block operations in a list of modules
        self.lateral_blocks = nn.ModuleList()
        for arch_i in config.architecture[1:]:
            self.lateral_blocks.append(self.build_block(arch_i,
                                                        self.encoder_skip_r,
                                                        self.encoder_skip_dims,
                                                        self.encoder_skip_layer,
                                                        config))
        # use resnetb before cat in first row of the architecture
        if self.use_resnetb:
            self.resnetb_blocks = nn.ModuleList()
            for arch_i in config.architecture[1:]:
                self.resnetb_blocks.append(block_decider('resnetb',
                                                         self.encoder_skip_r[1],
                                                         self.encoder_skip_dims[1],
                                                         self.encoder_skip_dims[1],
                                                         self.encoder_skip_layer[1],
                                                         config))

        self.last_upsample = block_decider('nearest_upsample',
                                           self.encoder_skip_r[1],
                                           self.encoder_skip_dims[1],
                                           self.encoder_skip_dims[0],
                                           self.encoder_skip_layer[1],
                                           config)
        if not self.use_multi_layer:
            self.last_block = block_decider('unary',
                                            self.encoder_skip_r[0],
                                            self.encoder_skip_dims[0] + self.encoder_skip_dims[1],
                                            self.encoder_skip_dims[0],
                                            self.encoder_skip_layer[0],
                                            config)
        else:
            self.last_block = block_decider('unary',
                                            self.encoder_skip_r[0],
                                            self.encoder_skip_dims[0] + 3*(2*config.first_features_dim),
                                            # + self.encoder_skip_dims[2] + self.encoder_skip_dims[3],
                                            self.encoder_skip_dims[0],
                                            self.encoder_skip_layer[0],
                                            config)


        # self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_mlp = UnaryBlock(self.encoder_skip_dims[0], config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0, no_relu=True)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])
        print('loss_type',config.loss_type)
        # Choose segmentation loss
        class_w = None
        if len(config.class_w) > 0:
            print('class_w is set', config.class_w)
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))

        if config.loss_type == 'focal_loss':
            self.criterion = FocalLoss(weight=class_w, reduction='sum', ignore_indx=-1)
        if config.loss_type == 'Dice_Ohem':
            # criterion = Combine_Ohem_Dice_Loss(mylambda_dice=1, ignore_label=cfg.ignored_label_mapping, thresh=0.7, min_kept=int(
            self.criterion = Combine_Ohem_Dice_Loss(mylambda_dice=1, mylambda_ohem=0.7,
                                               ignore_label=-1, thresh=0.7, min_kept=int(
                    config.batch_size * config.num_points // 16))
            # output_loss, loss_dice, loss_ohem = criterion(logits, labels)
        elif config.loss_type == 'Dice_CE':
            self.criterion = Combine_CE_Dice_Loss(mylambda=0.7, ignore_label=-1)
        elif config.loss_type == 'Dice_Focal':
            # criterion = Combine_Dice_Focal_Loss(mylambda_dice=0.7, mylambda_focal=1.3, ignore_label=cfg.ignored_label_mapping)
            self.criterion = Combine_Dice_Focal_Loss(mylambda_dice=0.7, mylambda_focal=1.3,
                                                ignore_label=-1, class_weights=class_w)
        else:  # CE
            if len(config.class_w) > 0:
                self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
            else:
                self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def build_block(self, architecture, rs, dims, layers, config):
        decoder_blocks = nn.ModuleList()

        for block_i, block in enumerate(architecture):
            # Apply the good block function defining tf ops
            if block_i == 0:
                in_dim = dims[block_i + 1] + dims[block_i + 2]
            else:
                in_dim = dims[block_i] + dims[block_i + 1] + dims[block_i + 2]
            decoder_blocks.append(block_decider(block,
                                                rs[block_i + 1],
                                                in_dim,
                                                dims[block_i + 1],
                                                layers[block_i + 1],
                                                config))
        return decoder_blocks

    def forward(self, batch, config):
        # torch.cuda.empty_cache()
        # Get input features
        x = batch.features.clone().detach()
        #print('x.shape',x.shape,'torch.unique(x)', torch.unique(x))
        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
        skip_x.append(x)
        # for block_i, block_op in enumerate(self.decoder_blocks):
        #     if block_i in self.decoder_concats:
        #         x = torch.cat([x, skip_x.pop()], dim=1)
        #     x = block_op(x, batch)
        # torch.cuda.empty_cache()

        stages_in_layer1_list = []
        for blocks_i, blocks in enumerate(self.lateral_blocks): # blocks_i index each stage
            for block_i, block_op in enumerate(blocks): # block_i index each layer
                if block_i == 0:
                    if self.use_resnetb:
                        res_op = self.resnetb_blocks[blocks_i]
                        skip_x[block_i + 1] = res_op(skip_x[block_i + 1], batch)
                    up_op = self.upsample_blocks[block_i]
                    feat = torch.cat([skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch)], dim=1)
                else:
                    up_op = self.upsample_blocks[block_i]
                    pool_op = self.maxpool_blocks[block_i - 1]
                    feat = torch.cat(
                        [pool_op(skip_x[block_i], batch), skip_x[block_i + 1], up_op(skip_x[block_i + 2], batch)],
                        dim=1)
                skip_x[block_i + 1] = block_op(feat, batch)
                if block_i==0:
                    stages_in_layer1_list.append(skip_x[block_i + 1])

        if self.use_multi_layer == False:
            feat = torch.cat([skip_x[0], self.last_upsample(skip_x[1], batch)], dim=1)
        else:
            # up_op_2to1 = self.upsample_blocks[0]
            # up_op_3to2 = self.upsample_blocks[1]
            feat = torch.cat([skip_x[0], #
                              self.last_upsample(skip_x[1], batch),
                              self.last_upsample(stages_in_layer1_list[-2], batch),
                              self.last_upsample(stages_in_layer1_list[-3], batch),
                              ],
                              # self.last_upsample(up_op_2to1(skip_x[2], batch), batch),
                              # self.last_upsample(up_op_2to1(up_op_3to2(skip_x[3], batch), batch), batch)],
                             dim=1)
        x = self.last_block(feat, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        labels = None
        if batch.labels is not None:
            labels = batch.labels
        if labels is None:
            return x
        else:
            return x, self.loss(x, labels), self.accuracy(x, labels)

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i
        
        #print('haha, outputs.shape',outputs.shape)
        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)
        if self.config.loss_type in ['Dice_Focal','Dice_Ohem','Dice_CE']:
            self.output_loss = 0.5*self.output_loss[0]
        #print('target',target.shape,torch.unique(target),'outputs',outputs.shape)
        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total







