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
#

import numpy as np
import torch
import torch.nn as nn

from models.blocks import KPConv, UnaryBlock, block_decider


def p2p_fitting_regularizer(net):
    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent**2)

            # Loss will be the square distance to closest input point.
            # We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat(
                    [KP_locs[:, :i, :], KP_locs[:, i + 1:, :]],
                    dim=1).detach()
                distances = torch.sqrt(
                    torch.sum((other_KP - KP_locs[:, i:i + 1, :])**2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances -
                                                     net.repulse_extent,
                                                     max=0.0)**2,
                                     dim=1)
                repulsive_loss += net.l1(rep_loss,
                                         torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


class KPCNN(nn.Module):
    """Class defining KPCNN."""
    def __init__(self, config):
        super(KPCNN, self).__init__()
        print('Welcome to KPCNN')
        #####################
        # Network opperations
        #####################

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
                raise ValueError(
                    'Equivariant block but features dimension is not a factor of 3'
                )

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(
                block_decider(block, r, in_dim, out_dim, layer, config))

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
        self.head_softmax = UnaryBlock(1024,
                                       config.num_classes,
                                       False,
                                       0,
                                       no_relu=True)

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
        torch.cuda.empty_cache()
        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)
        torch.cuda.empty_cache()
        # Head of network
        x = self.head_mlp(x, batch)
        torch.cuda.empty_cache()
        x = self.head_softmax(x, batch)
        torch.cuda.empty_cache()
        return x

    def loss(self, outputs, labels):
        """Runs the loss on outputs of the model.

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
            raise ValueError('Unknown fitting mode: ' +
                             self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        """Computes accuracy of the current batch.

        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total


class KPFCNN(nn.Module):
    """Class defining KPFCNN."""
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
                raise ValueError(
                    'Equivariant block but features dimension is not a factor of 3'
                )

            # Detect change to next layer for skip connection
            if np.any([
                    tmp in block
                    for tmp in ['pool', 'strided', 'upsample', 'global']
            ]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(
                block_decider(block, r, in_dim, out_dim, layer, config))

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
            if block_i > 0 and 'upsample' in config.architecture[start_i +
                                                                 block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(
                block_decider(block, r, in_dim, out_dim, layer, config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False,
                                   0)
        self.head_softmax = UnaryBlock(config.first_features_dim,
                                       self.C,
                                       False,
                                       0,
                                       no_relu=True)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort(
            [c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(
                np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w,
                                                       ignore_index=-1)
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

        # Get input features
        x = batch.features.clone().detach()
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

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """Runs the loss on outputs of the model.

        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = -torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' +
                             self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """Computes accuracy of the current batch.

        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = -torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total


class KP_Pyramid_V1(nn.Module):
    """Class defining KP_Pyramid_V1."""
    def __init__(self,
                 config,
                 lbl_values,
                 ign_lbls,
                 use_multi_layer=False,
                 use_resnetb=False):
        super(KP_Pyramid_V1, self).__init__()

        ############
        # Parameters
        ############
        print('Welcome to KP_Pyramid_V1, vertical')
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        self.use_multi_layer = use_multi_layer
        self.use_resnetb = use_resnetb

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
                raise ValueError(
                    'Equivariant block but features dimension is not a factor of 3'
                )

            # Detect change to next layer for skip connection
            if np.any([
                    tmp in block
                    for tmp in ['pool', 'strided', 'upsample', 'global']
            ]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)
                self.encoder_skip_r.append(r)
                self.encoder_skip_layer.append(layer)

            # Detect upsampling block to stop
            # if 'upsample' in block:
            #     break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(
                block_decider(block, r, in_dim, out_dim, layer, config))

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
            self.maxpool_blocks.append(
                block_decider('max_pool', self.encoder_skip_r[i + 1],
                              self.encoder_skip_dims[i + 1],
                              self.encoder_skip_dims[i + 2],
                              self.encoder_skip_layer[i], config))

        # Save all upsample operations
        self.upsample_blocks = nn.ModuleList()
        for i in range(self.num):
            self.upsample_blocks.append(
                block_decider('nearest_upsample', self.encoder_skip_r[i + 2],
                              self.encoder_skip_dims[i + 2],
                              self.encoder_skip_dims[i + 1],
                              self.encoder_skip_layer[i + 2], config))
        self.upsample_blocks.append(
            block_decider('nearest_upsample', r, out_dim,
                          self.encoder_skip_dims[-1], layer, config))

        # Save all block operations in a list of modules
        self.lateral_blocks = nn.ModuleList()
        for arch_i in config.architecture[1:]:
            self.lateral_blocks.append(
                self.build_block(arch_i, self.encoder_skip_r,
                                 self.encoder_skip_dims,
                                 self.encoder_skip_layer, config))
        # use resnetb before cat in first row of the architecture
        if self.use_resnetb:
            self.resnetb_blocks = nn.ModuleList()
            for arch_i in config.architecture[1:]:
                self.resnetb_blocks.append(
                    block_decider('resnetb', self.encoder_skip_r[1],
                                  self.encoder_skip_dims[1],
                                  self.encoder_skip_dims[1],
                                  self.encoder_skip_layer[1], config))

        self.last_upsample = block_decider('nearest_upsample',
                                           self.encoder_skip_r[1],
                                           self.encoder_skip_dims[1],
                                           self.encoder_skip_dims[0],
                                           self.encoder_skip_layer[1], config)
        if not self.use_multi_layer:
            self.last_block = block_decider(
                'resnetb', self.encoder_skip_r[0],
                self.encoder_skip_dims[0] + self.encoder_skip_dims[1],
                self.encoder_skip_dims[0], self.encoder_skip_layer[0], config)
        else:
            self.last_block = block_decider(
                'resnetb', self.encoder_skip_r[0],
                self.encoder_skip_dims[0] + self.encoder_skip_dims[1] +
                self.encoder_skip_dims[2] + self.encoder_skip_dims[3],
                self.encoder_skip_dims[0], self.encoder_skip_layer[0], config)

        self.head_mlp = UnaryBlock(self.encoder_skip_dims[0],
                                   config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim,
                                       self.C,
                                       False,
                                       0,
                                       no_relu=True)

        return

    def build_block(self, architecture, rs, dims, layers, config):
        decoder_blocks = nn.ModuleList()

        for block_i, block in enumerate(architecture):
            # Apply the good block function defining tf ops
            if block_i == 0:
                in_dim = dims[block_i + 1] + dims[block_i + 2]
            else:
                in_dim = dims[block_i] + dims[block_i + 1] + dims[block_i + 2]
            decoder_blocks.append(
                block_decider(block, rs[block_i + 1], in_dim,
                              dims[block_i + 1], layers[block_i + 1], config))
        return decoder_blocks

    def forward(self, batch, config):
        # Get input features
        x = batch.features.clone().detach()
        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
        skip_x.append(x)

        for blocks_i, blocks in enumerate(self.lateral_blocks):
            for block_i, block_op in enumerate(blocks):
                if block_i == 0:
                    if self.use_resnetb:
                        res_op = self.resnetb_blocks[blocks_i]
                        skip_x[block_i + 1] = res_op(skip_x[block_i + 1],
                                                     batch)
                    up_op = self.upsample_blocks[block_i]
                    feat = torch.cat([
                        skip_x[block_i + 1],
                        up_op(skip_x[block_i + 2], batch)
                    ],
                                     dim=1)
                else:
                    up_op = self.upsample_blocks[block_i]
                    pool_op = self.maxpool_blocks[block_i - 1]
                    feat = torch.cat([
                        pool_op(skip_x[block_i], batch), skip_x[block_i + 1],
                        up_op(skip_x[block_i + 2], batch)
                    ],
                                     dim=1)
                skip_x[block_i + 1] = block_op(feat, batch)

        if not self.use_multi_layer:
            feat = torch.cat(
                [skip_x[0], self.last_upsample(skip_x[1], batch)], dim=1)
        else:
            up_op_2to1 = self.upsample_blocks[0]
            up_op_3to2 = self.upsample_blocks[1]
            feat = torch.cat([
                skip_x[0],
                self.last_upsample(skip_x[1], batch),
                self.last_upsample(up_op_2to1(skip_x[2], batch), batch),
                self.last_upsample(
                    up_op_2to1(up_op_3to2(skip_x[3], batch), batch), batch)
            ],
                             dim=1)
        x = self.last_block(feat, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)
        return x

    def loss(self, outputs, labels):
        """Runs the loss on outputs of the model.

        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = -torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' +
                             self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """Computes accuracy of the current batch.

        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = -torch.ones_like(labels)
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
'''


class KP_Pyramid_V2(nn.Module):
    """Class defining KPFCNN."""
    def __init__(self,
                 config,
                 lbl_values,
                 ign_lbls,
                 use_multi_layer=False,
                 use_resnetb=False):
        super(KP_Pyramid_V2, self).__init__()

        ############
        # Parameters
        ############
        print('welcome to kP_Pyramid_V2, horizontal')
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        self.use_multi_layer = use_multi_layer
        self.use_resnetb = use_resnetb

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
                raise ValueError(
                    'Equivariant block but features dimension is not a factor of 3'
                )

            # Detect change to next layer for skip connection
            if np.any([
                    tmp in block
                    for tmp in ['pool', 'strided', 'upsample', 'global']
            ]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)
                self.encoder_skip_r.append(r)
                self.encoder_skip_layer.append(layer)

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(
                block_decider(block, r, in_dim, out_dim, layer, config))

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
            self.maxpool_blocks.append(
                block_decider('max_pool', self.encoder_skip_r[i + 1],
                              self.encoder_skip_dims[i + 1],
                              self.encoder_skip_dims[i + 2],
                              self.encoder_skip_layer[i], config))

        # Save all upsample operations
        self.upsample_blocks = nn.ModuleList()
        for i in range(self.num):
            self.upsample_blocks.append(
                block_decider('nearest_upsample', self.encoder_skip_r[i + 2],
                              self.encoder_skip_dims[i + 2],
                              self.encoder_skip_dims[i + 1],
                              self.encoder_skip_layer[i + 2], config))
        self.upsample_blocks.append(
            block_decider('nearest_upsample', r, out_dim,
                          self.encoder_skip_dims[-1], layer, config))

        # Save all block operations in a list of modules
        self.lateral_blocks = nn.ModuleList()
        for arch_i in config.architecture[1:]:
            self.lateral_blocks.append(
                self.build_block(arch_i, self.encoder_skip_r,
                                 self.encoder_skip_dims,
                                 self.encoder_skip_layer, config))
        # use resnetb before cat in first row of the architecture
        if self.use_resnetb:
            self.resnetb_blocks = nn.ModuleList()
            for arch_i in config.architecture[1:]:
                self.resnetb_blocks.append(
                    block_decider('resnetb', self.encoder_skip_r[1],
                                  self.encoder_skip_dims[1],
                                  self.encoder_skip_dims[1],
                                  self.encoder_skip_layer[1], config))

        self.last_upsample = block_decider('nearest_upsample',
                                           self.encoder_skip_r[1],
                                           self.encoder_skip_dims[1],
                                           self.encoder_skip_dims[0],
                                           self.encoder_skip_layer[1], config)
        if not self.use_multi_layer:
            self.last_block = block_decider(
                'unary', self.encoder_skip_r[0],
                self.encoder_skip_dims[0] + self.encoder_skip_dims[1],
                self.encoder_skip_dims[0], self.encoder_skip_layer[0], config)
        else:
            self.last_block = block_decider(
                'unary', self.encoder_skip_r[0], self.encoder_skip_dims[0] +
                3 * (2 * config.first_features_dim), self.encoder_skip_dims[0],
                self.encoder_skip_layer[0], config)

        # self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_mlp = UnaryBlock(self.encoder_skip_dims[0],
                                   config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim,
                                       self.C,
                                       False,
                                       0,
                                       no_relu=True)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort(
            [c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            print('class_w is set', config.class_w)
            class_w = torch.from_numpy(
                np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w,
                                                       ignore_index=-1)
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
            decoder_blocks.append(
                block_decider(block, rs[block_i + 1], in_dim,
                              dims[block_i + 1], layers[block_i + 1], config))
        return decoder_blocks

    def forward(self, batch, config):
        # Get input features
        x = batch.features.clone().detach()
        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
        skip_x.append(x)

        stages_in_layer1_list = []
        for blocks_i, blocks in enumerate(
                self.lateral_blocks):  # blocks_i index each stage
            for block_i, block_op in enumerate(
                    blocks):  # block_i index each layer
                if block_i == 0:
                    if self.use_resnetb:
                        res_op = self.resnetb_blocks[blocks_i]
                        skip_x[block_i + 1] = res_op(skip_x[block_i + 1],
                                                     batch)
                    up_op = self.upsample_blocks[block_i]
                    feat = torch.cat([
                        skip_x[block_i + 1],
                        up_op(skip_x[block_i + 2], batch)
                    ],
                                     dim=1)
                else:
                    up_op = self.upsample_blocks[block_i]
                    pool_op = self.maxpool_blocks[block_i - 1]
                    feat = torch.cat([
                        pool_op(skip_x[block_i], batch), skip_x[block_i + 1],
                        up_op(skip_x[block_i + 2], batch)
                    ],
                                     dim=1)
                skip_x[block_i + 1] = block_op(feat, batch)
                if block_i == 0:
                    stages_in_layer1_list.append(skip_x[block_i + 1])

        if not self.use_multi_layer:
            feat = torch.cat(
                [skip_x[0], self.last_upsample(skip_x[1], batch)], dim=1)
        else:
            feat = torch.cat(
                [
                    skip_x[0],  #
                    self.last_upsample(skip_x[1], batch),
                    self.last_upsample(stages_in_layer1_list[-2], batch),
                    self.last_upsample(stages_in_layer1_list[-3], batch),
                ],
                dim=1)
        x = self.last_block(feat, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)
        return x

    def loss(self, outputs, labels):
        """Runs the loss on outputs of the model.

        param:
            outputs: logits
            labels: labels
        return:
            loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = -torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' +
                             self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """Computes accuracy of the current batch.

        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = -torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total
