from utils.config import Config


class NPM3DConfig(Config):
    """Override the parameters you want to modify for this dataset."""
    use_potential = False
    if use_potential:
        # only necessary for potential sampling
        class_w = [1, 1, 1, 1, 5, 5, 1, 5, 1, 1]

    loss_type = 'Dice_Focal'

    world_size = 2
    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'NPM3D'

    # Number of classes in the dataset
    # (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 4

    #########################
    # Architecture definition
    #########################

    # # Define layers
    architecture = []
    use_multi_layer = True
    use_resnetb = False

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 4.0

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.08

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell".
    # Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell".
    # (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 64
    in_features_dim = 1  # 1 by default, 3 include color

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0  # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2  # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 600

    # Learning rate management
    base_learning_rate = 1e-2
    learning_rate = base_learning_rate * world_size
    momentum = 0.98
    lr_decays = {i: 0.1**(1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 2

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.01
    augment_color = 1.0

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted
    #              according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution
    #              (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = './result/npm3d_kpconv_plus_nie_allTrain_DiceFocal_lr0p02'

    # for temporary subdata
    subdata_path = 'data'

    # for data path
    data_path = '../Data/npm3d'

    debug = False
    if debug:
        print('debug mode on ')
        epoch_step = 10
        num_epoch = 2


class S3DISConfig(Config):
    """Override the parameters you want to modify for this dataset."""
    use_potential = False
    if use_potential:
        # only necessary for potential sampling
        class_w = [1, 1, 1, 1, 5, 5, 1, 5, 1, 1]

    world_size = 4
    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'S3DIS'

    validation_split = 4
    # Number of classes in the dataset (This value is overwritten by
    # dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 4

    loss_type = 'CE'

    #########################
    # Architecture definition
    #########################

    # # Define layers
    architecture = []
    use_multi_layer = True
    use_resnetb = False

    ###################
    # KPConv parameters
    ###################

    # tf version is heavy: 1/2,1/2,2; pytorch version is light: 1/4, 1/4, 1
    resblock = 'light'

    # Radius of the input sphere
    in_radius = 1.5

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.04

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell".
    # Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in
    # "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 64 if resblock == 'heavy' else 128

    in_features_dim = 5

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0  # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2  # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 500

    # Learning rate management
    base_learning_rate = 1e-2
    learning_rate = base_learning_rate * world_size
    momentum = 0.98
    lr_decays = {i: 0.1**(1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 8 if resblock == 'heavy' else 3

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted
    #              according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution
    #              (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = 'result/kpocnv_lr0p01_area{}'.format(validation_split)


class Semantic3DConfig(Config):
    """Override the parameters you want to modify for this dataset."""
    use_potential = False
    if use_potential:
        # only necessary for potential sampling
        class_w = [1, 1, 1, 1, 5, 5, 1, 5, 1, 1]

    world_size = 4
    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'Semantic3D'

    # Number of classes in the dataset (This value is overwritten by dataset
    # class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 4

    loss_type = 'CE'
    #########################
    # Architecture definition
    #########################

    # # Define layers
    architecture = []
    use_multi_layer = True
    use_resnetb = False

    ###################
    # KPConv parameters
    ###################

    # KPConv specific parameters
    num_kernel_points = 15
    first_subsampling_dl = 0.06
    in_radius = 3.0

    # Density of neighborhoods for deformable convs (which need bigger radiuses).
    # For normal conv we use KP_extent
    density_parameter = 5.0

    # Behavior of convolutions in ('constant', 'linear', gaussian)
    KP_influence = 'linear'
    KP_extent = 1.0

    # Behavior of convolutions in ('closest', 'sum')
    convolution_mode = 'sum'

    # Can the network learn modulations
    modulated = False

    # Offset loss
    # 'permissive' only constrains offsets inside the big radius
    # 'fitting' helps deformed kernels to adapt to the geometry
    # by penalizing distance to input points
    offsets_loss = 'fitting'
    offsets_decay = 0.1

    # Choice of input features
    in_features_dim = 4

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.98

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 600

    # Learning rate management
    base_learning_rate = 1e-2
    learning_rate = base_learning_rate * world_size
    momentum = 0.98
    lr_decays = {i: 0.1**(1 / 100) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 8

    # Number of steps per epochs (cannot be None for this dataset)
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each snapshot
    snapshot_gap = 10
    checkpoint_gap = 10

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_color = 1.0

    # Whether to use loss averaged on all points, or averaged per batch.
    batch_averaged_loss = False

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted
    #              according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution
    #              (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = 'result/semantic3d_kpocnv_bp_scale_lr0p02'


class SensatUrbanConfig(Config):
    """Override the parameters you want to modify for this dataset."""
    use_potential = False
    if use_potential:
        # only necessary for potential sampling
        class_w = [1, 1, 1, 1, 5, 5, 1, 5, 1, 1]

    world_size = 4
    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'SensatUrban'

    # Number of classes in the dataset (This value is overwritten by dataset
    # class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 4

    #########################
    # Architecture definition
    #########################

    # # Define layers
    architecture = []
    use_multi_layer = True
    use_resnetb = False

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 10.0

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.2

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell".
    # Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell".
    # (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 64
    in_features_dim = 4  # 1 by default, 4 include color

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0  # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2  # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 500

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1**(1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 4

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each checkpoint
    checkpoint_gap = 20

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.01
    augment_color = 1.0

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted
    #              according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution
    #              (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = './result/debug'

    # for temporary subdata
    subdata_path = 'data'

    # for data path
    data_path = 'data'

    debug = False
    if debug:
        print('debug mode on ')
        epoch_steps = 10
        num_epoch = 2

    data_path = ''
    input_path = ''

    trainlist = ''
    vallist = ''


class XMap3DConfig(Config):
    """Override the parameters you want to modify for this dataset."""
    use_potential = False
    if use_potential:
        # only necessary for potential sampling
        class_w = [1, 1, 1, 1, 5, 5, 1, 5, 1, 1]

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'XMap3D'

    # Number of classes in the dataset (This value is overwritten by dataset
    # class when Initializating dataset).
    num_classes = None

    loss_type = 'CE'
    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 8

    # define layers
    architecture = []
    use_multi_layer = True
    use_resnetb = False

    ###################
    # KPConv parameters
    ###################

    # tf version is heavy: 1/2,1/2,2; pytorch version is light: 1/4, 1/4, 1
    resblock = 'heavy'

    # Radius of the input sphere
    in_radius = 4.0

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.08

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell".
    # Larger so that deformed kernel can spread out
    deform_radius = 5.0

    # Radius of the area of influence of each kernel point in "number grid cell".
    # (1.0 is the standard value)
    KP_extent = 1.0

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 64 if resblock == 'heavy' else 128
    in_features_dim = 3  # 1 by default, 2 includes color,  3/4 includes color and z

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0  # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2  # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 1200

    # Learning rate management
    world_size = 4
    base_learning_rate = 1e-2
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1**(1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 6 if first_features_dim == 64 else 3
    loss_type = 'CE'

    # Number of steps per epochs
    epoch_steps = 1200

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted
    #              according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution
    #              (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = './result/kp_pyramid_v1_lr0p01_xmap3d'

    # for temporary subdata
    subdata_path = 'data'

    # for data path
    data_path = 'data'

    debug = False
    if debug:
        print('debug mode on ')
        epoch_steps = 10
        num_epoch = 2

    # on 83
    data_path = ''
    input_path = ''

    trainlist = ''
    vallist = ''
