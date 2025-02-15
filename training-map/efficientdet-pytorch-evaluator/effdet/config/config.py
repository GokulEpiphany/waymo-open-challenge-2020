"""EfficientDet Configurations

Adapted from official impl at https://github.com/google/automl/tree/master/efficientdet

TODO use a different config system, separate model from train specific hparams
"""

from omegaconf import OmegaConf


def default_detection_configs():
    """Returns a default detection configs."""
    h = OmegaConf.create()

    # model name.
    h.name = 'tf_efficientdet_d1'

    h.backbone_name = 'tf_efficientnet_b1'
    h.backbone_args = None  # FIXME sort out kwargs vs config for backbone creation

    # input preprocessing parameters
    h.image_size = 640
    h.input_rand_hflip = True
    h.train_scale_min = 0.1
    h.train_scale_max = 2.0
    h.autoaugment_policy = None

    # dataset specific parameters
    h.num_classes = 90
    h.skip_crowd_during_training = True

    # model architecture
    h.min_level = 3
    h.max_level = 7
    h.num_levels = h.max_level - h.min_level + 1
    h.num_scales = 3
    h.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    h.anchor_scale = 4.0
    h.pad_type = 'same'

    # For detection.
    h.box_class_repeats = 3
    h.fpn_cell_repeats = 3
    h.fpn_channels = 88
    h.separable_conv = True
    h.apply_bn_for_resampling = True
    h.conv_after_downsample = False
    h.conv_bn_relu_pattern = False
    h.use_native_resize_op = False
    h.pooling_type = None
    h.redundant_bias = True  # TF compatible models have back to back bias + BN layers

    # version.
    h.fpn_name = None
    h.fpn_config = None
    h.fpn_drop_path_rate = 0.  # No stochastic depth in default.

    # FIXME move config below this point to a different config, add hierarchy, or use args as I usually do?

    # optimization
    h.momentum = 0.9
    h.learning_rate = 0.08
    h.lr_warmup_init = 0.008
    h.lr_warmup_epoch = 1.0
    h.first_lr_drop_epoch = 5
    h.second_lr_drop_epoch = 10
    h.clip_gradients_norm = 10.0
    h.num_epochs = 15

    # regularization l2 loss.
    h.weight_decay = 4e-5

    # classification loss
    h.alpha = 0.25
    h.gamma = 1.5

    # localization loss
    h.delta = 0.1
    h.box_loss_weight = 50.0

    h.lr_decay_method = 'cosine'
    h.moving_average_decay = 0.9998
    h.ckpt_var_scope = None

    return h


efficientdet_model_param_dict = {
    'tf_efficientdet_d0':
        dict(
            name='efficientdet_d0',
            backbone_name='tf_efficientnet_b0',
            image_size=512,
            fpn_channels=64,
            fpn_cell_repeats=3,
            box_class_repeats=3,
            redundant_bias=True,
            backbone_args=dict(drop_rate=0.2, drop_path_rate=0.2),
        ),
    'tf_efficientdet_d1':
        dict(
            name='efficientdet_d1',
            backbone_name='tf_efficientnet_b1',
            image_size=640,
            fpn_channels=88,
            fpn_cell_repeats=4,
            box_class_repeats=3,
            redundant_bias=True,
            backbone_args=dict(drop_rate=0.2, drop_path_rate=0.2),
        ),
    'tf_efficientdet_d2':
        dict(
            name='efficientdet_d2',
            backbone_name='tf_efficientnet_b2',
            image_size=768,
            fpn_channels=112,
            fpn_cell_repeats=5,
            box_class_repeats=3,
            redundant_bias=True,
            backbone_args=dict(drop_rate=0.3, drop_path_rate=0.2),
        ),
    'tf_efficientdet_d3':
        dict(
            name='efficientdet_d3',
            backbone_name='tf_efficientnet_b3',
            image_size=896,
            fpn_channels=160,
            fpn_cell_repeats=6,
            box_class_repeats=4,
            redundant_bias=True,
            backbone_args=dict(drop_rate=0.3, drop_path_rate=0.2),
        ),
    'tf_efficientdet_d4':
        dict(
            name='efficientdet_d4',
            backbone_name='tf_efficientnet_b4',
            image_size=1024,
            fpn_channels=224,
            fpn_cell_repeats=7,
            box_class_repeats=4,
            redundant_bias=True,
            backbone_args=dict(drop_rate=0.4, drop_path_rate=0.2),
        ),
    'tf_efficientdet_d5':
        dict(
            name='efficientdet_d5',
            backbone_name='tf_efficientnet_b5',
            image_size=1280,
            fpn_channels=288,
            fpn_cell_repeats=7,
            box_class_repeats=4,
            redundant_bias=True,
            backbone_args=dict(drop_rate=0.4, drop_path_rate=0.2),
        ),
    'tf_efficientdet_d6':
        dict(
            name='efficientdet_d6',
            backbone_name='tf_efficientnet_b6',
            image_size=1280,
            fpn_channels=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
            redundant_bias=True,
            backbone_args=dict(drop_rate=0.5, drop_path_rate=0.2),
        ),
    'tf_efficientdet_d7':
        dict(
            name='efficientdet_d7',
            backbone_name='tf_efficientnet_b6',
            image_size=1536,
            fpn_channels=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            anchor_scale=5.0,
            fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
            redundant_bias=True,
            backbone_args=dict(drop_rate=0.5, drop_path_rate=0.2),
        ),
}


def get_efficientdet_config(model_name='efficientdet_d1'):
    """Get the default config for EfficientDet based on model name."""
    h = default_detection_configs()
    h.update(efficientdet_model_param_dict[model_name])
    return h


def bifpn_sum_config(base_reduction=8):
    """BiFPN config with sum."""
    p = OmegaConf.create()
    p.nodes = [
        {'reduction': base_reduction << 3, 'inputs_offsets': [3, 4]},
        {'reduction': base_reduction << 2, 'inputs_offsets': [2, 5]},
        {'reduction': base_reduction << 1, 'inputs_offsets': [1, 6]},
        {'reduction': base_reduction, 'inputs_offsets': [0, 7]},
        {'reduction': base_reduction << 1, 'inputs_offsets': [1, 7, 8]},
        {'reduction': base_reduction << 2, 'inputs_offsets': [2, 6, 9]},
        {'reduction': base_reduction << 3, 'inputs_offsets': [3, 5, 10]},
        {'reduction': base_reduction << 4, 'inputs_offsets': [4, 11]},
    ]
    p.weight_method = 'sum'
    return p


def bifpn_attn_config():
    """BiFPN config with fast weighted sum."""
    p = bifpn_sum_config()
    p.weight_method = 'attn'
    return p


def bifpn_fa_config():
    """BiFPN config with fast weighted sum."""
    p = bifpn_sum_config()
    p.weight_method = 'fastattn'
    return p


def get_fpn_config(fpn_name):
    if not fpn_name:
        fpn_name = 'bifpn_fa'
    name_to_config = {
        'bifpn_sum': bifpn_sum_config(),
        'bifpn_attn': bifpn_attn_config(),
        'bifpn_fa': bifpn_fa_config(),
    }
    return name_to_config[fpn_name]
