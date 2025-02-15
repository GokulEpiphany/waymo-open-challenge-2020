""" PyTorch EfficientDet model

Based on official Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
Paper: https://arxiv.org/abs/1911.09070

Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn
import logging
import math
from collections import OrderedDict
from typing import List
from timm import create_model
from timm.models.layers import create_conv2d, drop_path, create_pool2d, Swish

from .config.config import get_fpn_config

_DEBUG = False

_ACT_LAYER = Swish


class SequentialAppend(nn.Sequential):
    def __init__(self, *args):
        super(SequentialAppend, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]):
        for module in self:
            x.append(module(x))
        return x


class SequentialAppendLast(nn.Sequential):
    def __init__(self, *args):
        super(SequentialAppendLast, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]):
        for module in self:
            x.append(module(x[-1]))
        return x


class ConvBnAct2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding='', bias=False,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, act_layer=_ACT_LAYER):
        super(ConvBnAct2d, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)
        self.bn = None if norm_layer is None else norm_layer(out_channels, **norm_kwargs)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1, act_layer=_ACT_LAYER,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(SeparableConv2d, self).__init__()
        norm_kwargs = norm_kwargs or {}

        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)

        self.bn = None if norm_layer is None else norm_layer(out_channels, **norm_kwargs)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResampleFeatureMap(nn.Sequential):

    def __init__(self, in_channels, out_channels, reduction_ratio=1., pad_type='', pooling_type='max',
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, apply_bn=False, conv_after_downsample=False,
                 redundant_bias=False):
        super(ResampleFeatureMap, self).__init__()
        pooling_type = pooling_type or 'max'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.conv_after_downsample = conv_after_downsample

        conv = None
        if in_channels != out_channels:
            conv = ConvBnAct2d(
                in_channels, out_channels, kernel_size=1, padding=pad_type,
                norm_layer=norm_layer if apply_bn else None, norm_kwargs=norm_kwargs,
                bias=not apply_bn or redundant_bias, act_layer=None)

        if reduction_ratio > 1:
            stride_size = int(reduction_ratio)
            if conv is not None and not self.conv_after_downsample:
                self.add_module('conv', conv)
            self.add_module(
                'downsample',
                create_pool2d(
                    pooling_type, kernel_size=stride_size + 1, stride=stride_size, padding=pad_type))
            if conv is not None and self.conv_after_downsample:
                self.add_module('conv', conv)
        else:
            if conv is not None:
                self.add_module('conv', conv)
            if reduction_ratio < 1:
                scale = int(1 // reduction_ratio)
                self.add_module('upsample', nn.UpsamplingNearest2d(scale_factor=scale))

    # def forward(self, x):
    #     #  here for debugging only
    #     assert x.shape[1] == self.in_channels
    #     if self.reduction_ratio > 1:
    #         if hasattr(self, 'conv') and not self.conv_after_downsample:
    #             x = self.conv(x)
    #         x = self.downsample(x)
    #         if hasattr(self, 'conv') and self.conv_after_downsample:
    #             x = self.conv(x)
    #     else:
    #         if hasattr(self, 'conv'):
    #             x = self.conv(x)
    #         if self.reduction_ratio < 1:
    #             x = self.upsample(x)
    #     return x


class FpnCombine(nn.Module):
    def __init__(self, feature_info, fpn_config, fpn_channels, inputs_offsets, target_reduction, pad_type='',
                 pooling_type='max', norm_layer=nn.BatchNorm2d, norm_kwargs=None, apply_bn_for_resampling=False,
                 conv_after_downsample=False, redundant_bias=False, weight_method='attn'):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            in_channels = fpn_channels
            if offset < len(feature_info):
                in_channels = feature_info[offset]['num_chs']
                input_reduction = feature_info[offset]['reduction']
            else:
                node_idx = offset - len(feature_info)
                input_reduction = fpn_config.nodes[node_idx]['reduction']
            reduction_ratio = target_reduction / input_reduction
            self.resample[str(offset)] = ResampleFeatureMap(
                in_channels, fpn_channels, reduction_ratio=reduction_ratio, pad_type=pad_type,
                pooling_type=pooling_type, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                apply_bn=apply_bn_for_resampling, conv_after_downsample=conv_after_downsample,
                redundant_bias=redundant_bias)

        if weight_method == 'attn' or weight_method == 'fastattn':
            # WSM
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)
        else:
            self.edge_weights = None

    def forward(self, x):
        dtype = x[0].dtype
        nodes = []
        for offset in self.inputs_offsets:
            input_node = x[offset]
            input_node = self.resample[str(offset)](input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.type(dtype), dim=0)
            x = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.type(dtype))
            weights_sum = torch.sum(edge_weights)
            x = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)
        elif self.weight_method == 'sum':
            x = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        x = torch.sum(x, dim=-1)
        return x


class BiFpnLayer(nn.Module):
    def __init__(self, feature_info, fpn_config, fpn_channels, num_levels=5, pad_type='',
                 pooling_type='max', norm_layer=nn.BatchNorm2d, norm_kwargs=None, act_layer=_ACT_LAYER,
                 apply_bn_for_resampling=False, conv_after_downsample=True, conv_bn_relu_pattern=False,
                 separable_conv=True, redundant_bias=False):
        super(BiFpnLayer, self).__init__()
        self.fpn_config = fpn_config
        self.num_levels = num_levels
        self.conv_bn_relu_pattern = False

        self.feature_info = []
        self.fnode = SequentialAppend()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug('fnode {} : {}'.format(i, fnode_cfg))
            fnode_layers = OrderedDict()

            # combine features
            reduction = fnode_cfg['reduction']
            fnode_layers['combine'] = FpnCombine(
                feature_info, fpn_config, fpn_channels, fnode_cfg['inputs_offsets'], target_reduction=reduction,
                pad_type=pad_type, pooling_type=pooling_type, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                apply_bn_for_resampling=apply_bn_for_resampling, conv_after_downsample=conv_after_downsample,
                redundant_bias=redundant_bias, weight_method=fpn_config.weight_method)
            self.feature_info.append(dict(num_chs=fpn_channels, reduction=reduction))

            # after combine ops
            after_combine = OrderedDict()
            if not conv_bn_relu_pattern:
                after_combine['act'] = act_layer(inplace=True)
                conv_bias = redundant_bias
                conv_act = None
            else:
                conv_bias = False
                conv_act = act_layer
            conv_kwargs = dict(
                in_channels=fpn_channels, out_channels=fpn_channels, kernel_size=3, padding=pad_type,
                bias=conv_bias, norm_layer=norm_layer, norm_kwargs=norm_kwargs, act_layer=conv_act)
            after_combine['conv'] = SeparableConv2d(**conv_kwargs) if separable_conv else ConvBnAct2d(**conv_kwargs)
            fnode_layers['after_combine'] = nn.Sequential(after_combine)

            self.fnode.add_module(str(i), nn.Sequential(fnode_layers))

        self.feature_info = self.feature_info[-num_levels::]

    def forward(self, x):
        x = self.fnode(x)
        return x[-self.num_levels::]


class BiFpn(nn.Module):

    def __init__(self, config, feature_info, norm_layer=nn.BatchNorm2d, norm_kwargs=None, act_layer=_ACT_LAYER):
        super(BiFpn, self).__init__()
        self.config = config
        fpn_config = config.fpn_config or get_fpn_config(config.fpn_name)

        self.resample = SequentialAppendLast()
        for level in range(config.num_levels):
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                reduction = feature_info[level]['reduction']
            else:
                # Adds a coarser level by downsampling the last feature map
                reduction_ratio = 2
                self.resample.add_module(str(level), ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    pad_type=config.pad_type,
                    pooling_type=config.pooling_type,
                    norm_layer=norm_layer,
                    norm_kwargs=norm_kwargs,
                    reduction_ratio=reduction_ratio,
                    apply_bn=config.apply_bn_for_resampling,
                    conv_after_downsample=config.conv_after_downsample,
                    redundant_bias=config.redundant_bias,
                ))
                in_chs = config.fpn_channels
                reduction = int(reduction * reduction_ratio)
                feature_info.append(dict(num_chs=in_chs, reduction=reduction))

        self.cell = nn.Sequential()
        for rep in range(config.fpn_cell_repeats):
            logging.debug('building cell {}'.format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                pooling_type=config.pooling_type,
                norm_layer=norm_layer,
                norm_kwargs=norm_kwargs,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_bn_for_resampling=config.apply_bn_for_resampling,
                conv_after_downsample=config.conv_after_downsample,
                conv_bn_relu_pattern=config.conv_bn_relu_pattern,
                redundant_bias=config.redundant_bias,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x):
        assert len(self.resample) == self.config.num_levels - len(x)
        x = self.resample(x)
        x = self.cell(x)
        return x


class HeadNet(nn.Module):
    def __init__(self, config, num_outputs, norm_layer=nn.BatchNorm2d, norm_kwargs=None, act_layer=_ACT_LAYER):
        super(HeadNet, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.config = config
        num_anchors = len(config.aspect_ratios) * config.num_scales

        self.conv_rep = nn.ModuleList()
        self.bn_rep = nn.ModuleList()
        conv_kwargs = dict(
            in_channels=config.fpn_channels, out_channels=config.fpn_channels, kernel_size=3,
            padding=self.config.pad_type, bias=config.redundant_bias, act_layer=None, norm_layer=None)
        for i in range(config.box_class_repeats):
            conv = SeparableConv2d(**conv_kwargs) if config.separable_conv else ConvBnAct2d(**conv_kwargs)
            self.conv_rep.append(conv)

            bn_levels = []
            for _ in range(config.num_levels):
                bn_seq = nn.Sequential()
                bn_seq.add_module('bn', norm_layer(config.fpn_channels, **norm_kwargs))
                bn_levels.append(bn_seq)
            self.bn_rep.append(nn.ModuleList(bn_levels))

        self.act = act_layer(inplace=True)

        predict_kwargs = dict(
            in_channels=config.fpn_channels, out_channels=num_outputs * num_anchors, kernel_size=3,
            padding=self.config.pad_type, bias=True, norm_layer=None, act_layer=None)
        if config.separable_conv:
            self.predict = SeparableConv2d(**predict_kwargs)
        else:
            self.predict = ConvBnAct2d(**predict_kwargs)

    def forward(self, x):
        outputs = []
        for level in range(self.config.num_levels):
            x_level = x[level]
            for i in range(self.config.box_class_repeats):
                x_level_ident = x_level
                x_level = self.conv_rep[i](x_level)
                x_level = self.bn_rep[i][level](x_level)
                x_level = self.act(x_level)
                if i > 0 and self.config.fpn_drop_path_rate:
                    x_level = drop_path(x_level, self.config.fpn_drop_path_rate, self.training)
                    x_level += x_level_ident
            outputs.append(self.predict(x_level))
        return outputs


def _init_weight(m, n='', ):
    """ Weight initialization as per Tensorflow official implementations.
    """

    def _fan_in_out(w, groups=1):
        dimensions = w.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        num_input_fmaps = w.size(1)
        num_output_fmaps = w.size(0)
        receptive_field_size = 1
        if w.dim() > 2:
            receptive_field_size = w[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        fan_out //= groups
        return fan_in, fan_out

    def _glorot_uniform(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1., (fan_in + fan_out) / 2.)  # fan avg
        limit = math.sqrt(3.0 * gain)
        w.data.uniform_(-limit, limit)

    def _variance_scaling(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1., fan_in)  # fan in
        # gain /= max(1., (fan_in + fan_out) / 2.)  # fan

        # should it be normal or trunc normal? using normal for now since no good trunc in PT
        # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        # std = math.sqrt(gain) / .87962566103423978
        # w.data.trunc_normal(std=std)
        std = math.sqrt(gain)
        w.data.normal_(std=std)

    if isinstance(m, SeparableConv2d):
        if 'box_net' in n or 'class_net' in n:
            _variance_scaling(m.conv_dw.weight, groups=m.conv_dw.groups)
            _variance_scaling(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                if 'class_net.predict' in n:
                    m.conv_pw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_pw.bias.data.zero_()
        else:
            _glorot_uniform(m.conv_dw.weight, groups=m.conv_dw.groups)
            _glorot_uniform(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                m.conv_pw.bias.data.zero_()
    elif isinstance(m, ConvBnAct2d):
        if 'box_net' in n or 'class_net' in n:
            m.conv.weight.data.normal_(std=.01)
            if m.conv.bias is not None:
                if 'class_net.predict' in n:
                    m.conv.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv.bias.data.zero_()
        else:
            _glorot_uniform(m.conv.weight)
            if m.conv.bias is not None:
                m.conv.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        # looks like all bn init the same?
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


class EfficientDet(nn.Module):

    def __init__(self, config, norm_kwargs=None, pretrained_backbone=True):
        super(EfficientDet, self).__init__()
        norm_kwargs = norm_kwargs or dict(eps=.001, momentum=.01)
        self.backbone = create_model(
            config.backbone_name, features_only=True, out_indices=(2, 3, 4),
            pretrained=pretrained_backbone, **config.backbone_args)
        feature_info = [dict(num_chs=f['num_chs'], reduction=f['reduction'])
                        for i, f in enumerate(self.backbone.feature_info())]
        self.fpn = BiFpn(config, feature_info, norm_kwargs=norm_kwargs)
        self.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=norm_kwargs)
        self.box_net = HeadNet(config, num_outputs=4, norm_kwargs=norm_kwargs)

        for n, m in self.named_modules():
            if 'backbone' not in n:
                _init_weight(m, n)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x_class = self.class_net(x)
        x_box = self.box_net(x)
        return x_class, x_box
