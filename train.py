# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import os
import argparse
import json
import re


from dataclasses import dataclass, field, asdict

@dataclass
class GConfig:

@dataclass
class DConfig:

@dataclass
class OptimizerConfig:
    pass

@dataclass
class GOptConfig(OptimizerConfig):
    pass

@dataclass
class DOptConfig(OptimizerConfig):
    pass

@dataclass
class RunConfig:
    run_dir: str = '.',  # Output directory.
    G_args = {},  # Options for generator network.
    D_args = {},  # Options for discriminator network.
    G_opt_args = {},  # Options for generator optimizer.
    D_opt_args = {},  # Options for discriminator optimizer.
    loss_args = {},  # Options for loss function.
    train_dataset_args = {},  # Options for dataset to train with.
    metric_dataset_args = {},  # Options for dataset to evaluate metrics against.
    augment_args = {},  # Options for adaptive augmentations.
    metric_arg_list = [],  # Metrics to evaluate during training.
    num_gpus = 1,  # Number of GPUs to use.
    minibatch_size = 32,  # Global minibatch size.
    minibatch_gpu = 4,  # Number of samples processed at a time by one GPU.
    G_smoothing_kimg = 10,  # Half-life of the exponential moving average (EMA) of generator weights.
    G_smoothing_rampup = None,  # EMA ramp-up coefficient.
    minibatch_repeats = 4,  # Number of minibatches to run in the inner loop.
    lazy_regularization = True,  # Perform regularization as a separate training step?
    G_reg_interval = 4,  # How often the perform regularization for G? Ignored if lazy_regularization=False.
    D_reg_interval = 16,  # How often the perform regularization for D? Ignored if lazy_regularization=False.
    total_kimg = 25000,  # Total length of the training, measured in thousands of real images.
    kimg_per_tick = 4,  # Progress snapshot interval.
    image_snapshot_ticks = 50,  # How often to save image snapshots? None = only save 'reals.png' and 'fakes-init.png'.
    network_snapshot_ticks = 50,  # How often to save network snapshots? None = only save 'networks-final.pkl'.
    resume_pkl = None,  # Network pickle to resume training from.
    abort_fn = None,  # Callback function for determining whether to abort training.
    progress_fn = None,  # Callback function for updating training progress.


@dataclass
class TrainSpec:
    """
    バッチサイズ等を定義するデータクラス
    """

    mb: int = field(init=False)  # バッチサイズ
    mbstd: int = field(init=False) # バッチサイズのstd?
    fmaps: int = field(init=False) # ?
    lrate: float = field(init=False) # 学習率
    gamma: float = field(init=False) # ?
    ema  : float = field(init=False) # ?

    def __post_init__(self, gpus: int, res: int):

        """
        :param gpus: GPUの数
        :param res: 生成画像の解像度
        :return:
        """
        self.mb = max(min(gpus * min(4096 // res, 32), 64), gpus)  # keep gpu memory consumption at bay
        self.mbstd = min(self.mb // gpus, 4)  # other hyperparams behave more predictably if mbstd group size remains fixed
        self.fmaps = 1 if res >= 512 else 0.5
        self.lrate = 0.002 if res >= 1024 else 0.0025
        self.gamma = 0.0002 * (res ** 2) / spec.mb  # heuristic formula
        self.ema = self.mb * 10 / 32

@dataclass
class AugSetting:
    """
    データ拡張の手法をまとめて管理するデータクラス
    """

    @dataclass
    class AugmentConfig:
        """
        データ拡張の手法を管理するデータクラス
        """

        # Blit
        xflip:    bool = False # x反転
        yflip:    bool = False # y反転
        rotate90: bool = False # 90度回転
        xint:     bool = False # x方向平行移動
        yint:     bool = False # y方向平行移動

        # Geom
        scale:    bool = False #
        rotate:   bool = False # 回転
        aniso:    bool = False # アス比率が保存しない拡大縮小
        xfrac:    bool = False #

        # Color
        brightness: bool = False # 明るさ
        contrast: bool = False # コントラスト変換
        lumaflip: bool = False # 白黒反転
        hue:      bool = False # RGB入れ替え
        saturation: bool = False #

        # Filter
        imgfilter: bool = False

        # Noise
        noise: bool = False

        # Cutout
        cutout: bool = False

    b:   AugmentConfig = AugmentConfig(xflip=True, yflip=True, rotate90=True, xint=True, yint=True)
    bg:  AugmentConfig = AugmentConfig(xflip=True, yflip=True, rotate90=True, xint=True, yint=True, scale=True, rotate=True, aniso=True, xfrac=True)
    bgc: AugmentConfig = AugmentConfig(xflip=True, yflip=True, rotate90=True, xint=True, yint=True, scale=True, rotate=True, aniso=True, xfrac=True,
                                       brightness=True, contrast=True, lumaflip=True, hue=True, saturation=True)

#----------------------------------------------------------------------------