# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))
model = dict(
    type='SiamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='./pretrain_for_now/resnet18_v1c-b5776b93.pth',
    backbone=dict(
        type='mmseg.ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(type='FeatureFusionNeck', policy='concat'),
    decode_head=dict(
        type='STAHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=96,
        sa_mode='PAM',
        sa_in_channels=256,
        sa_ds=1,
        distance_threshold=1,
        out_channels=1,
        threshold=0.5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='BCLLoss', margin=2.0, loss_weight=1.0, ignore_index=255)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))