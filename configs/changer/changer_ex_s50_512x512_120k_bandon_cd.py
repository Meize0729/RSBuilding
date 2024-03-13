_base_ = [
    '../_base_/models/changer_s50.py', 
    '../common/standard_512x512_120k_bandon_cd.py']

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
    type='DIEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='IA_ResNeSt',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        stem_channels=64,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        interaction_cfg=(
            None,
            dict(type='SpatialExchange', p=1/2),
            dict(type='ChannelExchange', p=1/2),
            dict(type='ChannelExchange', p=1/2)
            ),
        ),
    decode_head=dict(
        type='Changer',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        sampler=dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



# compile = True # use PyTorch 2.x
optimizer=dict(
    type='AdamW', lr=0.002, betas=(0.9, 0.999), weight_decay=0.05)


wandb = 0
names = 'changer_ex_s50_512x512_120k_bandon_cd'
work_dir = '/mnt/public/usr/wangmingze/work_dir/CD_others/' + names


if wandb:
    vis_backends = [dict(type='CDLocalVisBackend'),
                    dict(
                        type='WandbVisBackend',
                        save_dir=
                        '/mnt/public/usr/wangmingze/opencd/wandb/try',
                        init_kwargs={
                            'entity': "wangmingze",
                            'project': "opencd_2",
                            'name': names,}
                            )
                    ]