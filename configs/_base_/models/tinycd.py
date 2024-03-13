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
    type='DIEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='TinyCD',
        in_channels=3,
        bkbn_name="efficientnet_b4",
        pretrained='/mnt/public/usr/wangmingze/opencd/pretrain_for_now/efficientnet_b4_rwightman-7eb33cd5.pth',
        output_layer_bkbn="3",
        freeze_backbone=False),
    decode_head=dict(
        type='IdentityHead',
        in_channels=1,
        in_index=-1,
        num_classes=2,
        out_channels=1, # support single class
        threshold=0.5,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))