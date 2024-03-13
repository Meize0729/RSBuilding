data_preprocessor = dict(
    type='FoundationInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='FoundationEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmseg.SwinTransformer',
        # init_cfg=dict(type='Pretrained', checkpoint='/mnt/public/usr/wangmingze/pretrain/swin_B_window12_imagenet22k_pre384_mmseg.pth'),
        pretrain_img_size=384,

        # # tiny
        # embed_dims=96,
        # depths=[2, 2, 6, 2],
        # num_heads=[3, 6, 12, 24],

        # base
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],

        # # large
        # embed_dims=192,
        # depths=[2, 2, 18, 2],
        # num_heads=[6, 12, 24, 48],

        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        ),
    decode_head=dict(
        type='Foundation_Decoder_swin_v1',
        in_channels=[128, 256, 512, 1024], 
        out_channels=256,
        drop=0.0,
        loss_type='BCELoss',
        loss_weight=[1,1,1],
    ),
    train_cfg=dict(),
    test_cfg=dict(),)



find_unused_parameters=True
