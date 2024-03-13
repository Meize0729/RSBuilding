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
        type='ViTSAM_Normal',
        arch='large',
        img_size=512,
        patch_size=16,
        out_channels=-1,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=7,
        drop_path_rate=0.125,
        # out_indices=[5,11,17,23],
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='pretrain/sam_vit_l_mm_allin.pth'),
    ),
    decode_head=dict(
        type='Foundation_Decoder_v1',
        # in_channels=1024,
        in_channels=1024,
        out_channels=256,
        drop=0.0,
        loss_type='BCELoss',
        loss_weight=[1,1,1],
    ),
    train_cfg=dict(),
    test_cfg=dict(),)



find_unused_parameters=True