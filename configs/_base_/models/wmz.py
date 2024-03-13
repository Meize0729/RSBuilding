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
    type='ThreeEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
            type='ViTSAM',
            arch='base',
            img_size=512,
            patch_size=16,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='pretrain/sam_vit_b_mm_new.pth'),
        ),
    decode_head=dict(
        type='Foundation_Decoder',
        channels=256,
        out_channels=256,
        patch_size=16,
        img_size=512, # support single class
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # DetrTransformerDecoder
            num_layers=6,
            layer_cfg=dict(  # DetrTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True))),
        ),
        upsample_decoder=dict(               
                 channels=256,
                 upsample_time=4,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU'),),
                 ),
    # model training and testing settings
    train_cfg='cp building',
    test_cfg=dict(),
)