_base_ = [
    '../_base_/models/changer_s50.py', 
    '../common/standard_512x512_40k_second.py']

model = dict(
    backbone=dict(
        interaction_cfg=(
            None,
            dict(type='SpatialExchange', p=1/2),
            dict(type='ChannelExchange', p=1/2),
            dict(type='ChannelExchange', p=1/2))
    ),
    decode_head=dict(
        num_classes=2,
        sampler=dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000)),
        # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
    )


# optimizer
optimizer=dict(
    type='AdamW', lr=0.005, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# compile = True # use PyTorch 2.x

wandb = 0
names = 'changer_ex_s50_512x512_40k_second'
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
