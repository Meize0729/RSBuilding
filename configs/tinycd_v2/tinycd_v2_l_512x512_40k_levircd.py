_base_ = [
    '../_base_/models/tinycd_v2.py',
    '../common/standard_512x512_40k_levircd.py']


# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.003,
    betas=(0.9, 0.999),
    weight_decay=0.05)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)
model = dict(
    backbone=dict(
        type='TinyNet',
        arch='L',
        stem_stack_nums=4,
        widen_factor=1.2),
    neck=dict(
        type='TinyFPN',
        in_channels=[24, 32, 40, 56],
        out_channels=24,
        num_outs=4),
    decode_head=dict(
        type='TinyHead',
        in_channels=[48, 24, 24, 24, 24],
        channels=24,
        dropout_ratio=0.))

wandb = 0
names = 'tinycd_v2_l_512x512_40k_levircd'
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
else:
        vis_backends = [dict(type='CDLocalVisBackend'),]
        
visualizer = dict(
    type='CDLocalVisualizer', 
    vis_backends=vis_backends, name='visualizer', alpha=1.0)