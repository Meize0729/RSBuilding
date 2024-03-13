_base_ = './changer_ex_s50_512x512_80k_s2looking.py'

model = dict(backbone=dict(depth=101, stem_channels=128))


wandb = 0
names = 'changer_ex_s101_512x512_80k_s2looking_v2'
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
optimizer=dict(
    type='AdamW', lr=0.075, betas=(0.9, 0.999), weight_decay=0.05)
