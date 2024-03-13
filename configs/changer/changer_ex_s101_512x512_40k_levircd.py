_base_ = './changer_ex_s50_512x512_40k_levircd.py'

model = dict(backbone=dict(depth=101, stem_channels=128))


wandb = 0
names = 'changer_ex_s101_512x512_40k_levircd'
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