_base_ = [
    '../_base_/models/snunet_c16.py',
    '../common/standard_512x512_40k_levircd.py']

base_channels = 32
model = dict(
    backbone=dict(base_channel=base_channels),
    decode_head=dict(
        in_channels=base_channels * 4,
        channels=base_channels * 4))

wandb = 0
names = 'snunet_c32_512x512_40k_levircd'
work_dir = '/mnt/public/usr/wangmingze/work_dir/CD_others/' + names
find_unused_parameters=True

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
