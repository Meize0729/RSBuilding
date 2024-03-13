_base_ = [
    '../_base_/models/hanet.py',
    '../common/standard_512x512_40k_levircd.py']

wandb = 0
names = 'hanet_512x512_40k_levircd'
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
