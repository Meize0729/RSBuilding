_base_ = [
    '../_base_/models/fc_siam_conc.py',
    '../common/standard_512x512_40k_levircd.py']

wandb = 0
names = 'fc_siam_conc_512x512_40k_levircd'
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