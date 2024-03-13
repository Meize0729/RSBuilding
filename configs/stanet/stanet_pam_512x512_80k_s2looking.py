_base_ = [
    '../_base_/models/stanet_r18.py',
    '../common/standard_512x512_80k_s2looking.py']

crop_size = (512, 512)
model = dict(
    decode_head=dict(sa_mode='None'),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
    )

wandb = 0
names = 'stanet_pam_512x512_80k_s2looking_v3'
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
