_base_ = [
    '../_base_/models/tinycd.py', 
    '../common/standard_512x512_120k_bandon_cd.py'
    # '../common/wmz.py'
    ]

crop_size = (512, 512)
model = dict(
    decode_head=dict(num_classes=2, out_channels=1),
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2))
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00356799066427741,
    betas=(0.9, 0.999),
    weight_decay=0.009449677083344786)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)

wandb = 0
names = 'tinycd_512x512_120k_bandon_cd'
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
