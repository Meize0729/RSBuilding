'''
python tools/train.py configs_finetune/ViT_B/ViT_B_whu.py
'''

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/standard_512x512_foundationdataset.py',
    '../_base_/models/ViT_B.py',
    '../_base_/schedules/schedule_default.py',
]

# You can change dataloader parameters here
bs=2
gpu_nums = 8
bs_mult = 1
num_workers = 8
persistent_workers = True

# data_list path
train_data_list = 'data_list/whu/train.txt'
test_data_list = 'data_list/whu/test.txt'

# training schedule for pretrain
max_iters = 4e4
val_interval = 200
logger_interval = 20
base_lr = 0.0001 * (bs * gpu_nums / 16) * bs_mult # lr is related to bs*gpu_num, default 16-0.0001


# If you want to train with some backbone init, you must change the dir for your personal save dir path
# But I think you will use our pretrained weight, you may do not need backbone_checkpoint
backbone_checkpoint = None
load_from = 'the checkpoint path' # !!!! must change this !!!!
resume_from = None

# If you want to use wandb, make it to 1
wandb = 0

# You can define which dir want to save checkpoint and loggings
names = 'ViT_B_whu'
work_dir = '/mnt/public/usr/wangmingze/work_dir/finetune/' + names



""" ************************** model **************************"""
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=backbone_checkpoint) if backbone_checkpoint else None
    ),
    finetune_cfg=None, #
)


""" ************************** data **************************"""
train_dataset = dict(
    dataset=dict(
        data_list=train_data_list,
    )
)
train_dataloader = dict(
    batch_size=bs,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
)

val_dataloader = dict(
    batch_size=bs,
    num_workers=num_workers,
    persistent_workers=persistent_workers, 
    dataset=dict(
        data_list=test_data_list,
    )
)

""" ************************** schedule **************************"""
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=base_lr
        ),
    # backbone lr_mult = 0.01
    )

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=logger_interval, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=10, img_shape=(512, 512, 3)))

""" ************************** visualization **************************"""
if wandb:
    vis_backends = [dict(type='CDLocalVisBackend'),
                    dict(
                        type='WandbVisBackend',
                        save_dir=
                        '/mnt/public/usr/wangmingze/opencd/wandb/try2',
                        init_kwargs={
                            'entity': "wangmingze",
                            'project': "opencd_all_v4",
                            'name': names,}
                            )
                    ]
