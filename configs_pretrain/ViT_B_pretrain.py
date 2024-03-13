'''
CUDA_VISBILE_DEIVCES=0,1,2,3,4,5,6,7 bash tools/dist_train.sh configs_pretrain/ViT_B_pretrain.py 8
python tools/train.py configs_pretrain/ViT_B_pretrain.py
'''

_base_ = [
    './_base_/default_runtime.py',
    './_base_/datasets/standard_512x512_foundationdataset.py',
    './_base_/models/ViT_B.py',
    './_base_/schedules/schedule_default.py',
]

# You can change dataloader parameters here
bs=2
gpu_nums = 8
bs_mult = 1
num_workers = 8
persistent_workers = True

# data_list path
train_data_list = 'data_list/pretrain/train.txt'
test_data_list = 'data_list/pretrain/test.txt'

# training schedule for pretrain
max_iters = 40e4
val_interval = 1000
base_lr = 0.0001 * (bs * gpu_nums / 16) * bs_mult # lr is related to bs*gpu_num, default 16 - 0.0001


# If you want to train with some backbone init, you must change the dir for your personal save dir path
# But I think you will use our pretrained weight, you may do not need backbone_checkpoint
backbone_checkpoint = 'pretrain/sam_vit_b_mm_allin.pth'
# load_from = 'the checkpoint path' # !!!! must change this !!!!
resume_from = None

# If you want to use wandb, make it to 1
wandb = 0

# You must change which dir want to save checkpoint and loggings
names = 'ViT_B'
work_dir = '/mnt/public/usr/wangmingze/work_dir/opencd_terminal/pretrain/' + names



""" ************************** model **************************"""
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=backbone_checkpoint) if backbone_checkpoint else None
    )
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
        lr=base_lr
    )
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=2000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=2000,
        end=max_iters,
        eta_min=base_lr*0.5,
        by_epoch=False,
    )
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval),
)

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
                            'name': names
                        }
                    )
    ]
else:
    vis_backends = [dict(type='CDLocalVisBackend')]

