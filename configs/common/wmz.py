_base_ = '../_base_/default_runtime.py'

dataset_type = 'FoundationDataset'

data_list = '/mnt/public/usr/wangmingze/opencd/data_list/data_list_all.txt'
data_type = 'all_label'

crop_size = (512, 512)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile_THREE'),
    dict(type='MultiImgMultiAnnLoadAnnotations_THREE'),
    dict(type='MultiImgResize', scale=[512, 512]),
    # dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=1),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    # dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs_THREE')
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile_THREE'),
    dict(type='MultiImgMultiAnnLoadAnnotations_THREE'),
    dict(type='MultiImgPackSegInputs_THREE')
]
img_ratios = [0.75, 1.0, 1.25]
tta_pipeline = [
    dict(type='MultiImgLoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='MultiImgResize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='MultiImgRandomFlip', prob=0., direction='horizontal'),
                dict(type='MultiImgRandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='MultiImgLoadAnnotations')],
            [dict(type='MultiImgPackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='Three_type_BatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_list=data_list,
        data_type=data_type,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='Three_type_BatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_list=data_list,
        data_type=data_type,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='Three_type_BatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_list=data_list,
        data_type=data_type,
        pipeline=test_pipeline))

val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = dict(
    type='mmseg.IoUMetric',
    iou_metrics=['mFscore', 'mIoU'])

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'picture_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'building_a_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'building_b_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'cd_embed': dict(lr_mult=1.0, decay_mult=0.0),
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))


# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000,
                    save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1, 
                       img_shape=(1024, 1024, 3)))