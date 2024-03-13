max_iters = 4e4
base_lr = 0.0001
bs_mult = 1
val_interval = 200
logger_interval = 20

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.01, decay_mult=1.0),
            'building_a_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'building_b_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'cd_embed': dict(lr_mult=1.0, decay_mult=0.0),
        },
        norm_decay_mult=0.0),
    accumulative_counts = bs_mult,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    )

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1000,
        end=max_iters,
        eta_min=1e-7,
        by_epoch=False,
    )
]


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