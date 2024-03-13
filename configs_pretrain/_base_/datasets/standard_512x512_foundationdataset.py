bs=4
num_workers = 8
persistent_workers = True

train_data_list = '/mnt/public/usr/wangmingze/opencd/data_list/data_list_all.txt'
test_data_list = '/mnt/public/usr/wangmingze/opencd/data_list/test_data_4.txt'

crop_size = (512, 512)

''' **************************** train ****************************'''
train_pipeline = [
    dict(type='RandomMosaic_Modified', img_scale=(512, 512), center_ratio_range=(0.25, 0.75), prob=0.5, ignore_type=[]),
    # dict(type='MultiImgRandomCrop_Modified', crop_size=crop_size, cat_max_ratio=1),
    dict(type='MultiImgPackSegInputs_Modified')
]
train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset_Modified',
    dataset=dict(
        type='FoundationDataset',
        data_list=train_data_list,
        pipeline=[
            dict(type='MultiImgLoadImageFromFile_Modified'),
            dict(type='MultiImgMultiAnnLoadAnnotations_Modified'),
            dict(type='MultiImgRandomResize_Modified', ratio_range=(0.8, 2), prob=0.1),
            dict(type='MultiImgRandomCrop_Modified', crop_size=(512, 512), cat_max_ratio=0.95),
            dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
            dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
            dict(type='MultiImgExchangeTime', prob=0.1),
            dict(
                type='MultiImgPhotoMetricDistortion',
                brightness_delta=10,
                contrast_range=(0.8, 1.2),
                saturation_range=(0.8, 1.2),
                hue_delta=10),
        ],
        backend_args=None),
    pipeline=train_pipeline)
train_dataloader = dict(
    batch_size=bs,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    batch_sampler=dict(type='BatchSampler_Modified'),
    dataset=train_dataset)

''' **************************** val and test ****************************'''
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile_Modified'),
    dict(type='MultiImgMultiAnnLoadAnnotations_Modified'),
    dict(type='MultiImgPackSegInputs_Modified')
]
val_dataloader = dict(
    batch_size=bs,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='BatchSampler_Modified'),
    dataset=dict(
        type='FoundationDataset',
        data_list=test_data_list,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoU_Base_Metric_Modified', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = val_evaluator
