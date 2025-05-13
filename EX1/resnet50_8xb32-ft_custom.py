_base_ = [
    '../_base_/models/resnet50.py',           # 模型设置
    '../_base_/datasets/imagenet_bs32.py',    # 数据设置
    '../_base_/schedules/imagenet_bs256.py',  # 训练策略设置
    '../_base_/default_runtime.py',           # 运行设置
]

# 模型设置
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=5),
)

# 数据设置
data_root = '../data'
train_dataloader = dict(
    batch_size=256,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',
        data_prefix='train',

    ))
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',
        data_prefix='val',

    ))
test_dataloader =None
test_cfg=None

# 训练策略设置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)