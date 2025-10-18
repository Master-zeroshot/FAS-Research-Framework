exp_num = 0

dataset = "celeba_spoof"

multi_task_learning = True

evaluation = True

test_steps = None

random_seed = 42

datasets = dict(
    Celeba_root="../CelebA-Spoof-zips/CelebA_Spoof",
    Casia_root="./CASIA",
    LCCFASD_root="../LCC_FASD",
)

img_norm_cfg = dict(mean=[0.5931, 0.4690, 0.4229], std=[0.2471, 0.2214, 0.2157])

optimizer = dict(lr=0.005, momentum=0.9, weight_decay=5e-4)

scheduler = dict(milestones=[20, 50], gamma=0.2)

data = dict(batch_size=64, data_loader_workers=8, sampler=True, pin_memory=True)

resize = dict(height=224, width=224)

checkpoint = dict(
    snapshot_name="mobilenetv4_small_224_antispoof.pth.tar",
    experiment_path="./logs_mobilenetv4_small",
)

loss = dict(
    loss_type="amsoftmax",
    amsoftmax=dict(
        m=0.5,
        s=1,
        margin_type="cross_entropy",
        label_smooth=False,
        smoothing=0.1,
        ratio=[1, 1],
        gamma=0,
    ),
)

epochs = dict(start_epoch=0, max_epoch=71)

model = dict(
    model_type="Mobilenet4",
    model_size="small",
    width_mult=1.0,
    pretrained=False,  # Set to False if pretrained weights not available
    embeding_dim=1024,
    imagenet_weights="./pretrained/mobilenetv4_small_imagenet.pth.tar",
)

aug = dict(type_aug=None, alpha=0.5, beta=0.5, aug_prob=0.7)

curves = dict(
    det_curve="det_curve_mobilenetv4_small.png",
    roc_curve="roc_curve_mobilenetv4_small.png",
)

dropout = dict(prob_dropout=0.2, classifier=0.3, type="gaussian", mu=0.5, sigma=0.3)

RSC = dict(use_rsc=False, p=0.333, b=0.333)

test_dataset = dict(type="LCC_FASD")

conv_cd = dict(theta=0)

test_file_name = "test_mobilenetv4_small"
