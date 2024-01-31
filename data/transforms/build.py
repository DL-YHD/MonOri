from . import transforms as T # 相对引用

# from data import transforms as T # 绝对引用

def build_transforms(cfg, is_train=True):
    to_bgr = cfg.INPUT.TO_BGR

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr=to_bgr
    )

    transform = T.Compose(
        [
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
