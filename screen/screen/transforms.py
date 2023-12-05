import albumentations as A

def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )
