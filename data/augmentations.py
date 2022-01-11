import albumentations
from albumentations.augmentations.transforms import (
    Blur,
    GaussianBlur,
    MotionBlur,
    MedianBlur,
)


class Albumentations:
    def __init__(self):
        self.augmentations = augmentations = albumentations.Compose(
            [
                MotionBlur(p=0.3, blur_limit=(7, 15)),
                Blur(p=0.3, blur_limit=(7, 15)),
                GaussianBlur(p=0.3, blur_limit=(7, 15)),
                MedianBlur(p=0.3, blur_limit=(7, 15)),
            ]
        )

    def __call__(self, image):
        return self.augmentations(image=image)["image"]
