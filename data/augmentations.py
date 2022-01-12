import albumentations
from PIL import Image
import numpy as np
from albumentations.augmentations.transforms import (
    Blur,
    GaussianBlur,
    MotionBlur,
    MedianBlur,
)


class Albumentations:
    def __init__(self):
        self.augmentations = albumentations.Compose(
            [
                MotionBlur(p=0.7, blur_limit=(7, 59)),
                Blur(p=0.7, blur_limit=(7, 59)),
                GaussianBlur(p=0.7, blur_limit=(7, 59)),
                MedianBlur(p=0.7, blur_limit=(7, 59)),
            ]
        )

    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.augmentations(image=image_np)["image"]
        return Image.fromarray(augmented)
