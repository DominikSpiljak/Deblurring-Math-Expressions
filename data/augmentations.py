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
                MotionBlur(p=0.9, blur_limit=(13, 21)),
                Blur(p=0.9, blur_limit=(13, 21)),
                GaussianBlur(p=0.9, blur_limit=(13, 21)),
                MedianBlur(p=0.9, blur_limit=(13, 21)),
            ]
        )

    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.augmentations(image=image_np)["image"]
        return Image.fromarray(augmented)
