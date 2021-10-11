import torchvision.transforms.functional as F


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max([w, h])

        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)

        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")
