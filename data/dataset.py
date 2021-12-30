import csv
import logging

from PIL import Image
from torch.utils import data
from torchvision import transforms

from data.transformations import SquarePad


def create_transforms(img_size, sigmas=None, kernel_size=None, artificial_blur=False):
    image_transformations = [
        transforms.Lambda(lambd=SquarePad()),
        transforms.Resize(img_size),
    ]
    if artificial_blur:
        logging.info(
            "Blur parameters are: kernel_size = %s, sigma = %s", kernel_size, sigmas
        )
        image_transformations.append(
            transforms.GaussianBlur(
                kernel_size=kernel_size,
                sigma=sigmas,
            )
        )

    tensor_transformations = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose([*image_transformations, *tensor_transformations])


def collect_images(dataset_path, validation=True):
    image_paths_train = []
    image_paths_val = []
    for dataset_csv in dataset_path.glob("*.csv"):
        with dataset_csv.open() as inp:
            reader = csv.reader(inp)
            for line in reader:
                if line[0].endswith("9") and validation:
                    image_paths_val.append(str(dataset_csv.parents[0] / line[1]))
                else:
                    image_paths_train.append(str(dataset_csv.parents[0] / line[1]))
    if validation:
        return image_paths_train, image_paths_val
    else:
        return image_paths_train


def get_dataset_deblur(dataset_path, img_size, kernel_size=None, sigmas=None):
    image_paths_train, image_paths_val = collect_images(dataset_path, validation=True)

    blur_transformations = create_transforms(
        img_size, artificial_blur=True, kernel_size=kernel_size, sigmas=sigmas
    )
    no_blur_transformations = create_transforms(img_size, artificial_blur=False)
    return (
        DeblurImageDataset(
            image_paths_train, blur_transformations, no_blur_transformations
        ),
        DeblurImageDataset(
            image_paths_val, blur_transformations, no_blur_transformations
        ),
    )


def get_dataset_blur(blurred_dataset_path, non_blurred_dataset_path, img_size):
    non_blurred_image_paths_train, non_blurred_image_paths_val = collect_images(
        non_blurred_dataset_path, validation=True
    )
    blurred_image_paths_train = collect_images(blurred_dataset_path, validation=False)

    transformations = create_transforms(img_size, artificial_blur=False)
    return (
        CompositeBlurImageDataset(
            [
                BlurImageDataset(non_blurred_image_paths_train, transformations),
                BlurImageDataset(blurred_image_paths_train, transformations),
            ]
        ),
        BlurImageDataset(non_blurred_image_paths_val, transformations),
    )


class DeblurImageDataset(data.Dataset):
    def __init__(self, image_paths, blur_transformations, no_blur_transformations):
        super().__init__()
        self.image_paths = image_paths
        self.blur_transformations = blur_transformations
        self.no_blur_transformations = no_blur_transformations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        return {
            "blurred": self.blur_transformations(image),
            "non_blurred": self.no_blur_transformations(image),
        }


class BlurImageDataset(data.Dataset):
    def __init__(self, image_paths, transformations):
        super().__init__()
        self.image_paths = image_paths
        self.transformations = transformations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        return self.transformations(image)


class CompositeBlurImageDataset(data.Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        return min([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        return [dataset[index] for dataset in self.datasets]
