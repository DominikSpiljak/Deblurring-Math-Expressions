import csv

from torchvision import transforms
from torch.utils import data
from PIL import Image

from data.transformations import SquarePad


def create_transforms(
    img_size,
    artificial_blur=False,
):
    image_transformations = [
        transforms.Lambda(lambd=SquarePad()),
        transforms.Resize(img_size),
    ]
    if artificial_blur:
        image_transformations.append(
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(2, 9))
        )

    tensor_transformations = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose([*image_transformations, *tensor_transformations])


def get_dataset(dataset_path, img_size):

    image_paths_train = []
    image_paths_val = []
    for dataset_csv in dataset_path.glob("**/*.csv"):
        with dataset_csv.open() as inp:
            reader = csv.reader(inp)
            for line in reader:
                if line[0].endswith("9"):
                    image_paths_val.append(str(dataset_csv.parents[0] / line[1]))
                else:
                    image_paths_train.append(str(dataset_csv.parents[0] / line[1]))

    blur_transformations = create_transforms(img_size, artificial_blur=True)
    no_blur_transformations = create_transforms(img_size, artificial_blur=False)
    return (
        ImageDataset(image_paths_train, blur_transformations, no_blur_transformations),
        ImageDataset(image_paths_val, blur_transformations, no_blur_transformations),
    )


class ImageDataset(data.Dataset):
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
            "non_blurred": self.blur_transformations(image),
        }
