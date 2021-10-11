import csv

from torchvision import transforms
from torch.utils import data
from PIL import Image

from data.transformations import SquarePad


def create_transforms(
    img_size,
):
    return transforms.Compose(
        [
            transforms.Lambda(lambd=SquarePad()),
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]
    )


def get_dataset(dataset_path, img_size, label=None):

    image_paths = []
    for dataset_csv in dataset_path.glob("**/*.csv"):
        with dataset_csv.open() as inp:
            reader = csv.reader(inp)
            for line in reader:
                if label is None or line[-1] == label:
                    image_paths.append(str(dataset_path / line[0]))

    transformations = create_transforms(img_size)
    return ImageDataset(image_paths, transformations)


class ImageDataset(data.Dataset):
    def __init__(self, image_paths, transformations):
        super().__init__()
        self.image_paths = image_paths
        self.transformations = transformations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        return self.transformations(image)
