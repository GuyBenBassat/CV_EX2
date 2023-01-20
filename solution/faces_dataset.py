"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

REAL_IMG_LBL = 0
FAKE_IMG_LBL = 1

class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform
        self.real_image_len = len(self.real_image_names)
        self.fake_image_len = len(self.fake_image_names)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        if index >= self.real_image_len:
            index = index - self.real_image_len
            img_path = self.root_path + '/fake/' + self.fake_image_names[index]
            image =  Image.open(img_path)
            label = FAKE_IMG_LBL
        else:
            img_path = self.root_path + '/real/' +self.real_image_names[index]
            image =  Image.open(img_path)
            label = REAL_IMG_LBL
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """Return the number of images in the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        return self.real_image_len + self.fake_image_len
