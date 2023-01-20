"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


#CONSTATNS
REAL_LABEL = 0
FAKE_LABEL = 1
IMG_NAME_INDEX = 0
IMG_LABEL_INDEX = 1
REAL_DIRECTORY = 'real'
FAKE_DIRECTORY = 'fake'


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
        self.real_images_size = len(self.real_image_names)
        self.fake_images_size = len(self.fake_image_names)
        self.transform = transform

    def __getitem__(self, index):
        """Get a sample and label from the dataset.""" 
        if index >= len(self.real_image_names):
            label = FAKE_LABEL
            directory_name = FAKE_DIRECTORY
            image_name = self.fake_image_names[index - self.real_images_size]   #get relative index in the fake directory
        else:
            label = REAL_LABEL
            directory_name = REAL_DIRECTORY
            image_name = self.real_image_names[index]
        img_path = os.path.join(self.root_path, directory_name, image_name)
        image = Image.open(img_path)
        if self.transform is not None:          
          image = self.transform(image)
        return image, label
                     

    def __len__(self):
        """Return the number of images in the dataset."""        
        return self.real_images_size + self.fake_images_size        
