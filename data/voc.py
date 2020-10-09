import torch.utils.data as data
from PIL import Image
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.image import MaskToTensor
import numpy as np


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None):

        self.root = root
        self.year = year

        self.transform = transform
        self.target_transform = target_transform
        self.image_set = image_set
        base_dir = 'VOC2012'
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        raw_img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transform is not None:
            img = self.transform(raw_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # maybe fix this step for faster speed
        raw_img = np.array(raw_img).astype(np.uint8)

        return img, target, raw_img

    def __len__(self):
        return len(self.images)

def _give_val_loader(root_path = './'):

    # Create transforms
    input_transform = transforms.Compose([ # No need to resize here in segmentation tasks
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # resnet normalizing values
    ])

    target_transform = MaskToTensor()

    val_data = VOCSegmentation(root_path,
                           image_set='val',
                           transform=input_transform,
                           target_transform=target_transform
                           )

    # Create data loader
    val_loader = DataLoader(val_data, batch_size=1, num_workers=4, shuffle=True)

    return val_loader