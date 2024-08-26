import os
import glob
from torch.utils.data import Dataset
from PIL import Image


class DIV2kDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform_lr=None, transform_hr=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
        self.lr_images = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

        # Debugging: Print the number of images found
        print(f'Found {len(self.hr_images)} HR images in {hr_dir}')
        print(f'Found {len(self.lr_images)} LR images in {lr_dir}')

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, index):
        hr_image = Image.open(self.hr_images[index])
        lr_image = Image.open(self.lr_images[index])

        if self.transform_lr:
            lr_image = self.transform_lr(lr_image)
        if self.transform_hr:
            hr_image = self.transform_hr(hr_image)

        return lr_image, hr_image
