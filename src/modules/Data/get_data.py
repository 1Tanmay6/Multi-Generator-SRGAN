from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from .div2k_dataset import DIV2kDataset
from ..Utils import get_config_files


class CustomDatasetGenerator:
    def __init__(self):
        data_config = get_config_files(key='dataset')
        self.hr_train_dir = str(data_config['hr_train_dir'])
        self.lr_train_dir = str(data_config['lr_train_dir'])
        self.hr_valid_dir = str(data_config['hr_valid_dir'])
        self.lr_valid_dir = str(data_config['lr_valid_dir'])
        self.batch_size = int(data_config['batch_size'])
        self.num_workers = int(data_config['num_workers'])
        self.shuffle = bool(data_config['shuffle'])
        self.input_size = int(data_config['input_size'])
        self.output_size = int(data_config['output_size'])

    def get_transformations(self, key='lr'):
        size = self.input_size if key == 'lr' else self.output_size
        transformations = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transformations

    def dataset_generator(self):
        train_dataset = DIV2kDataset(hr_dir=self.hr_train_dir, lr_dir=self.lr_train_dir,
                                     transform_lr=self.get_transformations(
                                         key='lr'),
                                     transform_hr=self.get_transformations(key='hr'))
        valid_dataset = DIV2kDataset(hr_dir=self.hr_valid_dir, lr_dir=self.lr_valid_dir,
                                     transform_lr=self.get_transformations(
                                         key='lr'),
                                     transform_hr=self.get_transformations(key='hr'))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                  shuffle=self.shuffle)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                  shuffle=self.shuffle)

        return train_loader, valid_loader
