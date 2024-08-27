import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torch.optim as optim

from ..Utils import get_config_files, print_progress
from ..Data.custom_dataset_generator import CustomDatasetGenerator
from ..Generator.generator import Generator


class PreTrainer:
    def __init__(self, generator: Generator, model_name='generator'):
        pretrain_config = get_config_files(key='pretrain')
        self.save_images = bool(pretrain_config['save_images'])
        if self.save_images:
            self.image_save_path = pretrain_config['image_save_path']
            os.makedirs(self.image_save_path, exist_ok=True)
        self.losses = []
        self.model_name = model_name
        self.criteria = nn.MSELoss()
        self.generator = generator.cuda()
        self.lr = float(pretrain_config['lr'])
        self.optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)

        self.epochs = int(pretrain_config['num_epochs'])

    def __call__(self):
        self.train()
        return self.get_model()

    def __str__(self):
        return "PreTrainer class for training the generator model"

    def __repr__(self):
        return "PreTrainer(generator=Generator())"

    def train(self, data_loader):
        self.generator.train()
        for epoch in range(self.epochs):
            for i, (imgs_lr, imgs_hr) in enumerate(train_loader):
                imgs_lr = imgs_lr.cuda()
                imgs_hr = imgs_hr.cuda()

                self.optimizer.zero_grad()

                gen_hr = self.generator(imgs_lr)

                loss = self.criteria(gen_hr, imgs_hr)
                self.losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                print_progress(epoch+1, self.epochs, i,
                               len(data_loader), loss.item())

            if self.save_images:
                save_image(gen_hr.data, f'''{
                    self.image_save_path}/generator_{epoch}.png''')

    def get_model(self):
        return self.generator

    def get_losses(self):
        return self.losses

    def get_optimizer(self):
        return self.optimizer

    def get_criteria(self):
        return self.criteria

    def save_model(self, path):
        torch.save(self.generator.state_dict(), path)

    def save_model_script(self, path):
        model_scripted = torch.jit.script(self.generator)
        model_scripted.save(os.path.join(path, f'{self.model_name}.pt'))


if __name__ == '__main__':
    dataset = CustomDatasetGenerator()
    train_loader, _ = dataset.dataset_generator()
    g = Generator()
    pre_trainer = PreTrainer(g)
    pre_trainer.train(train_loader)
    pre_trainer.save_model_script('src')
    # sys.stdout.write(pre_trainer.train(train_loader))
    # torch.save(g.state_dict(), 'generator.pth')
