import sys
import torch
import torch.nn as nn
import torch.optim as optim

from ..Utils import get_config_files, print_progress
from ..Data.custom_dataset_generator import CustomDatasetGenerator
from ..Generator.generator import Generator


class PreTrainer:
    def __init__(self, generator: Generator):
        pretrain_config = get_config_files(key='pretrain')
        self.losses = []
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

    def get_model(self):
        return self.generator

    def get_losses(self):
        return self.losses

    def get_optimizer(self):
        return self.optimizer

    def get_criteria(self):
        return self.criteria


if __name__ == '__main__':
    dataset = CustomDatasetGenerator()
    train_loader, _ = dataset.dataset_generator()
    g = Generator()
    pre_trainer = PreTrainer(g)
    sys.stdout.write(pre_trainer.train(train_loader))
    torch.save(g.state_dict(), 'generator.pth')
