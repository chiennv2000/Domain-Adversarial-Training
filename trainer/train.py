import sys, os
sys.path.append("..")

import torch
from torchvision import transforms, datasets
from trainer import Trainer
from dataset import MNDataset

src_data_name = '../data/MNIST'
tgt_data_name = '../data/mnist_m'

device = 'cuda'
batch_size = 32
lr = 1e-3
n_epochs = 100
image_size = 28


img_transform_src = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

img_transform_tgt = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

dataset_src = datasets.MNIST(
    root='data',
    train=True,
    transform=img_transform_src,
    download=True
)

dataloader_src = torch.utils.data.DataLoader(
    dataset=dataset_src,
    batch_size=batch_size,
    shuffle=True
)

dataset_target = MNDataset(
    data_root=os.path.join(tgt_data_name, 'mnist_m_train'),
    data_list=os.path.join(tgt_data_name, 'mnist_m_train_labels.txt'),
    transform=img_transform_tgt
)

dataloader_tgt = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True
)

my_trainer = Trainer(in_channels=3,
                     out_channels=50,
                     kernel_size=5,
                     n_classes=10,
                     device=device,
                     dataloader_source=dataloader_src,
                     dataloader_target=dataloader_tgt)

my_trainer.train(n_epochs=n_epochs, lr = lr)