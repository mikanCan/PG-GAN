import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DL
from torch.utils.data import Dataset as dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy
import torch
import glob


class DataLoader(object):
    def __init__(self, config, dataset):
        self.config = config
        self.batch_table = config.batch_table
        self.im_size = config.im_size
        self.batch_size = int(self.batch_table[self.im_size])
        self.num_workers = config.num_workers
        self.root = config.train_data_root

        self.dataset = dataset
        self.root = dataset.root

    def renew(self, resl):
        print('[*] Renew data_loader configuration, load image from {}.'.format(self.root))
        self.batch_size = int(self.batch_table[pow(2, resl)])
        self.im_size = int(pow(2, resl))
        print('[*] batch_size = {} , im_size = ({},{})'.format(self.batch_size, self.im_size, self.im_size))
        self.dataset.transforms = transforms.Compose([
            transforms.Resize(size=(self.im_size, self.im_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        print('[*] transforms = {})'.format(self.dataset.transforms))
        self.data_loader = DL(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def __iter__(self):
        return iter(self.data_loader)

    def __next__(self):
        return next(self.data_loader)

    def __len__(self):
        return len(self.data_loader.dataset)

    def get_batch(self):
        dataIter = iter(self.data_loader)
        return next(dataIter)  # pixel range [-1, 1]


class IdentityDataset(dataset):
    def __init__(self, config):
        self.config = config
        self.root = self.config.identity_image_root

        self.images = glob.glob(self.root)
        self.latent_dim = self.config.nz
        self.identity_vec = []
        self.gen_identity_vec()
        self.transforms = transforms.Compose([
            transforms.Resize(size=(self.config.im_size, self.config.im_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def gen_identity_vec(self):
        y = []
        for i in range(len(self.images)):
            base = numpy.ones(self.latent_dim) * (-1)
            base[i] = 1
            y.append(base)
        self.identity_vec = y

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = image.convert('RGB')
        image = self.transforms(image).float()
        return image, torch.Tensor([self.identity_vec[index]])


class TrainingDataset(dataset):
    def __init__(self, config):
        self.config = config
        self.root = self.config.train_data_root
        self.images = glob.glob(self.root)
        self.transforms = transforms.Compose([
            transforms.Resize(size=(self.config.im_size, self.config.im_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = image.convert('RGB')
        image = self.transforms(image).float()
        return image
