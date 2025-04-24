from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import pandas as pd


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class FFHQAgeDataset(data.Dataset):
    """Dataset class for the FFHQ age editing dataset."""
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the FFHQ age dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.age_bins = selected_attrs  # Use the selected attributes as age bins
        self.preprocess()
        
        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the aligned dataset CSV file."""
        # Read the CSV file
        df = pd.read_csv(self.attr_path)
        
        # Get mapping from age group to index
        self.age_group_to_idx = {bin_name: i for i, bin_name in enumerate(self.age_bins)}
        
        # Shuffle the dataframe with fixed seed for reproducibility
        df = df.sample(frac=1, random_state=1234).reset_index(drop=True)
        
        # Split into train and test - use 10% for testing, max 2000 images
        # test_size = min(2000, int(len(df) * 0.1))  
        test_size = min(1, len(df))
        
        print(f"Total images in dataset: {len(df)}")
        print(f"Using {test_size} images for testing")
        
        for i, row in df.iterrows():
            img_num = int(row['image_number'])
            filename = f"{img_num:05d}.png"  # Format as 00001.png, 00002.png, etc.
            age_group = row['age_group_binned']  # Use the binned age group
            
            # Create one-hot encoding for age group
            label = [False] * len(self.age_bins)
            if age_group in self.age_group_to_idx:
                label[self.age_group_to_idx[age_group]] = True
            else:
                # Skip images with age groups not in our selected attributes
                continue
            
            # Split into train and test
            if i < test_size:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
        
        print(f'Finished preprocessing the FFHQ dataset...')
        print(f'Number of training images: {len(self.train_dataset)}')
        print(f'Number of testing images: {len(self.test_dataset)}')

    def __getitem__(self, index):
        """Return one image and its corresponding age bin label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)
    elif dataset == 'FFHQ':
        dataset = FFHQAgeDataset(image_dir, attr_path, selected_attrs, transform, mode)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader