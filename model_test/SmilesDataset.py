import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from PIL import Image
from tqdm import tqdm

class SmilesDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        
        self.h = h5py.File('./backup/train_images.hdf5', 'r') #('/media/ksk/Backup/SMILES dataset/images.hdf5', 'r')
        #key = unicode('Annotation', "utf-8")
        #str(text, 'utf-8')
        #key = str('images', 'utf-8')
        self.imgs = self.h['images']

        with open(data_folder + data_name, 'r') as csv_file:
            data = csv_file.read()

        self.captions = []
        self.caplens = []
        
        self.stoi = {'<' : 0, '>' : 1, '$' : 2}
        self.itos = {0: '<', 1: '>', 2: '$'}

        for line in tqdm(data.split('\n')[1:101]):
            image_id, smiles = line.split(',')
            if len(smiles) > 78:
                continue

            caption = "<" + smiles + ">"

            '''
            full_image_path = data_folder + 'extra_train/' + image_id
            
            img = Image.open(os.path.join(full_image_path))
            img_clone = img.copy()
            img.close()
            
            self.imgs.append(img_clone)'''
            
            for char in list(caption):
                if char not in self.stoi:
                    self.stoi[char] = len(self.stoi)
                    self.itos[len(self.itos)] = char
            
            self.captions.append([self.stoi[char] for char in list(caption)] + [2] * (80 - len(caption)))
            
            assert len(caption) <= 80   ,'ssss'         
            
            
            self.caplens.append(len(caption))


        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i].reshape((224,224,3)))
        img = img.permute(2,0,1)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            all_captions = torch.LongTensor(self.captions[i])
            return img, caption, caplen, all_captions

        
    def get_images(self):
        return self.imgs
        
    def get_vocab(self):
        return self.stoi, self.itos
        
    def __len__(self):
        return self.dataset_size