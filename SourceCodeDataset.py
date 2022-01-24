import random
import pandas as pd
import numpy as np
import csv
import os
import torch
from utils.style import get_theme, getFonts, getFontSize
from utils.helper import check_open_utf8
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
from sklearn import preprocessing
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None

class SourceCodeDataset(Dataset):
    """Source Code dataset."""

    def __init__(
        self,
        csv_file,
        accepted,
        root_dir=None,
        transform=None,
        image_size=(512, 512),
        num_of_lines=1000,
        max_num_of_files=6000,
        offset_lines=40,
        min_num_lines=0
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Path to images folder.
            image_size (Tuple{int, int}): Image W x H
            transform (callable, optional): Optional transform to be applied
                on a sample.
            num_of_lines (int): Number of lines to be kept from csv file
            max_num_of_files (int): Max number of files per language
            offset_lines (int): Max number of lines source code per image
        """
        file_num = {}
        file_list = []
        file_ext = []
        self.root_dir = root_dir
        self.transform = transform
        with open(csv_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for lines_number, row in enumerate(csv_reader):
                if lines_number > num_of_lines:
                    break
                if len(row) > 1:
                    if row[0] and row[1]:
                        if os.path.isfile(row[0]):
                            if row[1] in accepted:
                                add = False
                                lenght_file = check_open_utf8(row[0])
                                if lenght_file != None and lenght_file > min_num_lines:
                                    if row[1] in file_num:
                                        if file_num[row[1]] < max_num_of_files:
                                            add = True
                                            file_num[row[1]] += 1
                                    else:
                                        add = True
                                        file_num[row[1]] = 1
                                    if add:
                                        for _ in range(0, 3):
                                            file_list.append(row[0])
                                            file_ext.append(row[1])
        self.file_num = file_num
        self.file_list = file_list
        self.le = preprocessing.LabelEncoder()
        self.file_ext = self.le.fit_transform(file_ext)
        self.len_dataset = len(self.file_list)
        self.transform = transform
        self.image_size = image_size
        self.offset_lines = offset_lines

    def get_frequency(self):
        return self.file_num

    def __len__(self):
        return self.len_dataset

    def num_classes(self):
        return len(list(self.le.classes_))

    def get_classes(self):
        return self.le.classes_
        
    def get_class(self, value):
        return self.le.inverse_transform([value])[0]

    def set_transform(self, trans):
        self.transform = trans

    def generate_image(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.root_dir is None:
            file_path = self.file_list[idx]
        else:
            file_path = os.path.join(self.root_dir,
                                     self.file_list.iloc[idx])
        # If lines > requested lines per image
        # Reduce number of lines
        lines = open(file_path, 'r').readlines()
        num_lines = sum(1 for _ in lines)
        if num_lines > self.offset_lines:
            n = random.randint(0, num_lines - self.offset_lines - 1)
            reduced_file = []
            for idx_line in range(n, n + self.offset_lines):
                if idx_line < num_lines:
                    reduced_file.append(lines[idx_line])
        # So
            if num_lines > self.offset_lines + 10:
                if np.random.uniform() > 0.2:
                    reduced_file = reduced_file[10:]
                    for idx_line in range(n + self.offset_lines, n + self.offset_lines + 10):
                        if idx_line < num_lines:
                            reduced_file.append(lines[idx_line])              
        else:
            reduced_file = lines
        # Get style image options
        theme = get_theme()
        list_font = getFonts()
        # Generate the image
        font = ImageFont.truetype(
            list_font[random.randint(0, len(list_font) - 1)], getFontSize())
        img = Image.new('RGB', self.image_size, color=theme[0])
        canvas = ImageDraw.Draw(img)
        canvas.text((10, 10), "".join(reduced_file), font=font, fill=theme[1])
            
        return img, self.file_ext[idx]

    def __getitem__(self, idx):

        img, label = self.generate_image(idx)

        if self.transform:
            img = self.transform(img)

        return img, label
