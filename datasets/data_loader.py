import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import transforms.ISTD_transforms as transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(train_A_dir, train_B_dir, train_C_dir):
    path_ABC = []
    if not os.path.isdir(train_A_dir):
        raise Exception('Data directory does not exist')
    for root, _, fnames in sorted(os.walk(train_A_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path_A = os.path.join(train_A_dir, fname)
                path_B = os.path.join(train_B_dir, fname)
                path_C = os.path.join(train_C_dir, fname)
                if not os.path.isfile(path_B):
                    raise Exception('%s does not exist' % path_B)
                if not os.path.isfile(path_C):
                    raise Exception('%s does not exist' % path_C)
                item = {'path_A': path_A, 'path_B': path_B, 'path_C': path_C}
                path_ABC.append(item)
    return path_ABC


class ISTD(data.Dataset):
    def __init__(self, dataroot, transform=None, split='train', seed=None):
        if split == 'train':
            name_A = 'input'
            name_B = 'mask'
            name_C = 'target'
        else:
            name_A = 'input'
            name_B = 'mask'
            name_C = 'target'
        self.A_dir = os.path.join(dataroot, name_A)
        self.B_dir = os.path.join(dataroot, name_B)
        self.C_dir = os.path.join(dataroot, name_C)
        self.path_ABC = make_dataset(self.A_dir, self.B_dir, self.C_dir)
        if len(self.path_ABC) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + dataroot + "\n"
                                                                                 "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.transform = transform

        if seed is not None:
            np.random.seed(seed)

    def __getitem__(self, index):
        path_A = self.path_ABC[index]['path_A']
        path_B = self.path_ABC[index]['path_B']
        path_C = self.path_ABC[index]['path_C']
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('L')
        img_C = Image.open(path_C).convert('RGB')
        filename = os.path.basename(path_A)
        if self.transform is not None:
            # NOTE preprocessing for each pair of images
            imgs = {'imgA': img_A, 'imgB': img_B, 'imgC': img_C}
            # img_A, img_B, img_C = self.transform(img_A, img_B, img_C)
            imgs = self.transform(imgs)
        # return img_A, img_B, img_C
        imgs['filename'] = filename
        return imgs

    def __len__(self):
        return len(self.path_ABC)
