import copy
import os.path

import torch
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import data.img_transforms as T
import random
import numpy as np

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def read_parsing_result(img_path):
    """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        super(ImageDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]

        if 'LTCC_ReID' in img_path:
            parsing_result_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'processed',
                                               os.path.basename(img_path))

        elif 'PRCC' in img_path:
            parsing_result_path = os.path.join('', '/'.join(img_path.split('/')[:7]), 'processed',
                                               img_path.split('/')[-2], img_path.split('/')[-1][:-4] + '.png')

        elif 'VC-Clothes' in img_path:
            parsing_result_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'processed',
                                              img_path.split('/')[-1][:-4] + '.png')

        elif 'DeepChange' in img_path:
            parsing_result_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'processed',
                                               img_path.split('/')[-1][:-4] + '.png')


        img = read_image(img_path)
        parsing_result = read_parsing_result(parsing_result_path)

        p1 = random.randint(0, 1)
        p2 = random.randint(0, 1)
        p3 = random.randint(0, 1)

        transform = T.Compose([
            T.Resize((384, 192)),
            T.RandomCroping(p=p1),
            T.RandomHorizontalFlip(p=p2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(probability=p3)
        ])

        transform_b = T.Compose([
            T.Resize((384, 192)),
            T.RandomCroping(p=p1),
            T.RandomHorizontalFlip(p=p2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(probability=p3)
        ])

        transform_parsing = T.Compose([
            T.Resize((384, 192)),
            T.Convet_ToTensor(),
        ])

        parsing_result_copy = torch.tensor(np.asarray(parsing_result, dtype=np.uint8)).unsqueeze(0).repeat(3, 1, 1)
        img_b = copy.deepcopy(img)
        img_b = np.asarray(img_b, dtype=np.uint8).transpose(2, 0, 1)
        img_b[(parsing_result_copy == 2) | (parsing_result_copy == 3) | (parsing_result_copy == 4) | (
                parsing_result_copy == 5) | (parsing_result_copy == 6) | (parsing_result_copy == 7) | (
                      parsing_result_copy == 10) | (
                      parsing_result_copy == 11)] = 0
        img = transform(img)
        img_b = img_b.transpose(1, 2, 0)
        img_b = Image.fromarray(img_b, mode='RGB')
        img_b = transform_b(img_b)
        parsing_result = transform_parsing(parsing_result)

        return parsing_result, img, img_b, pid, clothes_id


class ImageDataset_test(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        super(ImageDataset_test, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img_path, img, pid, camid, clothes_id


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
