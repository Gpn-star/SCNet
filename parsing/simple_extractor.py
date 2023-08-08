#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")
    parser.add_argument("--dataset", type=str, default='atr', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='exp-schp-201908301523-atr.pth',
                        help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='7', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette



def extractor():
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(root=args.input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    palette = get_palette(12)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image_copy, image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]
            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)

            # According to the parsing results of the SCHP model pre-trained on the ATR dataset, we merge some of the parsing results for later use.
            # For details about the SCHP model, please see: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing
            # In order to facilitate reproduction, we directly provide the parsing results in datasets.
            # If you are interested in obtaining other human parsing results, you can refer to this part of the code.

            parsing_result[
                (parsing_result == 1) | (parsing_result == 2) | (parsing_result == 3) | (parsing_result == 11)] = 1
            parsing_result[(parsing_result == 4)] = 2
            parsing_result[(parsing_result == 5)] = 3
            parsing_result[(parsing_result == 6)] = 4
            parsing_result[(parsing_result == 7)] = 5
            parsing_result[(parsing_result == 8)] = 6
            parsing_result[(parsing_result == 9) | (parsing_result == 10)] = 7
            parsing_result[(parsing_result == 12) | (parsing_result == 13)] = 8
            parsing_result[(parsing_result == 14) | (parsing_result == 15)] = 9
            parsing_result[(parsing_result == 16)] = 10
            parsing_result[(parsing_result == 17)] = 11

            parsing_result_path = os.path.join(args.output_dir, img_name[:-4] + '.png')
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(palette)
            output_img.save(parsing_result_path)

    return parsing_result


if __name__ == '__main__':
    extractor()
