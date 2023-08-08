#!/bin/bash
# For LTCC dataset
python main.py --dataset ltcc --gpu 0,1
# For PRCC dataset
#python main.py --dataset prcc --gpu 4,5
# For VC-Clothes dataset.
#python main.py --dataset vcclothes --gpu 6,7
# python main.py --dataset vcclothes_cc --gpu 4,5
# python main.py --dataset vcclothes_sc --gpu 4,5
# For DeepChange dataset.
#python main.py --dataset deepchange --gpu 6,7
