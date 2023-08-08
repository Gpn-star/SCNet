## Semantic-aware Consistency Network for Cloth-changing Person Re-Identification (ACM MM, 2023)
### Overall Framework
![Overall architecture of the proposed tri-stream semantic-aware consistency network (SCNet).](./model.png)
### Requirements
- Python 3.6
- Pytorch 1.6.0
- yacs
- apex

### Get Started
- Clone this repo:
  ```bash
  git clone https://github.com/Gpn-star/SCNet.git
  cd SCNet
  ```
- Download datasets: \
  LTCC: [[Offical Link]](https://naiq.github.io/LTCC_Perosn_ReID.html) \
  PRCC: [[Offical Link]](https://www.isee-ai.cn/~yangqize/clothing.html) \
  Vc-Clothes: [[Offical Link]](https://wanfb.github.io/dataset.html) \
  DeepChange: [[Offical Link]](https://github.com/PengBoXiangShang/deepchange)

- Download human parsing results: \
  LTCC: [[GoogleDrive]](https://drive.google.com/file/d/1in9e7pvKDxLP2G2W1eKrX-sksgIrlP5j/view?usp=sharing) \
  PRCC: [[GoogleDrive]](https://drive.google.com/file/d/1uAdP26CYBYM72E6x3CxM_yJb1As3z184/view?usp=sharing) \
  Vc-Clothes: [[GoogleDrive]](https://drive.google.com/file/d/1pEQ059XGSiBYqe6iWqOrPNalV1s0aNzS/view?usp=sharing) \
  DeepChange: [[GoogleDrive]](https://drive.google.com/file/d/1tD3_sIAqNxQPMCBtfdrLOAfdignDk6O0/view?usp=sharing)
  
- Arrange datasets according to the following structure：
```
Dataset/
├── LTCC_ReID/
│   ├── ...
│   └── processed
├── PRCC/
|   ├── rgb / processed
│   └── sketch
├── Vc-Clothes/
|   ├── ...
|   └── processed
└── DeepChange/
    ├── ...
    └── processed
```

- Replace `_C.DATA.ROOT` and `_C.OUTPUT` in `configs/default_img.py`with your own `data root path` and `output path`, respectively.


- Run `script.sh`

### Citation
If you find this code useful for your research, please cite our paper:

```
@inproceedings{guo2023SCNet,
  title={Semantic-aware Consistency Network for Cloth-changing Person Re-Identification},
  author={Guo, Peini and Liu, Hong and Wu, Jianbing and Wang, Guoquan and Wang, Tao},
  booktitle={Proceedings of the 31th ACM International Conference on Multimedia},
  year={2023}
}
```
### Contact Us
If you have any doubt about this code, please send emails to: `guopeini@stu.pku.edu.cn`.

