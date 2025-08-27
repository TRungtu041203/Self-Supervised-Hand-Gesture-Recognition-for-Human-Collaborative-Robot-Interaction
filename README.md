## Installation
  ```bash
# Install torchlight
$ cd torchlight
$ python setup.py install
$ cd ..
  
# Install other python libraries
$ pip install -r requirements.txt
  ```

## Unsupervised Pre-Training

Example for unsupervised pre-training of **3s-AimCLR++**. You can change some settings of `.yaml` files in `config/three-stream/pretext` folder.
```bash
# train on NTU RGB+D xview for three-stream
$ python main.py pretrain_aimclr_v2_3views --config config/three-stream/pretext/pretext_aimclr_v2_3views_ntu60_xview.yaml

# train on NTU RGB+D xsub for three-stream
$ python main.py pretrain_aimclr_v2_3views --config config/three-stream/pretext/pretext_aimclr_v2_3views_ntu60_xsub.yaml
```

## Linear Evaluation

Example for linear evaluation of **3s-AimCLR++**. You can change `.yaml` files in `config/three-stream/linear` folder.
```bash
# Linear_eval on NTU RGB+D xview for three-stream
$ python main.py linear_evaluation --config config/three-stream/linear/linear_eval_aimclr_v2_3views_ntu60_xview.yaml

# Linear_eval on NTU RGB+D xsub for three-stream
$ python main.py linear_evaluation --config config/three-stream/linear/linear_eval_aimclr_v2_3views_ntu60_xsub.yaml
```

## Linear Evaluation Results

|          Model          | NTU 60 xsub (%) | NTU 60 xview (%) | PKU Part I (%) | PKU Part II (%) |
| :---------------------: | :-------------: | :--------------: | :------------: | :-------------: |
|        3s-AimCLR        |      79.18      |      84.02       |     87.79      |      38.52      |
| 3s-AimCLR++ (This repo) |    **80.9**     |     **85.4**     |    **90.4**    |    **41.2**     |


## Citation
Please cite our paper if you find this repository useful in your resesarch:

```
@inproceedings{guo2022aimclr,
  Title= {Contrastive Learning from Extremely Augmented Skeleton Sequences for Self-supervised Action Recognition},
  Author= {Tianyu, Guo and Hong, Liu and Zhan, Chen and Mengyuan, Liu and Tao, Wang  and Runwei, Ding},
  Booktitle= {AAAI},
  Year= {2022}
}
```

## Licence

This project is licensed under the terms of the MIT license.
