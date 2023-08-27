# Experiments of Federated Learning for COVID-19 Chest X-ray Images

This repository contains the code and data for the paper ["Experiments of Federated Learning for COVID-19 Chest X-ray Images"](https://arxiv.org/pdf/2007.05592.pdf) by Boyi Liu, Bingjie Yan, Yize Zhou, et al.
## Abstract

AI plays an important role in COVID-19 identification. Computer vision and deep learning techniques can assist in determining COVID-19 infection with Chest X-ray Images. However, for the protection and respect of the privacy of patients, the hospital's specific medical-related data did not allow leakage and sharing without permission. Collecting such training data was a major challenge. To a certain extent, this has caused a lack of sufficient data samples when performing deep learning approaches to detect COVID-19. Federated Learning is an available way to address this issue. It can effectively address the issue of data silos and get a shared model without obtaining local data. In the work, we propose the use of federated learning for COVID-19 data training and deploy experiments to verify the effectiveness. And we also compare performances of four popular models (MobileNet, ResNet18, MoblieNet, and COVID-Net) with the federated learning framework and without the framework. This work aims to inspire more researches on federated learning about COVID-19.

## Contents

This will train and evaluate the four models (MobileNet, ResNet18, MoblieNet, and COVID-Net) with and without the federated learning framework.

PyTorch implementation of COVID-Net, training with CODIVx v3 dataset

## Acknowlegement
This repo is forked from <a href="https://github.com/lindawangg">https://github.com/lindawangg</a> and  <a href="https://github.com/IliasPap/COVIDNet">https://github.com/IliasPap/COVIDNet</a>

## Citation
If you find this code useful for your research, please cite our paper:

@article{liu2020experiments, <br>
  title={Experiments of federated learning for covid-19 chest x-ray images}, <br>
  author={Liu, Boyi and Yan, Bingjie and Zhou, Yize and Yang, Yifan and Zhang, Yixian}, <br>
  journal={arXiv preprint arXiv:2007.05592}, <br>
  year={2020} <br>
}

@inproceedings{yan2021experiments, <br>
  title={Experiments of federated learning for COVID-19 chest X-ray images}, <br>
  author={Yan, Bingjie and Wang, Jun and Cheng, Jieren and Zhou, Yize and Zhang, Yixian and Yang, Yifan and Liu, Li and Zhao, Haojiang and Wang, Chunjuan and Liu, Boyi}, <br>
  booktitle={Advances in Artificial Intelligence and Security: 7th International Conference, ICAIS 2021, Dublin, Ireland, July 19-23, 2021, Proceedings, Part II 7}, <br>
  pages={41--53}, <br>
  year={2021}, <br>
  organization={Springer} <br>
}
