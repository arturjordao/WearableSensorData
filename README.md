# WearableSensorData
This repository provides the codes and data used in our paper "Human Activity Recognition Based on Wearable Sensor Data: A Standardization of the State-of-the-Art", where we implement and evaluate several state-of-the-art approaches, ranging from handcrafted-based methods to convolutional neural networks. Also, we standardize a large number of datasets, which vary in terms of sampling rate, number of sensors, activities, and subjects.

## Requirements

- [Scikit-learn](http://scikit-learn.org/stable/)
- [Keras](https://github.com/fchollet/keras) (Recommended versions 2.0.4 or 2.1.2)
- [Tensorflow](https://www.tensorflow.org/) (Recommended version 1.3.0)
- [Python 3](https://www.python.org/)

## Quick Start
1. Clone this repository
2. Run
    ```bash
    python <Catal2015|...|ChenXue2015>.py data/<SNOW|FNOW|LOTO|LOSO>/<MHEALTH|USCHAD|UTD-MHAD1_1s|UTD-MHAD2_1s|WHARF|WISDM>.npz
    ```
	For example
	```bash
    python Catal2015.py data/LOSO/MHEALTH.npz
    ```
	
## Data Format
The raw signal provided by the original dataset was segmented by using a temporal sliding window of 5 seconds. 
Its format is (number of samples, 1, temporal window size, number of sensors)
	
## Contributing
Contributions to this repository are welcome. Examples of things you can contribute:
 * Implementation of other methods. See template_hancrafted.py and template_convNets.py
 * Accuracy Improvements.
 * Reporting bugs.

The table below shows the mean accuracy achieved by the methods using the Leave-One-Subject-Out (LOSO) as validation protocol. The symbol 'x' denotes which was not possible to execute the method on the respective dataset.

| Method | MHEALTH | PAMAP2 | USCHAD | UTD-MHAD1 | UTD-MHAD2 | WHARF | WISDM | Mean Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [Kwapisz et al.](https://dl.acm.org/citation.cfm?id=1964918) | 90.41 | 71.27 | 70.15 | 13.04 | 66.67 | 42.19 | 75.31 | 61.29 |
| [Catal et al.](https://www.sciencedirect.com/science/article/pii/S1568494615000447) | 94.66 | 85.25 | 75.89 | 32.45 | 74.67 | 46.84 | 74.96 | 69.29 |
| [Kim et al.](https://ieee0plore.ieee.org/document/6411901/) | 93.90 | 81.57 | 64.20 | 38.05 | 64.60 | 51.48 | 50.22 | 63.43 |
| [Chen and Xue](https://ieee0plore.ieee.org/document/7379395/) | 88.67 | 83.06 | 75.58 | x | x | 61.94 | 83.89 | 78.62 |
| [Jiang and Yin](https://dl.acm.org/citation.cfm?id=2806333) | 51.46 | x | 74.88 | x | x | 65.35 | 79.97 | 67.91 |
| [Ha et al.](https://ieee0plore.ieee.org/document/7379657/) | 88.34 | 73.79 | x | x | x | x | x | 81.06 |
| [Ha and Choi](https://ieee0plore.ieee.org/document/7727224/) | 84.23 | 74.21 | x | x | x | x | x | 79.21|
| Mean Accuracy[]() | 84.52 | 78.19 | 72.14 | 27.84 | 68.64 | 53.55 | 72.87 | x |

Please cite our paper in your publications if it helps your research.
```bash
@article{Jordao:2018,
author    = {Artur Jordao,
Antonio Carlos Nazare,
Jessica Sena and
William Robson Schwartz},
title     = {Human Activity Recognition Based on Wearable Sensor Data: A Standardization of the State-of-the-Art},
journal   = {In CoRR},
year      = {2018},
url       = {https://arxiv.org/abs/1806.05226},
archivePrefix = {arXiv},
eprint    = {1806.05226},
}
```
