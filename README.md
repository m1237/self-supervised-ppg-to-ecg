# self-supervised-ppg-to-ecg
This repo contains the implementation of End-to-end non-invasive ECG signal generation from PPG signal: a self-supervised learning approach


### End-to-end non-invasive ECG signal generation from PPG signal: a self-supervised learning approach

## [Paper](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1694995/full) - [Repository](https://github.com/m1237/self-supervised-ppg-to-ecg)

<hr style="height:1px" />

<img src="./images/game-scenes.jpg"/>

### Abstract

Electrocardiogram (ECG) signals are frequently utilized for detecting important cardiac events, such as variations in ECG intervals, as well as for monitoring essential physiological metrics, including heart rate (HR) and heart rate variability (HRV). However, the accurate measurement of ECG traditionally requires a clinical environment, thereby limiting its feasibility for continuous, everyday monitoring. In contrast, Photoplethysmography (PPG) offers a non-invasive, cost-effective optical method for capturing cardiac data in daily settings and is increasingly utilized in various clinical and commercial wearable devices. However, PPG measurements are significantly less detailed than those of ECG. In this study, we propose a novel approach to synthesize ECG signals from PPG signals, facilitating the generation of robust ECG waveforms using a simple, unobtrusive wearable setup. Our approach utilizes a Transformer-based Generative Adversarial Network model, designed to accurately capture ECG signal patterns and enhance generalization capabilities. Additionally, we incorporate self-supervised learning techniques to enable the model to learn diverse ECG patterns through specific tasks. Model performance is evaluated using various metrics, including heart rate calculation and root mean squared error (RMSE) on two different datasets. The comprehensive performance analysis demonstrates that our model exhibits superior efficacy in generating accurate ECG signals (with reducing 83.9% and 72.4% of the heart rate calculation error on MIMIC III and Who is Alyx? datasets, respectively), suggesting its potential application in the healthcare domain to enhance heart rate prediction and overall cardiac monitoring. As an empirical proof of concept, we also present an Atrial Fibrillation (AF) detection task, showcasing the practical utility of the generated ECG signals for cardiac diagnostic applications. To encourage replicability and reuse in future ECG generation studies, we have made both the dataset and the code publicly available.

### Prerequisites
- Pytorch
- Numpy
- Scikit-learn
- Scipy
- Pandas
- h5py
- Matplotlib

## Data-Processing

To get the datasets, run `train_model.py`.

## Train 

Run `train.py` for the training. For evaluation run `eval.py`. Cross-validation should be done within train set.
Run `self_supervised_transformation.py` to make self-supervised ECG transformations.



### Citation
Please cite our paper below when using or referring to our work.
```
@ARTICLE{10.3389/fphys.2026.1694995,
        AUTHOR={Yalcin, Murat  and Latoschik, Marc Erich},
        TITLE={End-to-end non-invasive ECG signal generation from PPG signal: a self-supervised learning approach},
        JOURNAL={Frontiers in Physiology},
        VOLUME={Volume 17 - 2026},
        YEAR={2026},
        URL={https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1694995},
        DOI={10.3389/fphys.2026.1694995},
        ISSN={1664-042X}
        }
```

