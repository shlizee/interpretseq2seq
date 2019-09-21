# Clustering and Recognition of Spatiotemporal Features through Interpretable Embedding of Sequence to Sequence Recurrent Neural Networks

## Introduction
This repository contains the code for utilization of RNN-Based Sequence to Sequence model (Seq2Seq) for unsupervised action recognition. It is tested on CMU Mocap Dataset.

## Abstract
Encoder-decoder recurrent neural network models (RNN Seq2Seq) have achieved great success in ubiquitous areas of computation and applications. It was shown to be successful in modeling data with both temporal and spatial dependencies for translation or prediction tasks. In this study, we propose an embedding approach to visualize and interpret the representation of data by these models. Furthermore, we show that the embedding is an effective method for unsupervised learning and can be utilized to estimate the optimality of model training. In particular, we demonstrate that embedding space projections of the decoder states of RNN Seq2Seq model trained on sequences prediction are organized in clusters capturing similarities and differences in the dynamics of these sequences. Such performance corresponds to an unsupervised clustering of any spatio-temporal features and can be employed for time-dependent problems such as temporal segmentation, clustering of dynamic activity, self-supervised classification, action recognition, failure prediction, etc. We test and demonstrate the application of the embedding methodology to time-sequences of 3D human body poses. We show that the methodology provides a high-quality unsupervised categorization of movements. 

## Examples
{% include youtubePlayer.html id="X9Oz81o55Gs" %}

## Datasets
We do expermients on two cases: 
(i) concatenated sequence (randomly select from 8 unique actions) and 
(ii) continous sequence (subject 86)

For case (i), we use CMU Mocap dataset for human motion prediction, as in [Li et al. CVPR 18](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics). Data is preprocessed using exponential map format. We provide the data in `data/cmu_mocap/train` folder. Alternatively, you can download the data from   [Li et al. CVPR 18] Github webpage. The data contains 8 unique actions (walking, running, jumping, basketball, soccer, washwindow, basketball signal, and directing traffic). We manually concatenate multiple actions together to demonstrate action recognition perfromance. To avoid biases on selecting the data, we specify the number of possible actions (repeat or non-repeat) and then randomly select the sequences and their ordering. These values are configurable. Please see the arguments below.

For case (ii), we use a continous sequence (subject 86) of CMU mocap dataset. The data is preprocessed with an exponential map format as in case (i) and is located in `data/cmu_mocap/subject_86` folder. Each sequence contains multiple actions and they are manually annotated, see [Barbic et al. Proc. of the Graph. Inter. 2004](https://www.researchgate.net/publication/221474944_Segmenting_Motion_Capture_Data_into_Distinct_Behaviors). Please find additional details in the `segmentation_ground_truth` folder.

## Requirements
1. Tensorflow 1.13
2. Python 3
3. scikit-learn 0.21.2

## Getting Started
Case (i): To train a new model from scratch, run
```bash
python src/main.py --dataset cmu --actions walking,running,jumping
```
Case (ii): We provide our pretrained models in the `demo` folder (download:[Google Drive](https://drive.google.com/file/d/1fIgWPqs7ukCGN5tW5ka-6F56j9wOyIwX/view?usp=sharing)). You can quickly check result by running
```bash
python src/subject_86.py --demo True --num_trial 01
```
To train a new model from scratch, run
```bash
python src/subject_86.py --demo False --num_trial 01
```

## Parameters for case (i)
`--dataset`: a string that is either `cmu` or `h3.6m`.

`--num_actions`: unique number of actions specified for tested sequence (2 to 8 for CMU, 2 to 15 for H3.6M).

`--total_num_actions`: integer, total number of actions, default is '3'.

`--actions`: a string to define your own actions, default is `random`. Instead of randomly choosing actions, you can specify a list that contains names of actions you want to show in the testing video. For CMU dataset, you can choose
any action from `['walking', 'running', 'jumping', 'soccer', 'basketball', 'directing_traffic', 'washwindow', 'basketball_signal']`.

`--learning_rate`: float number, default is `0.001`

`--batch_size`: interger number for choosing size of training batch,default is `16`

`--source_seq_len`:integer,the sequence length for encoder, default is `25`

`--target_seq_len`:integer,the sequence length for decoder, default is `25`

`iterations`:integer, iterations for training, default is `1e4`

`train_dir`:directory to save model

`gen_video`:boolean, whether to generate video, default is False

`video_dir`:directory to save generated video (only if you `generate_video` flag is `True`)

`load`: integer, the number of previous checkpoint you want to load.

## Parameters for case (ii)
`--demo`: boolean, default is `True`. Indicate whether to use saved models

`--num_trial`: string, default is `01`. The sequence number you want to check. range from `01` to `14`.

`--train_dir`: directory to save your new model
