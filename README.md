# Clustering and Recognition of Spatiotemporal Features through Interpretable Embedding of Sequence to Sequence Recurrent Neural Networks

## Introduction
This repository contains the code to use RNN-Based Sequence to Sequence model (Seq2Seq) for unsupervised action recognition tested on CMU Mocap Dataset.

## Abstract
We present that Seq2Seq model has capability to learn temporal embedding automatically. Given skeleton-based dataset, we train the network to do regression, but at the same time, the network itself is able to learn a embedding space that can be used to separate different type of actions. 
To demonstrate our method, we first concatenate multiple different actions together manually and use Seq2Seq to learn the representation. During testing,
we will perform motion prediction task and utilize the internal states in Recurrent Neural Network to do clustering. We only provide the total number of 
possible actions happening in the whole sequence and don't provide any label during training and testing.

## Examples
{% include youtubePlayer.html id="X9Oz81o55Gs" %}

## Datasets
We do expermients on two cases: concatenated sequence (randomly select from 8 unique actions) and continous sequence (subject 86).

For case one, we follow the work done by [Chen Li](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) to use CMU Mocap dataset for human motion prediction. Data are preprocessed on exponential map format. We provide the data in the `data/cmu_mocap/train` folder or you can download the data from their Github webpage. The data contains 8 unique actions (walking, running, jumping, basketball, soccer, washwindow, basketball signal, and directing traffic). We manually concatenate multiple actions together to demonstrate the action recognition. To avoid biases on selecting data, we only specify the number of possible actions (repeat or non-repeat) and then randomly select the sequences and orders of data. Each unique type of action can also be repeated. You can also specify the actions yourself. For details, please check the arguments below.

For case two, we use the continous sequence (subject 86) of CMU mocap dataset. The data are also preprocessed on exponential map format and you can find them in `data/cmu_mocap/subject_86` folder. Each sequence contains multiple actions and they are manually annotated by [Jernej Barbic](https://www.researchgate.net/publication/221474944_Segmenting_Motion_Capture_Data_into_Distinct_Behaviors). You can find details in `segmentation_ground_truth` folder.

## Requirements
1. Tensorflow 1.13
2. Python 3
3. scikit-learn 0.21.2

## Getting Started
Case one: To train a new model from scratch, run
```bash
python src/main.py --dataset cmu --actions walking,running,jumping
```
Case two: We provide saved models in the `demo` folder (download:[Google Drive](https://drive.google.com/file/d/1fIgWPqs7ukCGN5tW5ka-6F56j9wOyIwX/view?usp=sharing)). You can quickly check result by running
```bash
python src/subject_86.py --demo True --num_trial 01
```
To train a new model from scratch, run
```bash
python src/subject_86.py --demo False --num_trial 01
```

## Argument of Case one
`--dataset`: a string that is either `cmu` or `h3.6m`.

`--num_actions`: unique number of actions specified for tested sequence (2 to 8 for CMU, 2 to 15 for H3.6M).

`--total_num_actions`: integer, total number of actions, default is '3'.

`--actions`: a string to define your own actions, default is `random`. Instead of randomly choose actions, you can specify a list that contains names of actions you want to show in the testing video. For CMU dataset, you can choose
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

## Argument of Case two
`--demo`: boolean, default is `True`. Indicate whether to use saved models

`--num_trial`: string, default is `01`. The sequence number you want to check. range from `01` to `14`.

`--train_dir`: directory to save your new model
