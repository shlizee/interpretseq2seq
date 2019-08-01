# Temporal segmentation

## Introduction
This repository contains the code to use RNN-Based Sequence to Sequence model (Seq2Seq) for unsupervised temporal segmentation tested on CMU Mocap Dataset.

## Abstract
We present that Seq2Seq model has capability to learn temporal embedding automatically. Given skeleton-based dataset, we train the network to do regression, but at the same time, the network itself is able to learn a embedding space that can be used to separate different type of actions. 
To demonstrate our method, we first concatenate multiple different actions together manually and use Seq2Seq to learn the representation. During testing,
we will perform motion prediction task and utilize the internal states in Recurrent Neural Network to do clustering. We only provide the total number of 
possible actions happening in the whole sequence and don't provide any label during training and testing.

## Examples
{% include youtubePlayer.html id="X9Oz81o55Gs" %}

## Datasets
We do expermients on two human motion datasets: CMU Mocap Dataset(done) and H3.6M(coming soon).

We follow the work done by 
[Julieta Martinez](https://github.com/una-dinosauria/human-motion-prediction) and [Chen Li](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) to
use H3.6M dataset and CMU Mocap dataset which are two standard datasets for human motion prediction. Both datasets are preprocessed on exponential map format. We provide the data in the `data` folder or you can download the data from their Github webpage. H3.6M contains 15 different type of actions and for CMU Mocap we select 8 unique actions (walking, running, jumping, basketball, soccer, washwindow, basketball signal, and directing traffic). For both datasets, each sequence only contains one action. As a result, we manually concatenate multiple actions together to demonstrate the temporal segmentation. 
To avoid biases on selecting data, we only specify the number of possible actions (repeat or non-repeat) and then randomly select the sequences and orders of data. Each unique type of action can also be repeated. You can also specify the actions yourself. For details, please check the arguments below.

## Requirements
1. Tensorflow 1.13
2. Python 3
3. scikit-learn 0.21.2

## Getting Started
To train a new model from scratch, run
```bash
python src/main.py --dataset cmu --actions walking,running,jumping
```

## Argument
`--dataset`: a string that is either `cmu` or `h3.6m`.

`--num_actions`: unique number of actions specified for tested sequence (2 to 8 for CMU, 2 to 15 for H3.6M).

`--unique`: boolean, are all actions in the sequence uniuqe or not, default is 'True'.

`--actions`: a string to define your own actions, default is `random`. Instead of randomly choose actions, you can specify a list that contains names of actions you want to show in the testing video. For CMU dataset, you can choose
any action from `['walking', 'running', 'jumping', 'soccer', 'basketball', 'directing_traffic', 'washwindow', 'basketball_signal']`. For H3.6M dataset, you can choose any action from `['walking', 'directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting', 'sittingdown', 'smoking', 'takingphoto',
'waiting', 'walkingdog','walkingtogether']`.

`--learning_rate`: float number, default is `0.001`

`--batch_size`: interger number for choosing size of training batch,default is `16`

`--source_seq_len`:integer,the sequence length for encoder, default is `25`

`--target_seq_len`:integer,the sequence length for decoder, default is `25`

`iterations`:integer, iterations for training, default is `1e4`

`train_dir`:directory to save model

`gen_video`:boolean, whether to generate video, default is False

`video_dir`:directory to save generated video (only if you `generate_video` flag is `True`)

`load`: integer, the number of previous checkpoint you want to load.


## Files and functions
- [X] Seq2Seq model (class)
- [X] data_utils
  - [X] readCSV
  - [X] normalize data
  - [X] unormalize data
  - [X] expmap to rotation matrix
  - [X] get mini batch
- [X] Datasets
  - [X] CMU Mocap dataset
  - [ ] H3.6M dataset
- [X] experiments (checkpoint files)
- [X] forward kinematics
  - [X] some variables
  - [X] revert coordinate space
  - [X] fkl
- [X] visualize (class)
  - [X] axes
  - [X] update
- [X] main file
  - [X] get data (readCSV)
  - [X] preprocess data (normalization)
  - [X] train (seq2seq model, mini_batch)
  - [X] get checkpoint (save)
  - [X] find cuts
  - [X] clustering
  - [X] generate results
