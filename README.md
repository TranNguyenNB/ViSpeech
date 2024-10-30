# ViSpeech

## 1. Dataset

The ViSpeech Dataset is a collection of unscripted audio recordings designed for the classification of gender and Vietnamese dialects. The dataset comprises 10,686 mp3 files, totaling slightly over 14 hours of speech data from 449 speakers representing both genders across the three primary Vietnamese dialects: Northern, Central, and Southern. 
It is divided into three subsets: a training set with clean recordings and two test sets—one with clean recordings and the other with ambient noise. Notably, the speakers in the training set are independent of those in the test sets.

The dataset is avaliable at:
[[https://drive.google.com/drive/folders/1BA3d8eEiwt90YgLwcWuqbPy2OLtBieLr?usp=sharing](https://drive.google.com/file/d/1-BbOHf42o6eBje2WqQiiRKMtNxmZiRf9/view?usp=sharing)](https://drive.google.com/file/d/1-BbOHf42o6eBje2WqQiiRKMtNxmZiRf9/view?usp=drive_link)

**Details**
The drive contains 3 folder: `trainset`, `clean_testset` and `noise_testset`. There is also a `metadata` folder including information related to the three sets. The information includes following fields:
  - `audio`: Audio data with a sampling rate of 44,100 Hz.
  - `audio_name`: The name of the audio file.
  - `dialect`: The dialect of the speaker.
  - `gender`: The gender of the speaker.
  - `speaker`: The speaker's identifier.

## 2. Training, Evaluating, Infering
### Train
#### Requirements
Install the required packages:
   `requirements.txt`.
#### Data preparation
Download the datasets from the drive and place the folders `trainset`, `clean_testset`, and `noise_testset` inside the `dataset` directory. The text files in the `dataset` folder were used to generate the results reported in the paper. The `trainset` will be split into `train.txt` and `validation.txt`. Each text file contains two fields: audio path and numeric label. The gender and dialect information are encoded into numeric labels as follows, enabling training for both tasks without creating separate input files:
  - 0: female_north
  - 1: female_central
  - 2: female_south
  - 3: male_north
  - 4: male_central
  - 5: male_south

The gender label will derived using `number // 3` (0: female, 1: male) while the dialect label will be derived using `number % 3` (0: north, 1: central, 2: south).

#### Configuration and training
Modify `config.yaml` and run `train.py`
