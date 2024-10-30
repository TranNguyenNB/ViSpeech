# Dataset Card for the ViSpeech dataset

## Dataset Summary

The ViSpeech Dataset is a collection of unscripted audio recordings designed for the classification of gender and Vietnamese dialects. The dataset comprises 10,686 mp3 files, totaling slightly over 14 hours of speech data from 449 speakers representing both genders across the three primary Vietnamese dialects: Northern, Central, and Southern. It is divided into three subsets: a training set with clean recordings and two test sets—one with clean recordings and the other with ambient noise. Notably, the speakers in the training set are independent of those in the test sets. The dataset is designed to provide a diverse and comprehensive resource for audio classification research.

## Dataset Details dialect information.
  - `clean_test`: Clean test data for evaluation.
  - `noise_test`: Noisy test data for evaluation.

- **Features**:
  - `audio`: Audio data with a sampling rate of 16,000 Hz.
  - `audio_name`: The name of the audio file.
  - `dialect`: The dialect of the speaker.
  - `gender`: The gender of the speaker.
  - `speaker`: The speaker's identifier.


### Dataset Information

- **Configurations**: The dataset contains the following configurations:
  - `train`: Training data with gender and dialect information.
  - `validation`: Validation data with gender and