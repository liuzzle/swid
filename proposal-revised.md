# Paper Proposal: Evaluating Whisper Embeddings for Short-Utterance Speaker Identification

## Problem Setup

Modern systems for speaker identification often rely on specialized
speaker embeddings such as x-vectors and ECAPA-TDNN embeddings. However,
recent large speech models such as [OpenAI
Whisper](https://openai.com/index/whisper/) learn rich latent
representations from large multilingual speech datasets.
[@Radford2022RobustSR] Now this raises the question whether these
representations also encode speaker-specific information.

## Current Approaches

Traditional speaker identification systems use MFCC features, Gaussian
Mixture Models (GMMs) and Support Vector Machines (SVMs).
[@reynolds94_asriv]

Zhang et al.'s work suggests that pretrained multilingual speech models
such as Whisper may encode speaker-identifying information in their
latent representations. [@Zhang2024WhisperSVAW]

Current state-of-the-art systems rely on deep speaker embeddings like
x-vectors and Time Delay Neural Networks (ECAPA-TDNN).
[@Snyder2018XVectorsRD] [@Desplanques_2020]

Pre-trained implementations are available in
[SpeechBrain](https://speechbrain.github.io/) and
[PyTorch](https://pytorch.org/)

While recent work has begun to investigate Whisper-based speaker
embeddings [@Aldabergen_Kynabay_Kadyrov_2026]
[@emon2025whisperspeakeridentificationleveraging] , the behavior of
Whisper embeddings under very short utterance durations remains less
well explored.

## A tentative Setup

The core idea is to investigate whether embeddings extracted from OpenAI
Whisper can be used for speaker identification, especially when only
very short speech segments are available.

### Pipeline

To approach this I would compare dedicated speaker embeddings
(ECAPA/x-vectors) and Whisper embeddings under varying utterance
durations.

#### Dedicated Speaker Embeddings

- use pretrained ECAPA-TDNN embeddings

- cosine similarity, (alternatively: classifier-based identification)

#### Whisper Embeddings

- extract hidden state embeddings

- evaluate their usefulness for speaker identification

### Experiment Evaluation

- identification accuracy

- effect of utterance duration

### Datasets

- [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/): An
  audio-visual dataset consisting of short clips of human speech,
  extracted from interview videos uploaded to YouTube.

### Possible Durations

0.5s, 1s, 3s, 5s

### Possible Classifiers

Cosine similarity, k-NN and SVM.

## Expected Outcome

- Dedicated speaker embeddings outperform MFCCs.

- Whisper embeddings can capture (meaningful) speaker information
  despite being trained for ASR.

- Possibly a performance degradation below 1-2 seconds.

A nice side effect of this project could also be to find out which
Whisper layers encode speaker-related information the strongest.

## References
[1] Aimoldir Aldabergen, Bakdaulet Kynabay, and Shirali Kadyrov. Layer-wise probing of paralinguistic attributes in fine-tuned whisper for kazakh speech. Engineering, Technology amp; Applied Science Research,
16(2):33399–33404, Apr. 2026.

[2] Brecht Desplanques, Jenthe Thienpondt, and Kris Demuynck. Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification. In Interspeech 2020, page
3830–3834. ISCA, October 2020.

[3] Jakaria Islam Emon, Md Abu Salek,and Kazi Tamanna Alam. Whisper speaker identification: Leveraging
pre-trained multilingual transformers for robust speaker embeddings, 2025.

[4] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust
speech recognition via large-scale weak supervision. In International Conference on Machine Learning,
2022.

[5] Douglas A. Reynolds. Speaker identification and verification using gaussian mixture speaker models. In
ESCA Workshop on Automatic Speaker Recognition, Identification and Verification, pages 27–30, 1994.

[6] David Snyder, Daniel Garcia-Romero, Gregory Sell, Daniel Povey, and Sanjeev Khudanpur. X-vectors:
Robust dnn embeddings for speaker recognition. 2018 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), pages 5329–5333, 2018.

[7] Li Zhang, Ning Jiang, Qing Wang, Yuehong Li, Quan Lu, and Lei Xie. Whisper-sv: Adapting whisper
for low-data-resource speaker verification. ArXiv, abs/2407.10048, 2024.