# fixation-prediction

Goal: train a DINO-based scanpath-conditioned model for next-fixation prediction.

Current model:
- frozen DINOv2 visual patch encoder
- causal scanpath encoder
- cross-attention fusion
- spatial heatmap decoder
- KL training loss, NLL/NSS logging

| method | samples | KL ↓ | NLL ↓ | NSS ↑ |
|---|---:|---:|---:|---:|
| uniform | 1600 | 4.1031 | 8.3178 | 0.0000 |
| center gaussian | 1600 | 3.8116 | 8.0016 | 0.7858 |
| empirical density | 1600 | 3.6706 | 7.8690 | 0.9946 |
| model small_1k final | 1600 | 2.8684 | 6.9576 | 2.1569 |