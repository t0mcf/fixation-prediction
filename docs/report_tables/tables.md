# Table 1 — scaling ladder (canonical val, 5000 images)

| Images | ours NSS | ours LL | ours AUC | DG3 NSS | DG3 LL | DG3 AUC |
|---|---|---|---|---|---|---|
| 1k | 1.330 | 0.994 | 0.796 | 1.995 | 1.461 | 0.845 |
| 10k | 2.253 | 1.779 | 0.864 | 2.333 | 1.814 | 0.866 |
| 50k | 2.765 | 2.128 | 0.886 | 2.438 | 1.901 | 0.871 |
| 100k | 2.923 | 2.241 | 0.892 | 2.444 | 1.913 | 0.872 |
| 200k | 3.271 | 2.449 | 0.903 | 2.406 | 1.883 | 0.870 |
| 400k | 3.468 | 2.554 | 0.907 | — | — | — |
| 800k | 3.694 | 2.729 | 0.918 | — | — | — |

Caption: In-domain performance on the canonical validation split as a function of
training-set size. DeepGaze III was not trained beyond 200k images.

# Table 2 — loss ablation (200k images, identical config)

| Loss | NSS | LL | AUC |
|---|---|---|---|
| KL, σ=1.0 | 3.243 | 2.436 | 0.902 |
| KL, σ=2.0 | 3.061 | 2.348 | 0.899 |
| KL, σ=4.0 | 2.626 | 2.047 | 0.889 |
| **NLL** | **3.271** | **2.449** | **0.903** |

Caption: Effect of the training objective, all else equal. NLL requires no
smoothing hyperparameter.

# LaTeX

## Table 1

\begin{tabular}{lrrrrrr}
\toprule
 & \multicolumn{3}{c}{ours} & \multicolumn{3}{c}{DeepGaze III} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Images & NSS & LL & AUC & NSS & LL & AUC \\
\midrule
1k & 1.330 & 0.994 & 0.796 & 1.995 & 1.461 & 0.845 \\
10k & 2.253 & 1.779 & 0.864 & 2.333 & 1.814 & 0.866 \\
50k & 2.765 & 2.128 & 0.886 & 2.438 & 1.901 & 0.871 \\
100k & 2.923 & 2.241 & 0.892 & 2.444 & 1.913 & 0.872 \\
200k & 3.271 & 2.449 & 0.903 & 2.406 & 1.883 & 0.870 \\
400k & 3.468 & 2.554 & 0.907 & -- & -- & -- \\
800k & 3.694 & 2.729 & 0.918 & -- & -- & -- \\
\bottomrule
\end{tabular}

## Table 2

\begin{tabular}{lrrr}
\toprule
Loss & NSS & LL & AUC \\
\midrule
KL, $\sigma=1.0$ & 3.243 & 2.436 & 0.902 \\
KL, $\sigma=2.0$ & 3.061 & 2.348 & 0.899 \\
KL, $\sigma=4.0$ & 2.626 & 2.047 & 0.889 \\
\textbf{NLL} & \textbf{3.271} & \textbf{2.449} & \textbf{0.903} \\
\bottomrule
\end{tabular}
