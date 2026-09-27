# Whole-scanpath reporting change — 2026-09-23

Each separately scored sampling draw is averaged within its trained checkpoint; these per-run means are then averaged with equal weight across training runs. Reported training ranges span per-run means. Raw evaluations, checkpoint selection, and conditional metrics are unchanged. The released ScanDiff reference has one draw and is unchanged.

The full-test-set control and descriptive length statistics remain single-draw controls and are unchanged.

## realism_subset_v2_test.csv

| Condition | Metric | Previous | Revised | Change | Revised run range |
|---|---|---:|---:|---:|---|
| dg3 / 1000 | mm_kld | 0.11888 | 0.12336 | +0.00448 | [0.11608, 0.13063] |
| dg3 / 1000 | sm_kld | 0.59201 | 0.59603 | +0.00402 | [0.58532, 0.60674] |
| dg3 / 10000 | mm_kld | 0.09127 | 0.09332 | +0.00205 | [0.09191, 0.09473] |
| dg3 / 10000 | sm_kld | 0.50786 | 0.51158 | +0.00372 | [0.50615, 0.51702] |
| dg3 / 50000 | mm_kld | 0.08247 | 0.08531 | +0.00284 | [0.08531, 0.08531] |
| dg3 / 50000 | sm_kld | 0.46797 | 0.48223 | +0.01426 | [0.48223, 0.48223] |
| dg3 / 100000 | mm_kld | 0.08785 | 0.08709 | -0.00076 | [0.08709, 0.08709] |
| dg3 / 100000 | sm_kld | 0.46929 | 0.46965 | +0.00036 | [0.46965, 0.46965] |
| dg3 / 200000 | mm_kld | 0.08432 | 0.08336 | -0.00096 | [0.08336, 0.08336] |
| dg3 / 200000 | sm_kld | 0.47027 | 0.46787 | -0.00240 | [0.46787, 0.46787] |
| ours / 1000 | mm_kld | 0.17117 | 0.17486 | +0.00369 | [0.17048, 0.18072] |
| ours / 1000 | sm_kld | 0.78034 | 0.78175 | +0.00140 | [0.77075, 0.80253] |
| ours / 10000 | mm_kld | 0.06667 | 0.06730 | +0.00063 | [0.06468, 0.06976] |
| ours / 10000 | sm_kld | 0.41160 | 0.41783 | +0.00623 | [0.40238, 0.43681] |
| ours / 50000 | mm_kld | 0.05119 | 0.05301 | +0.00182 | [0.05173, 0.05432] |
| ours / 50000 | sm_kld | 0.35075 | 0.35427 | +0.00352 | [0.34171, 0.36149] |
| ours / 100000 | mm_kld | 0.04852 | 0.04827 | -0.00024 | [0.04606, 0.05049] |
| ours / 100000 | sm_kld | 0.33042 | 0.32947 | -0.00095 | [0.31790, 0.34105] |
| ours / 200000 | mm_kld | 0.04359 | 0.04209 | -0.00150 | [0.04084, 0.04334] |
| ours / 200000 | sm_kld | 0.29282 | 0.30131 | +0.00850 | [0.29599, 0.30664] |
| ours / 400000 | mm_kld | 0.02959 | 0.03188 | +0.00229 | [0.03188, 0.03188] |
| ours / 400000 | sm_kld | 0.25329 | 0.25996 | +0.00667 | [0.25996, 0.25996] |
| ours / 800000 | mm_kld | 0.02812 | 0.02927 | +0.00115 | [0.02927, 0.02927] |
| ours / 800000 | sm_kld | 0.24016 | 0.24669 | +0.00653 | [0.24669, 0.24669] |
| scandiff / 1000 | mm_kld | 0.13748 | 0.13970 | +0.00222 | [0.11592, 0.16347] |
| scandiff / 1000 | sm_kld | 0.57259 | 0.57892 | +0.00633 | [0.56693, 0.59091] |
| scandiff / 10000 | mm_kld | 0.06104 | 0.05962 | -0.00142 | [0.05790, 0.06134] |
| scandiff / 10000 | sm_kld | 0.21534 | 0.21738 | +0.00204 | [0.20769, 0.22707] |
| scandiff / 50000 | mm_kld | 0.04415 | 0.04274 | -0.00140 | [0.04264, 0.04285] |
| scandiff / 50000 | sm_kld | 0.13859 | 0.13848 | -0.00011 | [0.13629, 0.14066] |
| scandiff / 100000 | mm_kld | 0.04581 | 0.04705 | +0.00124 | [0.04705, 0.04705] |
| scandiff / 100000 | sm_kld | 0.14652 | 0.15032 | +0.00380 | [0.15032, 0.15032] |
| scandiff / 200000 | mm_kld | 0.04571 | 0.04645 | +0.00074 | [0.04645, 0.04645] |
| scandiff / 200000 | sm_kld | 0.16939 | 0.16731 | -0.00209 | [0.16731, 0.16731] |
## xfam_mit_v2_test.csv

| Condition | Metric | Previous | Revised | Change | Revised run range |
|---|---|---:|---:|---:|---|
| dg3 / mit_only / salicon / -2 | mm_kld | 0.03678 | 0.03589 | -0.00089 | [0.03453, 0.03725] |
| dg3 / mit_only / salicon / -2 | sm_kld_corr | 0.04733 | 0.04664 | -0.00069 | [0.04569, 0.04759] |
| dg3 / mit_only / scratch / 0 | mm_kld | 0.04370 | 0.04156 | -0.00215 | [0.04052, 0.04259] |
| dg3 / mit_only / scratch / 0 | sm_kld_corr | 0.06425 | 0.06001 | -0.00425 | [0.05601, 0.06400] |
| dg3 / mit_only / full1k / 1000 | mm_kld | 0.04219 | 0.03974 | -0.00245 | [0.03964, 0.03985] |
| dg3 / mit_only / full1k / 1000 | sm_kld_corr | 0.05607 | 0.05380 | -0.00227 | [0.05343, 0.05416] |
| dg3 / mit_only / full10k / 10000 | mm_kld | 0.04144 | 0.03993 | -0.00151 | [0.03985, 0.04001] |
| dg3 / mit_only / full10k / 10000 | sm_kld_corr | 0.05499 | 0.05379 | -0.00120 | [0.05199, 0.05559] |
| dg3 / mit_only / full50k / 50000 | mm_kld | 0.04309 | 0.04123 | -0.00186 | [0.03943, 0.04303] |
| dg3 / mit_only / full50k / 50000 | sm_kld_corr | 0.05919 | 0.05613 | -0.00306 | [0.05156, 0.06070] |
| dg3 / mit_only / full100k / 100000 | mm_kld | 0.04185 | 0.04090 | -0.00095 | [0.04018, 0.04161] |
| dg3 / mit_only / full100k / 100000 | sm_kld_corr | 0.05374 | 0.05337 | -0.00037 | [0.05130, 0.05544] |
| dg3 / mit_only / full200k / 200000 | mm_kld | 0.04595 | 0.04404 | -0.00191 | [0.04384, 0.04424] |
| dg3 / mit_only / full200k / 200000 | sm_kld_corr | 0.06676 | 0.06388 | -0.00288 | [0.06353, 0.06423] |
| ours / joint / scratch / 0 | mm_kld | 0.02832 | 0.02897 | +0.00064 | [0.02425, 0.03368] |
| ours / joint / scratch / 0 | sm_kld_corr | 0.03739 | 0.03597 | -0.00142 | [0.02526, 0.04669] |
| ours / joint / bv800k / 800000 | mm_kld | 0.01988 | 0.02095 | +0.00106 | [0.01831, 0.02359] |
| ours / joint / bv800k / 800000 | sm_kld_corr | 0.02442 | 0.02450 | +0.00008 | [0.01894, 0.03007] |
| ours / mit_only / scratch / 0 | mm_kld | 0.05752 | 0.06004 | +0.00251 | [0.04449, 0.07558] |
| ours / mit_only / scratch / 0 | sm_kld_corr | 0.05867 | 0.05677 | -0.00190 | [0.04146, 0.07209] |
| ours / mit_only / bv1k / 1000 | mm_kld | 0.05165 | 0.05395 | +0.00230 | [0.04385, 0.06405] |
| ours / mit_only / bv1k / 1000 | sm_kld_corr | 0.05797 | 0.05687 | -0.00110 | [0.04578, 0.06796] |
| ours / mit_only / bv10k / 10000 | mm_kld | 0.03075 | 0.03356 | +0.00281 | [0.02826, 0.03886] |
| ours / mit_only / bv10k / 10000 | sm_kld_corr | 0.03834 | 0.03773 | -0.00060 | [0.03080, 0.04467] |
| ours / mit_only / bv50k / 50000 | mm_kld | 0.03113 | 0.03181 | +0.00068 | [0.02844, 0.03519] |
| ours / mit_only / bv50k / 50000 | sm_kld_corr | 0.04091 | 0.03921 | -0.00171 | [0.03336, 0.04505] |
| ours / mit_only / bv100k / 100000 | mm_kld | 0.02875 | 0.02921 | +0.00045 | [0.02624, 0.03218] |
| ours / mit_only / bv100k / 100000 | sm_kld_corr | 0.03318 | 0.03362 | +0.00044 | [0.03068, 0.03656] |
| ours / mit_only / bv200k / 200000 | mm_kld | 0.02664 | 0.02905 | +0.00241 | [0.02679, 0.03131] |
| ours / mit_only / bv200k / 200000 | sm_kld_corr | 0.04002 | 0.03904 | -0.00098 | [0.03619, 0.04189] |
| ours / mit_only / bv400k / 400000 | mm_kld | 0.02245 | 0.02508 | +0.00263 | [0.02384, 0.02632] |
| ours / mit_only / bv400k / 400000 | sm_kld_corr | 0.02729 | 0.02819 | +0.00090 | [0.02748, 0.02890] |
| ours / mit_only / bv800k / 800000 | mm_kld | 0.02223 | 0.02382 | +0.00159 | [0.02129, 0.02635] |
| ours / mit_only / bv800k / 800000 | sm_kld_corr | 0.02700 | 0.02742 | +0.00042 | [0.02615, 0.02869] |
| scandiff / joint / released / -1 | mm_kld | 0.03012 | 0.03012 | +0.00000 | [0.03012, 0.03012] |
| scandiff / joint / released / -1 | sm_kld_corr | 0.06772 | 0.06772 | +0.00000 | [0.06772, 0.06772] |
| scandiff / joint / scratch / 0 | mm_kld | 0.02698 | 0.02573 | -0.00125 | [0.02521, 0.02625] |
| scandiff / joint / scratch / 0 | sm_kld_corr | 0.04272 | 0.04168 | -0.00104 | [0.03685, 0.04652] |
| scandiff / joint / av_pretrained / 100000 | mm_kld | 0.04018 | 0.03946 | -0.00073 | [0.03342, 0.04550] |
| scandiff / joint / av_pretrained / 100000 | sm_kld_corr | 0.07369 | 0.07218 | -0.00152 | [0.07209, 0.07226] |
| scandiff / mit_only / scratch / 0 | mm_kld | 0.05714 | 0.05753 | +0.00039 | [0.04752, 0.06754] |
| scandiff / mit_only / scratch / 0 | sm_kld_corr | 0.17587 | 0.17909 | +0.00322 | [0.13798, 0.22019] |
| scandiff / mit_only / av_pretrained / 1000 | mm_kld | 0.03310 | 0.03407 | +0.00097 | [0.03352, 0.03463] |
| scandiff / mit_only / av_pretrained / 1000 | sm_kld_corr | 0.06153 | 0.06152 | -0.00002 | [0.05825, 0.06478] |
| scandiff / mit_only / av_pretrained / 10000 | mm_kld | 0.05189 | 0.05136 | -0.00054 | [0.04985, 0.05286] |
| scandiff / mit_only / av_pretrained / 10000 | sm_kld_corr | 0.14808 | 0.15226 | +0.00418 | [0.11900, 0.18552] |
| scandiff / mit_only / av_pretrained / 50000 | mm_kld | 0.04373 | 0.04524 | +0.00150 | [0.04079, 0.04968] |
| scandiff / mit_only / av_pretrained / 50000 | sm_kld_corr | 0.10115 | 0.10053 | -0.00063 | [0.09850, 0.10256] |
| scandiff / mit_only / av_pretrained / 100000 | mm_kld | 0.04983 | 0.05099 | +0.00116 | [0.05034, 0.05164] |
| scandiff / mit_only / av_pretrained / 100000 | sm_kld_corr | 0.09181 | 0.09321 | +0.00140 | [0.09052, 0.09590] |
| scandiff / mit_only / av_pretrained / 200000 | mm_kld | 0.04120 | 0.03947 | -0.00173 | [0.03760, 0.04135] |
| scandiff / mit_only / av_pretrained / 200000 | sm_kld_corr | 0.08114 | 0.07769 | -0.00346 | [0.05267, 0.10270] |

## Interpretation and scope

- Our model remains monotonic on both in-domain divergences.
- On human transfer, our model's MultiMatch-KLD is now monotonic across pretraining scales; ScanMatch-KLD remains non-monotonic. Its 1k ScanMatch mean is very slightly above the no-pretraining mean (0.05687 versus 0.05677).
- DeepGaze III's lowest in-domain means now occur at 200k on both divergences, but this run is incomplete. Among completed runs, MultiMatch remains lowest at 50k; ScanMatch is lowest at 100k. Claims that neither metric improves after 50k must be revised.
- ScanDiff remains best at 1k for MIT-only transfer and 50k for in-domain evaluation.
- The main cross-family ranking comparisons and the 100k ScanDiff joint-training reversal are unchanged.
- Figures 2, 6 and 7 and the combined Figure 12 alternative were regenerated on TEST. Their bars now describe training-run means after within-run sampling averaging.
- Checkpoint selection, evaluators, raw scores, conditional metrics, single-draw full-test control, and descriptive length statistics were not changed.
- 86 reported per-run rows were checked against independent calculations from raw draw scores; identities and draw counts matched.
- Backups: docs/report_tables/archive/sampling_mean_20260923_before/
- Pending ScanDiff joint-1k jobs remain a separate follow-up; this reporting change does not alter their frozen training/evaluation protocol.
