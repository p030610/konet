# Prespecified cached random-offset score probe

Before computation, this probe fixes the candidate targets to the 12 cached signed lags `-320, -240, -160, -120, -80, -40, 40, 80, 120, 160, 240, 320` ms. They are the available six absolute distances in the released held-out score table, not a uniform sample over all offsets from 40 to 640 ms.

Primary result: for each model and window, average rank-win (tie = 0.5) and center-minus-target margin across all 12 targets; then aggregate window -> recording -> equal speaker over the unchanged 96 windows, 44 recordings, and 4 speakers. Near targets (`abs(lag) in {40,80,120}`) and far targets (`abs(lag) in {160,240,320}`) receive equal target weight within their separate reporting groups.

Secondary illustration: NumPy `default_rng(1701)` independently chooses one of the 12 targets per window. The exact selected lag is written to `random_offset_selection.csv` and used unchanged for Original, Full, and LoRA. It is not used to choose a model. `strict_single_target_win_tie_half` is explicitly a score-ordering quantity, not binary synchronization accuracy.

The score source is the released heldout_scores.csv; methods are Original, Full, and LoRA. This is neither new training nor a checkpoint/weight interpolation. No development score cache is available; all held-out comparisons are exploratory.

