# Cached random-offset score probe

| method | all-12 rank / strict / margin | near-6 rank / strict / margin | far-6 rank / strict / margin | fixed-draw rank / strict / margin |
|---|---|---|---|---|
| original | 0.4567 / 0.4567 / -0.0016 | 0.4726 / 0.4726 / 0.0074 | 0.4408 / 0.4408 / -0.0106 | 0.4932 / 0.4932 / 0.0096 |
| konet_full_epoch9 | 0.4459 / 0.4368 / -0.0199 | 0.4647 / 0.4503 / -0.0015 | 0.4272 / 0.4234 / -0.0383 | 0.4098 / 0.4041 / -0.0044 |
| konet_lora_r8a16_epoch5 | 0.4707 / 0.4707 / -0.0062 | 0.4889 / 0.4889 / 0.0077 | 0.4524 / 0.4524 / -0.0200 | 0.4776 / 0.4776 / -0.0055 |

Exact primary uses all 12 equally weighted targets. Fixed draw uses NumPy default_rng(1701) and counts {-320: 10, -240: 7, -160: 7, -120: 14, -80: 13, -40: 4, 40: 9, 80: 7, 120: 7, 160: 7, 240: 6, 320: 5}.
Rank assigns ties 0.5; strict assigns ties 0. This is not binary synchronization accuracy.
No method was selected from held-out outcomes.
