# §S181-AUDIT Wave 1 — Track B / B2 — buffer composition

Source log: `events_66cc066ac32549ebad6f882f9241d54c.jsonl` (6 snapshots).

## Position-class trajectory

| step | buffer size | n sampled | colony frac | extension frac | neither frac | colony mean v_target | extension mean v_target | neither mean v_target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 27743 | 5000 | 0.0938 | 0.1190 | 0.7872 | +0.0512 | +0.0605 | +0.0201 |
| 1000 | 53833 | 5000 | 0.0906 | 0.1144 | 0.7950 | +0.0177 | -0.0490 | +0.0438 |
| 1500 | 79256 | 5000 | 0.0924 | 0.1156 | 0.7920 | -0.0022 | -0.1176 | +0.0283 |
| 2000 | 105364 | 5000 | 0.0900 | 0.1248 | 0.7852 | +0.0556 | -0.0753 | +0.0458 |
| 2500 | 133860 | 5000 | 0.0912 | 0.1138 | 0.7950 | +0.0395 | -0.0035 | +0.0211 |
| 3000 | 160327 | 5000 | 0.0984 | 0.1116 | 0.7900 | +0.0081 | -0.0269 | +0.0248 |

## V-B-C verdict — feedback loop guard

From `B_launch_and_analysis_spec.md` §Aggregation: V-B-C fires if `colony_frac` > 0.50 by step 2000.

- step 500: colony_frac = 0.0938 (469/5000) 
- step 1000: colony_frac = 0.0906 (453/5000) 
- step 1500: colony_frac = 0.0924 (462/5000) 
- step 2000: colony_frac = 0.0900 (450/5000) 
- step 2500: colony_frac = 0.0912 (456/5000) 
- step 3000: colony_frac = 0.0984 (492/5000) 

**V-B-C trigger (literal):** NO

