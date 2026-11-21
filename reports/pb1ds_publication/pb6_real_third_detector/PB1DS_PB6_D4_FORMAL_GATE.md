# PB1-DS-PB6 — D4 Formal Post-D2 Gate

- Verdict: **PASS**
- Hard-stop rules enforced. On FAIL, no further scientific execution.

## Rule checks

- R1_categories_present: PASS (missing categories: none)
- R1_exact_70000_steps_seed0_complete: PASS (per-category final checkpoint@70000 + .complete + manifest complete/seed0/hash)
- R6_no_category_error: PASS (categories with error: none)
- R2_caches_complete: PASS (per-category score caches present on disk)
- R5_d3_gate_pass: PASS (D3 raw-score integrity audit == PASS)
- R5_all_cache_integrity: PASS (integrity passes 55/55)
- R3_teacher_sha256_matches_frozen: PASS (on_disk=f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec frozen=f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec)
- R4_1_config_num_steps_frozen: PASS (config num_steps=70000)
- R4_2_all_seed_zero: PASS (all manifest rows seed==0)
- R4_3_source_normal_training: PASS (per-category train/good > 0)
