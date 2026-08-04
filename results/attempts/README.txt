Old training runs kept for reference only (not for submission).

OFFICIAL submission models (in results/):
  section5_initial_model.keras  — Section 5 baseline (~72.6% test)
  section6_final_model.keras    — Section 6/7 final (77.75% test)

ATTEMPTS (extras renamed from confusing "best_*" / "initial_*" copies):
  attempt_1_25epochs.*              — early 25-epoch run
  attempt_2_30epochs.*              — 30-epoch run
  attempt_3_35epochs.*              — 35-epoch run
  attempt_4_final_source_35epochs.* — source weights copied into section6_final
  attempt_failed_bn_7epochs.*       — BatchNorm experiment (stopped early)
  attempt_old_checkpoint.*          — older checkpoint leftover
