# Archived notebooks

These notebooks produced the controlled experiment reported in the top-level README,
or predate the current entry points. They are preserved as they were.

- `legacy_distillation_notebooks/run_optuna.ipynb` — the paired comparison between
  the feature-only distillation arm (15 trials) and the fine-tuning-only control arm
  (6 trials), sharing a starting checkpoint, a budget, a TPE seed, and a decision
  margin of 0.30 WER fixed before the data were examined.
- `legacy_distillation_notebooks/run_distill.ipynb` — the earlier standalone run of
  the feature-only module.
- `legacy_distillation_notebooks/run_feature_distill.ipynb` — the original single
  distillation entry point, since superseded by the four `run_{forward,reverse}_
  {warm,scratch}_distill.ipynb` notebooks under `code/distillation/`.

Their prose and their printed output are in Italian, including the verdict the
README quotes:

```
VERDETTO: distillazione NEUTRA vs controllo (delta +0.16 <= margine 0.3).
```

which reads: *distillation is NEUTRAL against the control (delta +0.16 is within the
0.30 margin)*.

They are deliberately **not** translated. The cell outputs are the evidence for the
reported numbers; editing the sources without re-executing them would leave source
and output disagreeing, and re-executing them would overwrite the record. Treat these
files as read-only.

The paths inside them point at the machine they ran on and no longer resolve. The
maintained entry points are `code/csrl_skeleton/runner.ipynb`,
`code/signformer/runner.ipynb` and the four `run_*_distill.ipynb` notebooks under
`code/distillation/`.
