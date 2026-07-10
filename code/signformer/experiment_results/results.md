# Experiment results

The results reported for this repository live in the root [README](../../../README.md).

In brief, on PHOENIX-2014T: the Signformer baseline reaches 39.54 dev WER at beam 1
and 41.53 test WER at beam 4; adding the full FD-CMKD distillation moves test WER to
40.90, a change the controlled experiment finds indistinguishable from fine-tuning
the student for longer.

`dev.ph14t` and `test.ph14t` in this directory are the upstream Signformer outputs.
