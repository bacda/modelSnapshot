# Noisy-OR state-file examples

Files marked **cold start** can initialize a model without a previous state.
Files marked **patch** are intended for `loadModelState(path, &previous)`.

| File | Type | Demonstrates |
|---|---|---|
| `00_input_only_cold_start.state` | Cold start | Infer N; default K/order; random F/R |
| `01_input_only_patch.state` | Patch | Replace only X and retain the live model |
| `02_random_single_layer.state` | Cold start | Explicit Beta-random F/R |
| `03_full_inline_single_layer.state` | Cold start | Every layer field inline |
| `04_external_matrices.state` | Cold start | Relative external matrix files |
| `05_two_layer_random.state` | Cold start | Compatible two-layer hierarchy |
| `06_learning_enabled.state` | Cold start | Online EM enabled with soft observations |
| `07_candidate_policy_patch.state` | Patch | Change only candidate thresholds |
| `08_learning_patch.state` | Patch | Change only learning settings |
| `09_reinitialize_R_patch.state` | Patch | Replace only R with a new random draw |
| `10_checkpoint_patch.state` | Patch | Replace context, top-down support, and input index |
| `11_non_looping_input.state` | Cold start | Non-looping sequence and initial index |
| `12_inline_R_random_F.state` | Cold start | Mix initialization sources |

External matrix files contain numeric values only. Their shapes are declared in
`04_external_matrices.state`, and paths are resolved relative to that file.

The candidate-setting order is:

```text
context topDown observation activation useActivation maximumGenerators
```

The EM-setting order is:

```text
enabled R-rate F-rate base-rate epsilon binarize binarization-threshold
```
