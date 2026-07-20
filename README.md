# Streaming Hierarchical Bayesian Model for Musical Structure Inference

This repository contains a C++ implementation of a predictive processing bayesian inference model as part of my reasearch on computational modelling of musical structure perception. The model can be classed as a hierarchical latent-variable Noisy-OR statistical model for sequential processing of multivariate data, developed theoretically following predictive processing principles. It ingests (discrete-time) multivariate time series incrementally (one step at a time), learning along the way a set of sparse latent "generators" acting as temporal (autoregressive) predictors, and uses combinations of these generators to reconstruct/interpret current observations and predict future observations.

The code is being developed alongside a theoretical model and corresponding experimental work, and is written as a self-contained C++ library using Eigen, with a JUCE-based interface and logging/state-tracing tools built on top to facilitate experimentation.

#### How to use:

This implementation includes a state and tracing system designed for experimentation. It makes runs reproducible, editable, inspectable, and shareable:
- Model configurations and checkpoints are stored using `.state` files, which describe the current model state. Loading a `.state` file reconstructs the corresponding model state, including learned parameters and temporal context, so that a run can be resumed or reproduced from that point. The state loader also supports partial state descriptions, such that a `.state` file specifying, for example, only the input data or only the generator parameters, overwrites only the corresponding fields in current state.
- The interface provides a live view of the current model state. It displays the input window and, for each layer, the main quantities involved in inference and learning. Most of these quantities are shown as grayscale matrices or vector, and several parameters can also be edited directly from the GUI, allowing interactive experimentation while the model is running.  

To run the model:

- Load a `.state` file, or edit an existing pre-loaded one.
- Run the model interactively (step-wise) or run asynchronously for a predefined number of steps.
- Inspect the live layer display during learning, and/or enable logging to record the run, using the trace plotter to inspect a logged time series.
  
____
## Mathematical Model

#### Summary
The model can be understood as a factorial hierarchical latent-variable model: At each layer, the latent representation consists of multiple simultaneously active generators whose predictions compose through a Noisy-OR emission model.

#### Input
The input data to the model, referred to hereafter as the **sensory input**, is a time-discrete multivariate random process which is ingested sequentially one time step at a time.

```math
X := (X_t)_{t \in \mathbb{N}},
\qquad
X_t \in [0,1]^n,
```

where the sensory input is only observed by the bottom layer. All layers above the bottom layer likewise also have input $X := (X_t)_{t \in \mathbb{N}}$ which is given by the posterior expectation over hidden states of the layer below.

---

#### Input State Space

At a fixed time $t$ and layer $l$, the input state space consists of the current observation vector $\mathbf{x}^l_t \in [0,1]^{N_l}$ and an order-$o_l$ context matrix

```math
C^l_t =
[\mathbf{x}^l_{t-1};
 \mathbf{x}^l_{t-2};
 \cdots;
 \mathbf{x}^l_{t-o_l}]
\in
[0,1]^{N_l \times o_l},
```

whose columns contain the previous $o_l$ observation vectors.

---

#### Generators

Each layer contains a set of generators. Each generator $\mathbf{G}_k$ represents a sparse local temporal regularity represented as a pair

```math
G_k=(F_k,R_k),
```

where

- a **context detector** $F_k$ measures the presence of a characteristic pattern in the lagged context $C$;

- a **prediction vector** $R_k$ assigns a Bernoulli activation probability to each observation dimension in $x_t$, conditioned on the inferred presence of the context pattern measured by $F_k$.

Each active generator effectively gives a partial positive prediction over a subset of the observation vector's dimensions, based on the presence of a characteristic pattern inferred in the context $C$. A prediction of $0$ for a specific channel by a specific $GG therefore does not constitute negative evidence, only absence of positive evidence.

---

#### A factorial latent representation

We introduce a binary hidden state $z_k$ representing the presence of the pattern encoded by $G_k$ as an active explanation for the current observation $x_t$. Since many generators can jointly explain $x_t$, the corresponding latent representation is factorial. 

###### **Hidden States**

For each layer $l$, we introduce binary hidden variables

```math
z^l_{t,k} \in \{0,1\},
```

where $z^l_{t,k}=1$ means that generator $k$ is active as a latent cause of the current observation $\mathbf{x}^l_t$.

The full latent state is the binary vector

```math
\mathbf{z}^l_t =
(z^l_{t,1}, \dots, z^l_{t,K_l})
\in \{0,1\}^{K_l}.
```

This is a sparse distributed representation: multiple generators may be active simultaneously. This is what allows an observation to be represented compositionally, as a combination of latent causes rather than as a single mutually exclusive state.

---


---

#### Noisy-OR observation model

As seen above, a generator's prediction typically specifies only positive evidence over a subset of observation dimensions in $x_t$. The complete prediction for $x_t$ is obtained by composing the predictions of all inferred active generators via a Noisy-OR observation model.

Given the inferred active generators $z$, the predicted observation is

```math
\hat{x}_{t,i}
=
1-
\prod_k
\left(
1-z_{t,k}R_{k,i}
\right).
```

This is a Noisy-OR product taken elementwise over the observation dimensions.

---

#### Hierarchy

The posterior expectations of generator activations at one layer are represented as channels at the layer above, such that the sequence of inferred generator activations forms the input to the layer above.

The table below sets out the main objects for each layer, showing the shared dimensionalities across layers: the number of generators at one layer corresponds to the number of input channels at the layer above.

| layer | input state | output state | generators $G_k^l=(F_k^l,R_k^l)$ | hidden state | downward prior |
|------|-------------|--------------|----------------------------------|--------------|----------------|
| $l_2$ | $\mathbf{x}_t^1\in[0,1]^{K_1},\;C_t^2\in[0,1]^{K_1\times o_2}$ | $\mathbf{x}_t^2\in[0,1]^{K_2},\;C_t^3\in[0,1]^{K_2\times o_3}$ | $F_k^2\in\mathbb{R}^{K_1\times o_2},\quad R_k^2\in[0,1]^{K_1}$ | $\mathbf{z}_t^2\in\{0,1\}^{K_2}$ | $\boldsymbol{\tau}_t^2=f(\mathbf{x}_t^2)\in[0,1]^{K_2}$ |
| $l_1$ | $\mathbf{x}_t^0\in[0,1]^{n_0},\;C_t^1\in[0,1]^{n_0\times o_1}$ | $\mathbf{x}_t^1\in[0,1]^{K_1},\;C_t^2\in[0,1]^{K_1\times o_2}$ | $F_k^1\in\mathbb{R}^{n_0\times o_1},\quad R_k^1\in[0,1]^{n_0}$ | $\mathbf{z}_t^1\in\{0,1\}^{K_1}$ | $\boldsymbol{\tau}_t^1=f(\mathbf{x}_t^1)\in[0,1]^{K_1}$ |

##### Schedule

-New observation arrives at bottom layer. Posterior computed ad bottom layer, sent upwards.
-Repeat up the hierarchy.
-Top level reached. Current observation shifted to context, next-step prior computed
-Repeat down the hierarchy.
-bottom level reached. All layers now have the (predictive) prior ready for the next observation


###### **Generator Activation Prior**

Before observing $\mathbf{x}^l_t$, each generator has an activation prior

```math
\alpha^l_{t,k}
=
p(z^l_{t,k}=1 \mid C^l_t, \tau^l_t),
```

where $\tau^l_{t,k}$ is top-down support arriving from the layer above. In the current model, this prior is computed by combining bottom-up context support $q^l_{t,k}$, top-down support $\tau^l_{t,k}$, and learned/manual generator parameters.

Each generator has:

- base rate $b_k$,
- bottom-up weight $w_k$,
- evidence amplitude $A_k$,
- centering value $c_k$.

The combined evidence is

```math
e^l_{t,k}
=
w_k q^l_{t,k}
+
(1-w_k)\tau^l_{t,k}.
```

The activation prior is then

```math
\alpha^l_{t,k}
=
\sigma
\left(
\mathrm{logit}(b_k)
+
A_k(e^l_{t,k}-c_k)
\right).
```

Thus $F_k$ does not directly determine whether a generator is active. It contributes bottom-up evidence, which is combined with top-down evidence and transformed into an activation prior.

---

###### **Candidate Hidden States**

In principle, the latent state space has size

```math
2^{K_l}.
```

However, the current implementation does not usually evaluate all possible states. Instead, it first selects a candidate set of generators using thresholds on:

- context support $q_k$,
- top-down support $\tau_k$,
- singleton observation score,
- optionally, activation prior $\alpha_k$.

Let the selected generators be

```math
\mathcal{K}^l_t \subseteq \{1,\dots,K_l\}.
```

The model then enumerates candidate hidden states using subsets of these selected generators:

```math
\mathcal{S}^l_t \subseteq \{0,1\}^{K_l}.
```

If all generators are selected, this reduces to the full hidden state space. Otherwise, inference is performed only over the selected candidate states.

---

###### **Prior Over Hidden States**

For a state $\mathbf{z}_{t,m}^l \in \mathcal{S}^l_t$, the unnormalised Bernoulli product prior is

```math
\tilde{\pi}^l_{t,m}
=
\prod_k
(\alpha^l_{t,k})^{z^l_{t,m,k}}
(1-\alpha^l_{t,k})^{1-z^l_{t,m,k}}.
```

Because inference is restricted to the candidate set, this prior is normalised over $\mathcal{S}^l_t$:

```math
\pi^l_{t,m}
=
\frac{\tilde{\pi}^l_{t,m}}
{\sum_{\mathbf{z}_{t,m'}^l \in \mathcal{S}^l_t}
\tilde{\pi}^l_{t,m'}}.
```

This means the prior is a distribution over the current candidate states, not necessarily over all $2^{K_l}$ possible states.

---

###### **Noisy-OR Observation Model**

Given a candidate latent state $\mathbf{z}^l_{t,m}$, the model predicts each observation channel using a Noisy-OR composition of the active generators.

The current implementation also includes a leak probability $\lambda_i$ for each channel. The predicted probability of channel $i$ being active is

```math
\hat{x}^l_{t,m,i}
=
p(x^l_{t,i}=1 \mid \mathbf{z}^l_{t,m})
=
1
-
(1-\lambda_i)
\prod_k
(1-z^l_{t,m,k} r^l_{k,i}).
```

Thus, active generators jointly explain the observation vector. If different generators predict different channels, their predictions combine into a composite observation. If they predict the same channel, they compete through explaining-away rather than receiving independent additive credit.

The likelihood of the full observation vector is modelled channel-wise:

```math
\ell^l_{t,m}
=
p(\mathbf{x}^l_t \mid \mathbf{z}^l_{t,m})
=
\prod_i
(\hat{x}^l_{t,m,i})^{x^l_{t,i}}
(1-\hat{x}^l_{t,m,i})^{1-x^l_{t,i}}.
```

For binary observations this is the Bernoulli likelihood. For continuous values in $[0,1]$, it acts as a soft Bernoulli / cross-entropy likelihood.

---

###### **Posterior Inference**

For each candidate state, the posterior is

```math
\rho^l_{t,m}
=
p(\mathbf{z}^l_{t,m} \mid \mathbf{x}^l_t, C^l_t, \tau^l_t)
=
\frac{
\ell^l_{t,m}\pi^l_{t,m}
}{
\sum_{m'}
\ell^l_{t,m'}\pi^l_{t,m'}
}.
```

The posterior marginal activation of generator $k$ is

```math
\mu^l_{t,k}
=
\sum_m
\rho^l_{t,m}
z^l_{t,m,k}.
```

The posterior reconstruction of the observation is

```math
\hat{\mathbf{x}}^l_t
=
\sum_m
\rho^l_{t,m}
\hat{\mathbf{x}}^l_{t,m}.
```

The vector $\boldsymbol{\mu}^l_t$ is passed upward as the observation for the next layer.

---

###### **Top-Down Prediction**

After the upward inference and bayes update pass, the model performs a top-down prior step.

For a lower layer $l$, the top-down support used for the next timestep is the reconstruction produced by the layer above:

```math
\tau^l_{t+1}
=
\hat{\mathbf{x}}^{l+1}_t.
```

The top layer may receive external top-down support, or else a zero vector.

Each layer then shifts the current observation into its context window and computes the next activation prior:

```math
\alpha^l_{t+1,k}
=
p(z^l_{t+1,k}=1 \mid C^l_{t+1}, \tau^l_{t+1}).
```

So the model maintains two time-indexed quantities:

- $\alpha_t$: the inherited prior used for the current E-step,
- $\alpha_{t+1}$: the next prior prepared after observing the current timestep.

---

###### **Learning With Online EM / Generalized EM**

Learning is online. At each timestep, the model first performs an E-step by computing the posterior over candidate hidden states. It then updates generator parameters using posterior responsibilities.

The necessary E-step quantities are:

```math
\rho_m
=
p(\mathbf{z}_m \mid \mathbf{x}, C, \tau),
```

and the marginal generator activation

```math
\mu_k
=
\sum_m \rho_m z_{m,k}.
```

The prediction vectors $R$, filters $F$, and base rates $b$ are then updated.

The prediction update uses a Noisy-OR responsibility term. For generator $k$ and channel $i$, the expected causal credit is approximately

```math
\gamma_{k,i}
=
x_i
\sum_m
\rho_m
z_{m,k}
\frac{r_{k,i}}{\hat{x}_{m,i}}.
```

The prediction vector update is

```math
\Delta r_{k,i}
=
\eta_R
(\gamma_{k,i} - \mu_k r_{k,i}).
```

This increases $r_{k,i}$ when generator $k$ receives posterior responsibility for explaining an active channel, and decreases it when the generator is active but the channel is not supported.

The filter update tries to make the contextual activation prior $\alpha_k$ better match the posterior marginal $\mu_k$. The current update is a gradient-like rule:

```math
\Delta F_{k,a,b}
=
\eta_F
(\mu_k-\alpha_k)
A_k w_k
\frac{C_{a,b}-q_k}
{\epsilon + \sum_{a,b} F_{k,a,b}}.
```

The base rate update is

```math
\Delta b_k
=
\eta_b(\mu_k-b_k).
```

Thus, the model is not performing a closed-form batch M-step. It is better described as an online EM / generalized EM procedure.

The parameters $w_k$, $A_k$, $c_k$, candidate thresholds, and learning rates are currently user-editable controls rather than automatically learned parameters.

---

###### **Interpretation**

At a single layer, the model maps recent context and top-down support to a predictive distribution over the next/current observation:

```math
(C_t, \tau_t)
\mapsto
\alpha_t
\mapsto
p(\mathbf{z}_t \mid C_t,\tau_t)
\mapsto
p(\mathbf{x}_t \mid C_t,\tau_t).
```

The generators therefore serve two roles:

1. They are temporal detectors over recent context.
2. They are latent causes whose active combinations predict the observation vector.

This makes the representation compositional: an observation can be explained by several active generators, each accounting for part of the multivariate signal.

Rather than thinking of higher layers as simply storing nested chunks, a better picture is that each layer produces a moving field of posterior beliefs. These posterior beliefs become the signal observed by the next layer, while higher layers return top-down support that shapes the priors of lower-layer generators.

# Experiment framework

An experiment is a collection of independent runs derived from one shared model state.
The inference model itself does not contain an `Experiment` object. Experiments are an
orchestration and filesystem layer built on the existing patch-based `.state` format.

## Core model

Each run starts from:

```text
resolved run state = shared.state + variation.state
```

`shared.state` is a complete parent state. A run's `variation.state` is a child patch
that contains only the fields that differ for that run. Omitted fields are inherited
from `shared.state` through the normal state-loading rules.

A single-run experiment uses the same structure. Its variation may be empty, so the
resolved starting state is simply `shared.state`.

## Folder layout

Experiments are stored below the project-local library:

```text
Library/
  Experiments/
    exp_YY-MM-DD_HH-MM/
      experiment.json
      shared.state
      runs/
        000_baseline/
          variation.state
          resolved_start.state
          final.state
          log.jsonl
          diagnostics.json
        001_generated_seed_.../
          variation.state
          resolved_start.state
          final.state
          log.jsonl
          diagnostics.json
```

### `experiment.json`

The manifest is the GUI's table of contents. It stores the experiment name, steps per
run, run ordering, run status, seeds, and paths to each run's files. It does not replace
the `.state` format.

### `shared.state`

A complete state common to all runs. This normally contains the input sequence,
hierarchy dimensions, initial context, generator matrices and parameters, candidate
selection settings, EM settings, and the experiment comment.

### `variation.state`

A partial state patch for one run. It can be empty for a baseline run. Generated runs
materialize concrete variation files immediately, rather than storing an abstract
randomization recipe. This makes the experiment reproducible even if the generator UI
or code changes later.

### `resolved_start.state`

The complete state obtained after loading `shared.state` and applying the run's
`variation.state`. This is the exact checkpoint used to start the run.

### `final.state`

The learned state after the configured number of steps.

### `log.jsonl`

The full run trace, including state snapshots and per-step layer values.

### `diagnostics.json`

The summary diagnostics computed for the completed run.

## Run execution

Every experiment run is independent:

1. Load `shared.state`.
2. Apply the selected run's `variation.state` patch.
3. Save the result as `resolved_start.state`.
4. Execute the configured number of steps.
5. Save `final.state`, `log.jsonl`, and `diagnostics.json`.
6. Update the run entry in `experiment.json`.

A run never starts from another run's final state unless that final state is explicitly
made into a new shared state.

## Generate Runs

`Generate Runs...` creates many concrete child variation files. Matrix and scalar
variations can be combined in the same sweep.

### R and F

`R` and `F` can be randomized independently. Their entries use the model's existing
Beta initializer:

```text
R RANDOM alphaR betaR seedR
F RANDOM alphaF betaF seedF
```

Each layer receives deterministic seeds derived from the run seed. Filter rows are
normalized by the existing state loader after sampling.

`R` and `F` currently support Beta randomization only; they are not linear-grid
parameters.

### Scalar generator parameters

The following generator parameters can each be set to one of three modes:

- **Inherit**: keep the value from `shared.state`.
- **Random uniform**: sample independently for every generator in every layer from the
  selected `[minimum, maximum]` interval.
- **Linear grid**: use evenly spaced values between the selected minimum and maximum.
  A grid value is applied to all generators in all layers for that run.

Supported generator parameters:

- base rate
- bottom-up weight
- evidence amplitude
- centering

### EM parameters

The same inherit/random/grid modes are available for:

- EM enabled (`0` or `1`)
- R learning rate
- F learning rate
- base-rate learning rate
- EM epsilon
- observation binarization (`0` or `1`)
- observation threshold

A sampled or gridded EM value is applied to every layer in that run. Boolean settings
use `0` for false and `1` for true. Linear grids for boolean fields use the two endpoints;
random boolean values are sampled uniformly from the selected interval and thresholded
at `0.5`.

### Combining grids and randomization

All scalar parameters in **Linear grid** mode form a Cartesian product. For example:

```text
base grid:       0.05, 0.10, 0.15
eta R grid:      0.001, 0.005
```

creates `3 x 2 = 6` grid points.

If any parameter is random, the **Random samples / grid point** value controls how many
independent random runs are generated for every grid point. For example, six grid
points with four random samples per point produce 24 runs.

Random parameters include optional R/F randomization and scalar fields in **Random
uniform** mode. When no random parameter is selected, each grid point produces exactly
one run.

The generator enforces a 10,000-run safety limit.

## Reproducibility

Every generated run has a stored run seed. More importantly, the generated
`variation.state` is concrete:

- R/F random directives contain fixed seeds and Beta shapes.
- Random scalar generator values are written into a complete `PARAMETERS` matrix.
- Random and gridded EM values are written into a complete `EM` directive.

Therefore the experiment can be rerun from its folder without consulting the original
generation dialog.

## Editing and inspection

The main GUI continues to display one live model state. The experiment window manages
which run is selected and can load a run's resolved start or final state. Selecting a
completed run loads its final state automatically.

Use **Show Variation** to inspect the exact child patch associated with a run. This is
the clearest way to verify which fields differ from the shared state.

