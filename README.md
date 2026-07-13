# Streaming Hierarchical Bayesian Model for Musical Structure Inference

This repo contains a working prototype of an online bayesian inference model as part of my reasearch on computational modelling of musical structure perception. In essence, the model is a hierarchical latent-variable Noisy-OR statistical model for online processing of sequential multivariate data, developed theoretically following predictive-processing principles. It ingests a (discrete-time) multivariate time series incrementally (one step at a time), learns a set of sparse latent "generators" acting as temporal (autoregressive) predictors, and uses combinations of these generators to reconstruct/interpret current observations and predict future observations.

The code is being developed alongside a theoretical model and corresponding experimental work, and is written as a self-contained C++ library using Eigen, with a JUCE-based interface and logging/state-tracing tools built on top to facilitate experimentation.

###### **How to use**

This implementation includes a state and tracing system designed for flexible experimentation. It makes runs reproducible, editable, inspectable, and shareable:
- Model configurations and checkpoints are stored using `.state` files, which describe the current model state and are intended as self-contained checkpoints. Loading a `.state` file reconstructs the corresponding model state, including learned parameters and temporal context, so that a run can be resumed or reproduced from that point. The state loader also supports partial state descriptions, such that a `.state` file specifying, for example, only the input data or only the generator parameters, overwrites the corresponding fields in current state with those specifications.
- The interface provides a live view of the current model state so the interface can be used as a live diagnostic tool. It displays the input window and, for each layer, the main quantities involved in inference and learning. Most of these quantities are shown as grayscale matrices or vector, and several parameters can also be edited directly from the GUI, allowing interactive experimentation while the model is running.  

To run the model:

- Load a `.state` file, or edit an existing pre-loaded one.
- Run the model interactively (step-wise) or choose a number of steps and run asynchronously with `Fast N`.
- Inspect the live layer display during learning, and/or enable logging to record the run, using the trace plotter to inspect a logged time series.
  
____
## Mathematical Model

What follows is a brief, high-level description of the statistical model the code implements. Below it, a more detailed mathematical formulation.

### Summary of Mathematical Model

Each layer learns a set of 'generators' which represent learned sparse local temporal regularities over its input, and infers its structure as a composition of these regularities.

Generators at one layer are represented as channels at the layer above, such that the sequence of generator activations as a causal explanation for its layers input forms the input for the layer above.

As a whole, the set of generators forms a representation used for interence and prediction: the model interprets the current observation as a composition of generator patterns, and gives a corresponding prediction for the next timestep.


### Input

We take as input a time-discrete multivariate process, ingested sequentially.

```math
X := (X_t)_{t \in \mathbb{N}},
\qquad
X_t \in [0,1]^n.
```
At the sensory (input) layer, each element $X^0_{t,i}$ represents the presence and "intensity"/amplitude of an event in channel $i$.
In layers above it, elements $X^{l>0}_{t,i}$ represents the inferred probability that a generator's produced events at the layer below.

More speficially, At layer $l$, the observation vector is

```math
\mathbf{x}^l_t \in [0,1]^{N_l}.
```

For the bottom layer, $\mathbf{x}^0_t = X_t$. For higher layers, the observation is the posterior marginal activation vector produced by the layer below:

```math
\mathbf{x}^{l+1}_t = \boldsymbol{\mu}^l_t.
```

Thus, adjacent layers satisfy

```math
N_{l+1} = K_l,
```

where $K_l$ is the number of generators in layer $l$.


---

###### **Layers and Generators**



Each generator $k$ in layer $l$ has:

```math
G^l_k = (F^l_k, \mathbf{r}^l_k),
```

where

```math
F^l_k \in \mathbb{R}_{\geq 0}^{N_l \times o_l}
```

is an order-$o_l$ temporal filter over the recent context of layer $l$, and

```math
\mathbf{r}^l_k \in [0,1]^{N_l}
```

is a prediction vector over the current observation at that layer.

The context matrix is

```math
C^l_t =
\begin{bmatrix}
\mathbf{x}^l_{t-1} & \mathbf{x}^l_{t-2} & \cdots & \mathbf{x}^l_{t-o_l}
\end{bmatrix}
\in [0,1]^{N_l \times o_l}.
```

The filter match for generator $k$ is

```math
q^l_{t,k}
=
\frac{\langle F^l_k, C^l_t\rangle}
{\|F^l_k\|_1}.
```

This gives a bottom-up/contextual support value for the generator.

---

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

###### **Prior Over Candidate States**

For a candidate state $\mathbf{z}_{t,m}^l \in \mathcal{S}^l_t$, the unnormalised Bernoulli product prior is

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

