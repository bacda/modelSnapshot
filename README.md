We take as input a time-discrete multivariate process

$$
X := (X_t)_{t \in \mathbb{N}},
\qquad
X_t \in [0,1]^n.
$$

The binary case $X_t \in \{0,1\}^n$ is a special case. Each component $X_{t,i}$ represents the presence or intensity of an event in channel $i$. The signal is ingested sequentially.

The model is a hierarchical online Bayesian latent-variable model. Each layer learns a set of latent generators which act as sparse local temporal predictors. These generators infer which latent causes are currently active, reconstruct the current observation, and prepare a prediction/prior for the next timestep.

---

###### **Layers and Generators**

At layer $l$, the observation vector is

$$
\mathbf{x}^l_t \in [0,1]^{N_l}.
$$

For the bottom layer, $\mathbf{x}^0_t = X_t$. For higher layers, the observation is the posterior marginal activation vector produced by the layer below:

$$
\mathbf{x}^{l+1}_t = \boldsymbol{\mu}^l_t.
$$

Thus, adjacent layers satisfy

$$
N_{l+1} = K_l,
$$

where $K_l$ is the number of generators in layer $l$.

Each generator $k$ in layer $l$ has:

$$
G^l_k = (F^l_k, \mathbf{r}^l_k),
$$

where

$$
F^l_k \in \mathbb{R}_{\geq 0}^{N_l \times o_l}
$$

is an order-$o_l$ temporal filter over the recent context of layer $l$, and

$$
\mathbf{r}^l_k \in [0,1]^{N_l}
$$

is a prediction vector over the current observation at that layer.

The context matrix is

$$
C^l_t =
\begin{bmatrix}
\mathbf{x}^l_{t-1} & \mathbf{x}^l_{t-2} & \cdots & \mathbf{x}^l_{t-o_l}
\end{bmatrix}
\in [0,1]^{N_l \times o_l}.
$$

The filter match for generator $k$ is

$$
q^l_{t,k}
=
\frac{\langle F^l_k, C^l_t\rangle}
{\|F^l_k\|_1}.
$$

This gives a bottom-up/contextual support value for the generator.

---

###### **Hidden States**

For each layer $l$, we introduce binary hidden variables

$$
z^l_{t,k} \in \{0,1\},
$$

where $z^l_{t,k}=1$ means that generator $k$ is active as a latent cause of the current observation $\mathbf{x}^l_t$.

The full latent state is the binary vector

$$
\mathbf{z}^l_t =
(z^l_{t,1}, \dots, z^l_{t,K_l})
\in \{0,1\}^{K_l}.
$$

This is a sparse distributed representation: multiple generators may be active simultaneously. This is what allows an observation to be represented compositionally, as a combination of latent causes rather than as a single mutually exclusive state.

---

###### **Generator Activation Prior**

Before observing $\mathbf{x}^l_t$, each generator has an activation prior

$$
\alpha^l_{t,k}
=
p(z^l_{t,k}=1 \mid C^l_t, \tau^l_t),
$$

where $\tau^l_{t,k}$ is top-down support arriving from the layer above. In the current model, this prior is computed by combining bottom-up context support $q^l_{t,k}$, top-down support $\tau^l_{t,k}$, and learned/manual generator parameters.

Each generator has:

- base rate $b_k$,
- bottom-up weight $w_k$,
- evidence amplitude $A_k$,
- centering value $c_k$.

The combined evidence is

$$
e^l_{t,k}
=
w_k q^l_{t,k}
+
(1-w_k)\tau^l_{t,k}.
$$

The activation prior is then

$$
\alpha^l_{t,k}
=
\sigma
\left(
\operatorname{logit}(b_k)
+
A_k(e^l_{t,k}-c_k)
\right).
$$

Thus $F_k$ does not directly determine whether a generator is active. It contributes bottom-up evidence, which is combined with top-down evidence and transformed into an activation prior.

---

###### **Candidate Hidden States**

In principle, the latent state space has size

$$
2^{K_l}.
$$

However, the current implementation does not usually evaluate all possible states. Instead, it first selects a candidate set of generators using thresholds on:

- context support $q_k$,
- top-down support $\tau_k$,
- singleton observation score,
- optionally, activation prior $\alpha_k$.

Let the selected generators be

$$
\mathcal{K}^l_t \subseteq \{1,\dots,K_l\}.
$$

The model then enumerates candidate hidden states using subsets of these selected generators:

$$
\mathcal{S}^l_t \subseteq \{0,1\}^{K_l}.
$$

If all generators are selected, this reduces to the full hidden state space. Otherwise, inference is performed only over the selected candidate states.

---

###### **Prior Over Candidate States**

For a candidate state $\mathbf{z}_{t,m}^l \in \mathcal{S}^l_t$, the unnormalised Bernoulli product prior is

$$
\tilde{\pi}^l_{t,m}
=
\prod_k
(\alpha^l_{t,k})^{z^l_{t,m,k}}
(1-\alpha^l_{t,k})^{1-z^l_{t,m,k}}.
$$

Because inference is restricted to the candidate set, this prior is normalised over $\mathcal{S}^l_t$:

$$
\pi^l_{t,m}
=
\frac{\tilde{\pi}^l_{t,m}}
{\sum_{\mathbf{z}_{t,m'}^l \in \mathcal{S}^l_t}
\tilde{\pi}^l_{t,m'}}.
$$

This means the prior is a distribution over the current candidate states, not necessarily over all $2^{K_l}$ possible states.

---

###### **Noisy-OR Observation Model**

Given a candidate latent state $\mathbf{z}^l_{t,m}$, the model predicts each observation channel using a Noisy-OR composition of the active generators.

The current implementation also includes a leak probability $\lambda_i$ for each channel. The predicted probability of channel $i$ being active is

$$
\hat{x}^l_{t,m,i}
=
p(x^l_{t,i}=1 \mid \mathbf{z}^l_{t,m})
=
1
-
(1-\lambda_i)
\prod_k
(1-z^l_{t,m,k} r^l_{k,i}).
$$

Thus, active generators jointly explain the observation vector. If different generators predict different channels, their predictions combine into a composite observation. If they predict the same channel, they compete through explaining-away rather than receiving independent additive credit.

The likelihood of the full observation vector is modelled channel-wise:

$$
\ell^l_{t,m}
=
p(\mathbf{x}^l_t \mid \mathbf{z}^l_{t,m})
=
\prod_i
(\hat{x}^l_{t,m,i})^{x^l_{t,i}}
(1-\hat{x}^l_{t,m,i})^{1-x^l_{t,i}}.
$$

For binary observations this is the Bernoulli likelihood. For continuous values in $[0,1]$, it acts as a soft Bernoulli / cross-entropy likelihood.

---

###### **Posterior Inference**

For each candidate state, the posterior is

$$
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
$$

The posterior marginal activation of generator $k$ is

$$
\mu^l_{t,k}
=
\sum_m
\rho^l_{t,m}
z^l_{t,m,k}.
$$

The posterior reconstruction of the observation is

$$
\hat{\mathbf{x}}^l_t
=
\sum_m
\rho^l_{t,m}
\hat{\mathbf{x}}^l_{t,m}.
$$

The vector $\boldsymbol{\mu}^l_t$ is passed upward as the observation for the next layer.

---

###### **Top-Down Prediction**

After the upward inference pass, the model performs a top-down preparation step.

For a lower layer $l$, the top-down support used for the next timestep is the reconstruction produced by the layer above:

$$
\tau^l_{t+1}
=
\hat{\mathbf{x}}^{l+1}_t.
$$

The top layer may receive external top-down support, or else a zero vector.

Each layer then shifts the current observation into its context window and computes the next activation prior:

$$
\alpha^l_{t+1,k}
=
p(z^l_{t+1,k}=1 \mid C^l_{t+1}, \tau^l_{t+1}).
$$

So the model maintains two time-indexed quantities:

- $\alpha_t$: the inherited prior used for the current E-step,
- $\alpha_{t+1}$: the next prior prepared after observing the current timestep.

---

###### **Learning With Online EM / Generalized EM**

Learning is online. At each timestep, the model first performs an E-step by computing the posterior over candidate hidden states. It then updates generator parameters using posterior responsibilities.

The necessary E-step quantities are:

$$
\rho_m
=
p(\mathbf{z}_m \mid \mathbf{x}, C, \tau),
$$

and the marginal generator activation

$$
\mu_k
=
\sum_m \rho_m z_{m,k}.
$$

The prediction vectors $R$, filters $F$, and base rates $b$ are then updated.

The prediction update uses a Noisy-OR responsibility term. For generator $k$ and channel $i$, the expected causal credit is approximately

$$
\gamma_{k,i}
=
x_i
\sum_m
\rho_m
z_{m,k}
\frac{r_{k,i}}{\hat{x}_{m,i}}.
$$

The prediction vector update is

$$
\Delta r_{k,i}
=
\eta_R
(\gamma_{k,i} - \mu_k r_{k,i}).
$$

This increases $r_{k,i}$ when generator $k$ receives posterior responsibility for explaining an active channel, and decreases it when the generator is active but the channel is not supported.

The filter update tries to make the contextual activation prior $\alpha_k$ better match the posterior marginal $\mu_k$. The current update is a gradient-like rule:

$$
\Delta F_{k,a,b}
=
\eta_F
(\mu_k-\alpha_k)
A_k w_k
\frac{C_{a,b}-q_k}
{\epsilon + \sum_{a,b} F_{k,a,b}}.
$$

The base rate update is

$$
\Delta b_k
=
\eta_b(\mu_k-b_k).
$$

Thus, the model is not performing a closed-form batch M-step. It is better described as an online EM / generalized EM procedure.

The parameters $w_k$, $A_k$, $c_k$, candidate thresholds, and learning rates are currently user-editable controls rather than automatically learned parameters.

---

###### **Interpretation**

At a single layer, the model maps recent context and top-down support to a predictive distribution over the next/current observation:

$$
(C_t, \tau_t)
\mapsto
\alpha_t
\mapsto
p(\mathbf{z}_t \mid C_t,\tau_t)
\mapsto
p(\mathbf{x}_t \mid C_t,\tau_t).
$$

The generators therefore serve two roles:

1. They are temporal detectors over recent context.
2. They are latent causes whose active combinations predict the observation vector.

This makes the representation compositional: an observation can be explained by several active generators, each accounting for part of the multivariate signal.

Rather than thinking of higher layers as simply storing nested chunks, a better picture is that each layer produces a moving field of posterior beliefs. These posterior beliefs become the signal observed by the next layer, while higher layers return top-down support that shapes the priors of lower-layer generators.
