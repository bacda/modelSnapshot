#pragma once

#include "NoisyORUtilities.h"

#include <cstddef>
#include <vector>

namespace noisy_or {

struct CandidateSelectionPolicy {
    double contextThreshold = 0.01;
    double topDownThreshold = 0.01;

    // Average singleton log-likelihood improvement over the leak-only state.
    // Zero means that a generator must improve the observation likelihood.
    double observationThreshold = 0.01;

    // Alpha includes base rate, context, and top-down evidence. Keeping this
    // route prevents a substantial base rate from being ignored.
    bool useActivationSupport = false;
    double activationThreshold = 0.01;

    // A safety guard, not a silent truncation rule. Selection throws if more
    // generators pass the thresholds, because the power set has size 2^L.
    std::size_t maximumSelectedGenerators = 20;
};

struct CandidateSelectionResult {
    Eigen::VectorXd observationScores; // K average log-likelihood ratios
    std::vector<Eigen::Index> selectedGenerators;
    std::vector<State> candidates;     // Power set of selectedGenerators
};

// Selects generators supported by context, top-down input, activation
// probability, or the current observation. Observation support is measured
// with a leak-aware singleton log-likelihood ratio, not an inner product.
CandidateSelectionResult selectCandidates(
    const Eigen::VectorXd& filterMatch,
    const Eigen::VectorXd& topDownSupport,
    const Eigen::VectorXd& alpha,
    const Eigen::VectorXd& observation,
    const Eigen::MatrixXd& predictions,
    const Eigen::VectorXd& leak,
    const CandidateSelectionPolicy& policy);

} // namespace noisy_or
