#pragma once

#include "NoisyORLayer.h"

#include <vector>

namespace noisy_or {

struct OnlineEMOptions {
    double predictionLearningRate = 0.01;
    double filterLearningRate = 0.01;
    double baseRateLearningRate = 0.01;
    double epsilon = 1e-6;

    // The Max implementation thresholds x_i at 0.5. Disable this to use
    // fractional observations directly in the expected-count update.
    bool binarizeObservation = false;
    double observationThreshold = 0.5;
};

struct LearningDiagnostics {
    Eigen::VectorXd posteriorActivation; // K, mu_k
    Eigen::VectorXd activationError;     // K, mu_k - alpha_k
    Eigen::MatrixXd expectedCauseCounts; // K x N, gamma_{k,i}
    Eigen::MatrixXd predictionDelta;     // K x N
    std::vector<Eigen::MatrixXd> filterDelta;
    Eigen::VectorXd baseRateDelta;       // K
};

// Performs one online generalized-EM update using the most recent E-step in
// layer.output(). R follows the latent-cause responsibility update from the
// Max r_update script; F follows its gradient update; the base rate tracks mu
// by exponential moving average.
LearningDiagnostics updateParametersOnlineEM(
    NoisyORLayer& layer,
    const OnlineEMOptions& options = {});

} // namespace noisy_or
