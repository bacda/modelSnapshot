#include "NoisyORLearning.h"

#include "NoisyORUtilities.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace noisy_or {
namespace {

double clipProbability(double value, double epsilon)
{
    return std::max(epsilon, std::min(1.0 - epsilon, value));
}

void validateOptions(const OnlineEMOptions& options)
{
    if (!std::isfinite(options.predictionLearningRate) ||
        !std::isfinite(options.filterLearningRate) ||
        !std::isfinite(options.baseRateLearningRate) ||
        options.predictionLearningRate < 0.0 ||
        options.filterLearningRate < 0.0 ||
        options.baseRateLearningRate < 0.0) {
        throw std::invalid_argument(
            "Learning rates must be finite and nonnegative");
    }
    if (!std::isfinite(options.epsilon) ||
        options.epsilon <= 0.0 || options.epsilon >= 0.5) {
        throw std::invalid_argument("epsilon must lie in (0,0.5)");
    }
    if (!std::isfinite(options.observationThreshold)) {
        throw std::invalid_argument(
            "observationThreshold must be finite");
    }
}

} // namespace

LearningDiagnostics updateParametersOnlineEM(
    NoisyORLayer& layer,
    const OnlineEMOptions& options)
{
    validateOptions(options);

    const LayerOutput& output = layer.output_;
    const Eigen::Index K = layer.generatorCount();
    const Eigen::Index N = layer.channelCount();
    const Eigen::Index M = output.posterior.size();

    if (M == 0 || output.observation.size() != N ||
        output.alpha.size() != K ||
        output.filterMatch.size() != K ||
        output.states.rows() != M || output.states.cols() != K ||
        output.statePredictions.rows() != M ||
        output.statePredictions.cols() != N) {
        throw std::logic_error(
            "A completed observe() call is required before learning");
    }

    auto& configuration = layer.configuration_;
    const Eigen::MatrixXd oldPredictions = configuration.predictions;
    Eigen::MatrixXd newPredictions = oldPredictions;

    LearningDiagnostics diagnostics;
    diagnostics.posteriorActivation = output.marginals;
    diagnostics.activationError = output.marginals - output.alpha;
    diagnostics.expectedCauseCounts = Eigen::MatrixXd::Zero(K, N);
    diagnostics.predictionDelta = Eigen::MatrixXd::Zero(K, N);
    diagnostics.filterDelta.resize(static_cast<std::size_t>(K));
    diagnostics.baseRateDelta = Eigen::VectorXd::Zero(K);

    // Online EM update for R. gamma_{k,i} is the posterior expected count of
    // generator k causing observed channel i under the noisy-OR allocation.
    for (Eigen::Index k = 0; k < K; ++k) {
        const double mu = output.marginals(k);

        for (Eigen::Index i = 0; i < N; ++i) {
            const double x = options.binarizeObservation
                ? (output.observation(i) >= options.observationThreshold
                       ? 1.0 : 0.0)
                : output.observation(i);
            const double oldR = clipProbability(
                oldPredictions(k, i), options.epsilon);

            double responsibilitySum = 0.0;
            if (x > 0.0) {
                for (Eigen::Index m = 0; m < M; ++m) {
                    const double prediction = clipProbability(
                        output.statePredictions(m, i), options.epsilon);
                    responsibilitySum +=
                        output.posterior(m)
                        * static_cast<double>(output.states(m, k))
                        * oldR
                        / (prediction + options.epsilon);
                }
            }

            const double gamma = x * responsibilitySum;
            const double delta = options.predictionLearningRate
                * (gamma - mu * oldR);

            diagnostics.expectedCauseCounts(k, i) = gamma;
            diagnostics.predictionDelta(k, i) = delta;
            newPredictions(k, i) =
                clipProbability(oldR + delta, options.epsilon);
        }
    }

    // Gradient M-step for F, matching updateF.js:
    // F_{k,a} += eta_F (mu_k-alpha_k) A_k w_k
    //              (C_a-qF_k) / (epsilon + sum_b F_{k,b}).
    for (Eigen::Index k = 0; k < K; ++k) {
        Eigen::MatrixXd& filter =
            configuration.filters[static_cast<std::size_t>(k)];

        if ((filter.array() < 0.0).any()) {
            throw std::logic_error(
                "The F update requires nonnegative filter entries");
        }

        const double denominator = options.epsilon + filter.sum();
        const Eigen::MatrixXd localTerm =
            (output.context.array() - output.filterMatch(k))
                .matrix()
            / denominator;
        const Eigen::MatrixXd delta =
            options.filterLearningRate
            * diagnostics.activationError(k)
            * configuration.parameters.evidenceAmplitude(k)
            * configuration.parameters.bottomUpWeight(k)
            * localTerm;

        diagnostics.filterDelta[static_cast<std::size_t>(k)] = delta;
        filter = (filter + delta).array().max(0.0).min(1.0).matrix();
    }

    // Online base-rate estimate: beta_k <- beta_k + eta (mu_k-beta_k).
    for (Eigen::Index k = 0; k < K; ++k) {
        const double oldBase = configuration.parameters.baseRate(k);
        const double delta = options.baseRateLearningRate
            * (output.marginals(k) - oldBase);
        diagnostics.baseRateDelta(k) = delta;
        configuration.parameters.baseRate(k) =
            clipProbability(oldBase + delta, options.epsilon);
    }

    configuration.predictions = std::move(newPredictions);

    // If prepareNext() already advanced the layer, refresh alpha_{t+1} using
    // the learned F and base rates. If learning occurs immediately after
    // observe(), prepareNext() will perform this refresh itself.
    if (layer.phase_ == NoisyORLayer::Phase::ReadyToObserve) {
        layer.computeActivationEvidence();
    }

    return diagnostics;
}

} // namespace noisy_or
