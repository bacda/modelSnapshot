#include "CandidateSelection.h"

#include <cmath>
#include <stdexcept>

namespace noisy_or {
namespace {

Eigen::VectorXd singletonObservationScores(
    const Eigen::VectorXd& observation,
    const Eigen::MatrixXd& predictions,
    const Eigen::VectorXd& leak)
{
    const Eigen::Index K = predictions.rows();
    const Eigen::Index N = predictions.cols();
    Eigen::VectorXd scores(K);

    const Eigen::ArrayXd baseline = leak.array()
        .max(probabilityEpsilon)
        .min(1.0 - probabilityEpsilon);

    for (Eigen::Index k = 0; k < K; ++k) {
        const Eigen::ArrayXd singleton =
            (1.0 - (1.0 - leak.array())
                         * (1.0 - predictions.row(k).transpose().array()))
                .max(probabilityEpsilon)
                .min(1.0 - probabilityEpsilon);

        const Eigen::ArrayXd x = observation.array();
        const double logLikelihoodRatio =
            (x * (singleton.log() - baseline.log())
             + (1.0 - x)
                   * ((1.0 - singleton).log()
                      - (1.0 - baseline).log()))
                .sum();

        scores(k) = logLikelihoodRatio / static_cast<double>(N);
    }

    return scores;
}

} // namespace

CandidateSelectionResult selectCandidates(
    const Eigen::VectorXd& filterMatch,
    const Eigen::VectorXd& topDownSupport,
    const Eigen::VectorXd& alpha,
    const Eigen::VectorXd& observation,
    const Eigen::MatrixXd& predictions,
    const Eigen::VectorXd& leak,
    const CandidateSelectionPolicy& policy)
{
    const Eigen::Index K = predictions.rows();
    const Eigen::Index N = predictions.cols();

    if (K <= 0 || N <= 0) {
        throw std::invalid_argument(
            "predictions must have nonzero K x N shape");
    }
    if (filterMatch.size() != K ||
        topDownSupport.size() != K ||
        alpha.size() != K) {
        throw std::invalid_argument(
            "filterMatch, topDownSupport, and alpha must have length K");
    }
    if (observation.size() != N || leak.size() != N) {
        throw std::invalid_argument(
            "observation and leak must have length N");
    }
    if (!filterMatch.allFinite()) {
        throw std::invalid_argument("filterMatch must be finite");
    }
    requireProbabilityVector(topDownSupport, "topDownSupport");
    requireProbabilityVector(alpha, "alpha");
    requireProbabilityVector(observation, "observation");
    requireProbabilityVector(leak, "leak");
    if (!predictions.allFinite() ||
        (predictions.array() < 0.0).any() ||
        (predictions.array() > 1.0).any()) {
        throw std::invalid_argument(
            "predictions must contain values in [0,1]");
    }
    if (!std::isfinite(policy.contextThreshold) ||
        !std::isfinite(policy.topDownThreshold) ||
        !std::isfinite(policy.observationThreshold) ||
        !std::isfinite(policy.activationThreshold)) {
        throw std::invalid_argument("Selection thresholds must be finite");
    }

    CandidateSelectionResult result;
    result.observationScores = singletonObservationScores(
        observation, predictions, leak);

    for (Eigen::Index k = 0; k < K; ++k) {
        const bool contextSupported =
            filterMatch(k) > policy.contextThreshold;
        const bool topDownSupported =
            topDownSupport(k) > policy.topDownThreshold;
        const bool observationSupported =
            result.observationScores(k) > policy.observationThreshold;
        const bool activationSupported =
            policy.useActivationSupport &&
            alpha(k) > policy.activationThreshold;

        if (contextSupported || topDownSupported ||
            observationSupported || activationSupported) {
            result.selectedGenerators.push_back(k);
        }
    }

    if (result.selectedGenerators.size() >
        policy.maximumSelectedGenerators) {
        throw std::runtime_error(
            "Too many generators passed candidate thresholds; increase "
            "thresholds or maximumSelectedGenerators deliberately");
    }

    result.candidates = enumerateSelectedGeneratorSubsets(
        result.selectedGenerators,
        static_cast<std::size_t>(K));

    return result;
}

} // namespace noisy_or
