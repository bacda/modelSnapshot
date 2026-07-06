#include "NoisyORLayer.h"

#include "NoisyORUtilities.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace noisy_or {

NoisyORLayer::NoisyORLayer(LayerConfiguration configuration)
    : configuration_(std::move(configuration)),
      context_(configuration_.initialContext)
{
    validateConfiguration();
}

void NoisyORLayer::validateConfiguration() const
{
    const Eigen::Index K = configuration_.predictions.rows();
    const Eigen::Index N = configuration_.predictions.cols();
    const auto& parameters = configuration_.parameters;

    if (K <= 0 || N <= 0) {
        throw std::invalid_argument(
            "A layer must have at least one generator and channel");
    }
    if (K > 64) {
        throw std::invalid_argument(
            "The uint64_t State encoding supports at most 64 generators");
    }
    if (configuration_.initialContext.rows() != N) {
        throw std::invalid_argument(
            "initialContext.rows() must equal the channel count N");
    }
    if (configuration_.leak.size() != N) {
        throw std::invalid_argument("leak must have length N");
    }
    if (static_cast<Eigen::Index>(configuration_.filters.size()) != K) {
        throw std::invalid_argument("There must be one filter per generator");
    }

    for (const auto& filter : configuration_.filters) {
        if (filter.rows() != N ||
            filter.cols() != configuration_.initialContext.cols()) {
            throw std::invalid_argument(
                "Every filter must have shape N x contextOrder");
        }
        if (!filter.allFinite() ||
            filter.cwiseAbs().sum() <= probabilityEpsilon) {
            throw std::invalid_argument(
                "Every filter must be finite and have nonzero L1 norm");
        }
    }

    if (parameters.baseRate.size() != K ||
        parameters.bottomUpWeight.size() != K ||
        parameters.evidenceAmplitude.size() != K ||
        parameters.centering.size() != K) {
        throw std::invalid_argument(
            "Every generator parameter vector must have length K");
    }

    requireProbabilityVector(parameters.baseRate, "baseRate");
    requireProbabilityVector(parameters.bottomUpWeight, "bottomUpWeight");
    requireProbabilityVector(parameters.centering, "centering");
    requireProbabilityVector(configuration_.leak, "leak");

    if (!configuration_.predictions.allFinite() ||
        (configuration_.predictions.array() < 0.0).any() ||
        (configuration_.predictions.array() > 1.0).any()) {
        throw std::invalid_argument(
            "predictions must contain probabilities in [0,1]");
    }
    if (!parameters.evidenceAmplitude.allFinite() ||
        (parameters.evidenceAmplitude.array() < 0.0).any()) {
        throw std::invalid_argument(
            "evidenceAmplitude must be finite and nonnegative");
    }
}

void NoisyORLayer::initialize(const Eigen::VectorXd& topDownSupport)
{
    context_ = configuration_.initialContext;
    topDownSupport_ = topDownSupport.size() == 0
        ? Eigen::VectorXd::Zero(generatorCount())
        : topDownSupport;
    requireProbabilityVector(topDownSupport_, "topDownSupport");
    if (topDownSupport_.size() != generatorCount()) {
        throw std::invalid_argument("Top-down support must have length K");
    }

    lastObservation_.resize(0);
    output_ = {};
    computeActivationEvidence();
    phase_ = Phase::ReadyToObserve;
}

const LayerOutput& NoisyORLayer::observe(
    const Eigen::VectorXd& observation)
{
    if (phase_ == Phase::Uninitialized) {
        throw std::logic_error("initialize() must be called before observe()");
    }
    if (phase_ != Phase::ReadyToObserve) {
        throw std::logic_error(
            "prepareNext() must be called before the next observation");
    }
    if (observation.size() != channelCount()) {
        throw std::invalid_argument(
            "Observation length must equal the layer channel count N");
    }
    requireProbabilityVector(observation, "observation");

    output_.selection = selectCandidates(
        filterMatch_,
        topDownSupport_,
        alpha_,
        observation,
        configuration_.predictions,
        configuration_.leak,
        configuration_.candidateSelection);

    output_.observation = observation;
    output_.context = context_;
    output_.topDownSupport = topDownSupport_;
    output_.alpha = alpha_;
    output_.filterMatch = filterMatch_;
    output_.states = expandCandidates(
        output_.selection.candidates,
        static_cast<std::size_t>(generatorCount()));
    output_.statePredictions = computeNoisyOr(output_.states);
    const Eigen::VectorXd logPrior = computeLogPrior(output_.states);
    const Eigen::VectorXd logLikelihood =
        computeLogLikelihood(observation, output_.statePredictions);
    const Eigen::VectorXd logJoint = logPrior + logLikelihood;

    output_.prior = logPrior.array().exp().matrix();
    output_.likelihood = logLikelihood.array().exp().matrix();
    output_.logEvidence = logSumExp(logJoint);
    output_.posterior =
        (logJoint.array() - output_.logEvidence).exp().matrix();
    output_.marginals =
        output_.states.cast<double>().transpose() * output_.posterior;
    output_.reconstruction =
        output_.statePredictions.transpose() * output_.posterior;

    lastObservation_ = observation;
    phase_ = Phase::ReadyToPrepare;
    return output_;
}

void NoisyORLayer::prepareNext(
    const Eigen::VectorXd& topDownSupport)
{
    if (phase_ != Phase::ReadyToPrepare) {
        throw std::logic_error(
            "observe() must be called before prepareNext()");
    }

    topDownSupport_ = topDownSupport.size() == 0
        ? Eigen::VectorXd::Zero(generatorCount())
        : topDownSupport;
    if (topDownSupport_.size() != generatorCount()) {
        throw std::invalid_argument("Top-down support must have length K");
    }
    requireProbabilityVector(topDownSupport_, "topDownSupport");

    shiftObservationIntoContext();
    computeActivationEvidence();
    phase_ = Phase::ReadyToObserve;
}

void NoisyORLayer::computeActivationEvidence()
{
    filterMatch_.resize(generatorCount());
    for (Eigen::Index k = 0; k < generatorCount(); ++k) {
        const auto& filter =
            configuration_.filters[static_cast<std::size_t>(k)];
        const double l1Norm = filter.cwiseAbs().sum();
        filterMatch_(k) =
            filter.cwiseProduct(context_).sum() / l1Norm;
    }

    const auto& parameters = configuration_.parameters;
    const Eigen::ArrayXd base = parameters.baseRate.array()
        .max(probabilityEpsilon)
        .min(1.0 - probabilityEpsilon);
    const Eigen::ArrayXd logitBase =
        base.log() - (1.0 - base).log();
    const Eigen::ArrayXd combinedEvidence =
        parameters.bottomUpWeight.array() * filterMatch_.array()
        + (1.0 - parameters.bottomUpWeight.array())
              * topDownSupport_.array();
    const Eigen::ArrayXd logitAlpha =
        logitBase
        + parameters.evidenceAmplitude.array()
              * (combinedEvidence - parameters.centering.array());

    alpha_ = (1.0 / (1.0 + (-logitAlpha).exp())).matrix();
}

Eigen::MatrixXd NoisyORLayer::computeNoisyOr(
    const BinaryMatrix& states) const
{
    // P(m,i) = 1 - (1-leak(i)) product_k [1-Z(m,k)R(k,i)].
    const Eigen::ArrayXXd boundedR = configuration_.predictions.array()
        .min(1.0 - probabilityEpsilon);
    const Eigen::ArrayXd boundedLeak = configuration_.leak.array()
        .min(1.0 - probabilityEpsilon);
    const Eigen::MatrixXd logGeneratorFailure =
        (1.0 - boundedR).log().matrix();

    Eigen::MatrixXd logStateFailure =
        states.cast<double>() * logGeneratorFailure;
    logStateFailure.rowwise() +=
        (1.0 - boundedLeak).log().matrix().transpose();

    return (1.0 - logStateFailure.array().exp()).matrix();
}

Eigen::VectorXd NoisyORLayer::computeLogPrior(
    const BinaryMatrix& states) const
{
    const Eigen::ArrayXd a = alpha_.array()
        .max(probabilityEpsilon)
        .min(1.0 - probabilityEpsilon);
    const Eigen::VectorXd logOdds =
        (a.log() - (1.0 - a).log()).matrix();
    const double inactiveBaseline = (1.0 - a).log().sum();
    const Eigen::VectorXd logMass =
        (states.cast<double>() * logOdds).array()
        + inactiveBaseline;

    // Conditioning on membership in S_t makes this a simplex over candidates.
    return (logMass.array() - logSumExp(logMass)).matrix();
}

Eigen::VectorXd NoisyORLayer::computeLogLikelihood(
    const Eigen::VectorXd& observation,
    const Eigen::MatrixXd& statePredictions) const
{
    const Eigen::ArrayXXd P = statePredictions.array()
        .max(probabilityEpsilon)
        .min(1.0 - probabilityEpsilon);
    const Eigen::RowVectorXd x = observation.transpose();
    const Eigen::RowVectorXd notX =
        Eigen::RowVectorXd::Ones(channelCount()) - x;

    return P.log().matrix() * x.transpose()
         + (1.0 - P).log().matrix() * notX.transpose();
}

void NoisyORLayer::shiftObservationIntoContext()
{
    if (context_.cols() == 0) {
        return;
    }
    for (Eigen::Index lag = context_.cols() - 1; lag > 0; --lag) {
        context_.col(lag) = context_.col(lag - 1);
    }
    context_.col(0) = lastObservation_;
}

Eigen::Index NoisyORLayer::channelCount() const noexcept
{
    return configuration_.predictions.cols();
}

Eigen::Index NoisyORLayer::generatorCount() const noexcept
{
    return configuration_.predictions.rows();
}

Eigen::Index NoisyORLayer::candidateCount() const noexcept
{
    return static_cast<Eigen::Index>(
        output_.selection.candidates.size());
}

Eigen::Index NoisyORLayer::contextOrder() const noexcept
{
    return context_.cols();
}

bool NoisyORLayer::isInitialized() const noexcept
{
    return phase_ != Phase::Uninitialized;
}

bool NoisyORLayer::isAwaitingNextPrior() const noexcept
{
    return phase_ == Phase::ReadyToPrepare;
}

const LayerOutput& NoisyORLayer::output() const noexcept { return output_; }
const Eigen::VectorXd& NoisyORLayer::alpha() const noexcept { return alpha_; }
const Eigen::VectorXd& NoisyORLayer::prior() const noexcept
{
    return output_.prior;
}
const Eigen::VectorXd& NoisyORLayer::filterMatch() const noexcept
{
    return filterMatch_;
}
const Eigen::VectorXd& NoisyORLayer::topDownSupport() const noexcept
{
    return topDownSupport_;
}
const Eigen::MatrixXd& NoisyORLayer::context() const noexcept
{
    return context_;
}
const std::vector<State>& NoisyORLayer::candidates() const noexcept
{
    return output_.selection.candidates;
}
LayerConfiguration& NoisyORLayer::configuration() noexcept
{
    return configuration_;
}
const LayerConfiguration& NoisyORLayer::configuration() const noexcept
{
    return configuration_;
}

} // namespace noisy_or
