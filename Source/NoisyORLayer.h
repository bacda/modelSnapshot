#pragma once

#include "CandidateSelection.h"

#include <cstddef>
#include <vector>

namespace noisy_or {

class NoisyORLayer;
struct OnlineEMOptions;
struct LearningDiagnostics;
LearningDiagnostics updateParametersOnlineEM(
    NoisyORLayer& layer,
    const OnlineEMOptions& options);

struct GeneratorParameters {
    Eigen::VectorXd baseRate;          // K probabilities
    Eigen::VectorXd bottomUpWeight;    // K values in [0,1]
    Eigen::VectorXd evidenceAmplitude; // K nonnegative values
    Eigen::VectorXd centering;         // K values in [0,1]
};

struct LayerConfiguration {
    Eigen::MatrixXd predictions;          // K x N
    Eigen::VectorXd leak;                 // N probabilities
    std::vector<Eigen::MatrixXd> filters; // K matrices, each N x order
    GeneratorParameters parameters;
    CandidateSelectionPolicy candidateSelection;
    Eigen::MatrixXd initialContext;       // N x order
};

struct LayerOutput {
    CandidateSelectionResult selection;
    Eigen::VectorXd observation;     // N, observation used in this E-step
    Eigen::MatrixXd context;         // N x order, context used in this E-step
    Eigen::VectorXd alpha;           // K prior activations used in this E-step
    Eigen::VectorXd filterMatch;     // K qF values used in this E-step
    BinaryMatrix states;             // M x K expanded candidate states
    Eigen::MatrixXd statePredictions; // M x N Noisy-OR probabilities
    Eigen::VectorXd prior;           // M probabilities on selection.candidates
    Eigen::VectorXd posterior;       // M probabilities on selection.candidates
    Eigen::VectorXd marginals;       // K probabilities E[z_t | x_1:t]
    Eigen::VectorXd reconstruction;  // N probabilities E[x_t | x_1:t]
    double logEvidence = 0.0;
};

class NoisyORLayer {
public:
    explicit NoisyORLayer(LayerConfiguration configuration);

    // Preserves the evidence needed to select candidates when x_1 arrives.
    void initialize(const Eigen::VectorXd& topDownSupport = {});

    // Selects S_t, then evaluates prior_t, likelihood_t, and posterior_t on
    // that fixed support. prepareNext() is required before another observe().
    const LayerOutput& observe(const Eigen::VectorXd& observation);

    // Shifts x_t into context and preserves qF_{t+1}, tau_{t+1}, alpha_{t+1}.
    // The explicit candidate prior is deferred until x_{t+1} arrives.
    void prepareNext(const Eigen::VectorXd& topDownSupport);

    Eigen::Index channelCount() const noexcept;
    Eigen::Index generatorCount() const noexcept;
    Eigen::Index candidateCount() const noexcept;
    Eigen::Index contextOrder() const noexcept;

    bool isInitialized() const noexcept;
    bool isAwaitingNextPrior() const noexcept;

    const LayerOutput& output() const noexcept;
    const Eigen::VectorXd& alpha() const noexcept;
    const Eigen::VectorXd& prior() const noexcept;
    const Eigen::VectorXd& filterMatch() const noexcept;
    const Eigen::VectorXd& topDownSupport() const noexcept;
    const Eigen::MatrixXd& context() const noexcept;
    const std::vector<State>& candidates() const noexcept;
    const LayerConfiguration& configuration() const noexcept;

private:
    enum class Phase { Uninitialized, ReadyToObserve, ReadyToPrepare };

    LayerConfiguration configuration_;
    Eigen::MatrixXd context_;
    Eigen::VectorXd filterMatch_;
    Eigen::VectorXd topDownSupport_;
    Eigen::VectorXd alpha_;
    Eigen::VectorXd lastObservation_;
    LayerOutput output_;
    Phase phase_ = Phase::Uninitialized;

    void validateConfiguration() const;
    void computeActivationEvidence();
    Eigen::MatrixXd computeNoisyOr(const BinaryMatrix& states) const;
    Eigen::VectorXd computeLogPrior(const BinaryMatrix& states) const;
    Eigen::VectorXd computeLogLikelihood(
        const Eigen::VectorXd& observation,
        const Eigen::MatrixXd& statePredictions) const;
    void shiftObservationIntoContext();

    friend LearningDiagnostics updateParametersOnlineEM(
        NoisyORLayer& layer,
        const OnlineEMOptions& options);
};

} // namespace noisy_or
