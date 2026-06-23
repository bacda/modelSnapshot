#pragma once

#include "NoisyORLearning.h"
#include "NoisyORStack.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace noisy_or {

struct LayerState {
    LayerConfiguration configuration;
    OnlineEMOptions learning;
    bool learningEnabled = false;
    Eigen::VectorXd initialTopDownSupport; // K; empty means zeros
};

struct ModelState {
    Eigen::MatrixXd input; // T x N_0; may be empty for externally supplied x
    bool loopInput = true;
    std::size_t inputIndex = 0;
    std::vector<LayerState> layers;
};

struct StateLoadDefaults {
    std::size_t layerCount = 1;
    Eigen::Index generatorCount = 3;
    Eigen::Index contextOrder = 1;

    double leak = 0.01;
    double baseRate = 0.05;
    double bottomUpWeight = 0.7;
    double evidenceAmplitude = 4.0;
    double centering = 0.5;

    // R entries and unnormalized F entries are sampled independently from
    // Beta(shapeA, shapeB). Each F_k is subsequently L1-normalized.
    double betaShapeA = 0.05;
    double betaShapeB = 1.0;
    std::uint64_t randomSeed = 5489u;

    CandidateSelectionPolicy candidateSelection;
    OnlineEMOptions learning;
    bool learningEnabled = false;
};

struct StateLoadResult {
    ModelState state;
    std::vector<std::string> warnings;
};

// The text file is a patch. Explicit fields replace previous values; omitted
// compatible fields retain previous values; otherwise documented defaults and
// Beta random initialization are used. Every fallback emits a warning.
StateLoadResult loadModelState(
    const std::string& path,
    const ModelState* previous = nullptr,
    const StateLoadDefaults& defaults = {});

// Validates all dimensions, probability ranges, hierarchy connections, and
// learning settings. Throws std::invalid_argument on failure.
void validateModelState(const ModelState& state);

// Copies layer configurations into a newly initialized stack.
NoisyORStack makeInitializedStack(const ModelState& state);

// Captures learned parameters and temporal context from a running stack while
// retaining input and learning metadata from state.
ModelState snapshotModelState(
    const NoisyORStack& stack,
    const ModelState& state);

// Returns false when no input is stored or a non-looping sequence is exhausted.
bool readNextInput(ModelState& state, Eigen::VectorXd& observation);

} // namespace noisy_or
