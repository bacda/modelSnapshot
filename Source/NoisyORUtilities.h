#pragma once

#include "Eigen/Dense"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace noisy_or {

using State = std::uint64_t;
using BinaryMatrix =
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic,
                  Eigen::RowMajor>;

inline constexpr double probabilityEpsilon = 1e-12;

void requireProbabilityVector(
    const Eigen::VectorXd& values,
    const char* name);

double logSumExp(const Eigen::VectorXd& values);

BinaryMatrix expandCandidates(
    const std::vector<State>& candidates,
    std::size_t generatorCount);

// Enumerates the power set of selectedGenerators. Bits in each returned State
// retain their original generator indices, so the states still belong to
// {0,1}^K rather than to a reindexed smaller space.
std::vector<State> enumerateSelectedGeneratorSubsets(
    const std::vector<Eigen::Index>& selectedGenerators,
    std::size_t generatorCount);

// Intended for tests and small exact-inference comparisons.
std::vector<State> makeCompleteCandidates(std::size_t generatorCount);

} // namespace noisy_or
