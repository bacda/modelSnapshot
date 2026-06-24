#include "NoisyORUtilities.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace noisy_or {

void requireProbabilityVector(
    const Eigen::VectorXd& values,
    const char* name)
{
    if (!values.allFinite() ||
        (values.array() < 0.0).any() ||
        (values.array() > 1.0).any()) {
        throw std::invalid_argument(
            std::string(name) + " must contain values in [0,1]");
    }
}

double logSumExp(const Eigen::VectorXd& values)
{
    if (values.size() == 0) {
        throw std::invalid_argument("Cannot normalize an empty vector");
    }

    const double maximum = values.maxCoeff();
    return maximum
         + std::log((values.array() - maximum).exp().sum());
}

BinaryMatrix expandCandidates(
    const std::vector<State>& candidates,
    std::size_t generatorCount)
{
    if (generatorCount > std::numeric_limits<State>::digits) {
        throw std::invalid_argument(
            "Generator count exceeds the State bit width");
    }

    BinaryMatrix states(
        static_cast<Eigen::Index>(candidates.size()),
        static_cast<Eigen::Index>(generatorCount));

    for (Eigen::Index m = 0; m < states.rows(); ++m) {
        const State state = candidates[static_cast<std::size_t>(m)];

        if (generatorCount < std::numeric_limits<State>::digits &&
            (state >> generatorCount) != 0) {
            throw std::invalid_argument(
                "A candidate has bits outside the K-generator space");
        }

        for (std::size_t k = 0; k < generatorCount; ++k) {
            states(m, static_cast<Eigen::Index>(k)) =
                static_cast<std::uint8_t>(
                    (state >> k) & State{1});
        }
    }

    return states;
}

std::vector<State> enumerateSelectedGeneratorSubsets(
    const std::vector<Eigen::Index>& selectedGenerators,
    std::size_t generatorCount)
{
    constexpr std::size_t stateBits =
        std::numeric_limits<State>::digits;
    constexpr std::size_t sizeBits =
        std::numeric_limits<std::size_t>::digits;

    if (generatorCount > stateBits) {
        throw std::invalid_argument(
            "Generator count exceeds the State bit width");
    }
    if (selectedGenerators.size() >= sizeBits) {
        throw std::invalid_argument(
            "Selected generator set is too large to enumerate");
    }

    std::unordered_set<Eigen::Index> unique;
    for (const Eigen::Index generator : selectedGenerators) {
        if (generator < 0 ||
            generator >= static_cast<Eigen::Index>(generatorCount)) {
            throw std::invalid_argument(
                "Selected generator index is outside [0,K)");
        }
        if (!unique.insert(generator).second) {
            throw std::invalid_argument(
                "Selected generator indices must be unique");
        }
    }

    const std::size_t count =
        std::size_t{1} << selectedGenerators.size();
    std::vector<State> candidates(count, State{0});

    for (std::size_t subset = 0; subset < count; ++subset) {
        State state = 0;
        for (std::size_t local = 0;
             local < selectedGenerators.size();
             ++local) {
            if ((subset >> local) & std::size_t{1}) {
                state |= State{1}
                      << static_cast<std::size_t>(
                             selectedGenerators[local]);
            }
        }
        candidates[subset] = state;
    }

    return candidates;
}

std::vector<State> makeCompleteCandidates(std::size_t generatorCount)
{
    std::vector<Eigen::Index> generators(generatorCount);
    for (std::size_t k = 0; k < generatorCount; ++k) {
        generators[k] = static_cast<Eigen::Index>(k);
    }
    return enumerateSelectedGeneratorSubsets(generators, generatorCount);
}

} // namespace noisy_or
