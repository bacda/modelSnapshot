#include "noisy_or/NoisyORStack.h"

#include <stdexcept>
#include <utility>

namespace noisy_or {

NoisyORStack::NoisyORStack(
    std::vector<LayerConfiguration> configurations)
{
    if (configurations.empty()) {
        throw std::invalid_argument(
            "A NoisyORStack must contain at least one layer");
    }

    layers_.reserve(configurations.size());
    for (auto& configuration : configurations) {
        layers_.emplace_back(std::move(configuration));
    }

    outputs_.resize(layers_.size());
    validateConnections();
}

void NoisyORStack::validateConnections() const
{
    for (std::size_t lower = 0; lower + 1 < layers_.size(); ++lower) {
        if (layers_[lower].generatorCount() !=
            layers_[lower + 1].channelCount()) {
            throw std::invalid_argument(
                "Adjacent layers require lower K == upper N");
        }
    }
}

void NoisyORStack::initialize()
{
    for (auto& currentLayer : layers_) {
        currentLayer.initialize();
    }
    outputs_.assign(layers_.size(), LayerOutput{});
    initialized_ = true;
}

const std::vector<LayerOutput>& NoisyORStack::step(
    const Eigen::VectorXd& bottomObservation,
    const Eigen::VectorXd& externalTopDownSupport)
{
    if (!initialized_) {
        throw std::logic_error("initialize() must be called before step()");
    }

    Eigen::VectorXd currentObservation = bottomObservation;
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        outputs_[index] = layers_[index].observe(currentObservation);
        currentObservation = outputs_[index].marginals;
    }

    for (std::size_t reverse = layers_.size(); reverse-- > 0;) {
        if (reverse + 1 == layers_.size()) {
            layers_[reverse].prepareNext(externalTopDownSupport);
        } else {
            layers_[reverse].prepareNext(
                outputs_[reverse + 1].reconstruction);
        }
    }

    return outputs_;
}

std::size_t NoisyORStack::size() const noexcept { return layers_.size(); }

NoisyORLayer& NoisyORStack::layer(std::size_t index)
{
    return layers_.at(index);
}

const NoisyORLayer& NoisyORStack::layer(std::size_t index) const
{
    return layers_.at(index);
}

const std::vector<LayerOutput>& NoisyORStack::outputs() const noexcept
{
    return outputs_;
}

} // namespace noisy_or
