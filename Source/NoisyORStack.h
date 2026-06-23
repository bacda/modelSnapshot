#pragma once

#include "NoisyORLayer.h"

#include <vector>

namespace noisy_or {

class NoisyORStack {
public:
    explicit NoisyORStack(std::vector<LayerConfiguration> configurations);

    void initialize();
    void initialize(
        const std::vector<Eigen::VectorXd>& initialTopDownSupport);

    // Runs candidate selection and inference bottom-up, then preserves the
    // next-step activation evidence top-down.
    const std::vector<LayerOutput>& step(
        const Eigen::VectorXd& bottomObservation,
        const Eigen::VectorXd& externalTopDownSupport = {});

    std::size_t size() const noexcept;
    NoisyORLayer& layer(std::size_t index);
    const NoisyORLayer& layer(std::size_t index) const;
    const std::vector<LayerOutput>& outputs() const noexcept;

private:
    std::vector<NoisyORLayer> layers_;
    std::vector<LayerOutput> outputs_;
    bool initialized_ = false;

    void validateConnections() const;
};

} // namespace noisy_or
