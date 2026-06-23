#pragma once

#include <JuceHeader.h>
#include "NoisyOR.h"

#include <memory>

class MainComponent : public juce::Component
{
public:
    MainComponent();
    ~MainComponent() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    std::unique_ptr<noisy_or::NoisyORLayer> layer;

    void initialiseModel(
        Eigen::Index N,
        Eigen::Index K,
        Eigen::Index o);
    void runTestObservation();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
