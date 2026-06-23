#include "MainComponent.h"

//==============================================================================
MainComponent::MainComponent()
{
    setSize (600, 400);
}

MainComponent::~MainComponent()
{
}

//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

    g.setFont (juce::FontOptions (16.0f));
    g.setColour (juce::Colours::white);
    g.drawText ("Hello World!", getLocalBounds(), juce::Justification::centred, true);
}

void MainComponent::resized()
{
    // This is called when the MainComponent is resized.
    // If you add any child components, this is where you should
    // update their positions.
}
void MainComponent::initialiseModel(int _N, int _K, int o_){
    
      constexpr Eigen::Index N = _N;
      constexpr Eigen::Index K = _K;
      constexpr Eigen::Index order = _o;

      noisy_or::LayerConfiguration configuration;

      // R: K x N generator prediction matrix.
      configuration.predictions.resize(K, N);
      configuration.predictions <<
          0.90, 0.00, 0.00, 0.00,
          0.00, 0.85, 0.00, 0.00,
          0.00, 0.00, 0.80, 0.00;

      configuration.leak =
          Eigen::VectorXd::Constant(N, 0.01);

      // Initial context C_1: N x order.
      configuration.initialContext =
          Eigen::MatrixXd::Zero(N, order);

      // For this test, each filter has the same profile as its generator's
      // prediction vector. Every filter has shape N x order.
      configuration.filters.resize(K);

      for (Eigen::Index k = 0; k < K; ++k)
      {
          configuration.filters[static_cast<std::size_t>(k)] =
              configuration.predictions.row(k).transpose();
      }

      configuration.parameters.baseRate =
          Eigen::VectorXd::Constant(K, 0.05);

      configuration.parameters.bottomUpWeight =
          Eigen::VectorXd::Constant(K, 0.7);

      configuration.parameters.evidenceAmplitude =
          Eigen::VectorXd::Constant(K, 4.0);

      configuration.parameters.centering =
          Eigen::VectorXd::Constant(K, 0.5);

      auto& selection = configuration.candidateSelection;

      selection.contextThreshold = 0.01;
      selection.topDownThreshold = 0.01;
      selection.observationThreshold = 0.0;
      selection.useActivationSupport = false;
      selection.maximumSelectedGenerators = 10;

      layer = std::make_unique<noisy_or::NoisyORLayer>(
          std::move(configuration));

      // Uses zero top-down support for the first observation.
      layer->initialize();
}
