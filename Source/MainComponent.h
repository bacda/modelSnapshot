#pragma once

#include <JuceHeader.h>
#include "NoisyOR.h"
#include "LayerStateDisplay.h"

#include <fstream>
#include <memory>
#include <optional>
#include <vector>

class MainComponent : public juce::Component,
                      private juce::Timer,
                      private juce::KeyListener
{
public:
    MainComponent();
    ~MainComponent() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    bool keyPressed(const juce::KeyPress& key) override;

private:
    juce::TextButton loadStateButton { "Load State" };
    juce::TextButton saveStateButton { "Save State" };
    juce::TextButton stepButton { "Step" };
    juce::ToggleButton runButton { "Run" };
    juce::TextButton fastRunButton { "Fast N" };
    juce::TextEditor fastRunIterationsEditor;
    juce::ToggleButton logToggle { "Log" };
    juce::TextButton renameLogButton { "Rename" };
    juce::TextButton clearLogButton { "Clear" };
    juce::TextButton resetStepButton { "Reset Step" };
    juce::TextButton dimensionsButton { "Dims" };
    juce::TextButton traceButton { "Trace" };
    juce::Label statusLabel;
    juce::TextEditor stateSummaryEditor;
    juce::Viewport viewport;
    LayerStateDisplay stateDisplay;

    std::unique_ptr<juce::FileChooser> fileChooser;
    std::unique_ptr<juce::DocumentWindow> traceWindow;
    std::optional<noisy_or::ModelState> modelState;
    std::unique_ptr<noisy_or::NoisyORStack> model;
    std::optional<std::size_t> currentTimestep;
    std::vector<std::optional<noisy_or::LearningDiagnostics>>
        learningDiagnostics;
    std::unique_ptr<std::ofstream> logStream;
    juce::File currentLogFile;

    void chooseStateFile();
    void chooseSaveStateFile();
    void chooseRenameLogFile();
    void openTraceWindow();
    juce::File autoSaveFile() const;
    juce::File defaultLogFile() const;
    void loadAutoSavedState();
    void saveAutoSavedState();
    void loadStateFile(const juce::File& file);
    void saveStateFile(const juce::File& file);
    void stepModel();
    bool advanceModelOneStep(bool updateUi);
    void runOfflineIterations();
    void showDimensionsDialog();
    void applyDimensions(
        Eigen::Index inputLength,
        Eigen::Index inputDimension,
        const std::vector<Eigen::Index>& generatorCounts,
        Eigen::Index contextOrder);
    void applyRandomizeRequest(
        const LayerStateDisplay::RandomizeRequest& request);
    void updateDisplay();
    void updateStateSummary();
    juce::String stateSummaryText() const;
    void startLoggingToFile(
        const juce::File& file,
        bool writeInitialSnapshot = true);
    void stopLogging();
    void clearCurrentLogFile();
    void resetStepCounter();
    bool isLogging() const;
    void writeLogMetadata();
    void writeLogStateSnapshot(const char* role);
    void logStep(
        std::size_t inputRow,
        const Eigen::VectorXd& observation,
        const std::vector<noisy_or::LayerOutput>& outputs);
    void applyMatrixEdit(const LayerStateDisplay::MatrixEdit& edit);
    void applyParameterEdit(const LayerStateDisplay::ParameterEdit& edit);
    void applySettingEdit(const LayerStateDisplay::SettingEdit& edit);
    void rebuildModelAfterParameterEdit();
    void setRunEnabled(bool shouldRun);
    bool handleShortcutKey(const juce::KeyPress& key);
    bool keyPressed(
        const juce::KeyPress& key,
        juce::Component* originatingComponent) override;
    void timerCallback() override;


    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
