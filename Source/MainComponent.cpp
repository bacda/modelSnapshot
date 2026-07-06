#include "MainComponent.h"
#include "LogTracePlotter.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

class TraceDocumentWindow : public juce::DocumentWindow
{
public:
    TraceDocumentWindow()
        : juce::DocumentWindow(
            "Log Trace Plotter",
            juce::Colours::black,
            juce::DocumentWindow::closeButton)
    {
        setUsingNativeTitleBar(true);
        setResizable(true, true);

        plotter = new LogTracePlotter();
        plotter->setSize(1180, 760);
        setContentOwned(plotter, true);
        centreWithSize(1180, 760);
    }

    void closeButtonPressed() override
    {
        setVisible(false);
        auto close = onClose;
        juce::MessageManager::callAsync(
            [close]
            {
                if (close != nullptr)
                    close();
            });
    }

    std::function<void()> onClose;
    LogTracePlotter* plotter = nullptr;
};

Eigen::MatrixXd resizedMatrixCopy(
    const Eigen::MatrixXd& source,
    Eigen::Index rows,
    Eigen::Index columns,
    double fillValue = 0.0)
{
    Eigen::MatrixXd result =
        Eigen::MatrixXd::Constant(rows, columns, fillValue);
    const Eigen::Index copyRows = std::min(source.rows(), rows);
    const Eigen::Index copyColumns = std::min(source.cols(), columns);
    if (copyRows > 0 && copyColumns > 0)
        result.topLeftCorner(copyRows, copyColumns) =
            source.topLeftCorner(copyRows, copyColumns);
    return result;
}

Eigen::VectorXd resizedVectorCopy(
    const Eigen::VectorXd& source,
    Eigen::Index size,
    double fillValue)
{
    Eigen::VectorXd result = Eigen::VectorXd::Constant(size, fillValue);
    const Eigen::Index copySize = std::min(source.size(), size);
    if (copySize > 0)
        result.head(copySize) = source.head(copySize);
    return result;
}

juce::String generatorCountsText(
    const std::vector<Eigen::Index>& generatorCounts)
{
    juce::String text;
    for (std::size_t index = 0; index < generatorCounts.size(); ++index)
    {
        if (index > 0)
            text << ", ";
        text << static_cast<int>(generatorCounts[index]);
    }
    return text;
}

std::vector<Eigen::Index> parseGeneratorCounts(
    const juce::String& text,
    Eigen::Index layerCount,
    const std::vector<Eigen::Index>& fallback)
{
    if (layerCount <= 0)
        throw std::invalid_argument("Layer count must be positive");

    juce::StringArray tokens;
    tokens.addTokens(text, ",; ", "\"");
    tokens.trim();
    tokens.removeEmptyStrings();

    std::vector<Eigen::Index> result;
    result.reserve(static_cast<std::size_t>(layerCount));

    for (const auto& token : tokens)
    {
        const int value = token.getIntValue();
        if (value <= 0)
            throw std::invalid_argument(
                "Generator counts must be positive integers");
        result.push_back(static_cast<Eigen::Index>(value));
    }

    if (result.empty())
        result = fallback;

    while (result.size() < static_cast<std::size_t>(layerCount))
    {
        const Eigen::Index value = result.empty()
            ? Eigen::Index{3}
            : result.back();
        result.push_back(value);
    }

    if (result.size() > static_cast<std::size_t>(layerCount))
        result.resize(static_cast<std::size_t>(layerCount));

    return result;
}

std::uint64_t freshSeed()
{
    std::random_device device;
    return (static_cast<std::uint64_t>(device()) << 32)
        ^ static_cast<std::uint64_t>(device());
}

juce::String stepCountText(const noisy_or::ModelState& state)
{
    return juce::String(std::to_string(state.stepCount));
}

void writeJsonNumber(std::ostream& output, double value)
{
    if (!std::isfinite(value))
    {
        output << "null";
        return;
    }

    std::ostringstream formatted;
    formatted << std::setprecision(3);

    if (std::abs(value) > 0.0 && std::abs(value) < 0.001)
        formatted << std::scientific;

    formatted << value;
    output << formatted.str();
}

void writeJsonNumberPrecise(std::ostream& output, double value)
{
    if (!std::isfinite(value))
    {
        output << "null";
        return;
    }

    std::ostringstream formatted;
    formatted << std::setprecision(17) << value;
    output << formatted.str();
}

void writeJsonVector(std::ostream& output, const Eigen::VectorXd& values)
{
    output << '[';
    for (Eigen::Index index = 0; index < values.size(); ++index)
    {
        if (index > 0)
            output << ',';
        writeJsonNumber(output, values(index));
    }
    output << ']';
}

void writeJsonVectorPrecise(
    std::ostream& output,
    const Eigen::VectorXd& values)
{
    output << '[';
    for (Eigen::Index index = 0; index < values.size(); ++index)
    {
        if (index > 0)
            output << ',';
        writeJsonNumberPrecise(output, values(index));
    }
    output << ']';
}

void writeJsonIndexArray(
    std::ostream& output,
    const std::vector<Eigen::Index>& values)
{
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index > 0)
            output << ',';
        output << values[index];
    }
    output << ']';
}

void writeJsonBool(std::ostream& output, bool value)
{
    output << (value ? "true" : "false");
}

void writeJsonMatrix(std::ostream& output, const Eigen::MatrixXd& matrix)
{
    output << '[';
    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
    {
        if (row > 0)
            output << ',';

        output << '[';
        for (Eigen::Index column = 0; column < matrix.cols(); ++column)
        {
            if (column > 0)
                output << ',';
            writeJsonNumber(output, matrix(row, column));
        }
        output << ']';
    }
    output << ']';
}

void writeJsonMatrixPrecise(
    std::ostream& output,
    const Eigen::MatrixXd& matrix)
{
    output << '[';
    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
    {
        if (row > 0)
            output << ',';

        output << '[';
        for (Eigen::Index column = 0; column < matrix.cols(); ++column)
        {
            if (column > 0)
                output << ',';
            writeJsonNumberPrecise(output, matrix(row, column));
        }
        output << ']';
    }
    output << ']';
}

void writeJsonMatrixListPrecise(
    std::ostream& output,
    const std::vector<Eigen::MatrixXd>& matrices);

void writeJsonGeneratorParameters(
    std::ostream& output,
    const noisy_or::GeneratorParameters& parameters)
{
    output << "{\"base_rate\":";
    writeJsonVectorPrecise(output, parameters.baseRate);
    output << ",\"bottom_up_weight\":";
    writeJsonVectorPrecise(output, parameters.bottomUpWeight);
    output << ",\"evidence_amplitude\":";
    writeJsonVectorPrecise(output, parameters.evidenceAmplitude);
    output << ",\"centering\":";
    writeJsonVectorPrecise(output, parameters.centering);
    output << '}';
}

void writeJsonCandidateSelection(
    std::ostream& output,
    const noisy_or::CandidateSelectionPolicy& policy)
{
    output << "{\"context_threshold\":";
    writeJsonNumberPrecise(output, policy.contextThreshold);
    output << ",\"top_down_threshold\":";
    writeJsonNumberPrecise(output, policy.topDownThreshold);
    output << ",\"observation_threshold\":";
    writeJsonNumberPrecise(output, policy.observationThreshold);
    output << ",\"activation_threshold\":";
    writeJsonNumberPrecise(output, policy.activationThreshold);
    output << ",\"use_activation_support\":";
    writeJsonBool(output, policy.useActivationSupport);
    output << ",\"maximum_selected_generators\":"
           << policy.maximumSelectedGenerators << '}';
}

void writeJsonLearningOptions(
    std::ostream& output,
    const noisy_or::OnlineEMOptions& options,
    bool enabled)
{
    output << "{\"enabled\":";
    writeJsonBool(output, enabled);
    output << ",\"prediction_learning_rate\":";
    writeJsonNumberPrecise(output, options.predictionLearningRate);
    output << ",\"filter_learning_rate\":";
    writeJsonNumberPrecise(output, options.filterLearningRate);
    output << ",\"base_rate_learning_rate\":";
    writeJsonNumberPrecise(output, options.baseRateLearningRate);
    output << ",\"epsilon\":";
    writeJsonNumberPrecise(output, options.epsilon);
    output << ",\"binarize_observation\":";
    writeJsonBool(output, options.binarizeObservation);
    output << ",\"observation_threshold\":";
    writeJsonNumberPrecise(output, options.observationThreshold);
    output << '}';
}

void writeJsonMatrixList(
    std::ostream& output,
    const std::vector<Eigen::MatrixXd>& matrices);

void writeJsonModelState(
    std::ostream& output,
    const noisy_or::ModelState& state)
{
    output << "{\"input\":";
    writeJsonMatrixPrecise(output, state.input);
    output << ",\"loop_input\":";
    writeJsonBool(output, state.loopInput);
    output << ",\"input_index\":" << state.inputIndex
           << ",\"step_count\":" << state.stepCount
           << ",\"layers\":[";

    for (std::size_t layerIndex = 0;
         layerIndex < state.layers.size();
         ++layerIndex)
    {
        if (layerIndex > 0)
            output << ',';

        const auto& layer = state.layers[layerIndex];
        const auto& configuration = layer.configuration;
        const Eigen::Index K = configuration.predictions.rows();
        const Eigen::Index N = configuration.predictions.cols();
        const Eigen::Index order = configuration.initialContext.cols();
        const Eigen::VectorXd topDown =
            layer.initialTopDownSupport.size() == 0
                ? Eigen::VectorXd::Zero(K)
                : layer.initialTopDownSupport;

        output << "{\"layer\":" << layerIndex
               << ",\"N\":" << N
               << ",\"K\":" << K
               << ",\"order\":" << order
               << ",\"R\":";
        writeJsonMatrixPrecise(output, configuration.predictions);
        output << ",\"F\":";
        writeJsonMatrixListPrecise(output, configuration.filters);
        output << ",\"leak\":";
        writeJsonVectorPrecise(output, configuration.leak);
        output << ",\"parameters\":";
        writeJsonGeneratorParameters(output, configuration.parameters);
        output << ",\"context\":";
        writeJsonMatrixPrecise(output, configuration.initialContext);
        output << ",\"top_down\":";
        writeJsonVectorPrecise(output, topDown);
        output << ",\"candidate_selection\":";
        writeJsonCandidateSelection(output, configuration.candidateSelection);
        output << ",\"learning\":";
        writeJsonLearningOptions(
            output,
            layer.learning,
            layer.learningEnabled);
        output << '}';
    }

    output << "]}";
}

void writeJsonMatrixList(
    std::ostream& output,
    const std::vector<Eigen::MatrixXd>& matrices)
{
    output << '[';
    for (std::size_t index = 0; index < matrices.size(); ++index)
    {
        if (index > 0)
            output << ',';
        writeJsonMatrix(output, matrices[index]);
    }
    output << ']';
}

void writeJsonMatrixListPrecise(
    std::ostream& output,
    const std::vector<Eigen::MatrixXd>& matrices)
{
    output << '[';
    for (std::size_t index = 0; index < matrices.size(); ++index)
    {
        if (index > 0)
            output << ',';
        writeJsonMatrixPrecise(output, matrices[index]);
    }
    output << ']';
}

void writeJsonOptionalVector(
    std::ostream& output,
    const std::optional<noisy_or::LearningDiagnostics>& diagnostics,
    Eigen::VectorXd noisy_or::LearningDiagnostics::* member)
{
    if (diagnostics.has_value())
        writeJsonVector(output, (*diagnostics).*member);
    else
        output << "null";
}

void writeJsonOptionalMatrix(
    std::ostream& output,
    const std::optional<noisy_or::LearningDiagnostics>& diagnostics,
    Eigen::MatrixXd noisy_or::LearningDiagnostics::* member)
{
    if (diagnostics.has_value())
        writeJsonMatrix(output, (*diagnostics).*member);
    else
        output << "null";
}

void writeJsonOptionalMatrixList(
    std::ostream& output,
    const std::optional<noisy_or::LearningDiagnostics>& diagnostics,
    std::vector<Eigen::MatrixXd> noisy_or::LearningDiagnostics::* member)
{
    if (diagnostics.has_value())
        writeJsonMatrixList(output, (*diagnostics).*member);
    else
        output << "null";
}

} // namespace

//==============================================================================
MainComponent::MainComponent()
{
    setSize (1100, 760);
    setWantsKeyboardFocus(true);
    setMouseClickGrabsKeyboardFocus(true);

    addAndMakeVisible(loadStateButton);
    addAndMakeVisible(saveStateButton);
    addAndMakeVisible(stepButton);
    addAndMakeVisible(runButton);
    addAndMakeVisible(fastRunButton);
    addAndMakeVisible(fastRunIterationsEditor);
    addAndMakeVisible(logToggle);
    addAndMakeVisible(renameLogButton);
    addAndMakeVisible(clearLogButton);
    addAndMakeVisible(resetStepButton);
    addAndMakeVisible(dimensionsButton);
    addAndMakeVisible(traceButton);
    addAndMakeVisible(statusLabel);
    addAndMakeVisible(stateSummaryEditor);
    addAndMakeVisible(viewport);

    viewport.setViewedComponent(&stateDisplay, false);
    viewport.setScrollBarsShown(true, true);
    stateDisplay.onMatrixEdit = [this](const LayerStateDisplay::MatrixEdit& edit)
    {
        applyMatrixEdit(edit);
    };
    stateDisplay.onParameterEdit = [this](const LayerStateDisplay::ParameterEdit& edit)
    {
        applyParameterEdit(edit);
    };
    stateDisplay.onSettingEdit = [this](const LayerStateDisplay::SettingEdit& edit)
    {
        applySettingEdit(edit);
    };
    stateDisplay.onRandomizeRequest =
        [this](const LayerStateDisplay::RandomizeRequest& request)
    {
        applyRandomizeRequest(request);
    };

    addKeyListener(this);
    loadStateButton.addKeyListener(this);
    saveStateButton.addKeyListener(this);
    stepButton.addKeyListener(this);
    runButton.addKeyListener(this);
    fastRunButton.addKeyListener(this);
    logToggle.addKeyListener(this);
    renameLogButton.addKeyListener(this);
    clearLogButton.addKeyListener(this);
    resetStepButton.addKeyListener(this);
    dimensionsButton.addKeyListener(this);
    traceButton.addKeyListener(this);
    stateSummaryEditor.addKeyListener(this);
    viewport.addKeyListener(this);
    stateDisplay.addKeyListener(this);

    loadStateButton.onClick = [this] { chooseStateFile(); };
    saveStateButton.onClick = [this] { chooseSaveStateFile(); };
    stepButton.onClick = [this] { stepModel(); };
    fastRunButton.onClick = [this] { runOfflineIterations(); };
    logToggle.onClick = [this]
    {
        if (logToggle.getToggleState())
            startLoggingToFile(
                currentLogFile == juce::File{}
                    ? defaultLogFile()
                    : currentLogFile);
        else
            stopLogging();
    };
    logToggle.setColour( juce::ToggleButton::textColourId, juce::Colours::black); logToggle.setColour( juce::ToggleButton::tickColourId, juce::Colours::black); logToggle.setColour( juce::ToggleButton::tickDisabledColourId, juce::Colours::darkgrey);
    
    renameLogButton.onClick = [this] { chooseRenameLogFile(); };
    clearLogButton.onClick = [this] { clearCurrentLogFile(); };
    resetStepButton.onClick = [this] { resetStepCounter(); };
    dimensionsButton.onClick = [this] { showDimensionsDialog(); };
    traceButton.onClick = [this] { openTraceWindow(); };
    runButton.onClick = [this]
    {
        setRunEnabled(runButton.getToggleState());
    };

    fastRunIterationsEditor.setText("1000", juce::dontSendNotification);
    fastRunIterationsEditor.setInputRestrictions(9, "0123456789");
    fastRunIterationsEditor.setJustification(
        juce::Justification::centred);
    fastRunIterationsEditor.setSelectAllWhenFocused(true);

    saveStateButton.setEnabled(false);
    stepButton.setEnabled(false);
    runButton.setEnabled(false);
    fastRunButton.setEnabled(false);
    fastRunIterationsEditor.setEnabled(false);
    logToggle.setEnabled(false);
    renameLogButton.setEnabled(false);
    clearLogButton.setEnabled(false);
    resetStepButton.setEnabled(false);


    statusLabel.setText(
        "Load a state file to initialise the model",
        juce::dontSendNotification);
    statusLabel.setJustificationType(juce::Justification::centredLeft);

    stateSummaryEditor.setReadOnly(true);
    stateSummaryEditor.setMultiLine(true);
    stateSummaryEditor.setScrollbarsShown(false);
    stateSummaryEditor.setCaretVisible(false);
    stateSummaryEditor.setPopupMenuEnabled(false);
    stateSummaryEditor.setWantsKeyboardFocus(false);
    stateSummaryEditor.setColour(
        juce::TextEditor::backgroundColourId,
        juce::Colour::fromRGB(247, 249, 251));
    stateSummaryEditor.setColour(
        juce::TextEditor::outlineColourId,
        juce::Colour::fromRGB(220, 225, 230));
    stateSummaryEditor.setColour(
        juce::TextEditor::focusedOutlineColourId,
        juce::Colour::fromRGB(170, 180, 190));
    stateSummaryEditor.setColour(
        juce::TextEditor::textColourId,
        juce::Colour::fromRGB(22, 26, 30));
    updateStateSummary();

    loadAutoSavedState();
}

MainComponent::~MainComponent()
{
    stopTimer();
    stopLogging();
    traceWindow = nullptr;
    saveAutoSavedState();
    stateDisplay.removeKeyListener(this);
    viewport.removeKeyListener(this);
    dimensionsButton.removeKeyListener(this);
    traceButton.removeKeyListener(this);
    stateSummaryEditor.removeKeyListener(this);
    resetStepButton.removeKeyListener(this);
    clearLogButton.removeKeyListener(this);
    renameLogButton.removeKeyListener(this);
    logToggle.removeKeyListener(this);
    fastRunButton.removeKeyListener(this);
    runButton.removeKeyListener(this);
    stepButton.removeKeyListener(this);
    saveStateButton.removeKeyListener(this);
    loadStateButton.removeKeyListener(this);
    removeKeyListener(this);
    viewport.setViewedComponent(nullptr, false);
}

//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{
    g.fillAll(juce::Colours::white);
}

void MainComponent::resized()
{
    auto bounds = getLocalBounds().reduced(16);
    auto controlsArea = bounds.removeFromTop(70);
    auto controls = controlsArea.removeFromTop(32);

    loadStateButton.setBounds(controls.removeFromLeft(120));
    controls.removeFromLeft(8);
    saveStateButton.setBounds(controls.removeFromLeft(120));
    controls.removeFromLeft(8);
    stepButton.setBounds(controls.removeFromLeft(80));
    controls.removeFromLeft(8);
    runButton.setBounds(controls.removeFromLeft(80));
    controls.removeFromLeft(8);
    fastRunButton.setBounds(controls.removeFromLeft(80));
    controls.removeFromLeft(6);
    fastRunIterationsEditor.setBounds(controls.removeFromLeft(72));
    controls.removeFromLeft(8);
    dimensionsButton.setBounds(controls.removeFromLeft(70));
    controls.removeFromLeft(8);
    traceButton.setBounds(controls.removeFromLeft(70));

    controls.removeFromLeft(12);
    statusLabel.setBounds(controls);

    controlsArea.removeFromTop(6);
    auto logControls = controlsArea.removeFromTop(32);
    logToggle.setBounds(logControls.removeFromLeft(70));
    logControls.removeFromLeft(6);
    renameLogButton.setBounds(logControls.removeFromLeft(78));
    logControls.removeFromLeft(6);
    clearLogButton.setBounds(logControls.removeFromLeft(58));
    logControls.removeFromLeft(6);
    resetStepButton.setBounds(logControls.removeFromLeft(92));

    bounds.removeFromTop(12);
    auto summaryBounds = bounds.removeFromLeft(210);
    stateSummaryEditor.setBounds(summaryBounds);
    bounds.removeFromLeft(12);
    viewport.setBounds(bounds);

    const int contentWidth = std::max(
        1,
        std::max(stateDisplay.preferredWidth(), viewport.getWidth() - 18));
    const int contentHeight = std::max(
        1,
        std::max(stateDisplay.preferredHeight(contentWidth),
                 viewport.getHeight() - 18));
    stateDisplay.setSize(
        contentWidth,
        contentHeight);
}

bool MainComponent::keyPressed(const juce::KeyPress& key)
{
    return handleShortcutKey(key);
}

bool MainComponent::keyPressed(
    const juce::KeyPress& key,
    juce::Component* originatingComponent)
{
    juce::ignoreUnused(originatingComponent);
    return handleShortcutKey(key);
}

bool MainComponent::handleShortcutKey(const juce::KeyPress& key)
{
    if (key.getModifiers().isAnyModifierKeyDown())
        return false;

    if (key.getKeyCode() == juce::KeyPress::spaceKey)
    {
        if (runButton.isEnabled())
            setRunEnabled(! runButton.getToggleState());

        return true;
    }

    if (key.getKeyCode() == juce::KeyPress::rightKey)
    {
        if (stepButton.isEnabled())
            stepButton.triggerClick();

        return true;
    }

    return false;
}

void MainComponent::setRunEnabled(bool shouldRun)
{
    runButton.setToggleState(shouldRun, juce::dontSendNotification);

    if (shouldRun)
        startTimerHz(20);
    else
        stopTimer();
}

void MainComponent::chooseStateFile()
{
    fileChooser = std::make_unique<juce::FileChooser>(
        "Choose a Noisy-OR state file",
        juce::File{},
        "*.txt;*.state");

    juce::Component::SafePointer<MainComponent> safeThis(this);

    fileChooser->launchAsync(
        juce::FileBrowserComponent::openMode
            | juce::FileBrowserComponent::canSelectFiles,
        [safeThis](const juce::FileChooser& chooser)
        {
            if (safeThis == nullptr)
                return;

            const auto file = chooser.getResult();
            if (file.existsAsFile())
                safeThis->loadStateFile(file);
        });
}

juce::File MainComponent::autoSaveFile() const
{
    auto directory = juce::File::getSpecialLocation(
        juce::File::userApplicationDataDirectory)
        .getChildFile(ProjectInfo::projectName);

    return directory.getChildFile("autosave.state");
}

juce::File MainComponent::defaultLogFile() const
{
    return juce::File::getSpecialLocation(
        juce::File::userDocumentsDirectory)
        .getChildFile("noisy_or_trace.jsonl");
}

void MainComponent::loadAutoSavedState()
{
    const auto file = autoSaveFile();
    if (!file.existsAsFile())
        return;

    try
    {
        auto loaded = noisy_or::loadModelState(
            file.getFullPathName().toStdString());

        for (const auto& warning : loaded.warnings)
            DBG(juce::String("Autosave state warning: ")
                + juce::String(warning));

        auto loadedModel = noisy_or::makeInitializedStack(loaded.state);

        modelState = std::move(loaded.state);
        model = std::make_unique<noisy_or::NoisyORStack>(
            std::move(loadedModel));

        saveStateButton.setEnabled(true);
        stepButton.setEnabled(modelState->input.rows() > 0);
        runButton.setEnabled(modelState->input.rows() > 0);
        fastRunButton.setEnabled(modelState->input.rows() > 0);
        fastRunIterationsEditor.setEnabled(modelState->input.rows() > 0);
        logToggle.setEnabled(modelState->input.rows() > 0);
        renameLogButton.setEnabled(true);
        clearLogButton.setEnabled(true);
        resetStepButton.setEnabled(true);
        learningDiagnostics.assign(model->size(), std::nullopt);
        currentTimestep.reset();
        updateDisplay();

        statusLabel.setText(
            "Restored autosaved state",
            juce::dontSendNotification);
    }
    catch (const std::exception& error)
    {
        const juce::String message =
            "Could not restore autosaved state: "
            + juce::String(error.what());
        statusLabel.setText(message, juce::dontSendNotification);
        DBG(message);
    }
}

void MainComponent::saveAutoSavedState()
{
    if (model == nullptr || !modelState.has_value())
        return;

    try
    {
        const auto file = autoSaveFile();
        const auto directory = file.getParentDirectory();
        if (!directory.exists() &&
            directory.createDirectory().failed())
        {
            DBG("Could not create autosave directory: "
                + directory.getFullPathName());
            return;
        }

        noisy_or::ModelState snapshot =
            noisy_or::snapshotModelState(*model, *modelState);

        noisy_or::saveModelState(
            file.getFullPathName().toStdString(),
            snapshot);

        modelState = std::move(snapshot);
    }
    catch (const std::exception& error)
    {
        DBG("Could not autosave state: " + juce::String(error.what()));
    }
}

void MainComponent::chooseSaveStateFile()
{
    if (model == nullptr || !modelState.has_value())
        return;

    fileChooser = std::make_unique<juce::FileChooser>(
        "Save current Noisy-OR state",
        juce::File::getSpecialLocation(
            juce::File::userDocumentsDirectory).getChildFile("noisy_or_checkpoint.state"),
        "*.state;*.txt");

    juce::Component::SafePointer<MainComponent> safeThis(this);

    fileChooser->launchAsync(
        juce::FileBrowserComponent::saveMode
            | juce::FileBrowserComponent::canSelectFiles
            | juce::FileBrowserComponent::warnAboutOverwriting,
        [safeThis](const juce::FileChooser& chooser)
        {
            if (safeThis == nullptr)
                return;

            auto file = chooser.getResult();
            if (file == juce::File{})
                return;

            if (! file.hasFileExtension("state") &&
                ! file.hasFileExtension("txt"))
                file = file.withFileExtension("state");

            safeThis->saveStateFile(file);
        });
}

void MainComponent::chooseRenameLogFile()
{
    fileChooser = std::make_unique<juce::FileChooser>(
        "Rename or choose current Noisy-OR JSONL log file",
        currentLogFile == juce::File{}
            ? defaultLogFile()
            : currentLogFile,
        "*.jsonl;*.txt");

    juce::Component::SafePointer<MainComponent> safeThis(this);

    fileChooser->launchAsync(
        juce::FileBrowserComponent::saveMode
            | juce::FileBrowserComponent::canSelectFiles
            | juce::FileBrowserComponent::warnAboutOverwriting,
        [safeThis](const juce::FileChooser& chooser)
        {
            if (safeThis == nullptr)
                return;

            auto file = chooser.getResult();
            if (file == juce::File{})
                return;

            if (! file.hasFileExtension("jsonl") &&
                ! file.hasFileExtension("txt"))
                file = file.withFileExtension("jsonl");

            const bool wasLogging = safeThis->isLogging();
            if (wasLogging)
            {
                safeThis->logStream->flush();
                safeThis->logStream.reset();
            }

            if (safeThis->currentLogFile != juce::File{}
                && safeThis->currentLogFile.existsAsFile()
                && safeThis->currentLogFile != file)
            {
                if (!safeThis->currentLogFile.moveFileTo(file))
                {
                    safeThis->statusLabel.setText(
                        "Could not rename log file",
                        juce::dontSendNotification);
                    if (wasLogging)
                        safeThis->startLoggingToFile(
                            safeThis->currentLogFile,
                            false);
                    return;
                }
            }

            safeThis->currentLogFile = file;

            if (wasLogging)
                safeThis->startLoggingToFile(file, false);
            else
                safeThis->statusLabel.setText(
                    "Current log file: " + file.getFileName(),
                    juce::dontSendNotification);
        });
}

void MainComponent::openTraceWindow()
{
    if (isLogging())
        logStream->flush();

    if (traceWindow != nullptr)
    {
        if (auto* trace = dynamic_cast<TraceDocumentWindow*>(traceWindow.get()))
        {
            if (trace->plotter != nullptr && currentLogFile.existsAsFile())
                trace->plotter->loadFile(currentLogFile);
        }
        traceWindow->toFront(true);
        return;
    }

    auto window = std::make_unique<TraceDocumentWindow>();
    auto* rawWindow = window.get();
    juce::Component::SafePointer<MainComponent> safeThis(this);
    rawWindow->onClose = [safeThis]
    {
        if (safeThis != nullptr)
            safeThis->traceWindow = nullptr;
    };

    traceWindow = std::move(window);
    traceWindow->setVisible(true);
    if (rawWindow->plotter != nullptr && currentLogFile.existsAsFile())
        rawWindow->plotter->loadFile(currentLogFile);
    traceWindow->toFront(true);
}

void MainComponent::saveStateFile(const juce::File& file)
{
    if (model == nullptr || !modelState.has_value())
        return;

    try
    {
        stopTimer();
        runButton.setToggleState(false, juce::dontSendNotification);

        noisy_or::ModelState snapshot =
            noisy_or::snapshotModelState(*model, *modelState);

        noisy_or::saveModelState(
            file.getFullPathName().toStdString(),
            snapshot);

        // Keep the GUI's metadata in sync with what was written, including
        // learned F/R/base rates and the current contexts.
        modelState = std::move(snapshot);
        updateDisplay();

        statusLabel.setText(
            "Saved " + file.getFileName(),
            juce::dontSendNotification);
    }
    catch (const std::exception& error)
    {
        const juce::String message =
            "Could not save state: " + juce::String(error.what());
        statusLabel.setText(message, juce::dontSendNotification);
        DBG(message);
    }
}

void MainComponent::loadStateFile(const juce::File& file)
{
    try
    {
        stopTimer();
        stopLogging();
        runButton.setToggleState(false, juce::dontSendNotification);
        // Capture learned parameters and temporal context before applying a
        // partial file. If no model exists, missing fields use defaults.
        std::optional<noisy_or::ModelState> previous;

        if (model != nullptr && modelState.has_value())
            previous = noisy_or::snapshotModelState(*model, *modelState);

        auto loaded = noisy_or::loadModelState(
            file.getFullPathName().toStdString(),
            previous.has_value() ? &*previous : nullptr);

        for (const auto& warning : loaded.warnings)
            DBG(juce::String("State warning: ") + juce::String(warning));

        auto loadedModel = noisy_or::makeInitializedStack(loaded.state);

        modelState = std::move(loaded.state);
        model = std::make_unique<noisy_or::NoisyORStack>(
            std::move(loadedModel));

        saveStateButton.setEnabled(true);
        stepButton.setEnabled(modelState->input.rows() > 0);
        runButton.setEnabled(modelState->input.rows() > 0);
        fastRunButton.setEnabled(modelState->input.rows() > 0);
        fastRunIterationsEditor.setEnabled(modelState->input.rows() > 0);
        logToggle.setEnabled(modelState->input.rows() > 0);
        renameLogButton.setEnabled(true);
        clearLogButton.setEnabled(true);
        resetStepButton.setEnabled(true);
        learningDiagnostics.assign(model->size(), std::nullopt);
        currentTimestep.reset();
        updateDisplay();

        juce::String status;
        status << "Loaded " << file.getFileName()
               << " | layers: " << static_cast<int>(model->size())
               << " | observations: "
               << static_cast<int>(modelState->input.rows())
               << " | warnings: " << static_cast<int>(loaded.warnings.size());
        statusLabel.setText(status, juce::dontSendNotification);
    }
    catch (const std::exception& error)
    {
        model.reset();
        modelState.reset();
        currentTimestep.reset();
        learningDiagnostics.clear();
        saveStateButton.setEnabled(false);
        stepButton.setEnabled(false);
        runButton.setEnabled(false);
        fastRunButton.setEnabled(false);
        fastRunIterationsEditor.setEnabled(false);
        logToggle.setEnabled(false);
        renameLogButton.setEnabled(false);
        clearLogButton.setEnabled(false);
        resetStepButton.setEnabled(false);
        stopLogging();
        runButton.setToggleState(false, juce::dontSendNotification);
        stopTimer();

        const juce::String message =
            "Could not load state: " + juce::String(error.what());
        statusLabel.setText(message, juce::dontSendNotification);
        DBG(message);
    }
}

void MainComponent::stepModel()
{
    if (model == nullptr || !modelState.has_value())
        return;

    try
    {
        advanceModelOneStep(true);
    }
    catch (const std::exception& error)
    {
        stopTimer();
        runButton.setToggleState(false, juce::dontSendNotification);
        const juce::String message =
            "Model step failed: " + juce::String(error.what());
        statusLabel.setText(message, juce::dontSendNotification);
        DBG(message);
    }
}

bool MainComponent::advanceModelOneStep(bool updateUi)
{
    if (model == nullptr || !modelState.has_value())
        return false;

    Eigen::VectorXd observation;
    const std::size_t rowCount =
        static_cast<std::size_t>(modelState->input.rows());
    const std::size_t timestep = rowCount == 0
        ? 0 : modelState->inputIndex % rowCount;

    if (!noisy_or::readNextInput(*modelState, observation))
    {
        if (updateUi)
        {
            stopTimer();
            runButton.setToggleState(false, juce::dontSendNotification);
            statusLabel.setText(
                "Input sequence exhausted",
                juce::dontSendNotification);
        }
        return false;
    }

    const auto& outputs = model->step(observation);
    currentTimestep = timestep;
    if (learningDiagnostics.size() != model->size())
        learningDiagnostics.assign(model->size(), std::nullopt);
    else
        std::fill(
            learningDiagnostics.begin(),
            learningDiagnostics.end(),
            std::nullopt);

    for (std::size_t index = 0; index < modelState->layers.size(); ++index)
    {
        const auto& layerState = modelState->layers[index];
        if (layerState.learningEnabled)
        {
            learningDiagnostics[index] =
                noisy_or::updateParametersOnlineEM(
                model->layer(index),
                layerState.learning);
        }
    }

    ++modelState->stepCount;
    logStep(timestep, observation, outputs);

    if (updateUi)
    {
        const auto& bottom = outputs.front();
        juce::String status;
        status << "step: " << stepCountText(*modelState)
               << " | input row: " << static_cast<int>(timestep)
               << " | candidates: "
               << static_cast<int>(bottom.posterior.size())
               << " | selected generators: "
               << static_cast<int>(bottom.selection.selectedGenerators.size())
               << " | log evidence: "
               << juce::String(bottom.logEvidence, 3);
        statusLabel.setText(status, juce::dontSendNotification);
        updateDisplay();
    }

    return true;
}

void MainComponent::runOfflineIterations()
{
    if (model == nullptr || !modelState.has_value())
        return;

    const int requested = juce::jlimit(
        1,
        10000000,
        fastRunIterationsEditor.getText().getIntValue());
    fastRunIterationsEditor.setText(
        juce::String(requested),
        juce::dontSendNotification);

    setRunEnabled(false);

    std::size_t completed = 0;
    bool exhausted = false;
    const auto start = std::chrono::steady_clock::now();

    try
    {
        for (; completed < static_cast<std::size_t>(requested); ++completed)
        {
            if (!advanceModelOneStep(false))
            {
                exhausted = true;
                break;
            }
        }

        if (completed > 0)
            updateDisplay();

        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();

        juce::String status;
        status << "Fast ran " << static_cast<int>(completed)
               << " / " << requested << " iterations"
               << " | total step: " << stepCountText(*modelState);
        if (elapsed > 0.0)
        {
            status << " in " << juce::String(elapsed, 3) << " s"
                   << " | "
                   << juce::String(static_cast<double>(completed) / elapsed, 1)
                   << " it/s";
        }
        if (exhausted)
            status << " | input sequence exhausted";

        statusLabel.setText(status, juce::dontSendNotification);
    }
    catch (const std::exception& error)
    {
        runButton.setToggleState(false, juce::dontSendNotification);
        const juce::String message =
            "Fast run failed: " + juce::String(error.what());
        statusLabel.setText(message, juce::dontSendNotification);
        DBG(message);
    }
}

void MainComponent::showDimensionsDialog()
{
    Eigen::Index inputLength = 16;
    Eigen::Index inputDimension = 4;
    Eigen::Index layerCount = 1;
    std::vector<Eigen::Index> generatorCounts { 3 };
    Eigen::Index contextOrder = 1;

    if (modelState.has_value())
    {
        inputLength = std::max<Eigen::Index>(1, modelState->input.rows());
        inputDimension = std::max<Eigen::Index>(1, modelState->input.cols());
        layerCount = std::max<Eigen::Index>(
            1,
            static_cast<Eigen::Index>(modelState->layers.size()));
        generatorCounts.clear();
        generatorCounts.reserve(modelState->layers.size());
        if (!modelState->layers.empty())
        {
            for (const auto& layer : modelState->layers)
            {
                generatorCounts.push_back(std::max<Eigen::Index>(
                    1,
                    layer.configuration.predictions.rows()));
            }
            contextOrder =
                std::max<Eigen::Index>(
                    1,
                    modelState->layers.front()
                        .configuration.initialContext.cols());
        }
    }

    auto* alert = new juce::AlertWindow(
        "Model dimensions",
        "Set input sequence and layer dimensions",
        juce::AlertWindow::NoIcon);
    alert->addTextEditor("inputLength", juce::String(inputLength),
                         "input length T");
    alert->addTextEditor("inputDimension", juce::String(inputDimension),
                         "input dimension N");
    alert->addTextEditor("layerCount", juce::String(layerCount),
                         "number of layers");
    alert->addTextEditor("generatorCounts", generatorCountsText(generatorCounts),
                         "generators per layer, e.g. 4, 3, 2");
    alert->addTextEditor("contextOrder", juce::String(contextOrder),
                         "context order");
    alert->addButton("Apply", 1, juce::KeyPress(juce::KeyPress::returnKey));
    alert->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));

    juce::Component::SafePointer<MainComponent> safeThis(this);
    alert->enterModalState(
        true,
        juce::ModalCallbackFunction::create(
            [safeThis, alert, generatorCounts](int result)
            {
                std::unique_ptr<juce::AlertWindow> alertOwner(alert);
                if (safeThis == nullptr || result != 1)
                    return;

                const auto readIndex =
                    [alert](const char* name, Eigen::Index fallback)
                    {
                        const int value =
                            alert->getTextEditorContents(name).getIntValue();
                        return value > 0
                            ? static_cast<Eigen::Index>(value)
                            : fallback;
                    };

                try
                {
                    safeThis->applyDimensions(
                        readIndex("inputLength", 16),
                        readIndex("inputDimension", 4),
                        parseGeneratorCounts(
                            alert->getTextEditorContents("generatorCounts"),
                            readIndex("layerCount", 1),
                            generatorCounts),
                        readIndex("contextOrder", 1));
                }
                catch (const std::exception& error)
                {
                    const juce::String message =
                        "Could not apply dimensions: "
                        + juce::String(error.what());
                    safeThis->statusLabel.setText(
                        message,
                        juce::dontSendNotification);
                    DBG(message);
                }
            }),
        false);
}

void MainComponent::applyDimensions(
    Eigen::Index inputLength,
    Eigen::Index inputDimension,
    const std::vector<Eigen::Index>& generatorCounts,
    Eigen::Index contextOrder)
{
    if (inputLength <= 0 || inputDimension <= 0 ||
        generatorCounts.empty() || contextOrder <= 0)
    {
        throw std::invalid_argument("All dimensions must be positive");
    }
    for (const auto generatorCount : generatorCounts)
    {
        if (generatorCount <= 0)
            throw std::invalid_argument("All generator counts must be positive");
    }

    stopTimer();
    stopLogging();
    runButton.setToggleState(false, juce::dontSendNotification);

    noisy_or::ModelState edited;
    if (model != nullptr && modelState.has_value())
        edited = noisy_or::snapshotModelState(*model, *modelState);
    else if (modelState.has_value())
        edited = *modelState;

    edited.input = resizedMatrixCopy(
        edited.input,
        inputLength,
        inputDimension,
        0.0);
    edited.inputIndex = edited.input.rows() == 0
        ? 0
        : std::min<std::size_t>(
            edited.inputIndex,
            static_cast<std::size_t>(edited.input.rows() - 1));

    edited.layers.resize(generatorCounts.size());

    for (std::size_t layerIndex = 0;
         layerIndex < generatorCounts.size();
         ++layerIndex)
    {
        const Eigen::Index channelCount = layerIndex == 0
            ? inputDimension
            : generatorCounts[layerIndex - 1];
        const Eigen::Index generatorCount = generatorCounts[layerIndex];

        auto& layer = edited.layers[layerIndex];
        auto& configuration = layer.configuration;

        const Eigen::MatrixXd previousPredictions =
            configuration.predictions;
        const auto previousFilters = configuration.filters;

        Eigen::MatrixXd nextPredictions = noisy_or::makeRandomPredictions(
            generatorCount,
            channelCount,
            0.05,
            1.0,
            freshSeed());
        const Eigen::Index copyPredictionRows =
            std::min(previousPredictions.rows(), nextPredictions.rows());
        const Eigen::Index copyPredictionCols =
            std::min(previousPredictions.cols(), nextPredictions.cols());
        if (copyPredictionRows > 0 && copyPredictionCols > 0)
        {
            nextPredictions.topLeftCorner(
                copyPredictionRows,
                copyPredictionCols) =
                previousPredictions.topLeftCorner(
                    copyPredictionRows,
                    copyPredictionCols);
        }
        configuration.predictions = std::move(nextPredictions);

        auto nextFilters = noisy_or::makeRandomFilters(
            generatorCount,
            channelCount,
            contextOrder,
            0.05,
            1.0,
            freshSeed());
        const std::size_t copyFilterCount = std::min(
            previousFilters.size(),
            nextFilters.size());
        for (std::size_t k = 0; k < copyFilterCount; ++k)
        {
            const Eigen::Index copyRows = std::min(
                previousFilters[k].rows(),
                nextFilters[k].rows());
            const Eigen::Index copyCols = std::min(
                previousFilters[k].cols(),
                nextFilters[k].cols());
            if (copyRows > 0 && copyCols > 0)
            {
                nextFilters[k].topLeftCorner(copyRows, copyCols) =
                    previousFilters[k].topLeftCorner(copyRows, copyCols);
            }
        }
        configuration.filters = std::move(nextFilters);

        configuration.leak = resizedVectorCopy(
            configuration.leak,
            channelCount,
            0.01);

        configuration.parameters.baseRate = resizedVectorCopy(
            configuration.parameters.baseRate,
            generatorCount,
            0.05);
        configuration.parameters.bottomUpWeight = resizedVectorCopy(
            configuration.parameters.bottomUpWeight,
            generatorCount,
            0.7);
        configuration.parameters.evidenceAmplitude = resizedVectorCopy(
            configuration.parameters.evidenceAmplitude,
            generatorCount,
            4.0);
        configuration.parameters.centering = resizedVectorCopy(
            configuration.parameters.centering,
            generatorCount,
            0.5);

        configuration.initialContext = resizedMatrixCopy(
            configuration.initialContext,
            channelCount,
            contextOrder,
            0.0);
        layer.initialTopDownSupport = resizedVectorCopy(
            layer.initialTopDownSupport,
            generatorCount,
            0.0);
    }

    noisy_or::validateModelState(edited);
    modelState = std::move(edited);
    rebuildModelAfterParameterEdit();
    saveStateButton.setEnabled(true);

    juce::String status;
    status << "Set dimensions: T=" << static_cast<int>(inputLength)
           << " N=" << static_cast<int>(inputDimension)
           << " layers=" << static_cast<int>(generatorCounts.size())
           << " K=[" << generatorCountsText(generatorCounts) << "]"
           << " order=" << static_cast<int>(contextOrder);
    statusLabel.setText(status, juce::dontSendNotification);
}

void MainComponent::applyRandomizeRequest(
    const LayerStateDisplay::RandomizeRequest& request)
{
    if (model == nullptr || !modelState.has_value())
        return;

    try
    {
        noisy_or::ModelState edited =
            noisy_or::snapshotModelState(*model, *modelState);
        if (request.layerIndex >= edited.layers.size())
            return;

        std::random_device device;
        auto& configuration =
            edited.layers[request.layerIndex].configuration;

        if (request.matrix == LayerStateDisplay::EditableMatrix::Predictions)
        {
            configuration.predictions = noisy_or::makeRandomPredictions(
                configuration.predictions.rows(),
                configuration.predictions.cols(),
                request.betaShapeA,
                request.betaShapeB,
                (static_cast<std::uint64_t>(device()) << 32)
                    ^ static_cast<std::uint64_t>(device())
                    ^ static_cast<std::uint64_t>(request.layerIndex));
        }
        else if (request.matrix == LayerStateDisplay::EditableMatrix::Filters)
        {
            configuration.filters = noisy_or::makeRandomFilters(
                configuration.predictions.rows(),
                configuration.predictions.cols(),
                configuration.initialContext.cols(),
                request.betaShapeA,
                request.betaShapeB,
                (static_cast<std::uint64_t>(device()) << 32)
                    ^ static_cast<std::uint64_t>(device())
                    ^ static_cast<std::uint64_t>(request.layerIndex));
        }
        else
        {
            return;
        }

        noisy_or::validateModelState(edited);
        modelState = std::move(edited);
        rebuildModelAfterParameterEdit();

        const juce::String matrixName =
            request.matrix == LayerStateDisplay::EditableMatrix::Predictions
                ? "R"
                : "F";
        statusLabel.setText(
            "Randomized layer "
            + juce::String(static_cast<int>(request.layerIndex))
            + " " + matrixName + " from Beta("
            + juce::String(request.betaShapeA, 3)
            + ", "
            + juce::String(request.betaShapeB, 3)
            + ")",
            juce::dontSendNotification);
    }
    catch (const std::exception& error)
    {
        statusLabel.setText(
            "Could not randomize matrix: " + juce::String(error.what()),
            juce::dontSendNotification);
        DBG(statusLabel.getText());
    }
}

void MainComponent::updateDisplay()
{
    if (model == nullptr || !modelState.has_value())
        return;

    stateDisplay.update(
        *modelState,
        *model,
        currentTimestep,
        learningDiagnostics);
    updateStateSummary();

    const int contentWidth = std::max(
        1,
        std::max(stateDisplay.preferredWidth(), viewport.getWidth() - 18));
    const int contentHeight = std::max(
        1,
        std::max(stateDisplay.preferredHeight(contentWidth),
                 viewport.getHeight() - 18));
    stateDisplay.setSize(
        contentWidth,
        contentHeight);
}

void MainComponent::updateStateSummary()
{
    stateSummaryEditor.setText(stateSummaryText(), juce::dontSendNotification);
}

juce::String MainComponent::stateSummaryText() const
{
    if (!modelState.has_value())
        return "Current .state\n\nNo state loaded.";

    const auto& state = *modelState;
    juce::String text;
    text << "Current .state\n\n";
    text << "step: " << stepCountText(state) << "\n";
    text << "input rows: " << static_cast<int>(state.input.rows()) << "\n";
    text << "input dim: " << static_cast<int>(state.input.cols()) << "\n";
    text << "input index: " << static_cast<int>(state.inputIndex) << "\n";
    text << "loop input: " << (state.loopInput ? "yes" : "no") << "\n";
    text << "layers: " << static_cast<int>(state.layers.size()) << "\n\n";

    for (std::size_t layerIndex = 0;
         layerIndex < state.layers.size();
         ++layerIndex)
    {
        const auto& layer = state.layers[layerIndex];
        const auto& config = layer.configuration;
        const Eigen::Index k = config.predictions.rows();
        const Eigen::Index n = config.predictions.cols();
        const Eigen::Index order = config.filters.empty()
            ? config.initialContext.cols()
            : config.filters.front().cols();

        text << "L" << static_cast<int>(layerIndex)
             << ": N=" << static_cast<int>(n)
             << " K=" << static_cast<int>(k)
             << " order=" << static_cast<int>(order) << "\n";
        text << "  learn: " << (layer.learningEnabled ? "on" : "off")
             << "  etaR=" << juce::String(
                    layer.learning.predictionLearningRate, 3)
             << " etaF=" << juce::String(
                    layer.learning.filterLearningRate, 3)
             << "\n";
        text << "  cand qF>="
             << juce::String(
                    config.candidateSelection.contextThreshold, 3)
             << " obs>="
             << juce::String(
                    config.candidateSelection.observationThreshold, 3)
             << "\n\n";
    }

    if (currentLogFile != juce::File{})
    {
        text << "log:\n";
        text << currentLogFile.getFileName() << "\n";
        text << (isLogging() ? "logging on" : "logging off") << "\n";
    }

    return text;
}

void MainComponent::startLoggingToFile(
    const juce::File& file,
    bool writeInitialSnapshot)
{
    if (model == nullptr || !modelState.has_value())
        return;

    const auto directory = file.getParentDirectory();
    if (!directory.exists() && directory.createDirectory().failed())
    {
        statusLabel.setText(
            "Could not create log directory: "
                + directory.getFullPathName(),
            juce::dontSendNotification);
        return;
    }

    const bool shouldWriteMetadata =
        !file.existsAsFile() || file.getSize() == 0;

    auto stream = std::make_unique<std::ofstream>(
        file.getFullPathName().toStdString(),
        std::ios::out | std::ios::app);

    if (!*stream)
    {
        statusLabel.setText(
            "Could not open log file: " + file.getFileName(),
            juce::dontSendNotification);
        logToggle.setToggleState(false, juce::dontSendNotification);
        return;
    }

    logStream = std::move(stream);
    currentLogFile = file;
    if (shouldWriteMetadata)
        writeLogMetadata();
    if (writeInitialSnapshot)
        writeLogStateSnapshot("initial");

    logToggle.setToggleState(true, juce::dontSendNotification);
    statusLabel.setText(
        "Logging to " + file.getFileName(),
        juce::dontSendNotification);
    updateStateSummary();
}

void MainComponent::stopLogging()
{
    if (!isLogging())
        return;

    const auto fileName = currentLogFile.getFileName();
    writeLogStateSnapshot("final");
    logStream->flush();
    logStream.reset();
    logToggle.setToggleState(false, juce::dontSendNotification);

    if (fileName.isNotEmpty())
    {
        statusLabel.setText(
            "Stopped logging " + fileName,
            juce::dontSendNotification);
    }
    updateStateSummary();
}

bool MainComponent::isLogging() const
{
    return logStream != nullptr && *logStream;
}

void MainComponent::clearCurrentLogFile()
{
    const auto file = currentLogFile == juce::File{}
        ? defaultLogFile()
        : currentLogFile;
    const bool wasLogging = isLogging();

    if (wasLogging)
    {
        logStream->flush();
        logStream.reset();
    }

    const auto directory = file.getParentDirectory();
    if (!directory.exists() && directory.createDirectory().failed())
    {
        statusLabel.setText(
            "Could not create log directory: "
                + directory.getFullPathName(),
            juce::dontSendNotification);
        if (wasLogging)
            startLoggingToFile(file);
        return;
    }

    std::ofstream clearStream(
        file.getFullPathName().toStdString(),
        std::ios::out | std::ios::trunc);

    if (!clearStream)
    {
        statusLabel.setText(
            "Could not clear log file: " + file.getFileName(),
            juce::dontSendNotification);
        if (wasLogging)
            startLoggingToFile(file);
        return;
    }

    clearStream.close();
    currentLogFile = file;

    if (wasLogging)
    {
        startLoggingToFile(file);
        statusLabel.setText(
            "Cleared and resumed logging " + file.getFileName(),
            juce::dontSendNotification);
    }
    else
    {
        statusLabel.setText(
            "Cleared log file " + file.getFileName(),
            juce::dontSendNotification);
    }
    updateStateSummary();
}

void MainComponent::resetStepCounter()
{
    if (!modelState.has_value())
        return;

    modelState->stepCount = 0;
    updateDisplay();
    statusLabel.setText(
        "Reset step counter to 0",
        juce::dontSendNotification);
}

void MainComponent::writeLogMetadata()
{
    if (!isLogging() || model == nullptr || !modelState.has_value())
        return;

    auto& output = *logStream;
    output << std::setprecision(17);
    output << "{\"type\":\"metadata\","
           << "\"schema\":\"noisy_or_step_log_v2\","
           << "\"step_count_start\":" << modelState->stepCount << ','
           << "\"input_rows\":" << modelState->input.rows() << ','
           << "\"input_columns\":" << modelState->input.cols() << ','
           << "\"loop_input\":" << (modelState->loopInput ? "true" : "false")
           << ",\"layers\":[";

    for (std::size_t layerIndex = 0;
         layerIndex < modelState->layers.size();
         ++layerIndex)
    {
        if (layerIndex > 0)
            output << ',';

        const auto& configuration =
            modelState->layers[layerIndex].configuration;
        output << "{\"layer\":" << layerIndex
               << ",\"N\":" << configuration.predictions.cols()
               << ",\"K\":" << configuration.predictions.rows()
               << ",\"order\":"
               << (configuration.filters.empty()
                   ? 0
                   : configuration.filters.front().cols())
               << '}';
    }

    output << "]}\n";
    output.flush();
}

void MainComponent::writeLogStateSnapshot(const char* role)
{
    if (!isLogging() || model == nullptr || !modelState.has_value())
        return;

    try
    {
        const auto snapshot =
            noisy_or::snapshotModelState(*model, *modelState);

        auto& output = *logStream;
        output << std::setprecision(17);
        output << "{\"type\":\"state\","
               << "\"role\":\"" << role << "\","
               << "\"step\":" << snapshot.stepCount
               << ",\"state\":";
        writeJsonModelState(output, snapshot);
        output << "}\n";
        output.flush();
    }
    catch (const std::exception& error)
    {
        DBG("Could not write log state snapshot: "
            + juce::String(error.what()));
    }
}

void MainComponent::logStep(
    std::size_t inputRow,
    const Eigen::VectorXd& observation,
    const std::vector<noisy_or::LayerOutput>& outputs)
{
    if (!isLogging() || model == nullptr || !modelState.has_value())
        return;

    auto& output = *logStream;
    output << std::setprecision(17);
    output << "{\"type\":\"step\","
           << "\"step\":" << modelState->stepCount << ','
           << "\"input_row\":" << inputRow << ','
           << "\"x\":";
    writeJsonVector(output, observation);
    output << ",\"layers\":[";

    for (std::size_t layerIndex = 0; layerIndex < outputs.size(); ++layerIndex)
    {
        if (layerIndex > 0)
            output << ',';

        const auto& layerOutput = outputs[layerIndex];
        output << "{\"layer\":" << layerIndex
               << ",\"log_evidence\":";
        writeJsonNumber(output, layerOutput.logEvidence);
        output << ",\"candidate_count\":" << layerOutput.posterior.size()
               << ",\"selected_generators\":";
        writeJsonIndexArray(output, layerOutput.selection.selectedGenerators);
        output << ",\"top_down_prior\":";
        writeJsonVector(output, layerOutput.topDownSupport);
        output << ",\"alpha_inherited\":";
        writeJsonVector(output, layerOutput.alpha);
        output << ",\"alpha_next\":";
        writeJsonVector(output, model->layer(layerIndex).alpha());
        output << ",\"qF\":";
        writeJsonVector(output, layerOutput.filterMatch);
        output << ",\"mu\":";
        writeJsonVector(output, layerOutput.marginals);
        output << ",\"reconstruction\":";
        writeJsonVector(output, layerOutput.reconstruction);
        output << ",\"R\":";
        writeJsonMatrix(
            output,
            model->layer(layerIndex).configuration().predictions);
        output << ",\"F\":";
        writeJsonMatrixList(
            output,
            model->layer(layerIndex).configuration().filters);

        if (layerIndex < learningDiagnostics.size())
        {
            output << ",\"learning\":{"
                   << "\"activation_error\":";
            writeJsonOptionalVector(
                output,
                learningDiagnostics[layerIndex],
                &noisy_or::LearningDiagnostics::activationError);
            output << ",\"base_rate_delta\":";
            writeJsonOptionalVector(
                output,
                learningDiagnostics[layerIndex],
                &noisy_or::LearningDiagnostics::baseRateDelta);
            output << ",\"R_delta\":";
            writeJsonOptionalMatrix(
                output,
                learningDiagnostics[layerIndex],
                &noisy_or::LearningDiagnostics::predictionDelta);
            output << ",\"F_delta\":";
            writeJsonOptionalMatrixList(
                output,
                learningDiagnostics[layerIndex],
                &noisy_or::LearningDiagnostics::filterDelta);
            output << '}';
        }

        output << '}';
    }

    output << "]}\n";

    if (!output)
    {
        logStream.reset();
        logToggle.setToggleState(false, juce::dontSendNotification);
        statusLabel.setText(
            "Log write failed; logging stopped",
            juce::dontSendNotification);
    }
}


void MainComponent::applyMatrixEdit(const LayerStateDisplay::MatrixEdit& edit)
{
    if (!modelState.has_value())
        return;

    try
    {
        const double value = juce::jlimit(0.0, 1.0, edit.value);

        if (edit.matrix == LayerStateDisplay::EditableMatrix::Input)
        {
            if (edit.row < 0 || edit.column < 0 ||
                edit.row >= modelState->input.rows() ||
                edit.column >= modelState->input.cols())
                return;

            modelState->input(edit.row, edit.column) = value;
            statusLabel.setText(
                "Edited X[" + juce::String(static_cast<int>(edit.row))
                + ", " + juce::String(static_cast<int>(edit.column))
                + "] = " + juce::String(value, 3),
                juce::dontSendNotification);
            updateDisplay();
            return;
        }

        noisy_or::ModelState edited = model != nullptr
            ? noisy_or::snapshotModelState(*model, *modelState)
            : *modelState;

        if (edit.layerIndex >= edited.layers.size())
            return;

        auto& layerConfiguration =
            edited.layers[edit.layerIndex].configuration;

        if (edit.matrix == LayerStateDisplay::EditableMatrix::Predictions)
        {
            if (edit.row < 0 || edit.column < 0 ||
                edit.row >= layerConfiguration.predictions.rows() ||
                edit.column >= layerConfiguration.predictions.cols())
                return;

            layerConfiguration.predictions(edit.row, edit.column) = value;
        }
        else if (edit.matrix == LayerStateDisplay::EditableMatrix::Filters)
        {
            if (edit.row < 0 || edit.column < 0 ||
                edit.row >= static_cast<Eigen::Index>(
                    layerConfiguration.filters.size()))
                return;

            const Eigen::Index contextOrder =
                layerConfiguration.initialContext.cols();
            if (contextOrder <= 0 || edit.column < 0)
                return;

            const Eigen::Index channel = edit.column / contextOrder;
            const Eigen::Index lag = edit.column % contextOrder;

            auto& filter =
                layerConfiguration.filters[static_cast<std::size_t>(edit.row)];
            if (channel >= filter.rows() || lag >= filter.cols())
                return;

            filter(channel, lag) = value;
        }

        noisy_or::validateModelState(edited);
        modelState = std::move(edited);
        rebuildModelAfterParameterEdit();

        juce::String matrixName = edit.matrix ==
            LayerStateDisplay::EditableMatrix::Predictions ? "R" : "F";
        statusLabel.setText(
            "Edited layer " + juce::String(static_cast<int>(edit.layerIndex))
            + " " + matrixName + "["
            + juce::String(static_cast<int>(edit.row)) + ", "
            + juce::String(static_cast<int>(edit.column)) + "] = "
            + juce::String(value, 3),
            juce::dontSendNotification);
    }
    catch (const std::exception& error)
    {
        statusLabel.setText(
            "Matrix edit rejected: " + juce::String(error.what()),
            juce::dontSendNotification);
        DBG(statusLabel.getText());
    }
}

void MainComponent::applyParameterEdit(
    const LayerStateDisplay::ParameterEdit& edit)
{
    if (!modelState.has_value())
        return;

    try
    {
        if (edit.layerIndex >= modelState->layers.size())
            return;

        auto& parameters =
            modelState->layers[edit.layerIndex].configuration.parameters;

        if (!edit.applyToAllGenerators &&
            (edit.generator < 0 ||
             edit.generator >= parameters.baseRate.size()))
            return;

        double value = edit.value;
        juce::String parameterName;
        const auto applyValue =
            [&edit, &value](Eigen::VectorXd& values)
            {
                if (edit.applyToAllGenerators)
                    values.setConstant(value);
                else
                    values(edit.generator) = value;
            };

        switch (edit.parameter)
        {
            case LayerStateDisplay::EditableParameter::BaseRate:
                value = juce::jlimit(0.0, 1.0, value);
                applyValue(parameters.baseRate);
                parameterName = "base";
                break;

            case LayerStateDisplay::EditableParameter::BottomUpWeight:
                value = juce::jlimit(0.0, 1.0, value);
                applyValue(parameters.bottomUpWeight);
                parameterName = "w";
                break;

            case LayerStateDisplay::EditableParameter::EvidenceAmplitude:
                value = std::max(0.0, value);
                applyValue(parameters.evidenceAmplitude);
                parameterName = "A";
                break;

            case LayerStateDisplay::EditableParameter::Centering:
                value = juce::jlimit(0.0, 1.0, value);
                applyValue(parameters.centering);
                parameterName = "c";
                break;
        }

        if (model != nullptr && edit.layerIndex < model->size())
        {
            model->layer(edit.layerIndex)
                .configuration()
                .parameters = parameters;
        }

        noisy_or::validateModelState(*modelState);

        if (runButton.getToggleState())
            stateDisplay.repaint();
        else
            updateDisplay();

        juce::String targetText = edit.applyToAllGenerators
            ? " all generators "
            : " generator "
                + juce::String(static_cast<int>(edit.generator))
                + " ";

        statusLabel.setText(
            "Edited layer "
            + juce::String(static_cast<int>(edit.layerIndex))
            + targetText
            + " " + parameterName + " = "
            + juce::String(value, 3),
            juce::dontSendNotification);
    }
    catch (const std::exception& error)
    {
        statusLabel.setText(
            "Parameter edit rejected: " + juce::String(error.what()),
            juce::dontSendNotification);
        DBG(statusLabel.getText());
    }
}

void MainComponent::applySettingEdit(
    const LayerStateDisplay::SettingEdit& edit)
{
    if (!modelState.has_value())
        return;

    try
    {
        if (edit.layerIndex >= modelState->layers.size())
            return;

        auto& layerState = modelState->layers[edit.layerIndex];
        auto& policy = layerState.configuration.candidateSelection;
        auto& learning = layerState.learning;

        double value = edit.value;
        juce::String settingName;

        switch (edit.setting)
        {
            case LayerStateDisplay::EditableSetting::CandidateContextThreshold:
                value = juce::jlimit(0.0, 1.0, value);
                policy.contextThreshold = value;
                settingName = "threshold qF";
                break;

            case LayerStateDisplay::EditableSetting::CandidateTopDownThreshold:
                value = juce::jlimit(0.0, 1.0, value);
                policy.topDownThreshold = value;
                settingName = "threshold tau";
                break;

            case LayerStateDisplay::EditableSetting::CandidateObservationThreshold:
                value = juce::jlimit(-1.0, 1.0, value);
                policy.observationThreshold = value;
                settingName = "threshold observation";
                break;

            case LayerStateDisplay::EditableSetting::CandidateUseActivationSupport:
                policy.useActivationSupport = value >= 0.5;
                value = policy.useActivationSupport ? 1.0 : 0.0;
                settingName = "alpha support";
                break;

            case LayerStateDisplay::EditableSetting::CandidateActivationThreshold:
                value = juce::jlimit(0.0, 1.0, value);
                policy.activationThreshold = value;
                settingName = "threshold alpha";
                break;

            case LayerStateDisplay::EditableSetting::CandidateMaximumSelectedGenerators:
                value = juce::jlimit(0.0, 62.0, std::round(value));
                policy.maximumSelectedGenerators = static_cast<std::size_t>(value);
                settingName = "maximum generators";
                break;

            case LayerStateDisplay::EditableSetting::LearningEnabled:
                layerState.learningEnabled = value >= 0.5;
                value = layerState.learningEnabled ? 1.0 : 0.0;
                settingName = "learning enabled";
                break;

            case LayerStateDisplay::EditableSetting::LearningPredictionRate:
                value = juce::jlimit(0.0, 1.0, value);
                learning.predictionLearningRate = value;
                settingName = "eta R";
                break;

            case LayerStateDisplay::EditableSetting::LearningFilterRate:
                value = juce::jlimit(0.0, 1.0, value);
                learning.filterLearningRate = value;
                settingName = "eta F";
                break;

            case LayerStateDisplay::EditableSetting::LearningBaseRate:
                value = juce::jlimit(0.0, 1.0, value);
                learning.baseRateLearningRate = value;
                settingName = "eta base";
                break;

            case LayerStateDisplay::EditableSetting::LearningEpsilon:
                value = juce::jlimit(1.0e-12, 1.0, value);
                learning.epsilon = value;
                settingName = "epsilon";
                break;

            case LayerStateDisplay::EditableSetting::LearningBinarizeObservation:
                learning.binarizeObservation = value >= 0.5;
                value = learning.binarizeObservation ? 1.0 : 0.0;
                settingName = "binarize";
                break;

            case LayerStateDisplay::EditableSetting::LearningObservationThreshold:
                value = juce::jlimit(0.0, 1.0, value);
                learning.observationThreshold = value;
                settingName = "learning threshold";
                break;
        }

        if (model != nullptr && edit.layerIndex < model->size())
        {
            model->layer(edit.layerIndex)
                .configuration()
                .candidateSelection = policy;
        }

        noisy_or::validateModelState(*modelState);

        if (runButton.getToggleState())
            stateDisplay.repaint();
        else
            updateDisplay();

        statusLabel.setText(
            "Edited layer "
            + juce::String(static_cast<int>(edit.layerIndex))
            + " " + settingName + " = "
            + juce::String(value, 6),
            juce::dontSendNotification);
    }
    catch (const std::exception& error)
    {
        statusLabel.setText(
            "Setting edit rejected: " + juce::String(error.what()),
            juce::dontSendNotification);
        DBG(statusLabel.getText());
    }
}

void MainComponent::rebuildModelAfterParameterEdit()
{
    if (!modelState.has_value())
        return;

    const bool wasRunning = runButton.getToggleState();
    stopTimer();

    auto rebuilt = noisy_or::makeInitializedStack(*modelState);
    model = std::make_unique<noisy_or::NoisyORStack>(std::move(rebuilt));

    saveStateButton.setEnabled(true);
    stepButton.setEnabled(modelState->input.rows() > 0);
    const bool canRun = modelState->input.rows() > 0;
    runButton.setEnabled(canRun);
    fastRunButton.setEnabled(canRun);
    fastRunIterationsEditor.setEnabled(canRun);
    logToggle.setEnabled(canRun);
    renameLogButton.setEnabled(true);
    clearLogButton.setEnabled(true);
    resetStepButton.setEnabled(true);
    learningDiagnostics.assign(model->size(), std::nullopt);
    currentTimestep.reset();
    updateDisplay();

    if (wasRunning && canRun)
    {
        runButton.setToggleState(true, juce::dontSendNotification);
        startTimerHz(4);
    }
    else
    {
        runButton.setToggleState(false, juce::dontSendNotification);
    }
}

void MainComponent::timerCallback()
{
    stepModel();
}
