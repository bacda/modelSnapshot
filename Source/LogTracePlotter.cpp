#include "LogTracePlotter.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace
{
bool isNumericVar(const juce::var& value)
{
    return value.isInt() || value.isInt64() || value.isDouble()
        || value.isBool();
}

double numericValue(const juce::var& value)
{
    if (value.isBool())
        return static_cast<bool>(value) ? 1.0 : 0.0;

    return static_cast<double>(value);
}

bool isNumericArray(const juce::var& value)
{
    const auto* array = value.getArray();
    if (array == nullptr || array->isEmpty())
        return false;

    for (const auto& item : *array)
        if (!isNumericVar(item))
            return false;

    return true;
}

bool isNumericMatrix(const juce::var& value)
{
    const auto* array = value.getArray();
    if (array == nullptr || array->isEmpty())
        return false;

    for (const auto& row : *array)
        if (!isNumericArray(row))
            return false;

    return true;
}

std::vector<double> numericVectorFromArray(const juce::var& value)
{
    std::vector<double> result;

    if (const auto* array = value.getArray())
    {
        result.reserve(static_cast<std::size_t>(array->size()));
        for (const auto& item : *array)
            result.push_back(numericValue(item));
    }

    return result;
}

std::vector<double> numericVectorFromMatrix(const juce::var& value)
{
    std::vector<double> result;

    if (const auto* rows = value.getArray())
    {
        for (const auto& row : *rows)
        {
            if (const auto* columns = row.getArray())
            {
                result.reserve(
                    result.size()
                    + static_cast<std::size_t>(columns->size()));

                for (const auto& item : *columns)
                    result.push_back(numericValue(item));
            }
        }
    }

    return result;
}

bool shouldFlattenMatrixAtPath(const juce::String& path)
{
    return path.contains(".F[")
        || path.contains(".F_delta[");
}

juce::String indexedPath(const juce::String& prefix, int index)
{
    return prefix + "[" + juce::String(index) + "]";
}

void writeExportJsonString(std::ostream& output, const juce::String& text)
{
    output << '"';
    const auto utf8 = text.toRawUTF8();
    for (const char* c = utf8; *c != '\0'; ++c)
    {
        switch (*c)
        {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (static_cast<unsigned char>(*c) < 0x20)
                {
                    output << "\\u"
                           << std::hex << std::setw(4)
                           << std::setfill('0')
                           << static_cast<int>(
                               static_cast<unsigned char>(*c))
                           << std::dec << std::setfill(' ');
                }
                else
                {
                    output << *c;
                }
                break;
        }
    }
    output << '"';
}

void writeExportJsonNumber(std::ostream& output, double value)
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

void writeExportJsonValues(
    std::ostream& output,
    const std::vector<double>& values)
{
    if (values.size() == 1)
    {
        writeExportJsonNumber(output, values.front());
        return;
    }

    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index > 0)
            output << ',';
        writeExportJsonNumber(output, values[index]);
    }
    output << ']';
}

juce::var propertyOf(const juce::var& value, const juce::String& name)
{
    if (const auto* object = value.getDynamicObject())
        return object->getProperty(name);

    return {};
}

int arraySizeOf(const juce::var& value)
{
    if (const auto* array = value.getArray())
        return array->size();

    return 0;
}

juce::var arrayItemOf(const juce::var& value, int index)
{
    if (const auto* array = value.getArray())
    {
        if (index >= 0 && index < array->size())
            return array->getReference(index);
    }

    return {};
}

double stateNumber(const juce::var& value)
{
    return numericValue(value);
}

int stateInt(const juce::var& value)
{
    return static_cast<int>(std::llround(stateNumber(value)));
}

bool stateBool(const juce::var& value)
{
    if (value.isBool())
        return static_cast<bool>(value);

    return stateNumber(value) != 0.0;
}

void writeStateNumber(std::ostream& output, const juce::var& value)
{
    output << std::setprecision(17) << stateNumber(value);
}

void writeStateMatrixValues(std::ostream& output, const juce::var& matrix)
{
    const int rows = arraySizeOf(matrix);

    for (int row = 0; row < rows; ++row)
    {
        const auto rowValue = arrayItemOf(matrix, row);
        const int columns = arraySizeOf(rowValue);

        for (int column = 0; column < columns; ++column)
        {
            if (column > 0)
                output << ' ';

            writeStateNumber(output, arrayItemOf(rowValue, column));
        }

        output << '\n';
    }
}

void writeStateMatrix(
    std::ostream& output,
    const char* name,
    const juce::var& matrix)
{
    const int rows = arraySizeOf(matrix);
    const int columns = rows > 0 ? arraySizeOf(arrayItemOf(matrix, 0)) : 0;

    output << name << " INLINE " << rows << ' ' << columns << '\n';
    writeStateMatrixValues(output, matrix);
}

void writeStateRowVector(
    std::ostream& output,
    const char* name,
    const juce::var& vector)
{
    const int columns = arraySizeOf(vector);

    output << name << " INLINE 1 " << columns << '\n';
    for (int column = 0; column < columns; ++column)
    {
        if (column > 0)
            output << ' ';

        writeStateNumber(output, arrayItemOf(vector, column));
    }
    output << '\n';
}

void writeStateFlattenedFilters(
    std::ostream& output,
    const juce::var& filters,
    int K,
    int N,
    int order)
{
    output << "F INLINE " << (K * N) << ' ' << order << '\n';

    for (int k = 0; k < K; ++k)
    {
        const auto filter = arrayItemOf(filters, k);

        for (int row = 0; row < N; ++row)
        {
            const auto rowValue = arrayItemOf(filter, row);

            for (int column = 0; column < order; ++column)
            {
                if (column > 0)
                    output << ' ';

                writeStateNumber(output, arrayItemOf(rowValue, column));
            }

            output << '\n';
        }
    }
}

void writeStateParameters(
    std::ostream& output,
    const juce::var& parameters,
    int K)
{
    const auto baseRate = propertyOf(parameters, "base_rate");
    const auto bottomUpWeight = propertyOf(parameters, "bottom_up_weight");
    const auto evidenceAmplitude =
        propertyOf(parameters, "evidence_amplitude");
    const auto centering = propertyOf(parameters, "centering");

    output << "PARAMETERS INLINE " << K << " 4\n";
    for (int k = 0; k < K; ++k)
    {
        writeStateNumber(output, arrayItemOf(baseRate, k));
        output << ' ';
        writeStateNumber(output, arrayItemOf(bottomUpWeight, k));
        output << ' ';
        writeStateNumber(output, arrayItemOf(evidenceAmplitude, k));
        output << ' ';
        writeStateNumber(output, arrayItemOf(centering, k));
        output << '\n';
    }
}

bool writeStateFileFromJson(std::ostream& output, const juce::var& state)
{
    const auto layers = propertyOf(state, "layers");
    const int layerCount = arraySizeOf(layers);

    if (layerCount <= 0)
        return false;

    output << std::setprecision(17);
    output << "NOISY_OR_STATE 1\n\n";

    writeStateMatrix(output, "INPUT", propertyOf(state, "input"));
    output << "LOOP " << (stateBool(propertyOf(state, "loop_input")) ? 1 : 0)
           << '\n';
    output << "INPUT_INDEX " << stateInt(propertyOf(state, "input_index"))
           << '\n';
    output << "STEP_COUNT " << stateInt(propertyOf(state, "step_count"))
           << "\n\n";

    output << "LAYERS " << layerCount << "\n\n";

    for (int layerIndex = 0; layerIndex < layerCount; ++layerIndex)
    {
        const auto layer = arrayItemOf(layers, layerIndex);
        const int N = stateInt(propertyOf(layer, "N"));
        const int K = stateInt(propertyOf(layer, "K"));
        const int order = stateInt(propertyOf(layer, "order"));
        const auto candidateSelection =
            propertyOf(layer, "candidate_selection");
        const auto learning = propertyOf(layer, "learning");

        if (N <= 0 || K <= 0 || order <= 0)
            return false;

        output << "LAYER " << layerIndex << '\n';
        output << "DIMENSIONS " << N << ' ' << K << ' ' << order << '\n';

        writeStateMatrix(output, "R", propertyOf(layer, "R"));
        writeStateFlattenedFilters(output, propertyOf(layer, "F"), K, N, order);
        writeStateRowVector(output, "LEAK", propertyOf(layer, "leak"));
        writeStateParameters(output, propertyOf(layer, "parameters"), K);
        writeStateMatrix(output, "CONTEXT", propertyOf(layer, "context"));
        writeStateRowVector(output, "TOP_DOWN", propertyOf(layer, "top_down"));

        output << "CANDIDATE_SELECTION\n";
        writeStateNumber(
            output,
            propertyOf(candidateSelection, "context_threshold"));
        output << ' ';
        writeStateNumber(
            output,
            propertyOf(candidateSelection, "top_down_threshold"));
        output << ' ';
        writeStateNumber(
            output,
            propertyOf(candidateSelection, "observation_threshold"));
        output << ' ';
        writeStateNumber(
            output,
            propertyOf(candidateSelection, "activation_threshold"));
        output << ' '
               << (stateBool(propertyOf(
                       candidateSelection,
                       "use_activation_support")) ? 1 : 0)
               << ' '
               << stateInt(propertyOf(
                       candidateSelection,
                       "maximum_selected_generators"))
               << '\n';

        output << "EM\n";
        output << (stateBool(propertyOf(learning, "enabled")) ? 1 : 0)
               << ' ';
        writeStateNumber(
            output,
            propertyOf(learning, "prediction_learning_rate"));
        output << ' ';
        writeStateNumber(
            output,
            propertyOf(learning, "filter_learning_rate"));
        output << ' ';
        writeStateNumber(
            output,
            propertyOf(learning, "base_rate_learning_rate"));
        output << ' ';
        writeStateNumber(output, propertyOf(learning, "epsilon"));
        output << ' '
               << (stateBool(propertyOf(
                       learning,
                       "binarize_observation")) ? 1 : 0)
               << ' ';
        writeStateNumber(
            output,
            propertyOf(learning, "observation_threshold"));
        output << "\nEND_LAYER\n\n";
    }

    return static_cast<bool>(output);
}
} // namespace

LogTracePlotter::PlotCanvas::PlotCanvas(LogTracePlotter& ownerRef)
    : owner(ownerRef)
{
    setSize(900, 520);
}

LogTracePlotter::SeriesRowComponent::SeriesRowComponent(
    LogTracePlotter& ownerRef)
    : owner(ownerRef)
{
    addAndMakeVisible(toggle);
    addAndMakeVisible(disclosure);
    addAndMakeVisible(label);
    label.setInterceptsMouseClicks(false, false);
    label.setColour(juce::Label::textColourId, juce::Colours::white);
    label.setJustificationType(juce::Justification::centredLeft);
    disclosure.setColour(juce::TextButton::buttonColourId,
                         juce::Colours::transparentBlack);
    disclosure.setColour(juce::TextButton::buttonOnColourId,
                         juce::Colours::transparentBlack);
}

void LogTracePlotter::SeriesRowComponent::configureGroup(
    int rowIndex,
    const juce::String& text,
    bool checked,
    bool expanded)
{
    row = rowIndex;
    group = true;
    disclosure.setVisible(true);
    disclosure.setButtonText(expanded ? "-" : "+");
    toggle.setButtonText({});
    toggle.setToggleState(checked, juce::dontSendNotification);
    label.setText(text, juce::dontSendNotification);
    label.setFont(juce::Font(15.0f, juce::Font::bold));
}

void LogTracePlotter::SeriesRowComponent::configureSeries(
    int rowIndex,
    const juce::String& text,
    bool checked)
{
    row = rowIndex;
    group = false;
    disclosure.setVisible(false);
    disclosure.setButtonText({});
    toggle.setButtonText({});
    toggle.setToggleState(checked, juce::dontSendNotification);
    label.setText("  " + text, juce::dontSendNotification);
    label.setFont(juce::Font(14.0f));
}

void LogTracePlotter::SeriesRowComponent::resized()
{
    auto area = getLocalBounds().reduced(2, 1);
    disclosure.setBounds(area.removeFromLeft(24));
    area.removeFromLeft(2);
    toggle.setBounds(area.removeFromLeft(28));
    area.removeFromLeft(4);
    label.setBounds(area);
}

LogTracePlotter::LogTracePlotter()
    : canvas(*this)
{
    setSize(1180, 760);

    loadButton.addListener(this);
    addAndMakeVisible(loadButton);

    showValuesButton.setToggleState(false, juce::dontSendNotification);
    showValuesButton.addListener(this);
    addAndMakeVisible(showValuesButton);

    timeZoomOutButton.addListener(this);
    addAndMakeVisible(timeZoomOutButton);

    timeZoomInButton.addListener(this);
    addAndMakeVisible(timeZoomInButton);

    cellZoomOutButton.addListener(this);
    addAndMakeVisible(cellZoomOutButton);

    cellZoomInButton.addListener(this);
    addAndMakeVisible(cellZoomInButton);

    exportButton.addListener(this);
    addAndMakeVisible(exportButton);

    saveStartStateButton.addListener(this);
    addAndMakeVisible(saveStartStateButton);

    fileLabel.setText("No log loaded", juce::dontSendNotification);
    fileLabel.setJustificationType(juce::Justification::centredLeft);
    fileLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(fileLabel);

    seriesList.setRowHeight(25);
    seriesList.setColour(juce::ListBox::backgroundColourId,
                         juce::Colour(0xff1e2b30));
    seriesList.setColour(juce::ListBox::outlineColourId,
                         juce::Colours::transparentBlack);
    addAndMakeVisible(seriesList);

    viewport.setViewedComponent(&canvas, false);
    viewport.setScrollBarsShown(true, true);
    addAndMakeVisible(viewport);
}

LogTracePlotter::~LogTracePlotter()
{
    loadButton.removeListener(this);
    showValuesButton.removeListener(this);
    timeZoomOutButton.removeListener(this);
    timeZoomInButton.removeListener(this);
    cellZoomOutButton.removeListener(this);
    cellZoomInButton.removeListener(this);
    exportButton.removeListener(this);
    saveStartStateButton.removeListener(this);
    viewport.setViewedComponent(nullptr, false);
}

void LogTracePlotter::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
}

void LogTracePlotter::resized()
{
    auto area = getLocalBounds().reduced(10);
    auto top = area.removeFromTop(34);

    loadButton.setBounds(top.removeFromLeft(110));
    top.removeFromLeft(8);
    showValuesButton.setBounds(top.removeFromLeft(84));
    top.removeFromLeft(8);
    timeZoomOutButton.setBounds(top.removeFromLeft(64));
    top.removeFromLeft(4);
    timeZoomInButton.setBounds(top.removeFromLeft(64));
    top.removeFromLeft(8);
    cellZoomOutButton.setBounds(top.removeFromLeft(66));
    top.removeFromLeft(4);
    cellZoomInButton.setBounds(top.removeFromLeft(66));
    top.removeFromLeft(8);
    exportButton.setBounds(top.removeFromLeft(132));
    top.removeFromLeft(8);
    saveStartStateButton.setBounds(top.removeFromLeft(164));
    top.removeFromLeft(12);
    fileLabel.setBounds(top);

    area.removeFromTop(10);

    auto left = area.removeFromLeft(leftPanelWidth);
    seriesList.setBounds(left);

    area.removeFromLeft(10);
    viewport.setBounds(area);

    refreshCanvasSize();
}

void LogTracePlotter::buttonClicked(juce::Button* button)
{
    if (button == &loadButton)
    {
        loadLogFile();
        return;
    }

    if (button == &showValuesButton)
    {
        canvas.repaint();
        return;
    }

    if (button == &timeZoomOutButton)
    {
        adjustTimeZoom(-1);
        return;
    }

    if (button == &timeZoomInButton)
    {
        adjustTimeZoom(1);
        return;
    }

    if (button == &cellZoomOutButton)
    {
        adjustCellZoom(-1);
        return;
    }

    if (button == &cellZoomInButton)
    {
        adjustCellZoom(1);
        return;
    }

    if (button == &exportButton)
    {
        exportCheckedFields();
        return;
    }

    if (button == &saveStartStateButton)
        saveStartingState();
}

void LogTracePlotter::loadLogFile()
{
    fileChooser = std::make_unique<juce::FileChooser>(
        "Choose a JSONL log file",
        juce::File{},
        "*.jsonl;*.log;*.txt");

    juce::Component::SafePointer<LogTracePlotter> safeThis(this);

    fileChooser->launchAsync(
        juce::FileBrowserComponent::openMode
            | juce::FileBrowserComponent::canSelectFiles,
        [safeThis](const juce::FileChooser& chooser)
        {
            if (safeThis == nullptr)
                return;

            const auto file = chooser.getResult();
            if (file.existsAsFile())
                safeThis->parseJsonLog(file);
        });
}

void LogTracePlotter::exportCheckedFields()
{
    if (!currentFile.existsAsFile())
    {
        fileLabel.setText(
            "Load a log before exporting checked fields",
            juce::dontSendNotification);
        return;
    }

    if (enabledSeriesCount() == 0)
    {
        fileLabel.setText(
            "Check at least one field before exporting",
            juce::dontSendNotification);
        return;
    }

    const auto defaultFile =
        currentFile.getSiblingFile(
            currentFile.getFileNameWithoutExtension()
                + "_checked.jsonl");

    fileChooser = std::make_unique<juce::FileChooser>(
        "Export checked trace fields",
        defaultFile,
        "*.jsonl;*.txt");

    juce::Component::SafePointer<LogTracePlotter> safeThis(this);

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

            if (!file.hasFileExtension("jsonl") &&
                !file.hasFileExtension("txt"))
                file = file.withFileExtension("jsonl");

            safeThis->writeCheckedFieldExport(file);
        });
}

bool LogTracePlotter::writeCheckedFieldExport(const juce::File& file)
{
    std::ofstream output(
        file.getFullPathName().toStdString(),
        std::ios::out | std::ios::trunc);

    if (!output)
    {
        fileLabel.setText(
            "Could not write export: " + file.getFileName(),
            juce::dontSendNotification);
        return false;
    }

    output << "{\"type\":\"trace_export\",";
    output << "\"source\":";
    writeExportJsonString(output, currentFile.getFullPathName());
    output << ",\"fields\":[";

    bool firstField = true;
    for (const auto& spec : series)
    {
        if (!spec.enabled)
            continue;

        if (!firstField)
            output << ',';
        firstField = false;
        writeExportJsonString(output, spec.path);
    }
    output << "]}\n";

    juce::StringArray lines;
    lines.addLines(currentFile.loadFileAsString());

    int stepIndex = 0;
    for (const auto& line : lines)
    {
        const auto trimmed = line.trim();
        if (trimmed.isEmpty())
            continue;

        const auto parsed = juce::JSON::parse(trimmed);
        const auto* object = parsed.getDynamicObject();

        if (object == nullptr ||
            object->getProperty("type").toString() != "step")
        {
            output << trimmed << '\n';
            continue;
        }

        if (stepIndex >= static_cast<int>(timeSlices.size()))
            break;

        const auto& slice = timeSlices[static_cast<std::size_t>(stepIndex)];
        output << "{\"type\":\"step\",\"step\":";
        output << (slice.step.isEmpty() ? juce::String("null") : slice.step);
        output << ",\"input_row\":";
        output << (slice.inputRow.isEmpty()
            ? juce::String("null")
            : slice.inputRow);

        for (const auto& spec : series)
        {
            if (!spec.enabled)
                continue;

            output << ',';
            writeExportJsonString(output, spec.path);
            output << ':';

            if (stepIndex < static_cast<int>(spec.values.size()))
                writeExportJsonValues(
                    output,
                    spec.values[static_cast<std::size_t>(stepIndex)]);
            else
                output << "null";
        }

        output << "}\n";
        ++stepIndex;
    }

    if (!output)
    {
        fileLabel.setText(
            "Export failed while writing " + file.getFileName(),
            juce::dontSendNotification);
        return false;
    }

    fileLabel.setText(
        "Exported checked fields to " + file.getFileName(),
        juce::dontSendNotification);
    return true;
}

void LogTracePlotter::saveStartingState()
{
    if (!currentFile.existsAsFile())
    {
        fileLabel.setText(
            "Load a log before saving the starting state",
            juce::dontSendNotification);
        return;
    }

    juce::var state;
    if (!findInitialState(state))
    {
        fileLabel.setText(
            "No initial state snapshot found in this log",
            juce::dontSendNotification);
        return;
    }

    const auto defaultFile =
        currentFile.getSiblingFile(
            currentFile.getFileNameWithoutExtension()
                + "_starting.state");

    fileChooser = std::make_unique<juce::FileChooser>(
        "Save starting Noisy-OR state",
        defaultFile,
        "*.state;*.txt");

    juce::Component::SafePointer<LogTracePlotter> safeThis(this);

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

            if (!file.hasFileExtension("state") &&
                !file.hasFileExtension("txt"))
                file = file.withFileExtension("state");

            safeThis->writeStartingStateFile(file);
        });
}

bool LogTracePlotter::writeStartingStateFile(const juce::File& file)
{
    juce::var state;
    if (!findInitialState(state))
    {
        fileLabel.setText(
            "No initial state snapshot found in this log",
            juce::dontSendNotification);
        return false;
    }

    std::ofstream output(
        file.getFullPathName().toStdString(),
        std::ios::out | std::ios::trunc);

    if (!output)
    {
        fileLabel.setText(
            "Could not write state: " + file.getFileName(),
            juce::dontSendNotification);
        return false;
    }

    if (!writeStateFileFromJson(output, state))
    {
        fileLabel.setText(
            "Could not convert initial state snapshot",
            juce::dontSendNotification);
        return false;
    }

    fileLabel.setText(
        "Saved starting state to " + file.getFileName(),
        juce::dontSendNotification);
    return true;
}

bool LogTracePlotter::findInitialState(juce::var& stateOut) const
{
    if (!currentFile.existsAsFile())
        return false;

    juce::StringArray lines;
    lines.addLines(currentFile.loadFileAsString());

    for (const auto& line : lines)
    {
        const auto trimmed = line.trim();
        if (trimmed.isEmpty())
            continue;

        const auto parsed = juce::JSON::parse(trimmed);
        const auto* object = parsed.getDynamicObject();
        if (object == nullptr)
            continue;

        if (object->getProperty("type").toString() != "state" ||
            object->getProperty("role").toString() != "initial")
            continue;

        stateOut = object->getProperty("state");
        return !stateOut.isVoid();
    }

    return false;
}

bool LogTracePlotter::loadFile(const juce::File& file)
{
    if (!file.existsAsFile())
    {
        fileLabel.setText(
            "Log file not found: " + file.getFullPathName(),
            juce::dontSendNotification);
        return false;
    }

    return parseJsonLog(file);
}

void LogTracePlotter::clearData()
{
    series.clear();
    groups.clear();
    visibleRows.clear();
    timeSlices.clear();
    currentFile = juce::File{};
    fileLabel.setText("No log loaded", juce::dontSendNotification);
    seriesList.updateContent();
    refreshCanvasSize();
    canvas.repaint();
}

bool LogTracePlotter::parseJsonLog(const juce::File& file)
{
    clearData();
    currentFile = file;
    fileLabel.setText(file.getFullPathName(), juce::dontSendNotification);

    juce::StringArray lines;
    lines.addLines(file.loadFileAsString());

    for (const auto& line : lines)
    {
        const auto trimmed = line.trim();
        if (trimmed.isEmpty())
            continue;

        const auto parsed = juce::JSON::parse(trimmed);
        const auto* object = parsed.getDynamicObject();
        if (object == nullptr)
            continue;

        if (object->getProperty("type").toString() == "step")
            ingestStepObject(parsed);
    }

    recomputeSeriesRanges();
    rebuildGroups();
    seriesList.updateContent();
    refreshCanvasSize();
    canvas.repaint();
    return true;
}

void LogTracePlotter::collectNumericSeries(
    const juce::var& node,
    const juce::String& pathPrefix,
    RowSeries& rowSeries,
    juce::StringArray& discoveryOrder)
{
    if (pathPrefix.isNotEmpty() && isNumericVar(node))
    {
        rowSeries[pathPrefix.toStdString()] = { numericValue(node) };
        if (!discoveryOrder.contains(pathPrefix))
            discoveryOrder.add(pathPrefix);
        return;
    }

    if (pathPrefix.isNotEmpty() && isNumericArray(node))
    {
        rowSeries[pathPrefix.toStdString()] = numericVectorFromArray(node);
        if (!discoveryOrder.contains(pathPrefix))
            discoveryOrder.add(pathPrefix);
        return;
    }

    if (pathPrefix.isNotEmpty() && isNumericMatrix(node))
    {
        if (shouldFlattenMatrixAtPath(pathPrefix))
        {
            rowSeries[pathPrefix.toStdString()] =
                numericVectorFromMatrix(node);
            if (!discoveryOrder.contains(pathPrefix))
                discoveryOrder.add(pathPrefix);
            return;
        }

        const auto* array = node.getArray();
        for (int row = 0; row < array->size(); ++row)
        {
            const auto childPath = indexedPath(pathPrefix, row);
            rowSeries[childPath.toStdString()] =
                numericVectorFromArray(array->getReference(row));
            if (!discoveryOrder.contains(childPath))
                discoveryOrder.add(childPath);
        }
        return;
    }

    if (const auto* object = node.getDynamicObject())
    {
        const auto& properties = object->getProperties();
        for (int index = 0; index < properties.size(); ++index)
        {
            const auto name = properties.getName(index).toString();
            const auto childPath = pathPrefix.isEmpty()
                ? name
                : pathPrefix + "." + name;

            collectNumericSeries(
                properties.getValueAt(index),
                childPath,
                rowSeries,
                discoveryOrder);
        }
        return;
    }

    if (const auto* array = node.getArray())
    {
        for (int index = 0; index < array->size(); ++index)
        {
            collectNumericSeries(
                array->getReference(index),
                indexedPath(pathPrefix, index),
                rowSeries,
                discoveryOrder);
        }
    }
}

void LogTracePlotter::ingestStepObject(const juce::var& stepObject)
{
    RowSeries rowSeries;
    juce::StringArray discoveryOrder;
    collectNumericSeries(stepObject, {}, rowSeries, discoveryOrder);

    const int rowIndex = static_cast<int>(timeSlices.size());

    TimeSlice slice;
    if (const auto* object = stepObject.getDynamicObject())
    {
        slice.step = object->getProperty("step").toString();
        slice.inputRow = object->getProperty("input_row").toString();
    }
    timeSlices.push_back(slice);

    for (const auto& path : discoveryOrder)
    {
        if (path == "step" || path == "input_row")
            continue;

        const auto found = rowSeries.find(path.toStdString());
        if (found != rowSeries.end())
            addSeriesIfNeeded(path, static_cast<int>(found->second.size()));
    }

    for (auto& spec : series)
    {
        auto values = std::vector<double>(
            static_cast<std::size_t>(spec.vectorLength), 0.0);
        const auto found = rowSeries.find(spec.path.toStdString());
        if (found != rowSeries.end())
        {
            values = found->second;
            values.resize(static_cast<std::size_t>(spec.vectorLength), 0.0);
        }

        while (static_cast<int>(spec.values.size()) < rowIndex)
        {
            spec.values.push_back(std::vector<double>(
                static_cast<std::size_t>(spec.vectorLength), 0.0));
        }
        spec.values.push_back(std::move(values));
    }
}

void LogTracePlotter::addSeriesIfNeeded(
    const juce::String& path,
    int vectorLength)
{
    if (findSeriesIndex(path) >= 0)
        return;

    SeriesSpec spec;
    spec.path = path;
    spec.displayName = prettyNameForPath(path);
    spec.vectorLength = std::max(1, vectorLength);
    spec.enabled = shouldEnableByDefault(path);
    spec.probabilityLike = isProbabilityLikePath(path);
    spec.values.resize(timeSlices.empty() ? 0 : timeSlices.size() - 1);
    for (auto& row : spec.values)
        row.resize(static_cast<std::size_t>(spec.vectorLength), 0.0);

    series.push_back(std::move(spec));
}

int LogTracePlotter::findSeriesIndex(const juce::String& path) const
{
    for (int index = 0; index < static_cast<int>(series.size()); ++index)
        if (series[static_cast<std::size_t>(index)].path == path)
            return index;

    return -1;
}

void LogTracePlotter::rebuildGroups()
{
    groups.clear();

    for (int seriesIndex = 0;
         seriesIndex < static_cast<int>(series.size());
         ++seriesIndex)
    {
        const auto key = groupKeyForPath(
            series[static_cast<std::size_t>(seriesIndex)].path);

        auto found = std::find_if(
            groups.begin(),
            groups.end(),
            [&key](const SeriesGroup& group)
            {
                return group.key == key;
            });

        if (found == groups.end())
        {
            SeriesGroup group;
            group.key = key;
            group.displayName = groupNameForKey(key);
            group.expanded = !key.contains("R_delta")
                && !key.contains("F_delta");
            group.seriesIndices.push_back(seriesIndex);
            groups.push_back(std::move(group));
        }
        else
        {
            found->seriesIndices.push_back(seriesIndex);
        }
    }

    rebuildVisibleRows();
}

void LogTracePlotter::rebuildVisibleRows()
{
    visibleRows.clear();

    for (int groupIndex = 0;
         groupIndex < static_cast<int>(groups.size());
         ++groupIndex)
    {
        visibleRows.push_back({ true, groupIndex, -1 });

        const auto& group = groups[static_cast<std::size_t>(groupIndex)];
        if (!group.expanded)
            continue;

        for (const auto seriesIndex : group.seriesIndices)
            visibleRows.push_back({ false, groupIndex, seriesIndex });
    }
}

void LogTracePlotter::recomputeSeriesRanges()
{
    for (auto& spec : series)
    {
        if (spec.probabilityLike)
        {
            spec.minimum = 0.0;
            spec.maximum = 1.0;
            continue;
        }

        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();

        for (const auto& row : spec.values)
        {
            for (const auto value : row)
            {
                if (!std::isfinite(value))
                    continue;

                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
        }

        if (!std::isfinite(minimum) || !std::isfinite(maximum)
            || std::abs(maximum - minimum) < 1.0e-12)
        {
            minimum = 0.0;
            maximum = std::max(1.0, std::abs(maximum));
        }

        spec.minimum = minimum;
        spec.maximum = maximum;
    }
}

void LogTracePlotter::refreshCanvasSize()
{
    const auto size = canvas.preferredSize();
    canvas.setSize(size.first, size.second);
}

bool LogTracePlotter::shouldEnableByDefault(const juce::String& path)
{
    return path == "x"
        || path.endsWith(".mu")
        || path.endsWith(".alpha_next")
        || path.endsWith(".reconstruction")
        || path.endsWith(".log_evidence")
        || path.endsWith(".candidate_count");
}

bool LogTracePlotter::isProbabilityLikePath(const juce::String& path)
{
    return path == "x"
        || path.contains("alpha")
        || path.contains(".R[")
        || path.contains(".F[")
        || path.endsWith(".mu")
        || path.contains("top_down_prior")
        || path.endsWith(".qF")
        || path.endsWith(".reconstruction");
}

juce::String LogTracePlotter::groupKeyForPath(const juce::String& path)
{
    if (!path.startsWith("layers["))
        return "root";

    const auto close = path.indexOfChar(']');
    if (close < 0)
        return "layers";

    const auto layerKey = path.substring(0, close + 1);
    const auto afterLayer = path.substring(close + 2);

    if (afterLayer.startsWith("learning.R_delta"))
        return layerKey + ".learning.R_delta";

    if (afterLayer.startsWith("learning.F_delta"))
        return layerKey + ".learning.F_delta";

    if (afterLayer.startsWith("learning."))
        return layerKey + ".learning";

    if (afterLayer.startsWith("R"))
        return layerKey + ".R";

    if (afterLayer.startsWith("F"))
        return layerKey + ".F";

    return layerKey;
}

juce::String LogTracePlotter::groupNameForKey(const juce::String& key)
{
    if (key == "root")
        return "run";

    return prettyNameForPath(key);
}

juce::String LogTracePlotter::prettyNameForPath(const juce::String& path)
{
    if (path == "step")
        return "step (axis)";

    if (path == "input_row")
        return "input row (axis)";

    return path
        .replace("layers[", "L")
        .replace("]", "")
        .replace(".", " / ");
}

juce::String LogTracePlotter::compactNumber(double value)
{
    if (!std::isfinite(value))
        return "nan";

    if (std::abs(value) > 0.0 && std::abs(value) < 0.001)
        return juce::String(value, 2, true);

    return juce::String(value, 3);
}

bool LogTracePlotter::isGroupFullyEnabled(int groupIndex) const
{
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groups.size()))
        return false;

    const auto& group = groups[static_cast<std::size_t>(groupIndex)];
    if (group.seriesIndices.empty())
        return false;

    for (const auto seriesIndex : group.seriesIndices)
    {
        if (seriesIndex < 0 || seriesIndex >= static_cast<int>(series.size()))
            continue;

        if (!series[static_cast<std::size_t>(seriesIndex)].enabled)
            return false;
    }

    return true;
}

void LogTracePlotter::setGroupEnabled(int groupIndex, bool enabled)
{
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groups.size()))
        return;

    for (const auto seriesIndex :
         groups[static_cast<std::size_t>(groupIndex)].seriesIndices)
    {
        if (seriesIndex >= 0 && seriesIndex < static_cast<int>(series.size()))
            series[static_cast<std::size_t>(seriesIndex)].enabled = enabled;
    }

    seriesList.updateContent();
    refreshCanvasSize();
    canvas.repaint();
}

void LogTracePlotter::toggleGroupExpanded(int groupIndex)
{
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groups.size()))
        return;

    auto& group = groups[static_cast<std::size_t>(groupIndex)];
    group.expanded = !group.expanded;
    rebuildVisibleRows();
    seriesList.updateContent();
}

void LogTracePlotter::adjustTimeZoom(int delta)
{
    columnWidth = juce::jlimit(3, 64, columnWidth + delta * 3);
    refreshCanvasSize();
    canvas.repaint();
}

void LogTracePlotter::adjustCellZoom(int delta)
{
    cellSize = juce::jlimit(3, 24, cellSize + delta * 2);
    cellGap = cellSize <= 5 ? 0 : 1;
    seriesGap = juce::jlimit(4, 16, cellSize);
    refreshCanvasSize();
    canvas.repaint();
}

double LogTracePlotter::normalizedValue(
    const SeriesSpec& spec,
    double value) const
{
    if (!std::isfinite(value))
        return 0.0;

    const auto denominator = spec.maximum - spec.minimum;
    if (std::abs(denominator) < 1.0e-12)
        return 0.0;

    return juce::jlimit(0.0, 1.0, (value - spec.minimum) / denominator);
}

juce::Colour LogTracePlotter::colourForValue(
    const SeriesSpec& spec,
    double value) const
{
    if (!std::isfinite(value))
        return juce::Colour(0xff7a1d1d);

    const auto normalized = normalizedValue(spec, value);
    const auto channel = static_cast<juce::uint8>(
        juce::roundToInt(normalized * 235.0 + 16.0));

    return juce::Colour(channel, channel, channel);
}

int LogTracePlotter::enabledSeriesCount() const
{
    int count = 0;
    for (const auto& spec : series)
        if (spec.enabled)
            ++count;
    return count;
}

int LogTracePlotter::plotHeight() const
{
    int height = headerHeight + 8;
    for (const auto& spec : series)
    {
        if (!spec.enabled)
            continue;

        height += 17
            + spec.vectorLength * (cellSize + cellGap)
            + seriesGap;
    }

    return std::max(260, height + 16);
}

int LogTracePlotter::plotWidth() const
{
    return std::max(
        520,
        labelGutterWidth + 16
            + static_cast<int>(timeSlices.size()) * columnWidth + 24);
}

int LogTracePlotter::getNumRows()
{
    return static_cast<int>(visibleRows.size());
}

void LogTracePlotter::paintListBoxItem(
    int,
    juce::Graphics& g,
    int width,
    int height,
    bool rowSelected)
{
    if (rowSelected)
    {
        g.setColour(juce::Colours::white.withAlpha(0.08f));
        g.fillRect(0, 0, width, height);
    }
}

juce::Component* LogTracePlotter::refreshComponentForRow(
    int row,
    bool,
    juce::Component* existingComponentToUpdate)
{
    auto* rowComponent =
        dynamic_cast<SeriesRowComponent*>(existingComponentToUpdate);

    if (rowComponent == nullptr)
        rowComponent = new SeriesRowComponent(*this);

    if (row >= 0 && row < static_cast<int>(visibleRows.size()))
    {
        const auto visible = visibleRows[static_cast<std::size_t>(row)];

        if (visible.isGroup)
        {
            const auto& group = groups[static_cast<std::size_t>(
                visible.groupIndex)];
            const auto checked = isGroupFullyEnabled(visible.groupIndex);
            int enabledCount = 0;
            for (const auto seriesIndex : group.seriesIndices)
            {
                if (seriesIndex >= 0
                    && seriesIndex < static_cast<int>(series.size())
                    && series[static_cast<std::size_t>(seriesIndex)].enabled)
                {
                    ++enabledCount;
                }
            }
            rowComponent->configureGroup(
                row,
                group.displayName + "  (" + juce::String(enabledCount)
                    + "/" + juce::String(static_cast<int>(
                        group.seriesIndices.size())) + ")",
                checked,
                group.expanded);

            rowComponent->disclosure.onClick = [this, visible]
            {
                toggleGroupExpanded(visible.groupIndex);
            };

            rowComponent->toggle.onClick = [this, visible]
            {
                setGroupEnabled(
                    visible.groupIndex,
                    !isGroupFullyEnabled(visible.groupIndex));
            };
        }
        else if (visible.seriesIndex >= 0
            && visible.seriesIndex < static_cast<int>(series.size()))
        {
            auto& spec = series[static_cast<std::size_t>(
                visible.seriesIndex)];
            rowComponent->configureSeries(row, spec.displayName, spec.enabled);
            rowComponent->disclosure.onClick = nullptr;
            rowComponent->toggle.onClick = [this, visible, rowComponent]
            {
                const auto seriesIndex = visible.seriesIndex;
                if (seriesIndex < 0
                    || seriesIndex >= static_cast<int>(series.size()))
                    return;

                series[static_cast<std::size_t>(seriesIndex)].enabled =
                    rowComponent->toggle.getToggleState();
                seriesList.updateContent();
                refreshCanvasSize();
                canvas.repaint();
            };
        }
    }

    return rowComponent;
}

std::pair<int, int> LogTracePlotter::PlotCanvas::preferredSize() const
{
    return { owner.plotWidth(), owner.plotHeight() };
}

void LogTracePlotter::PlotCanvas::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setFont(juce::Font(11.0f));

    const auto clip = g.getClipBounds();
    const int plotLeft = owner.labelGutterWidth;
    const int plotTop = owner.headerHeight;
    const int timeCount = static_cast<int>(owner.timeSlices.size());
    const int visibleFirstTime = timeCount <= 0
        ? 0
        : juce::jlimit(
            0,
            timeCount - 1,
            static_cast<int>(std::floor(
                static_cast<double>(clip.getX() - plotLeft)
                    / static_cast<double>(owner.columnWidth))));
    const int visibleLastTime = timeCount <= 0
        ? -1
        : juce::jlimit(
            0,
            timeCount - 1,
            static_cast<int>(std::ceil(
                static_cast<double>(clip.getRight() - plotLeft)
                    / static_cast<double>(owner.columnWidth))) - 1);
    const int labelStride = owner.columnWidth >= 18
        ? 1
        : std::max(1, static_cast<int>(
            std::ceil(42.0 / static_cast<double>(owner.columnWidth))));

    g.setColour(juce::Colours::white.withAlpha(0.75f));
    g.drawText("step", 8, 6, owner.labelGutterWidth - 16, 14,
               juce::Justification::centredRight);
    g.drawText("input row", 8, 24, owner.labelGutterWidth - 16, 14,
               juce::Justification::centredRight);

    for (int t = visibleFirstTime; t <= visibleLastTime; ++t)
    {
        const int x = plotLeft + t * owner.columnWidth;
        if ((t % 2) == 0)
        {
            g.setColour(juce::Colours::white.withAlpha(0.025f));
            g.fillRect(x - 2, plotTop, owner.columnWidth, getHeight() - plotTop);
        }

        if (owner.columnWidth < 8 || (t % labelStride) != 0)
            continue;

        const auto& slice = owner.timeSlices[static_cast<std::size_t>(t)];
        const bool showStep = t == 0
            || slice.step != owner.timeSlices[static_cast<std::size_t>(t - 1)].step
            || owner.columnWidth >= 28;

        g.setColour(juce::Colours::white.withAlpha(0.82f));
        if (showStep && owner.columnWidth >= 12)
            g.drawText(slice.step, x - 4, 6, owner.columnWidth + 8, 14,
                       juce::Justification::centred);

        g.setColour(juce::Colours::white.withAlpha(0.55f));
        if (owner.columnWidth >= 10)
            g.drawText(slice.inputRow, x - 4, 24, owner.columnWidth + 8, 14,
                       juce::Justification::centred);
    }

    g.setColour(juce::Colours::white.withAlpha(0.14f));
    g.drawHorizontalLine(plotTop - 4, 0.0f, static_cast<float>(getWidth()));

    int y = plotTop + 8;
    int enabledIndex = 0;
    const int totalEnabledSeries = owner.enabledSeriesCount();
    const int rowPitch = owner.cellSize + owner.cellGap;
    const int cellWidth = std::min(
        owner.cellSize,
        std::max(1, owner.columnWidth - owner.cellGap));
    const bool drawCellBorders =
        owner.cellSize >= 6 && cellWidth >= 6 && owner.columnWidth >= 7;
    const bool drawCellValues =
        owner.showValuesButton.getToggleState()
        && owner.cellSize >= 13
        && owner.columnWidth >= 18;

    for (const auto& spec : owner.series)
    {
        if (!spec.enabled)
            continue;

        const int seriesHeight =
            spec.vectorLength * rowPitch;
        const int labelTop = y;
        const int dataTop = y + 17;
        const auto labelBounds = juce::Rectangle<int>(
            0, labelTop, owner.labelGutterWidth, 30);
        const auto dataBounds = juce::Rectangle<int>(
            plotLeft, dataTop,
            std::max(0, getWidth() - plotLeft),
            seriesHeight);
        const bool labelsVisible = clip.intersects(labelBounds);
        const bool dataVisible = clip.intersects(dataBounds);

        if (labelsVisible)
        {
            g.setColour(juce::Colours::white.withAlpha(0.86f));
            g.drawText(spec.displayName, 8, y, owner.labelGutterWidth - 16, 14,
                       juce::Justification::centredRight, true);

            if (!spec.probabilityLike)
            {
                const auto range = owner.compactNumber(spec.minimum)
                    + " .. " + owner.compactNumber(spec.maximum);
                g.setColour(juce::Colours::white.withAlpha(0.38f));
                g.drawText(range, 8, y + 13, owner.labelGutterWidth - 16, 13,
                           juce::Justification::centredRight, true);
            }
        }

        y += 17;

        if (dataVisible && visibleLastTime >= visibleFirstTime)
        {
            const int firstRow = juce::jlimit(
                0,
                std::max(0, spec.vectorLength - 1),
                static_cast<int>(std::floor(
                    static_cast<double>(clip.getY() - y)
                        / static_cast<double>(rowPitch))));
            const int lastRow = juce::jlimit(
                0,
                std::max(0, spec.vectorLength - 1),
                static_cast<int>(std::ceil(
                    static_cast<double>(clip.getBottom() - y)
                        / static_cast<double>(rowPitch))) - 1);
            const int lastTime = std::min(
                visibleLastTime,
                static_cast<int>(spec.values.size()) - 1);

            for (int t = visibleFirstTime; t <= lastTime; ++t)
            {
                const int x = plotLeft + t * owner.columnWidth;
                const auto& values = spec.values[static_cast<std::size_t>(t)];
                const int rowLimit = std::min(
                    lastRow,
                    static_cast<int>(values.size()) - 1);

                for (int row = firstRow; row <= rowLimit; ++row)
                {
                    const auto value = values[static_cast<std::size_t>(row)];
                    const auto cell = juce::Rectangle<int>(
                        x,
                        y + row * rowPitch,
                        cellWidth,
                        owner.cellSize);

                    const auto colour = owner.colourForValue(spec, value);
                    g.setColour(colour);
                    g.fillRect(cell);

                    if (drawCellBorders)
                    {
                        g.setColour(juce::Colours::white.withAlpha(0.22f));
                        g.drawRect(cell);
                    }

                    if (drawCellValues && spec.vectorLength <= 4)
                    {
                        const auto normalized = owner.normalizedValue(spec, value);
                        g.setColour(normalized > 0.55
                            ? juce::Colours::black
                            : juce::Colours::white);
                        g.setFont(juce::Font(8.0f));
                        g.drawText(owner.compactNumber(value), cell,
                                   juce::Justification::centred, false);
                        g.setFont(juce::Font(11.0f));
                    }
                }
            }
        }

        y += seriesHeight + owner.seriesGap;
        ++enabledIndex;

        if (enabledIndex < totalEnabledSeries
            && clip.intersects(juce::Rectangle<int>(
                0,
                y - owner.seriesGap / 2 - 1,
                getWidth(),
                3)))
        {
            g.setColour(juce::Colours::white.withAlpha(0.08f));
            g.drawHorizontalLine(y - owner.seriesGap / 2,
                                 0.0f,
                                 static_cast<float>(getWidth()));
        }
    }

    if (timeCount == 0)
    {
        g.setColour(juce::Colours::white.withAlpha(0.65f));
        g.drawText("Load a JSONL log to inspect a run",
                   getLocalBounds(),
                   juce::Justification::centred);
    }
}
