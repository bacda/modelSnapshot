#include "LayerStateDisplay.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

namespace {

constexpr float margin = 12.0f;
constexpr float sectionGap = 18.0f;
constexpr float titleHeight = 22.0f;
constexpr float lineHeight = 17.0f;
constexpr float parameterHeaderHeight = 12.0f;
constexpr float maximumInputCell = 16.0f;
constexpr float maximumParameterCell = 18.0f;
constexpr float maximumProbabilityCell = 12.0f;
constexpr float maximumDeltaCell = 14.0f;
constexpr float maximumParameterSliderWidth = 50.0f;
constexpr float parameterSliderGap = 8.0f;
constexpr float compactMatrixGap = 14.0f;
constexpr float settingSliderWidth = 72.0f;
constexpr float randomButtonWidth = 34.0f;
constexpr float randomSettingsButtonWidth = 62.0f;
constexpr float randomControlGap = 5.0f;
constexpr float layerPanelWidth = 300.0f;
constexpr float dragPixelsForFullScale = 120.0f;
constexpr float amplitudeDragUnitsForFullScale = 5.0f;
constexpr Eigen::Index maximumDisplayedCandidates = 64;

juce::Colour panelColour()
{
    return juce::Colour::fromRGB(230, 234, 238);
}

juce::Colour primaryTextColour()
{
    return juce::Colour::fromRGB(20, 24, 28);
}

juce::Colour secondaryTextColour()
{
    return juce::Colour::fromRGB(82, 90, 98);
}

double meanAbsolute(const Eigen::MatrixXd& matrix)
{
    return matrix.size() == 0 ? 0.0 : matrix.cwiseAbs().mean();
}

double maxAbsolute(const Eigen::MatrixXd& matrix)
{
    return matrix.size() == 0 ? 0.0 : matrix.cwiseAbs().maxCoeff();
}

double meanAbsolute(const std::vector<Eigen::MatrixXd>& matrices)
{
    double sum = 0.0;
    Eigen::Index count = 0;
    for (const auto& matrix : matrices)
    {
        sum += matrix.cwiseAbs().sum();
        count += matrix.size();
    }
    return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

bool hasVisibleSupport(const Eigen::VectorXd& vector)
{
    return vector.size() > 0 && vector.cwiseAbs().maxCoeff() > 1.0e-12;
}

bool intersectsClip(juce::Graphics& g, juce::Rectangle<float> bounds)
{
    return g.getClipBounds().toFloat().intersects(bounds);
}

float matrixHeightForShape(
    Eigen::Index rows,
    Eigen::Index columns,
    float availableWidth,
    float maximumCellSize)
{
    if (rows <= 0 || columns <= 0)
        return 0.0f;

    const float cell = std::min(
        maximumCellSize,
        availableWidth / static_cast<float>(columns));
    return cell * static_cast<float>(rows);
}

float rowMatrixHeight(
    Eigen::Index columns,
    float availableWidth,
    float maximumCellSize)
{
    return matrixHeightForShape(1, columns, availableWidth, maximumCellSize);
}

std::pair<Eigen::Index, Eigen::Index> visibleCellRange(
    float clipStart,
    float clipEnd,
    float boundsStart,
    float cellSize,
    Eigen::Index count)
{
    if (count <= 0 || cellSize <= 0.0f)
        return { 0, -1 };

    const auto first = static_cast<Eigen::Index>(
        std::floor((clipStart - boundsStart) / cellSize));
    const auto last = static_cast<Eigen::Index>(
        std::ceil((clipEnd - boundsStart) / cellSize)) - 1;

    return {
        juce::jlimit<Eigen::Index>(0, count - 1, first),
        juce::jlimit<Eigen::Index>(0, count - 1, last)
    };
}


float naturalMatrixWidth(const Eigen::MatrixXd& matrix, float maximumCellSize)
{
    return matrix.cols() <= 0
        ? 0.0f
        : maximumCellSize * static_cast<float>(matrix.cols());
}

float naturalMatrixWidth(Eigen::Index columns, float maximumCellSize)
{
    return columns <= 0
        ? 0.0f
        : maximumCellSize * static_cast<float>(columns);
}

std::pair<float, float> compactMatrixWidths(
    const Eigen::MatrixXd& left,
    const Eigen::MatrixXd& right,
    float availableWidth,
    float leftMaximumCellSize,
    float rightMaximumCellSize)
{
    const float leftNatural = naturalMatrixWidth(left, leftMaximumCellSize);
    const float rightNatural = naturalMatrixWidth(right, rightMaximumCellSize);
    const bool hasLeft = leftNatural > 0.0f;
    const bool hasRight = rightNatural > 0.0f;
    const float gap = hasLeft && hasRight ? compactMatrixGap : 0.0f;
    const float totalNatural = leftNatural + rightNatural;

    if (totalNatural <= 0.0f)
        return { 0.0f, 0.0f };

    const float scale = std::min(
        1.0f,
        std::max(1.0f, availableWidth - gap) / totalNatural);

    return { leftNatural * scale, rightNatural * scale };
}

std::pair<float, float> compactMatrixWidths(
    Eigen::Index leftColumns,
    Eigen::Index rightColumns,
    float availableWidth,
    float leftMaximumCellSize,
    float rightMaximumCellSize)
{
    const float leftNatural = naturalMatrixWidth(
        leftColumns, leftMaximumCellSize);
    const float rightNatural = naturalMatrixWidth(
        rightColumns, rightMaximumCellSize);
    const bool hasLeft = leftNatural > 0.0f;
    const bool hasRight = rightNatural > 0.0f;
    const float gap = hasLeft && hasRight ? compactMatrixGap : 0.0f;
    const float totalNatural = leftNatural + rightNatural;

    if (totalNatural <= 0.0f)
        return { 0.0f, 0.0f };

    const float scale = std::min(
        1.0f,
        std::max(1.0f, availableWidth - gap) / totalNatural);

    return { leftNatural * scale, rightNatural * scale };
}

} // namespace

LayerStateDisplay::LayerStateDisplay()
{
    setOpaque(true);
    setBufferedToImage(true);
}

void LayerStateDisplay::update(
    const noisy_or::ModelState& state,
    const noisy_or::NoisyORStack& model,
    std::optional<std::size_t> currentTimestep,
    const std::vector<std::optional<noisy_or::LearningDiagnostics>>& diagnostics)
{
    constexpr std::size_t windowLength = 64;

    inputWindow_.resize(0, 0);
    inputIndices_.clear();
    highlightedInputColumn_.reset();

    const std::size_t timestepCount =
        static_cast<std::size_t>(state.input.rows());
    const Eigen::Index channelCount = state.input.cols();

    if (timestepCount > 0 && channelCount > 0)
    {
        const std::size_t count = std::min(windowLength, timestepCount);
        inputWindow_.resize(channelCount, static_cast<Eigen::Index>(count));
        inputIndices_.resize(count);

        if (currentTimestep.has_value())
        {
            const std::size_t current = *currentTimestep % timestepCount;
            const std::size_t first =
                (current + timestepCount - (count - 1)) % timestepCount;

            for (std::size_t column = 0; column < count; ++column)
            {
                const std::size_t source = (first + column) % timestepCount;
                inputIndices_[column] = source;
                inputWindow_.col(static_cast<Eigen::Index>(column)) =
                    state.input.row(static_cast<Eigen::Index>(source)).transpose();
            }
            highlightedInputColumn_ = count - 1;
        }
        else
        {
            for (std::size_t column = 0; column < count; ++column)
            {
                inputIndices_[column] = column;
                inputWindow_.col(static_cast<Eigen::Index>(column)) =
                    state.input.row(static_cast<Eigen::Index>(column)).transpose();
            }
        }
    }

    layers_.clear();
    layers_.reserve(model.size());
    if (randomControls_.size() < model.size())
        randomControls_.resize(model.size());
    else if (randomControls_.size() > model.size())
        randomControls_.resize(model.size());
    if (sectionStates_.size() < model.size())
        sectionStates_.resize(model.size());
    else if (sectionStates_.size() > model.size())
        sectionStates_.resize(model.size());

    for (std::size_t index = 0; index < model.size(); ++index)
    {
        const auto& layer = model.layer(index);
        const auto& configuration = layer.configuration();
        const auto& output = layer.output();

        LayerSnapshot snapshot;
        snapshot.R = configuration.predictions;
        snapshot.parameters = configuration.parameters;
        snapshot.candidatePolicy = configuration.candidateSelection;
        snapshot.alpha = layer.alpha();

        const Eigen::Index K = configuration.predictions.rows();
        const Eigen::Index N = configuration.predictions.cols();
        const Eigen::Index order = configuration.initialContext.cols();
        snapshot.F.resize(K, N * order);

        for (Eigen::Index k = 0; k < K; ++k)
        {
            const auto& filter =
                configuration.filters[static_cast<std::size_t>(k)];
            for (Eigen::Index i = 0; i < N; ++i)
                for (Eigen::Index lag = 0; lag < order; ++lag)
                    snapshot.F(k, i * order + lag) = filter(i, lag);
        }

        if (index < state.layers.size())
        {
            snapshot.learning = state.layers[index].learning;
            snapshot.learningEnabled = state.layers[index].learningEnabled;
        }

        if (output.posterior.size() > 0)
        {
            snapshot.hasInference = true;
            snapshot.inheritedTopDownSupport = output.topDownSupport;
            snapshot.inheritedAlpha = output.alpha;
            snapshot.marginals = output.marginals;
            snapshot.observationScores = output.selection.observationScores;
            snapshot.selectedGenerators = output.selection.selectedGenerators;
            snapshot.candidates = output.selection.candidates;
            snapshot.candidatePrior = output.prior;
            snapshot.candidateLikelihood = output.likelihood;
            snapshot.candidatePosterior = output.posterior;
            snapshot.candidateCount = output.selection.candidates.size();
            snapshot.logEvidence = output.logEvidence;
        }

        if (index < diagnostics.size())
            snapshot.learningDiagnostics = diagnostics[index];

        layers_.push_back(std::move(snapshot));
    }

    setSize(getWidth(), preferredHeight(getWidth()));
    repaint();
}

juce::String LayerStateDisplay::number(double value)
{
    if (!std::isfinite(value))
        return "n/a";

    if (value > 0.0 && value < 0.001)
        return "<0.001";
    if (value < 0.0 && value > -0.001)
        return ">-0.001";

    auto text = juce::String(value, 3);
    while (text.containsChar('.') && text.endsWithChar('0'))
        text = text.dropLastCharacters(1);
    if (text.endsWithChar('.'))
        text = text.dropLastCharacters(1);
    return text;
}

juce::String LayerStateDisplay::cellLabel(double value)
{
    if (!std::isfinite(value))
        return "n/a";

    const double clipped = juce::jlimit(0.0, 1.0, value);

    if (clipped < 0.005)
        return "0";
    if (clipped > 0.995)
        return "1";

    auto text = juce::String(clipped, 2);

    // Probabilities are in [0,1]. For fractional values, display .53
    // rather than 0.53 to keep labels compact inside small cells.
    if (text.startsWithChar('0'))
        text = text.substring(1);

    return text;
}

juce::String LayerStateDisplay::signedCellLabel(double value)
{
    if (!std::isfinite(value))
        return "n/a";

    if (std::abs(value) < 0.005)
        return "0";

    const bool negative = value < 0.0;
    const double magnitude = std::abs(value);
    auto text = juce::String(magnitude, 2);

    if (text.startsWithChar('0'))
        text = text.substring(1);

    return juce::String(negative ? "-" : "+") + text;
}

float LayerStateDisplay::matrixHeight(
    const Eigen::MatrixXd& matrix,
    float availableWidth,
    float maximumCellSize)
{
    if (matrix.rows() == 0 || matrix.cols() == 0)
        return 0.0f;

    const float cell = std::min(
        maximumCellSize,
        availableWidth / static_cast<float>(matrix.cols()));
    return cell * static_cast<float>(matrix.rows());
}

juce::Rectangle<float> LayerStateDisplay::drawMatrix(
    juce::Graphics& g,
    const Eigen::MatrixXd& matrix,
    juce::Rectangle<float> area,
    float maximumCellSize,
    std::optional<std::size_t> highlightedColumn)
{
    if (matrix.rows() == 0 || matrix.cols() == 0)
        return {};

    const float cell = std::min(
        maximumCellSize,
        std::min(
            area.getWidth() / static_cast<float>(matrix.cols()),
            area.getHeight() / static_cast<float>(matrix.rows())));

    const float width = cell * static_cast<float>(matrix.cols());
    const float height = cell * static_cast<float>(matrix.rows());
    const auto bounds = juce::Rectangle<float>(
        area.getX(), area.getY(), width, height);

    const auto clip = g.getClipBounds().toFloat();
    const auto visibleBounds = bounds.getIntersection(clip);
    if (visibleBounds.isEmpty())
        return bounds;

    g.setColour(juce::Colours::black);
    g.fillRect(visibleBounds);

    const auto rowRange = visibleCellRange(
        clip.getY(), clip.getBottom(), bounds.getY(), cell, matrix.rows());
    const auto columnRange = visibleCellRange(
        clip.getX(), clip.getRight(), bounds.getX(), cell, matrix.cols());
    const auto visibleCellCount =
        (rowRange.second - rowRange.first + 1)
        * (columnRange.second - columnRange.first + 1);
    const bool drawLabels = cell >= 11.0f && visibleCellCount <= 320;
    const float labelFontSize = std::max(7.0f, std::min(11.0f, cell * 0.70f));
    if (drawLabels)
        g.setFont(juce::Font(juce::FontOptions(labelFontSize)));
    
    for (Eigen::Index row = rowRange.first; row <= rowRange.second; ++row)
    {
        for (Eigen::Index column = columnRange.first;
             column <= columnRange.second;
             ++column)
        {
            const double rawValue = matrix(row, column);
            const float value = static_cast<float>(juce::jlimit(
                0.0, 1.0, rawValue));
            const float x = bounds.getX() + static_cast<float>(column) * cell;
            const float y = bounds.getY() + static_cast<float>(row) * cell;

            g.setColour(juce::Colour::fromFloatRGBA(value, value, value, 1.0f));
            g.fillRect(x, y, cell, cell);

            if (drawLabels)
            {
                g.setColour(value >= 0.55f
                    ? juce::Colours::black.withAlpha(0.82f)
                    : juce::Colours::white.withAlpha(0.88f));
                g.drawText(
                    cellLabel(rawValue),
                    x + 1.0f,
                    y,
                    std::max(1.0f, cell - 2.0f),
                    cell,
                    juce::Justification::centred,
                    false);
            }
        }
    }

    g.setColour(juce::Colour::fromRGB(90, 94, 100));
    g.drawRect(bounds, 1.0f);

    if (highlightedColumn.has_value() &&
        *highlightedColumn < static_cast<std::size_t>(matrix.cols()))
    {
        const auto highlight = juce::Rectangle<float>(
            bounds.getX() + static_cast<float>(*highlightedColumn) * cell,
            bounds.getY(), cell, bounds.getHeight());
        g.setColour(juce::Colour::fromRGB(255, 196, 64));
        g.drawRect(highlight, 2.0f);
    }

    return bounds;
}

juce::Rectangle<float> LayerStateDisplay::drawSignedMatrix(
    juce::Graphics& g,
    const Eigen::MatrixXd& matrix,
    juce::Rectangle<float> area,
    float maximumCellSize)
{
    if (matrix.rows() == 0 || matrix.cols() == 0)
        return {};

    const float cell = std::min(
        maximumCellSize,
        std::min(
            area.getWidth() / static_cast<float>(matrix.cols()),
            area.getHeight() / static_cast<float>(matrix.rows())));

    const float width = cell * static_cast<float>(matrix.cols());
    const float height = cell * static_cast<float>(matrix.rows());
    const auto bounds = juce::Rectangle<float>(
        area.getX(), area.getY(), width, height);

    const auto clip = g.getClipBounds().toFloat();
    const auto visibleBounds = bounds.getIntersection(clip);
    if (visibleBounds.isEmpty())
        return bounds;

    g.setColour(juce::Colour::fromFloatRGBA(0.5f, 0.5f, 0.5f, 1.0f));
    g.fillRect(visibleBounds);

    const auto rowRange = visibleCellRange(
        clip.getY(), clip.getBottom(), bounds.getY(), cell, matrix.rows());
    const auto columnRange = visibleCellRange(
        clip.getX(), clip.getRight(), bounds.getX(), cell, matrix.cols());
    const auto visibleCellCount =
        (rowRange.second - rowRange.first + 1)
        * (columnRange.second - columnRange.first + 1);
    const bool drawLabels = cell >= 11.0f && visibleCellCount <= 320;
    const double scale = std::max(1.0e-12, maxAbsolute(matrix));
    const float labelFontSize = std::max(7.0f, std::min(11.0f, cell * 0.70f));
    if (drawLabels)
        g.setFont(juce::Font(juce::FontOptions(labelFontSize)));

    for (Eigen::Index row = rowRange.first; row <= rowRange.second; ++row)
    {
        for (Eigen::Index column = columnRange.first;
             column <= columnRange.second;
             ++column)
        {
            const double rawValue = matrix(row, column);
            const float normalised = static_cast<float>(juce::jlimit(
                0.0, 1.0, 0.5 + 0.5 * rawValue / scale));
            const float x = bounds.getX() + static_cast<float>(column) * cell;
            const float y = bounds.getY() + static_cast<float>(row) * cell;

            // Signed grayscale: black = most negative update,
            // mid-gray = zero, white = most positive update.
            g.setColour(juce::Colour::fromFloatRGBA(
                normalised, normalised, normalised, 1.0f));
            g.fillRect(x, y, cell, cell);

            if (drawLabels)
            {
                g.setColour(normalised >= 0.62f
                    ? juce::Colours::black.withAlpha(0.82f)
                    : juce::Colours::white.withAlpha(0.88f));
                g.drawText(
                    signedCellLabel(rawValue),
                    x + 1.0f,
                    y,
                    std::max(1.0f, cell - 2.0f),
                    cell,
                    juce::Justification::centred,
                    false);
            }
        }
    }

    g.setColour(juce::Colour::fromRGB(90, 94, 100));
    g.drawRect(bounds, 1.0f);

    return bounds;
}

juce::String LayerStateDisplay::selectedGeneratorText(
    const std::vector<Eigen::Index>& generators)
{
    if (generators.empty())
        return "none";

    juce::String text;
    for (std::size_t index = 0; index < generators.size(); ++index)
    {
        if (index > 0)
            text << ", ";
        text << static_cast<int>(generators[index]);
    }
    return text;
}


juce::String LayerStateDisplay::parameterName(
    EditableParameter parameter)
{
    switch (parameter)
    {
        case EditableParameter::BaseRate: return "base";
        case EditableParameter::BottomUpWeight: return "w";
        case EditableParameter::EvidenceAmplitude: return "A";
        case EditableParameter::Centering: return "c";
    }

    return "param";
}

double LayerStateDisplay::parameterMinimum(
    EditableParameter parameter)
{
    switch (parameter)
    {
        case EditableParameter::BaseRate:
        case EditableParameter::BottomUpWeight:
        case EditableParameter::Centering:
            return 0.0;
        case EditableParameter::EvidenceAmplitude:
            return 0.0;
    }

    return 0.0;
}

double LayerStateDisplay::parameterMaximum(
    EditableParameter parameter)
{
    switch (parameter)
    {
        case EditableParameter::BaseRate:
        case EditableParameter::BottomUpWeight:
        case EditableParameter::Centering:
            return 1.0;
        case EditableParameter::EvidenceAmplitude:
            return 1000.0;
    }

    return 1.0;
}

double LayerStateDisplay::parameterDragSensitivity(
    EditableParameter parameter)
{
    switch (parameter)
    {
        case EditableParameter::BaseRate:
        case EditableParameter::BottomUpWeight:
        case EditableParameter::Centering:
            return 1.0 / static_cast<double>(dragPixelsForFullScale);
        case EditableParameter::EvidenceAmplitude:
            return static_cast<double>(amplitudeDragUnitsForFullScale)
                 / static_cast<double>(dragPixelsForFullScale);
    }

    return 1.0 / static_cast<double>(dragPixelsForFullScale);
}

juce::String LayerStateDisplay::settingName(EditableSetting setting)
{
    switch (setting)
    {
        case EditableSetting::CandidateContextThreshold: return "threshold qF";
        case EditableSetting::CandidateTopDownThreshold: return "threshold tau";
        case EditableSetting::CandidateObservationThreshold: return "threshold obs";
        case EditableSetting::CandidateUseActivationSupport: return "alpha support";
        case EditableSetting::CandidateActivationThreshold: return "threshold alpha";
        case EditableSetting::CandidateMaximumSelectedGenerators: return "max generators";

        case EditableSetting::LearningEnabled: return "enabled";
        case EditableSetting::LearningPredictionRate: return "eta R";
        case EditableSetting::LearningFilterRate: return "eta F";
        case EditableSetting::LearningBaseRate: return "eta base";
        case EditableSetting::LearningEpsilon: return "epsilon";
        case EditableSetting::LearningBinarizeObservation: return "binarize";
        case EditableSetting::LearningObservationThreshold: return "threshold";
    }

    return "setting";
}

double LayerStateDisplay::settingMinimum(EditableSetting setting)
{
    switch (setting)
    {
        case EditableSetting::CandidateObservationThreshold:
            return -1.0;

        case EditableSetting::CandidateMaximumSelectedGenerators:
            return 0.0;

        case EditableSetting::LearningEpsilon:
            return 1.0e-12;

        default:
            return 0.0;
    }
}

double LayerStateDisplay::settingMaximum(EditableSetting setting)
{
    switch (setting)
    {
        case EditableSetting::CandidateMaximumSelectedGenerators:
            return 62.0;

        case EditableSetting::LearningPredictionRate:
        case EditableSetting::LearningFilterRate:
        case EditableSetting::LearningBaseRate:
            return 1.0;

        case EditableSetting::LearningEpsilon:
            return 1.0;

        default:
            return 1.0;
    }
}

double LayerStateDisplay::settingVisualMaximum(
    EditableSetting setting,
    double value)
{
    switch (setting)
    {
        case EditableSetting::LearningPredictionRate:
        case EditableSetting::LearningFilterRate:
        case EditableSetting::LearningBaseRate:
            return std::max(0.01, value);

        case EditableSetting::LearningEpsilon:
            return std::max(1.0e-4, value);

        default:
            return settingMaximum(setting);
    }
}

double LayerStateDisplay::settingDragSensitivity(EditableSetting setting)
{
    switch (setting)
    {
        case EditableSetting::CandidateContextThreshold:
        case EditableSetting::CandidateTopDownThreshold:
        case EditableSetting::CandidateActivationThreshold:
        case EditableSetting::LearningObservationThreshold:
            return 1.0 / (2.0 * static_cast<double>(dragPixelsForFullScale));

        case EditableSetting::CandidateObservationThreshold:
            return 0.25 / static_cast<double>(dragPixelsForFullScale);

        case EditableSetting::CandidateMaximumSelectedGenerators:
            return 10.0 / static_cast<double>(dragPixelsForFullScale);

        case EditableSetting::LearningPredictionRate:
        case EditableSetting::LearningFilterRate:
        case EditableSetting::LearningBaseRate:
            return 0.01 / static_cast<double>(dragPixelsForFullScale);

        case EditableSetting::LearningEpsilon:
            return 1.0e-4 / static_cast<double>(dragPixelsForFullScale);

        case EditableSetting::CandidateUseActivationSupport:
        case EditableSetting::LearningEnabled:
        case EditableSetting::LearningBinarizeObservation:
            return 1.0;
    }

    return 1.0 / static_cast<double>(dragPixelsForFullScale);
}

bool LayerStateDisplay::settingIsInteger(EditableSetting setting)
{
    return setting == EditableSetting::CandidateMaximumSelectedGenerators;
}

bool LayerStateDisplay::settingIsToggle(EditableSetting setting)
{
    return setting == EditableSetting::CandidateUseActivationSupport
        || setting == EditableSetting::LearningEnabled
        || setting == EditableSetting::LearningBinarizeObservation;
}

juce::String LayerStateDisplay::settingValueText(
    EditableSetting setting,
    double value)
{
    if (settingIsToggle(setting))
        return value >= 0.5 ? "yes" : "no";

    if (settingIsInteger(setting))
        return juce::String(static_cast<int>(std::round(value)));

    return number(value);
}

juce::String LayerStateDisplay::stateActiveSetText(
    noisy_or::State state,
    Eigen::Index generatorCount)
{
    if (state == 0)
        return "{}";

    juce::String text("{");
    bool first = true;
    for (Eigen::Index k = 0; k < generatorCount; ++k)
    {
        if ((state >> static_cast<std::size_t>(k)) & noisy_or::State{1})
        {
            if (!first)
                text << ",";
            text << static_cast<int>(k);
            first = false;
        }
    }
    text << "}";
    return text;
}

juce::String LayerStateDisplay::vectorText(
    const Eigen::VectorXd& vector,
    Eigen::Index maximumEntries)
{
    if (vector.size() == 0)
        return "[]";

    const Eigen::Index count = std::min(vector.size(), maximumEntries);
    juce::String text("[");
    for (Eigen::Index i = 0; i < count; ++i)
    {
        if (i > 0)
            text << ", ";
        text << number(vector(i));
    }
    if (count < vector.size())
        text << ", ...";
    text << "]";
    return text;
}

Eigen::MatrixXd LayerStateDisplay::vectorAsRowMatrix(
    const Eigen::VectorXd& vector,
    Eigen::Index maximumEntries)
{
    if (vector.size() == 0 || maximumEntries <= 0)
        return Eigen::MatrixXd();

    const Eigen::Index count = std::min(vector.size(), maximumEntries);
    Eigen::MatrixXd row(1, count);
    for (Eigen::Index i = 0; i < count; ++i)
        row(0, i) = vector(i);
    return row;
}

Eigen::MatrixXd LayerStateDisplay::orderedVectorAsRowMatrix(
    const Eigen::VectorXd& vector,
    const std::vector<Eigen::Index>& order)
{
    if (vector.size() == 0 || order.empty())
        return Eigen::MatrixXd();

    Eigen::MatrixXd row(1, static_cast<Eigen::Index>(order.size()));
    for (Eigen::Index column = 0;
         column < static_cast<Eigen::Index>(order.size());
         ++column)
    {
        const Eigen::Index source = order[static_cast<std::size_t>(column)];
        row(0, column) = source >= 0 && source < vector.size()
            ? vector(source)
            : 0.0;
    }
    return row;
}

std::vector<Eigen::Index> LayerStateDisplay::sortedCandidateRows(
    const LayerSnapshot& layer,
    Eigen::Index maximumRows)
{
    std::vector<Eigen::Index> rows;
    if (!layer.hasInference || layer.candidatePosterior.size() == 0 ||
        maximumRows <= 0)
        return rows;

    rows.reserve(static_cast<std::size_t>(layer.candidatePosterior.size()));
    for (Eigen::Index m = 0; m < layer.candidatePosterior.size(); ++m)
        rows.push_back(m);

    std::sort(rows.begin(), rows.end(),
        [&layer](Eigen::Index a, Eigen::Index b)
        {
            return layer.candidatePosterior(a) > layer.candidatePosterior(b);
        });

    if (rows.size() > static_cast<std::size_t>(maximumRows))
        rows.resize(static_cast<std::size_t>(maximumRows));

    return rows;
}

Eigen::MatrixXd LayerStateDisplay::flattenFilterDelta(
    const std::vector<Eigen::MatrixXd>& filterDelta)
{
    if (filterDelta.empty())
        return Eigen::MatrixXd();

    const Eigen::Index K = static_cast<Eigen::Index>(filterDelta.size());
    const Eigen::Index N = filterDelta.front().rows();
    const Eigen::Index order = filterDelta.front().cols();

    if (N <= 0 || order <= 0)
        return Eigen::MatrixXd();

    Eigen::MatrixXd flattened(K, N * order);
    flattened.setZero();

    for (Eigen::Index k = 0; k < K; ++k)
    {
        const auto& matrix = filterDelta[static_cast<std::size_t>(k)];
        if (matrix.rows() != N || matrix.cols() != order)
            continue;

        for (Eigen::Index i = 0; i < N; ++i)
            for (Eigen::Index lag = 0; lag < order; ++lag)
                flattened(k, i * order + lag) = matrix(i, lag);
    }

    return flattened;
}

juce::String LayerStateDisplay::candidateOrderText(
    const LayerSnapshot& layer,
    const std::vector<Eigen::Index>& order,
    Eigen::Index maximumEntries)
{
    if (order.empty())
        return "order: none";

    const Eigen::Index count = std::min<Eigen::Index>(
        static_cast<Eigen::Index>(order.size()),
        maximumEntries);

    juce::String text("order: ");
    for (Eigen::Index i = 0; i < count; ++i)
    {
        if (i > 0)
            text << "  ";

        const Eigen::Index m = order[static_cast<std::size_t>(i)];
        const noisy_or::State state =
            m >= 0 && m < static_cast<Eigen::Index>(layer.candidates.size())
                ? layer.candidates[static_cast<std::size_t>(m)]
                : noisy_or::State{0};

        text << static_cast<int>(m)
             << ":"
             << static_cast<juce::int64>(state)
             << stateActiveSetText(state, layer.R.rows());
    }

    if (count < static_cast<Eigen::Index>(order.size()))
        text << "  ...";

    return text;
}

juce::Rectangle<float> LayerStateDisplay::drawEditableMatrix(
    juce::Graphics& g,
    EditableMatrix matrixType,
    std::size_t layerIndex,
    const Eigen::MatrixXd& matrix,
    juce::Rectangle<float> area,
    float maximumCellSize,
    std::optional<std::size_t> highlightedColumn)
{
    const auto bounds = drawMatrix(
        g, matrix, area, maximumCellSize, highlightedColumn);

    if (!bounds.isEmpty())
    {
        MatrixRegion region;
        region.matrix = matrixType;
        region.layerIndex = layerIndex;
        region.bounds = bounds.translated(activePaintOffsetX_, 0.0f);
        region.rows = matrix.rows();
        region.columns = matrix.cols();
        region.cellSize = bounds.getWidth()
            / static_cast<float>(std::max<Eigen::Index>(1, matrix.cols()));
        matrixRegions_.push_back(region);

        if (!intersectsClip(g, bounds))
            return bounds;

        const auto clip = g.getClipBounds().toFloat();
        const auto rowRange = visibleCellRange(
            clip.getY(), clip.getBottom(),
            bounds.getY(), region.cellSize, matrix.rows());
        const auto columnRange = visibleCellRange(
            clip.getX(), clip.getRight(),
            bounds.getX(), region.cellSize, matrix.cols());

        g.setColour(juce::Colours::white.withAlpha(0.08f));
        for (Eigen::Index row = std::max<Eigen::Index>(1, rowRange.first);
             row <= rowRange.second;
             ++row)
        {
            const float y = bounds.getY() + static_cast<float>(row) * region.cellSize;
            g.drawHorizontalLine(static_cast<int>(std::round(y)),
                                 bounds.getX(), bounds.getRight());
        }
        for (Eigen::Index column = std::max<Eigen::Index>(1, columnRange.first);
             column <= columnRange.second;
             ++column)
        {
            const float x = bounds.getX() + static_cast<float>(column) * region.cellSize;
            g.drawVerticalLine(static_cast<int>(std::round(x)),
                               bounds.getY(), bounds.getBottom());
        }
    }

    return bounds;
}


juce::Rectangle<float> LayerStateDisplay::drawParameterSlider(
    juce::Graphics& g,
    EditableParameter parameter,
    std::size_t layerIndex,
    Eigen::Index generator,
    double value,
    juce::Rectangle<float> bounds)
{
    if (bounds.isEmpty())
        return {};

    ParameterRegion region;
    region.parameter = parameter;
    region.layerIndex = layerIndex;
    region.generator = generator;
    region.bounds = bounds.translated(activePaintOffsetX_, 0.0f);
    region.minimum = parameterMinimum(parameter);
    region.maximum = parameterMaximum(parameter);
    region.dragSensitivity = parameterDragSensitivity(parameter);
    parameterRegions_.push_back(region);

    if (!intersectsClip(g, bounds))
        return bounds;

    const double visualMaximum = parameter == EditableParameter::EvidenceAmplitude
        ? std::max(10.0, value)
        : region.maximum;
    const double denominator = std::max(1.0e-12, visualMaximum - region.minimum);
    const float proportion = static_cast<float>(juce::jlimit(
        0.0,
        1.0,
        (value - region.minimum) / denominator));

    g.setColour(juce::Colour::fromRGB(48, 52, 58));
    g.fillRoundedRectangle(bounds, 3.0f);

    const auto fill = bounds.withWidth(bounds.getWidth() * proportion);
    g.setColour(juce::Colour::fromRGB(92, 104, 118));
    g.fillRoundedRectangle(fill, 3.0f);

    g.setColour(juce::Colour::fromRGB(105, 110, 116));
    g.drawRoundedRectangle(bounds, 3.0f, 1.0f);

    g.setColour(juce::Colours::white.withAlpha(0.92f));
    g.setFont(juce::Font(juce::FontOptions(10.5f)));
    g.drawText(
        number(value),
        bounds.reduced(3.0f, 0.0f),
        juce::Justification::centred,
        false);

    return bounds;
}

juce::Rectangle<float> LayerStateDisplay::drawSettingSlider(
    juce::Graphics& g,
    EditableSetting setting,
    std::size_t layerIndex,
    double value,
    juce::Rectangle<float> bounds)
{
    if (bounds.isEmpty())
        return {};

    SettingRegion region;
    region.setting = setting;
    region.layerIndex = layerIndex;
    region.bounds = bounds.translated(activePaintOffsetX_, 0.0f);
    region.minimum = settingMinimum(setting);
    region.maximum = settingMaximum(setting);
    region.dragSensitivity = settingDragSensitivity(setting);
    region.integer = settingIsInteger(setting);
    region.toggle = settingIsToggle(setting);
    settingRegions_.push_back(region);

    if (!intersectsClip(g, bounds))
        return bounds;

    if (region.integer)
        value = std::round(value);
    if (region.toggle)
        value = value >= 0.5 ? 1.0 : 0.0;

    const double visualMaximum = settingVisualMaximum(setting, value);
    const double denominator = std::max(1.0e-12, visualMaximum - region.minimum);
    const float proportion = static_cast<float>(juce::jlimit(
        0.0,
        1.0,
        (value - region.minimum) / denominator));

    g.setColour(juce::Colour::fromRGB(48, 52, 58));
    g.fillRoundedRectangle(bounds, 3.0f);

    const auto fill = bounds.withWidth(bounds.getWidth() * proportion);
    g.setColour(region.toggle
        ? juce::Colour::fromRGB(88, 104, 88)
        : juce::Colour::fromRGB(92, 104, 118));
    g.fillRoundedRectangle(fill, 3.0f);

    g.setColour(juce::Colour::fromRGB(105, 110, 116));
    g.drawRoundedRectangle(bounds, 3.0f, 1.0f);

    g.setColour(juce::Colours::white.withAlpha(0.92f));
    g.setFont(juce::Font(juce::FontOptions(10.5f)));
    g.drawText(
        settingValueText(setting, value),
        bounds.reduced(3.0f, 0.0f),
        juce::Justification::centred,
        false);

    return bounds;
}

juce::Rectangle<float> LayerStateDisplay::drawSmallButton(
    juce::Graphics& g,
    const juce::String& text,
    juce::Rectangle<float> bounds,
    bool active)
{
    if (bounds.isEmpty())
        return {};

    if (!intersectsClip(g, bounds))
        return bounds;

    g.setColour(active
        ? juce::Colour::fromRGB(78, 92, 108)
        : juce::Colour::fromRGB(48, 52, 58));
    g.fillRoundedRectangle(bounds, 3.0f);
    g.setColour(active
        ? juce::Colour::fromRGB(140, 154, 172)
        : juce::Colour::fromRGB(105, 110, 116));
    g.drawRoundedRectangle(bounds, 3.0f, 1.0f);
    g.setColour(juce::Colours::white.withAlpha(0.92f));
    g.setFont(juce::Font(juce::FontOptions(9.5f)));
    g.drawText(text, bounds.reduced(2.0f, 0.0f),
               juce::Justification::centred, false);
    return bounds;
}

float LayerStateDisplay::drawGenerationControls(
    juce::Graphics& g,
    std::size_t layerIndex,
    juce::Rectangle<float> predictionArea,
    juce::Rectangle<float> filterArea,
    juce::Rectangle<float> fullArea,
    float y)
{
    if (layerIndex >= randomControls_.size())
        return 0.0f;

    const float buttonHeight = 16.0f;
    const float buttonY = y + 1.0f;
    float x = predictionArea.getX();

    const auto addButton =
        [this, &g, buttonY, buttonHeight, layerIndex]
        (const juce::String& text,
         float buttonX,
         EditableMatrix matrix,
         RandomControl control,
         float width,
         bool active)
        {
            RandomControlRegion region;
            region.control = control;
            region.matrix = matrix;
            region.layerIndex = layerIndex;
            const auto visualBounds = juce::Rectangle<float>(
                buttonX, buttonY, width, buttonHeight);
            region.bounds = visualBounds.translated(activePaintOffsetX_, 0.0f);
            randomControlRegions_.push_back(region);
            drawSmallButton(g, text, visualBounds, active);
        };

    addButton("rand", x, EditableMatrix::Predictions,
              RandomControl::Randomize, randomButtonWidth, false);
    addButton("rand", filterArea.getX(), EditableMatrix::Filters,
              RandomControl::Randomize, randomButtonWidth, false);

    RandomControlRegion settingsRegion;
    settingsRegion.control = RandomControl::ToggleSettings;
    settingsRegion.matrix = EditableMatrix::Predictions;
    settingsRegion.layerIndex = layerIndex;
    const auto settingsVisualBounds = juce::Rectangle<float>(
        std::max(
            predictionArea.getX() + randomButtonWidth + randomControlGap,
            fullArea.getRight() - randomSettingsButtonWidth),
        buttonY,
        randomSettingsButtonWidth,
        buttonHeight);
    settingsRegion.bounds =
        settingsVisualBounds.translated(activePaintOffsetX_, 0.0f);
    randomControlRegions_.push_back(settingsRegion);
    drawSmallButton(g, "rand set", settingsVisualBounds, false);

    return lineHeight;
}

std::optional<LayerStateDisplay::MatrixRegion>
LayerStateDisplay::findMatrixRegion(juce::Point<float> point) const
{
    for (auto it = matrixRegions_.rbegin(); it != matrixRegions_.rend(); ++it)
    {
        if (it->bounds.contains(point))
            return *it;
    }
    return std::nullopt;
}

std::optional<LayerStateDisplay::RandomControlRegion>
LayerStateDisplay::findRandomControlRegion(juce::Point<float> point) const
{
    for (auto it = randomControlRegions_.rbegin();
         it != randomControlRegions_.rend(); ++it)
    {
        if (it->bounds.contains(point))
            return *it;
    }
    return std::nullopt;
}

std::optional<LayerStateDisplay::SectionRegion>
LayerStateDisplay::findSectionRegion(juce::Point<float> point) const
{
    for (auto it = sectionRegions_.rbegin(); it != sectionRegions_.rend(); ++it)
    {
        if (it->bounds.contains(point))
            return *it;
    }
    return std::nullopt;
}


std::optional<LayerStateDisplay::ParameterRegion>
LayerStateDisplay::findParameterRegion(juce::Point<float> point) const
{
    for (auto it = parameterRegions_.rbegin(); it != parameterRegions_.rend(); ++it)
    {
        if (it->bounds.contains(point))
            return *it;
    }
    return std::nullopt;
}

std::optional<LayerStateDisplay::SettingRegion>
LayerStateDisplay::findSettingRegion(juce::Point<float> point) const
{
    for (auto it = settingRegions_.rbegin(); it != settingRegions_.rend(); ++it)
    {
        if (it->bounds.contains(point))
            return *it;
    }
    return std::nullopt;
}

Eigen::Index LayerStateDisplay::cellRowAt(
    const MatrixRegion& region,
    float y) const
{
    const auto row = static_cast<Eigen::Index>(
        std::floor((y - region.bounds.getY()) / region.cellSize));
    return juce::jlimit<Eigen::Index>(0, region.rows - 1, row);
}

Eigen::Index LayerStateDisplay::cellColumnAt(
    const MatrixRegion& region,
    float x) const
{
    const auto column = static_cast<Eigen::Index>(
        std::floor((x - region.bounds.getX()) / region.cellSize));
    return juce::jlimit<Eigen::Index>(0, region.columns - 1, column);
}

double LayerStateDisplay::matrixValueAt(
    EditableMatrix matrix,
    std::size_t layerIndex,
    Eigen::Index row,
    Eigen::Index column) const
{
    switch (matrix)
    {
        case EditableMatrix::Input:
            if (row >= 0 && row < inputWindow_.rows() &&
                column >= 0 && column < inputWindow_.cols())
                return inputWindow_(row, column);
            break;

        case EditableMatrix::Predictions:
            if (layerIndex < layers_.size())
            {
                const auto& R = layers_[layerIndex].R;
                if (row >= 0 && row < R.rows() &&
                    column >= 0 && column < R.cols())
                    return R(row, column);
            }
            break;

        case EditableMatrix::Filters:
            if (layerIndex < layers_.size())
            {
                const auto& F = layers_[layerIndex].F;
                if (row >= 0 && row < F.rows() &&
                    column >= 0 && column < F.cols())
                    return F(row, column);
            }
            break;
    }

    return 0.0;
}


double LayerStateDisplay::parameterValueAt(
    EditableParameter parameter,
    std::size_t layerIndex,
    Eigen::Index generator) const
{
    if (layerIndex >= layers_.size())
        return 0.0;

    const auto& parameters = layers_[layerIndex].parameters;
    const auto vectorValue =
        [generator](const Eigen::VectorXd& values)
        {
            if (values.size() <= 0)
                return 0.0;
            if (generator < 0)
                return values.mean();
            return generator < values.size() ? values(generator) : 0.0;
        };

    switch (parameter)
    {
        case EditableParameter::BaseRate:
            return vectorValue(parameters.baseRate);
        case EditableParameter::BottomUpWeight:
            return vectorValue(parameters.bottomUpWeight);
        case EditableParameter::EvidenceAmplitude:
            return vectorValue(parameters.evidenceAmplitude);
        case EditableParameter::Centering:
            return vectorValue(parameters.centering);
    }

    return 0.0;
}

double LayerStateDisplay::settingValueAt(
    EditableSetting setting,
    std::size_t layerIndex) const
{
    if (layerIndex >= layers_.size())
        return 0.0;

    const auto& layer = layers_[layerIndex];
    const auto& policy = layer.candidatePolicy;
    const auto& learning = layer.learning;

    switch (setting)
    {
        case EditableSetting::CandidateContextThreshold:
            return policy.contextThreshold;
        case EditableSetting::CandidateTopDownThreshold:
            return policy.topDownThreshold;
        case EditableSetting::CandidateObservationThreshold:
            return policy.observationThreshold;
        case EditableSetting::CandidateUseActivationSupport:
            return policy.useActivationSupport ? 1.0 : 0.0;
        case EditableSetting::CandidateActivationThreshold:
            return policy.activationThreshold;
        case EditableSetting::CandidateMaximumSelectedGenerators:
            return static_cast<double>(policy.maximumSelectedGenerators);

        case EditableSetting::LearningEnabled:
            return layer.learningEnabled ? 1.0 : 0.0;
        case EditableSetting::LearningPredictionRate:
            return learning.predictionLearningRate;
        case EditableSetting::LearningFilterRate:
            return learning.filterLearningRate;
        case EditableSetting::LearningBaseRate:
            return learning.baseRateLearningRate;
        case EditableSetting::LearningEpsilon:
            return learning.epsilon;
        case EditableSetting::LearningBinarizeObservation:
            return learning.binarizeObservation ? 1.0 : 0.0;
        case EditableSetting::LearningObservationThreshold:
            return learning.observationThreshold;
    }

    return 0.0;
}

void LayerStateDisplay::sendRandomizeRequest(
    EditableMatrix matrix,
    std::size_t layerIndex)
{
    if (!onRandomizeRequest || layerIndex >= randomControls_.size())
        return;

    const auto& state = randomControls_[layerIndex];
    const bool isR = matrix == EditableMatrix::Predictions;

    RandomizeRequest request;
    request.matrix = matrix;
    request.layerIndex = layerIndex;
    request.betaShapeA = isR ? state.predictionAlpha : state.filterAlpha;
    request.betaShapeB = isR ? state.predictionBeta : state.filterBeta;
    onRandomizeRequest(request);
}

void LayerStateDisplay::editGenerationSettingsAsText(std::size_t layerIndex)
{
    if (layerIndex >= randomControls_.size())
        return;

    const auto state = randomControls_[layerIndex];

    auto* alert = new juce::AlertWindow(
        "Random generation settings",
        "Layer " + juce::String(static_cast<int>(layerIndex))
            + " Beta distribution parameters",
        juce::AlertWindow::NoIcon);

    alert->addTextEditor("alphaR", juce::String(state.predictionAlpha, 6),
                         "alphaR");
    alert->addTextEditor("betaR", juce::String(state.predictionBeta, 6),
                         "betaR");
    alert->addTextEditor("alphaF", juce::String(state.filterAlpha, 6),
                         "alphaF");
    alert->addTextEditor("betaF", juce::String(state.filterBeta, 6),
                         "betaF");
    alert->addButton("OK", 1, juce::KeyPress(juce::KeyPress::returnKey));
    alert->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));

    juce::Component::SafePointer<LayerStateDisplay> safeThis(this);
    alert->enterModalState(
        true,
        juce::ModalCallbackFunction::create(
            [safeThis, alert, layerIndex](int result)
            {
                std::unique_ptr<juce::AlertWindow> alertOwner(alert);
                if (safeThis == nullptr || result != 1)
                    return;

                if (layerIndex >= safeThis->randomControls_.size())
                    return;

                const auto readShape =
                    [alert](const char* name, double fallback)
                    {
                        const double value =
                            alert->getTextEditorContents(name).getDoubleValue();
                        return std::isfinite(value) && value > 0.0
                            ? juce::jlimit(0.001, 50.0, value)
                            : fallback;
                    };

                auto& state = safeThis->randomControls_[layerIndex];
                state.predictionAlpha =
                    readShape("alphaR", state.predictionAlpha);
                state.predictionBeta =
                    readShape("betaR", state.predictionBeta);
                state.filterAlpha =
                    readShape("alphaF", state.filterAlpha);
                state.filterBeta =
                    readShape("betaF", state.filterBeta);
                safeThis->repaint();
            }),
        false);
}

bool LayerStateDisplay::isSectionExpanded(
    CollapsibleSection section,
    std::size_t layerIndex) const
{
    if (layerIndex >= sectionStates_.size())
        return false;

    const auto& state = sectionStates_[layerIndex];
    return section == CollapsibleSection::CandidateSelection
        ? state.candidateSelectionExpanded
        : state.onlineEMExpanded;
}

void LayerStateDisplay::toggleSection(
    CollapsibleSection section,
    std::size_t layerIndex)
{
    if (layerIndex >= sectionStates_.size())
        return;

    auto& state = sectionStates_[layerIndex];
    if (section == CollapsibleSection::CandidateSelection)
        state.candidateSelectionExpanded = !state.candidateSelectionExpanded;
    else
        state.onlineEMExpanded = !state.onlineEMExpanded;

    setSize(getWidth(), preferredHeight(getWidth()));
    repaint();
}

juce::Rectangle<float> LayerStateDisplay::drawSectionToggle(
    juce::Graphics& g,
    CollapsibleSection section,
    std::size_t layerIndex,
    juce::Rectangle<float> headerBounds)
{
    const bool expanded = isSectionExpanded(section, layerIndex);
    const float size = 16.0f;
    SectionRegion region;
    region.section = section;
    region.layerIndex = layerIndex;
    const auto visualBounds = juce::Rectangle<float>(
        headerBounds.getRight() - size,
        headerBounds.getY() + 3.0f,
        size,
        size);
    region.bounds = visualBounds.translated(activePaintOffsetX_, 0.0f);
    sectionRegions_.push_back(region);
    drawSmallButton(g, expanded ? "-" : "+", visualBounds, expanded);
    return visualBounds;
}

void LayerStateDisplay::applyEditFromMouse(
    const MatrixRegion& region,
    Eigen::Index row,
    Eigen::Index column,
    double value)
{
    if (!onMatrixEdit)
        return;

    MatrixEdit edit;
    edit.matrix = region.matrix;
    edit.layerIndex = region.layerIndex;
    edit.row = row;
    edit.column = column;
    edit.value = juce::jlimit(0.0, 1.0, value);

    // Input is displayed as channels x time-window, so map the visible column
    // back to the actual row in the stored T x N input sequence.
    if (region.matrix == EditableMatrix::Input)
    {
        if (column < 0 ||
            column >= static_cast<Eigen::Index>(inputIndices_.size()))
            return;
        edit.row = static_cast<Eigen::Index>(inputIndices_[static_cast<std::size_t>(column)]);
        edit.column = row;
    }

    onMatrixEdit(edit);
}


void LayerStateDisplay::applyParameterEditFromMouse(
    const ParameterRegion& region,
    double value)
{
    if (!onParameterEdit)
        return;

    ParameterEdit edit;
    edit.parameter = region.parameter;
    edit.layerIndex = region.layerIndex;
    edit.generator = region.generator;
    edit.value = juce::jlimit(region.minimum, region.maximum, value);
    edit.applyToAllGenerators = region.generator < 0;

    onParameterEdit(edit);
}

void LayerStateDisplay::applySettingEditFromMouse(
    const SettingRegion& region,
    double value)
{
    if (!onSettingEdit)
        return;

    value = juce::jlimit(region.minimum, region.maximum, value);
    if (region.integer)
        value = std::round(value);
    if (region.toggle)
        value = value >= 0.5 ? 1.0 : 0.0;

    SettingEdit edit;
    edit.setting = region.setting;
    edit.layerIndex = region.layerIndex;
    edit.value = value;

    onSettingEdit(edit);
}

void LayerStateDisplay::mouseDown(const juce::MouseEvent& event)
{
    matrixDragState_.reset();
    parameterDragState_.reset();
    settingDragState_.reset();

    auto sectionRegion = findSectionRegion(event.position);
    if (sectionRegion.has_value())
    {
        toggleSection(sectionRegion->section, sectionRegion->layerIndex);
        return;
    }

    auto randomRegion = findRandomControlRegion(event.position);
    if (randomRegion.has_value())
    {
        if (randomRegion->control == RandomControl::Randomize)
        {
            sendRandomizeRequest(
                randomRegion->matrix,
                randomRegion->layerIndex);
            return;
        }

        if (randomRegion->control == RandomControl::ToggleSettings)
        {
            editGenerationSettingsAsText(randomRegion->layerIndex);
            return;
        }

        return;
    }

    auto parameterRegion = findParameterRegion(event.position);
    if (parameterRegion.has_value())
    {
        ParameterDragState state;
        state.region = *parameterRegion;
        state.originalValue = parameterValueAt(
            parameterRegion->parameter,
            parameterRegion->layerIndex,
            parameterRegion->generator);
        parameterDragState_ = state;
        return;
    }

    auto settingRegion = findSettingRegion(event.position);
    if (settingRegion.has_value())
    {
        const double originalValue = settingValueAt(
            settingRegion->setting,
            settingRegion->layerIndex);

        if (settingRegion->toggle)
        {
            applySettingEditFromMouse(
                *settingRegion,
                originalValue >= 0.5 ? 0.0 : 1.0);
            return;
        }

        SettingDragState state;
        state.region = *settingRegion;
        state.originalValue = originalValue;
        settingDragState_ = state;
        return;
    }

    auto region = findMatrixRegion(event.position);
    if (!region.has_value())
        return;

    const Eigen::Index row = cellRowAt(*region, event.position.y);
    const Eigen::Index column = cellColumnAt(*region, event.position.x);
    const double oldValue = matrixValueAt(region->matrix, region->layerIndex,
                                          row, column);

    MatrixDragState state;
    state.region = *region;
    state.row = row;
    state.column = column;
    state.originalValue = oldValue;
    matrixDragState_ = state;

    const double toggledValue = oldValue < 0.5 ? 1.0 : 0.0;
    applyEditFromMouse(*region, row, column, toggledValue);
}

void LayerStateDisplay::mouseDrag(const juce::MouseEvent& event)
{
    if (settingDragState_.has_value())
    {
        auto& state = *settingDragState_;
        const float dy = event.getDistanceFromDragStartY();
        if (!state.dragStarted && std::abs(dy) < 2.0f)
            return;

        state.dragStarted = true;
        const double value = state.originalValue
            - static_cast<double>(dy) * state.region.dragSensitivity;
        applySettingEditFromMouse(state.region, value);
        return;
    }

    if (parameterDragState_.has_value())
    {
        auto& state = *parameterDragState_;
        const float dy = event.getDistanceFromDragStartY();
        if (!state.dragStarted && std::abs(dy) < 2.0f)
            return;

        state.dragStarted = true;
        const double value = state.originalValue
            - static_cast<double>(dy) * state.region.dragSensitivity;
        applyParameterEditFromMouse(state.region, value);
        return;
    }

    if (!matrixDragState_.has_value())
        return;

    auto& state = *matrixDragState_;
    const float dy = event.getDistanceFromDragStartY();
    if (!state.fractionalDragStarted && std::abs(dy) < 3.0f)
        return;

    state.fractionalDragStarted = true;
    const double value = juce::jlimit(
        0.0,
        1.0,
        state.originalValue - static_cast<double>(dy) / dragPixelsForFullScale);

    applyEditFromMouse(state.region, state.row, state.column, value);
}

void LayerStateDisplay::mouseUp(const juce::MouseEvent&)
{
    matrixDragState_.reset();
    parameterDragState_.reset();
    settingDragState_.reset();
}


int LayerStateDisplay::preferredHeight(int width) const
{
    const float inputContentWidth =
        std::max(1.0f, static_cast<float>(width) - 2.0f * margin);
    float inputHeight = margin + titleHeight;

    if (inputWindow_.size() > 0)
        inputHeight += matrixHeight(inputWindow_, inputContentWidth, maximumInputCell)
                    + titleHeight + sectionGap;

    float maxLayerHeight = 0.0f;
    const float contentWidth = layerPanelWidth;
    for (std::size_t layerIndex = 0; layerIndex < layers_.size(); ++layerIndex)
    {
        float height = 0.0f;
        const auto& layer = layers_[layerIndex];
        const auto matrixWidths = compactMatrixWidths(
            layer.R, layer.F, contentWidth,
            maximumParameterCell, maximumParameterCell);
        const float matrices = std::max(
            matrixHeight(layer.R, matrixWidths.first, maximumParameterCell),
            matrixHeight(layer.F, matrixWidths.second, maximumParameterCell));
        const float randomControlsHeight = lineHeight;
        float generatorVectors = 0.0f;
        if (layer.alpha.size() > 0)
        {
            generatorVectors += lineHeight
                + matrixHeight(vectorAsRowMatrix(layer.alpha),
                               contentWidth, maximumProbabilityCell)
                + 4.0f;
        }
        if (layer.hasInference &&
            hasVisibleSupport(layer.inheritedTopDownSupport))
        {
            generatorVectors += lineHeight
                + matrixHeight(vectorAsRowMatrix(layer.inheritedTopDownSupport),
                               contentWidth, maximumProbabilityCell)
                + 4.0f;
        }
        if (layer.hasInference && layer.inheritedAlpha.size() > 0)
        {
            generatorVectors += lineHeight
                + matrixHeight(vectorAsRowMatrix(layer.inheritedAlpha),
                               contentWidth, maximumProbabilityCell)
                + 4.0f;
        }
        if (layer.hasInference && layer.marginals.size() > 0)
        {
            generatorVectors += lineHeight
                + matrixHeight(vectorAsRowMatrix(layer.marginals),
                               contentWidth, maximumProbabilityCell)
                + 4.0f;
        }

        const float generatorText =
            titleHeight
            + parameterHeaderHeight
            + lineHeight * static_cast<float>(layer.R.rows())
            + lineHeight
            + generatorVectors;
        const bool candidateExpanded = isSectionExpanded(
            CollapsibleSection::CandidateSelection,
            layerIndex);
        const bool emExpanded = isSectionExpanded(
            CollapsibleSection::OnlineEM,
            layerIndex);
        const float candidateSectionHeight = titleHeight
            + (candidateExpanded ? lineHeight * 9.0f : lineHeight);
        const float emSectionHeight = titleHeight
            + (emExpanded ? lineHeight * 10.0f : lineHeight);
        const float diagnosticsText =
            std::max(candidateSectionHeight, emSectionHeight);
        const float emDeltaMatrices =
            emExpanded && layer.learningDiagnostics.has_value()
            ? titleHeight
                + titleHeight
                + std::max(
                    matrixHeight(layer.learningDiagnostics->predictionDelta,
                                 compactMatrixWidths(
                                     layer.learningDiagnostics->predictionDelta,
                                     flattenFilterDelta(layer.learningDiagnostics->filterDelta),
                                     contentWidth, maximumDeltaCell, maximumDeltaCell).first,
                                 maximumDeltaCell),
                    matrixHeight(flattenFilterDelta(
                                     layer.learningDiagnostics->filterDelta),
                                 compactMatrixWidths(
                                     layer.learningDiagnostics->predictionDelta,
                                     flattenFilterDelta(layer.learningDiagnostics->filterDelta),
                                     contentWidth, maximumDeltaCell, maximumDeltaCell).second,
                                 maximumDeltaCell))
                + sectionGap
            : 0.0f;
        const float posteriorVectors = layer.hasInference
            ? titleHeight
                + lineHeight
                + 3.0f * (lineHeight + maximumProbabilityCell + 4.0f)
                + lineHeight
            : 0.0f;

        height += titleHeight + titleHeight
               + matrices + 4.0f + randomControlsHeight
               + generatorText + sectionGap + diagnosticsText + sectionGap
               + emDeltaMatrices
               + posteriorVectors + sectionGap;
        maxLayerHeight = std::max(maxLayerHeight, height);
    }

    return static_cast<int>(std::ceil(inputHeight + maxLayerHeight + margin));
}

int LayerStateDisplay::preferredWidth() const
{
    const float inputWidth = inputWindow_.cols() > 0
        ? 2.0f * margin
            + std::min(
                maximumInputCell * static_cast<float>(inputWindow_.cols()),
                layerPanelWidth)
        : 0.0f;

    const float layersWidth = layers_.empty()
        ? 0.0f
        : 2.0f * margin
            + static_cast<float>(layers_.size()) * layerPanelWidth
            + static_cast<float>(layers_.size() - 1) * sectionGap;

    return static_cast<int>(std::ceil(
        std::max({ 320.0f, inputWidth, layersWidth })));
}

void LayerStateDisplay::paint(juce::Graphics& g)
{
    matrixRegions_.clear();
    parameterRegions_.clear();
    settingRegions_.clear();
    randomControlRegions_.clear();
    sectionRegions_.clear();
    g.fillAll(juce::Colours::white);
    g.setFont(juce::Font(juce::FontOptions(13.0f)));

    float y = margin;
    const float contentWidth = static_cast<float>(getWidth()) - 2.0f * margin;
    activePaintOffsetX_ = 0.0f;

    g.setColour(primaryTextColour());
    g.setFont(juce::Font(juce::FontOptions(16.0f)));
    g.drawText("Input sequence", margin, y, contentWidth, titleHeight,
               juce::Justification::centredLeft);
    y += titleHeight;

    if (inputWindow_.size() > 0)
    {
        const float height = matrixHeight(
            inputWindow_, contentWidth, maximumInputCell);
        const auto drawn = drawEditableMatrix(
            g, EditableMatrix::Input, 0, inputWindow_,
            { margin, y, contentWidth, height },
            maximumInputCell, highlightedInputColumn_);
        y += drawn.getHeight() + 4.0f;

        g.setColour(secondaryTextColour());
        g.setFont(juce::Font(juce::FontOptions(12.0f)));
        juce::String range;
        if (!inputIndices_.empty())
            range << "timesteps " << static_cast<int>(inputIndices_.front())
                  << " ... " << static_cast<int>(inputIndices_.back())
                  << " (rolling window, max 64)";
        g.drawText(range, margin, y, contentWidth, titleHeight,
                   juce::Justification::centredLeft);
        y += titleHeight + sectionGap;
    }

    const float layerTop = y;
    for (std::size_t layerIndex = 0; layerIndex < layers_.size(); ++layerIndex)
    {
        y = layerTop;
        const float layerLeft = margin
            + static_cast<float>(layerIndex) * (layerPanelWidth + sectionGap);
        const float contentWidth = layerPanelWidth;
        const auto layerBounds = juce::Rectangle<float>(
            layerLeft - 8.0f,
            layerTop - 8.0f,
            layerPanelWidth + 16.0f,
            static_cast<float>(getHeight()) - layerTop + 8.0f);
        if (!intersectsClip(g, layerBounds))
            continue;

        activePaintOffsetX_ = layerLeft - margin;
        juce::Graphics::ScopedSaveState saved(g);
        g.reduceClipRegion(juce::Rectangle<int>(
            static_cast<int>(std::floor(layerLeft - 8.0f)),
            static_cast<int>(std::floor(layerTop - 8.0f)),
            static_cast<int>(std::ceil(layerPanelWidth + 16.0f)),
            std::max(1, getHeight())));
        g.addTransform(juce::AffineTransform::translation(
            layerLeft - margin,
            0.0f));

        const auto& layer = layers_[layerIndex];
        const auto clip = g.getClipBounds().toFloat();

        const auto panelBounds = juce::Rectangle<float>(
            margin - 8.0f,
            y - 6.0f,
            contentWidth + 16.0f,
            static_cast<float>(getHeight()) - y - margin);
        const auto visiblePanelBounds =
            panelBounds.getIntersection(clip).expanded(0.0f, 2.0f);
        g.setColour(juce::Colour::fromRGB(247, 249, 251));
        g.fillRect(visiblePanelBounds);
        g.setColour(panelColour());
        g.drawVerticalLine(static_cast<int>(std::round(panelBounds.getX())),
                           visiblePanelBounds.getY(),
                           visiblePanelBounds.getBottom());
        g.drawVerticalLine(static_cast<int>(std::round(panelBounds.getRight())),
                           visiblePanelBounds.getY(),
                           visiblePanelBounds.getBottom());

        g.setColour(panelColour());
        g.fillRect(margin, y, contentWidth, 2.0f);
        y += 8.0f;

        g.setColour(primaryTextColour());
        g.setFont(juce::Font(juce::FontOptions(16.0f)));
        g.drawText("Layer " + juce::String(static_cast<int>(layerIndex)),
                   margin, y, contentWidth, titleHeight,
                   juce::Justification::centredLeft);
        y += titleHeight;

        const auto matrixWidths = compactMatrixWidths(
            layer.R, layer.F, contentWidth,
            maximumParameterCell, maximumParameterCell);
        const float rWidth = matrixWidths.first;
        const float fWidth = matrixWidths.second;
        const float fX = margin + rWidth + compactMatrixGap;
        const float matrixAreaHeight = std::max(
            matrixHeight(layer.R, rWidth, maximumParameterCell),
            matrixHeight(layer.F, fWidth, maximumParameterCell));

        g.setFont(juce::Font(juce::FontOptions(13.0f)));
        g.setColour(secondaryTextColour());
        g.drawText("R  [K x N] = ["
                       + juce::String(static_cast<int>(layer.R.rows())) + " x "
                       + juce::String(static_cast<int>(layer.R.cols())) + "]",
                   margin, y, rWidth, titleHeight,
                   juce::Justification::centredLeft);
        g.drawText("F  [K x (N*order)] = ["
                       + juce::String(static_cast<int>(layer.F.rows()))
                       + " x " + juce::String(static_cast<int>(layer.F.cols()))
                       + "]",
                   fX, y, fWidth, titleHeight,
                   juce::Justification::centredLeft);
        y += titleHeight;

        drawEditableMatrix(g, EditableMatrix::Predictions, layerIndex, layer.R,
                           { margin, y, rWidth, matrixAreaHeight },
                           maximumParameterCell);
        drawEditableMatrix(g, EditableMatrix::Filters, layerIndex, layer.F,
                           { fX, y, fWidth, matrixAreaHeight },
                           maximumParameterCell);
        y += matrixAreaHeight + 4.0f;

        const float generationControlHeight = drawGenerationControls(
            g,
            layerIndex,
            { margin, y, rWidth, lineHeight },
            { fX, y, fWidth, lineHeight },
            { margin, y, contentWidth, lineHeight },
            y);
        y += generationControlHeight + sectionGap;

        g.setColour(primaryTextColour());
        g.setFont(juce::Font(juce::FontOptions(14.0f)));
        g.drawText("Generator parameters", margin, y,
                   contentWidth, titleHeight,
                   juce::Justification::centredLeft);
        y += titleHeight;

        const float generatorLabelWidth = 44.0f;
        const float availableForSliders = std::max(
            1.0f,
            contentWidth - generatorLabelWidth - 3.0f * parameterSliderGap);
        const float sliderWidth = std::min(
            maximumParameterSliderWidth,
            availableForSliders / 4.0f);

        // Parameter names are column headers, not repeated inside each slider.
        // The labels sit in a small top margin directly above the k=0 row.
        if (intersectsClip(g, { margin, y, contentWidth, parameterHeaderHeight }))
        {
            g.setColour(secondaryTextColour());
            g.setFont(juce::Font(juce::FontOptions(10.5f)));
            float headerX = margin + generatorLabelWidth;
            const juce::String parameterHeaders[] { "base", "w", "A", "c" };
            for (const auto& header : parameterHeaders)
            {
                g.drawText(header, headerX, y, sliderWidth, parameterHeaderHeight,
                           juce::Justification::centred, false);
                headerX += sliderWidth + parameterSliderGap;
            }
        }
        y += parameterHeaderHeight;

        g.setFont(juce::Font(juce::FontOptions(12.5f)));
        const float generatorRowsTop = y;
        const float generatorRowsHeight =
            lineHeight * static_cast<float>(layer.R.rows());
        if (intersectsClip(
                g,
                { margin, generatorRowsTop, contentWidth, generatorRowsHeight }))
        {
            const auto rowRange = visibleCellRange(
                clip.getY(),
                clip.getBottom(),
                generatorRowsTop,
                lineHeight,
                layer.R.rows());
            for (Eigen::Index k = rowRange.first; k <= rowRange.second; ++k)
            {
                const float rowY =
                    generatorRowsTop + static_cast<float>(k) * lineHeight;
                g.setColour(k % 2 == 0 ? primaryTextColour() : secondaryTextColour());
                g.drawText("k=" + juce::String(static_cast<int>(k)),
                           margin, rowY, generatorLabelWidth - 4.0f, lineHeight,
                           juce::Justification::centredLeft, false);

                float sliderX = margin + generatorLabelWidth;
                const auto sliderBounds = [&sliderX, rowY, sliderWidth]()
                {
                    const auto bounds = juce::Rectangle<float>(
                        sliderX, rowY + 1.0f, sliderWidth, lineHeight - 2.0f);
                    sliderX += sliderWidth + parameterSliderGap;
                    return bounds;
                };

                drawParameterSlider(g, EditableParameter::BaseRate,
                                    layerIndex, k,
                                    layer.parameters.baseRate(k),
                                    sliderBounds());
                drawParameterSlider(g, EditableParameter::BottomUpWeight,
                                    layerIndex, k,
                                    layer.parameters.bottomUpWeight(k),
                                    sliderBounds());
                drawParameterSlider(g, EditableParameter::EvidenceAmplitude,
                                    layerIndex, k,
                                    layer.parameters.evidenceAmplitude(k),
                                    sliderBounds());
                drawParameterSlider(g, EditableParameter::Centering,
                                    layerIndex, k,
                                    layer.parameters.centering(k),
                                    sliderBounds());
            }
        }
        y = generatorRowsTop + generatorRowsHeight;

        if (intersectsClip(g, { margin, y, contentWidth, lineHeight }))
        {
            g.setColour(secondaryTextColour());
            g.setFont(juce::Font(juce::FontOptions(12.5f)));
            g.drawText("all",
                       margin, y, generatorLabelWidth - 4.0f, lineHeight,
                       juce::Justification::centredLeft, false);
        }

        if (intersectsClip(g, { margin, y, contentWidth, lineHeight }))
        {
            float sliderX = margin + generatorLabelWidth;
            const auto sliderBounds = [&sliderX, y, sliderWidth]()
            {
                const auto bounds = juce::Rectangle<float>(
                    sliderX, y + 1.0f, sliderWidth, lineHeight - 2.0f);
                sliderX += sliderWidth + parameterSliderGap;
                return bounds;
            };

            drawParameterSlider(g, EditableParameter::BaseRate,
                                layerIndex, -1,
                                parameterValueAt(
                                    EditableParameter::BaseRate,
                                    layerIndex,
                                    -1),
                                sliderBounds());
            drawParameterSlider(g, EditableParameter::BottomUpWeight,
                                layerIndex, -1,
                                parameterValueAt(
                                    EditableParameter::BottomUpWeight,
                                    layerIndex,
                                    -1),
                                sliderBounds());
            drawParameterSlider(g, EditableParameter::EvidenceAmplitude,
                                layerIndex, -1,
                                parameterValueAt(
                                    EditableParameter::EvidenceAmplitude,
                                    layerIndex,
                                    -1),
                                sliderBounds());
            drawParameterSlider(g, EditableParameter::Centering,
                                layerIndex, -1,
                                parameterValueAt(
                                    EditableParameter::Centering,
                                    layerIndex,
                                    -1),
                                sliderBounds());
        }
        y += lineHeight;

        const auto drawVectorBlock =
            [this, &g, &y, contentWidth]
            (const juce::String& label, const Eigen::VectorXd& vector)
            {
                if (vector.size() <= 0)
                    return;

                const float vectorHeight = rowMatrixHeight(
                    vector.size(), contentWidth, maximumProbabilityCell);
                const auto blockBounds = juce::Rectangle<float>(
                    margin,
                    y,
                    contentWidth,
                    lineHeight + vectorHeight + 4.0f);

                if (intersectsClip(g, blockBounds))
                {
                    g.setColour(secondaryTextColour());
                    g.setFont(juce::Font(juce::FontOptions(12.5f)));
                    g.drawText(label + vectorText(vector),
                               margin, y, contentWidth, lineHeight,
                               juce::Justification::centredLeft, false);

                    const auto row = vectorAsRowMatrix(vector);
                    drawMatrix(
                        g, row,
                        { margin, y + lineHeight, contentWidth, vectorHeight },
                        maximumProbabilityCell);
                }

                y += lineHeight + vectorHeight + 4.0f;
            };

        if (layer.hasInference &&
            hasVisibleSupport(layer.inheritedTopDownSupport))
        {
            drawVectorBlock(
                "top-down prior [1 x K]  ",
                layer.inheritedTopDownSupport);
        }

        if (layer.hasInference)
            drawVectorBlock("alpha inherited [1 x K]  ", layer.inheritedAlpha);

        drawVectorBlock("alpha next [1 x K]  ", layer.alpha);

        if (layer.hasInference)
            drawVectorBlock("mu [1 x K]  ", layer.marginals);

        y += sectionGap;

        const float candidateColumnWidth = contentWidth * 0.5f - sectionGap * 0.5f;
        const float emX = margin + candidateColumnWidth + sectionGap;
        const float emColumnWidth = contentWidth - candidateColumnWidth - sectionGap;
        const bool candidateExpanded = isSectionExpanded(
            CollapsibleSection::CandidateSelection,
            layerIndex);
        const bool emExpanded = isSectionExpanded(
            CollapsibleSection::OnlineEM,
            layerIndex);

        g.setColour(primaryTextColour());
        g.setFont(juce::Font(juce::FontOptions(14.0f)));
        const auto candidateHeader =
            juce::Rectangle<float>(margin, y, candidateColumnWidth, titleHeight);
        const auto emHeader =
            juce::Rectangle<float>(emX, y, emColumnWidth, titleHeight);
        g.drawText("Candidate selection", margin, y,
                   candidateColumnWidth - 20.0f, titleHeight,
                   juce::Justification::centredLeft);
        drawSectionToggle(
            g, CollapsibleSection::CandidateSelection,
            layerIndex, candidateHeader);
        g.drawText("Online EM", emX, y,
                   emColumnWidth - 20.0f, titleHeight,
                   juce::Justification::centredLeft);
        drawSectionToggle(
            g, CollapsibleSection::OnlineEM,
            layerIndex, emHeader);
        y += titleHeight;

        const auto settingRow = [&g](EditableSetting setting,
                                     float x,
                                     float& rowY,
                                     float width)
        {
            const float sliderWidth = std::min(
                settingSliderWidth,
                std::max(1.0f, width * 0.45f));
            const float labelWidth = std::max(
                1.0f,
                width - sliderWidth - parameterSliderGap);

            if (intersectsClip(g, { x, rowY, width, lineHeight }))
            {
                g.setColour(secondaryTextColour());
                g.setFont(juce::Font(juce::FontOptions(12.5f)));
                g.drawText(settingName(setting),
                           x, rowY, labelWidth, lineHeight,
                           juce::Justification::centredLeft, false);
            }

            auto bounds = juce::Rectangle<float>(
                x + labelWidth + parameterSliderGap,
                rowY + 1.0f,
                sliderWidth,
                lineHeight - 2.0f);
            rowY += lineHeight;
            return bounds;
        };

        const auto& policy = layer.candidatePolicy;
        float candidateY = y;
        g.setColour(secondaryTextColour());
        g.setFont(juce::Font(juce::FontOptions(12.5f)));
        if (candidateExpanded)
        {
            drawSettingSlider(g, EditableSetting::CandidateContextThreshold,
                              layerIndex, policy.contextThreshold,
                              settingRow(EditableSetting::CandidateContextThreshold, margin, candidateY, candidateColumnWidth));
            drawSettingSlider(g, EditableSetting::CandidateTopDownThreshold,
                              layerIndex, policy.topDownThreshold,
                              settingRow(EditableSetting::CandidateTopDownThreshold, margin, candidateY, candidateColumnWidth));
            drawSettingSlider(g, EditableSetting::CandidateObservationThreshold,
                              layerIndex, policy.observationThreshold,
                              settingRow(EditableSetting::CandidateObservationThreshold, margin, candidateY, candidateColumnWidth));
            drawSettingSlider(g, EditableSetting::CandidateUseActivationSupport,
                              layerIndex, policy.useActivationSupport ? 1.0 : 0.0,
                              settingRow(EditableSetting::CandidateUseActivationSupport, margin, candidateY, candidateColumnWidth));
            drawSettingSlider(g, EditableSetting::CandidateActivationThreshold,
                              layerIndex, policy.activationThreshold,
                              settingRow(EditableSetting::CandidateActivationThreshold, margin, candidateY, candidateColumnWidth));
            drawSettingSlider(g, EditableSetting::CandidateMaximumSelectedGenerators,
                              layerIndex,
                              static_cast<double>(policy.maximumSelectedGenerators),
                              settingRow(EditableSetting::CandidateMaximumSelectedGenerators, margin, candidateY, candidateColumnWidth));
            g.setColour(secondaryTextColour());
            g.setFont(juce::Font(juce::FontOptions(12.5f)));
            g.drawText("selected: " + selectedGeneratorText(layer.selectedGenerators),
                       margin, candidateY, candidateColumnWidth, lineHeight,
                       juce::Justification::centredLeft, false);
            candidateY += lineHeight;
            g.drawText(
                "candidate states: "
                    + juce::String(static_cast<int>(layer.candidateCount))
                    + "  log evidence: " + number(layer.logEvidence),
                margin, candidateY, candidateColumnWidth, lineHeight,
                juce::Justification::centredLeft, false);
            candidateY += lineHeight;
            if (layer.observationScores.size() > 0)
            {
                g.drawText(
                    "observation score min/mean/max: "
                    + number(layer.observationScores.minCoeff()) + " / "
                    + number(layer.observationScores.mean()) + " / "
                    + number(layer.observationScores.maxCoeff()),
                    margin, candidateY, candidateColumnWidth, lineHeight,
                    juce::Justification::centredLeft, false);
                candidateY += lineHeight;
            }
        }
        else
        {
            g.drawText(
                "M=" + juce::String(static_cast<int>(layer.candidateCount))
                    + "  selected="
                    + juce::String(static_cast<int>(layer.selectedGenerators.size()))
                    + "  logZ=" + number(layer.logEvidence),
                margin, candidateY, candidateColumnWidth, lineHeight,
                juce::Justification::centredLeft, false);
            candidateY += lineHeight;
        }

        const auto& learning = layer.learning;
        float emY = y;
        g.setColour(secondaryTextColour());
        g.setFont(juce::Font(juce::FontOptions(12.5f)));
        if (emExpanded)
        {
            drawSettingSlider(g, EditableSetting::LearningEnabled,
                              layerIndex, layer.learningEnabled ? 1.0 : 0.0,
                              settingRow(EditableSetting::LearningEnabled, emX, emY, emColumnWidth));
            drawSettingSlider(g, EditableSetting::LearningPredictionRate,
                              layerIndex, learning.predictionLearningRate,
                              settingRow(EditableSetting::LearningPredictionRate, emX, emY, emColumnWidth));
            drawSettingSlider(g, EditableSetting::LearningFilterRate,
                              layerIndex, learning.filterLearningRate,
                              settingRow(EditableSetting::LearningFilterRate, emX, emY, emColumnWidth));
            drawSettingSlider(g, EditableSetting::LearningBaseRate,
                              layerIndex, learning.baseRateLearningRate,
                              settingRow(EditableSetting::LearningBaseRate, emX, emY, emColumnWidth));
            drawSettingSlider(g, EditableSetting::LearningEpsilon,
                              layerIndex, learning.epsilon,
                              settingRow(EditableSetting::LearningEpsilon, emX, emY, emColumnWidth));
            drawSettingSlider(g, EditableSetting::LearningBinarizeObservation,
                              layerIndex, learning.binarizeObservation ? 1.0 : 0.0,
                              settingRow(EditableSetting::LearningBinarizeObservation, emX, emY, emColumnWidth));
            drawSettingSlider(g, EditableSetting::LearningObservationThreshold,
                              layerIndex, learning.observationThreshold,
                              settingRow(EditableSetting::LearningObservationThreshold, emX, emY, emColumnWidth));
            g.setColour(secondaryTextColour());
            g.setFont(juce::Font(juce::FontOptions(12.5f)));
            if (layer.learningDiagnostics.has_value())
            {
                const auto& d = *layer.learningDiagnostics;
                g.drawText(
                    "mean|max delta R: " + number(meanAbsolute(d.predictionDelta))
                    + " | " + number(maxAbsolute(d.predictionDelta)),
                    emX, emY, emColumnWidth, lineHeight,
                    juce::Justification::centredLeft, false);
                emY += lineHeight;
                g.drawText(
                    "mean |delta F|: " + number(meanAbsolute(d.filterDelta))
                    + "  mean |delta base|: "
                    + number(d.baseRateDelta.size() == 0
                        ? 0.0 : d.baseRateDelta.cwiseAbs().mean()),
                    emX, emY, emColumnWidth, lineHeight,
                    juce::Justification::centredLeft, false);
                emY += lineHeight;
            }
            else
            {
                g.drawText("diagnostics: no learning update yet",
                           emX, emY, emColumnWidth, lineHeight,
                           juce::Justification::centredLeft, false);
                emY += lineHeight;
            }
        }
        else
        {
            g.drawText(
                juce::String(layer.learningEnabled ? "enabled" : "disabled")
                    + "  etaR=" + number(learning.predictionLearningRate)
                    + " etaF=" + number(learning.filterLearningRate),
                       emX, emY, emColumnWidth, lineHeight,
                       juce::Justification::centredLeft, false);
            emY += lineHeight;
        }

        y = std::max(candidateY, emY) + sectionGap;

        if (emExpanded && layer.learningDiagnostics.has_value())
        {
            const auto& d = *layer.learningDiagnostics;
            const Eigen::Index deltaFRows =
                static_cast<Eigen::Index>(d.filterDelta.size());
            const Eigen::Index deltaFColumns = d.filterDelta.empty()
                ? 0
                : d.filterDelta.front().rows() * d.filterDelta.front().cols();
            const auto deltaWidths = compactMatrixWidths(
                d.predictionDelta.cols(), deltaFColumns, contentWidth,
                maximumDeltaCell, maximumDeltaCell);
            const float deltaRWidth = deltaWidths.first;
            const float deltaFWidth = deltaWidths.second;
            const float deltaAreaHeight = std::max(
                matrixHeightForShape(
                    d.predictionDelta.rows(),
                    d.predictionDelta.cols(),
                    deltaRWidth,
                    maximumDeltaCell),
                matrixHeightForShape(
                    deltaFRows,
                    deltaFColumns,
                    deltaFWidth,
                    maximumDeltaCell));
            const float deltaBlockHeight =
                titleHeight + titleHeight + deltaAreaHeight + sectionGap;

            if (!intersectsClip(
                    g,
                    { margin, y, contentWidth, deltaBlockHeight }))
            {
                y += deltaBlockHeight;
            }
            else
            {
                const Eigen::MatrixXd deltaF = flattenFilterDelta(d.filterDelta);
                const float deltaFX = margin + deltaRWidth + compactMatrixGap;

                g.setColour(primaryTextColour());
                g.setFont(juce::Font(juce::FontOptions(14.0f)));
                g.drawText("Online EM update matrices",
                           margin, y, contentWidth, titleHeight,
                           juce::Justification::centredLeft);
                y += titleHeight;

                g.setColour(secondaryTextColour());
                g.setFont(juce::Font(juce::FontOptions(12.5f)));
                g.drawText("delta R [K x N], max |delta| = "
                               + number(maxAbsolute(d.predictionDelta)),
                           margin, y, deltaRWidth, titleHeight,
                           juce::Justification::centredLeft, false);
                g.drawText("delta F [K x (N*order)], max |delta| = "
                               + number(maxAbsolute(deltaF)),
                           deltaFX, y, deltaFWidth, titleHeight,
                           juce::Justification::centredLeft, false);
                y += titleHeight;

                if (d.predictionDelta.size() > 0)
                {
                    drawSignedMatrix(
                        g, d.predictionDelta,
                        { margin, y, deltaRWidth, deltaAreaHeight },
                        maximumDeltaCell);
                }

                if (deltaF.size() > 0)
                {
                    drawSignedMatrix(
                        g, deltaF,
                        { deltaFX, y, deltaFWidth, deltaAreaHeight },
                        maximumDeltaCell);
                }

                y += deltaAreaHeight + sectionGap;
            }
        }

        if (layer.hasInference)
        {
            const Eigen::Index candidateRowCount = std::min<Eigen::Index>(
                layer.candidatePosterior.size(),
                maximumDisplayedCandidates);
            const float candidateVectorHeight = rowMatrixHeight(
                candidateRowCount,
                contentWidth,
                maximumProbabilityCell);
            const float candidateProbabilitiesHeight =
                titleHeight
                + lineHeight
                + 3.0f * (lineHeight
                    + (candidateRowCount > 0
                        ? candidateVectorHeight + 4.0f
                        : 0.0f))
                + (candidateRowCount > 0 ? lineHeight : 0.0f)
                + sectionGap;

            if (!intersectsClip(
                    g,
                    { margin, y, contentWidth, candidateProbabilitiesHeight }))
            {
                y += candidateProbabilitiesHeight;
                continue;
            }

            const auto candidateRows = sortedCandidateRows(
                layer, maximumDisplayedCandidates);

            g.setColour(primaryTextColour());
            g.setFont(juce::Font(juce::FontOptions(14.0f)));
            g.drawText("Candidate probability vectors",
                       margin, y, contentWidth, titleHeight,
                       juce::Justification::centredLeft);
            y += titleHeight;

            g.setColour(secondaryTextColour());
            g.setFont(juce::Font(juce::FontOptions(12.5f)));
            juce::String summary;
            summary << "vectors are sorted by posterior; showing "
                    << static_cast<int>(candidateRows.size())
                    << " of "
                    << static_cast<int>(layer.candidatePosterior.size())
                    << " candidates";
            g.drawText(summary, margin, y, contentWidth, lineHeight,
                       juce::Justification::centredLeft, false);
            y += lineHeight;

            const Eigen::MatrixXd priorRow = orderedVectorAsRowMatrix(
                layer.candidatePrior, candidateRows);
            const Eigen::MatrixXd likelihoodRow = orderedVectorAsRowMatrix(
                layer.candidateLikelihood, candidateRows);
            const Eigen::MatrixXd posteriorRow = orderedVectorAsRowMatrix(
                layer.candidatePosterior, candidateRows);

            struct NamedVector
            {
                const char* label;
                const Eigen::MatrixXd& row;
            };

            const NamedVector vectors[] {
                { "prior", priorRow },
                { "likelihood", likelihoodRow },
                { "posterior", posteriorRow }
            };

            for (const auto& item : vectors)
            {
                g.setColour(secondaryTextColour());
                g.setFont(juce::Font(juce::FontOptions(12.5f)));
                g.drawText(juce::String(item.label) + " [1 x M]",
                           margin, y, contentWidth, lineHeight,
                           juce::Justification::centredLeft, false);
                y += lineHeight;

                if (item.row.size() > 0)
                {
                    const float vectorHeight = matrixHeight(
                        item.row, contentWidth, maximumProbabilityCell);
                    const auto drawn = drawMatrix(
                        g, item.row, { margin, y, contentWidth, vectorHeight },
                        maximumProbabilityCell);
                    y += drawn.getHeight() + 4.0f;
                }
            }

            if (!candidateRows.empty())
            {
                g.setColour(secondaryTextColour());
                g.setFont(juce::Font(juce::FontOptions(12.0f)));
                g.drawText(candidateOrderText(layer, candidateRows),
                           margin, y, contentWidth, lineHeight,
                           juce::Justification::centredLeft, false);
                y += lineHeight;
            }

            y += sectionGap;
        }
    }

    activePaintOffsetX_ = 0.0f;
}
