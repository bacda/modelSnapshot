#pragma once

#include <JuceHeader.h>
#include "NoisyOR.h"

#include <functional>
#include <limits>
#include <optional>
#include <vector>

class LayerStateDisplay : public juce::Component
{
public:
    enum class EditableMatrix
    {
        Input,
        Predictions,
        Filters
    };

    enum class EditableParameter
    {
        BaseRate,
        BottomUpWeight,
        EvidenceAmplitude,
        Centering
    };

    enum class EditableSetting
    {
        CandidateContextThreshold,
        CandidateTopDownThreshold,
        CandidateObservationThreshold,
        CandidateUseActivationSupport,
        CandidateActivationThreshold,
        CandidateMaximumSelectedGenerators,

        LearningEnabled,
        LearningPredictionRate,
        LearningFilterRate,
        LearningBaseRate,
        LearningEpsilon,
        LearningBinarizeObservation,
        LearningObservationThreshold
    };

    struct MatrixEdit
    {
        EditableMatrix matrix = EditableMatrix::Input;
        std::size_t layerIndex = 0;
        Eigen::Index row = 0;
        Eigen::Index column = 0;
        double value = 0.0;
    };

    struct ParameterEdit
    {
        EditableParameter parameter = EditableParameter::BaseRate;
        std::size_t layerIndex = 0;
        Eigen::Index generator = 0;
        double value = 0.0;
        bool applyToAllGenerators = false;
    };

    struct SettingEdit
    {
        EditableSetting setting = EditableSetting::CandidateContextThreshold;
        std::size_t layerIndex = 0;
        double value = 0.0;
    };

    struct RandomizeRequest
    {
        EditableMatrix matrix = EditableMatrix::Predictions;
        std::size_t layerIndex = 0;
        double betaShapeA = 0.05;
        double betaShapeB = 1.0;
    };

    std::function<void(const MatrixEdit&)> onMatrixEdit;
    std::function<void(const ParameterEdit&)> onParameterEdit;
    std::function<void(const SettingEdit&)> onSettingEdit;
    std::function<void(const RandomizeRequest&)> onRandomizeRequest;

    LayerStateDisplay();

    void update(
        const noisy_or::ModelState& state,
        const noisy_or::NoisyORStack& model,
        std::optional<std::size_t> currentTimestep,
        const std::vector<std::optional<noisy_or::LearningDiagnostics>>&
            learningDiagnostics);

    int preferredHeight(int width) const;
    int preferredWidth() const;
    void paint(juce::Graphics& g) override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;

private:
    struct LayerSnapshot
    {
        Eigen::MatrixXd R;
        Eigen::MatrixXd F;
        noisy_or::GeneratorParameters parameters;
        noisy_or::CandidateSelectionPolicy candidatePolicy;
        noisy_or::OnlineEMOptions learning;
        bool learningEnabled = false;

        Eigen::VectorXd alpha;
        Eigen::VectorXd inheritedTopDownSupport;
        Eigen::VectorXd inheritedAlpha;
        Eigen::VectorXd marginals;
        Eigen::VectorXd observationScores;
        std::vector<Eigen::Index> selectedGenerators;
        std::vector<noisy_or::State> candidates;
        Eigen::VectorXd candidatePrior;
        Eigen::VectorXd candidateLikelihood;
        Eigen::VectorXd candidatePosterior;
        std::size_t candidateCount = 0;
        double logEvidence = 0.0;
        bool hasInference = false;

        std::optional<noisy_or::LearningDiagnostics> learningDiagnostics;
    };

    struct MatrixRegion
    {
        EditableMatrix matrix = EditableMatrix::Input;
        std::size_t layerIndex = 0;
        juce::Rectangle<float> bounds;
        float cellSize = 1.0f;
        Eigen::Index rows = 0;
        Eigen::Index columns = 0;
    };

    struct ParameterRegion
    {
        EditableParameter parameter = EditableParameter::BaseRate;
        std::size_t layerIndex = 0;
        Eigen::Index generator = 0;
        juce::Rectangle<float> bounds;
        double minimum = 0.0;
        double maximum = 1.0;
        double dragSensitivity = 1.0 / 120.0;
    };

    struct SettingRegion
    {
        EditableSetting setting = EditableSetting::CandidateContextThreshold;
        std::size_t layerIndex = 0;
        juce::Rectangle<float> bounds;
        double minimum = 0.0;
        double maximum = 1.0;
        double dragSensitivity = 1.0 / 120.0;
        bool integer = false;
        bool toggle = false;
    };

    struct MatrixDragState
    {
        MatrixRegion region;
        Eigen::Index row = 0;
        Eigen::Index column = 0;
        double originalValue = 0.0;
        bool fractionalDragStarted = false;
    };

    struct ParameterDragState
    {
        ParameterRegion region;
        double originalValue = 0.0;
        bool dragStarted = false;
    };

    struct SettingDragState
    {
        SettingRegion region;
        double originalValue = 0.0;
        bool dragStarted = false;
    };

    enum class RandomControl
    {
        Randomize,
        ToggleSettings
    };

    enum class CollapsibleSection
    {
        CandidateSelection,
        OnlineEM
    };

    struct RandomControlState
    {
        bool showGenerationSettings = false;
        double predictionAlpha = 5.;
        double predictionBeta = 5.;
        double filterAlpha = 5.;
        double filterBeta = 5.;
    };

    struct RandomControlRegion
    {
        RandomControl control = RandomControl::Randomize;
        EditableMatrix matrix = EditableMatrix::Predictions;
        std::size_t layerIndex = 0;
        juce::Rectangle<float> bounds;
    };

    struct SectionState
    {
        bool candidateSelectionExpanded = false;
        bool onlineEMExpanded = false;
    };

    struct SectionRegion
    {
        CollapsibleSection section = CollapsibleSection::CandidateSelection;
        std::size_t layerIndex = 0;
        juce::Rectangle<float> bounds;
    };

    Eigen::MatrixXd inputWindow_; // channels x displayed timesteps
    std::vector<std::size_t> inputIndices_;
    std::optional<std::size_t> highlightedInputColumn_;
    std::vector<LayerSnapshot> layers_;
    std::vector<RandomControlState> randomControls_;
    std::vector<SectionState> sectionStates_;
    std::vector<MatrixRegion> matrixRegions_;
    std::vector<ParameterRegion> parameterRegions_;
    std::vector<SettingRegion> settingRegions_;
    std::vector<RandomControlRegion> randomControlRegions_;
    std::vector<SectionRegion> sectionRegions_;
    std::optional<MatrixDragState> matrixDragState_;
    std::optional<ParameterDragState> parameterDragState_;
    std::optional<SettingDragState> settingDragState_;
    float activePaintOffsetX_ = 0.0f;

    static juce::String number(double value);
    static juce::String cellLabel(double value);
    static juce::String signedCellLabel(double value);
    static float matrixHeight(
        const Eigen::MatrixXd& matrix,
        float availableWidth,
        float maximumCellSize);
    static juce::Rectangle<float> drawMatrix(
        juce::Graphics& g,
        const Eigen::MatrixXd& matrix,
        juce::Rectangle<float> area,
        float maximumCellSize,
        std::optional<std::size_t> highlightedColumn = std::nullopt);
    static juce::Rectangle<float> drawSignedMatrix(
        juce::Graphics& g,
        const Eigen::MatrixXd& matrix,
        juce::Rectangle<float> area,
        float maximumCellSize);
    static juce::String selectedGeneratorText(
        const std::vector<Eigen::Index>& generators);
    static juce::String parameterName(EditableParameter parameter);
    static double parameterMinimum(EditableParameter parameter);
    static double parameterMaximum(EditableParameter parameter);
    static double parameterDragSensitivity(EditableParameter parameter);
    static juce::String settingName(EditableSetting setting);
    static double settingMinimum(EditableSetting setting);
    static double settingMaximum(EditableSetting setting);
    static double settingVisualMaximum(EditableSetting setting, double value);
    static double settingDragSensitivity(EditableSetting setting);
    static bool settingIsInteger(EditableSetting setting);
    static bool settingIsToggle(EditableSetting setting);
    static juce::String settingValueText(EditableSetting setting, double value);
    static juce::String stateActiveSetText(
        noisy_or::State state,
        Eigen::Index generatorCount);
    static juce::String vectorText(
        const Eigen::VectorXd& vector,
        Eigen::Index maximumEntries = 16);
    static Eigen::MatrixXd vectorAsRowMatrix(
        const Eigen::VectorXd& vector,
        Eigen::Index maximumEntries = std::numeric_limits<Eigen::Index>::max());
    static Eigen::MatrixXd orderedVectorAsRowMatrix(
        const Eigen::VectorXd& vector,
        const std::vector<Eigen::Index>& order);
    static std::vector<Eigen::Index> sortedCandidateRows(
        const LayerSnapshot& layer,
        Eigen::Index maximumRows);
    static Eigen::MatrixXd flattenFilterDelta(
        const std::vector<Eigen::MatrixXd>& filterDelta);
    static juce::String candidateOrderText(
        const LayerSnapshot& layer,
        const std::vector<Eigen::Index>& order,
        Eigen::Index maximumEntries = 16);

    juce::Rectangle<float> drawEditableMatrix(
        juce::Graphics& g,
        EditableMatrix matrixType,
        std::size_t layerIndex,
        const Eigen::MatrixXd& matrix,
        juce::Rectangle<float> area,
        float maximumCellSize,
        std::optional<std::size_t> highlightedColumn = std::nullopt);

    juce::Rectangle<float> drawParameterSlider(
        juce::Graphics& g,
        EditableParameter parameter,
        std::size_t layerIndex,
        Eigen::Index generator,
        double value,
        juce::Rectangle<float> bounds);

    juce::Rectangle<float> drawSettingSlider(
        juce::Graphics& g,
        EditableSetting setting,
        std::size_t layerIndex,
        double value,
        juce::Rectangle<float> bounds);
    juce::Rectangle<float> drawSmallButton(
        juce::Graphics& g,
        const juce::String& text,
        juce::Rectangle<float> bounds,
        bool active);
    float drawGenerationControls(
        juce::Graphics& g,
        std::size_t layerIndex,
        juce::Rectangle<float> predictionArea,
        juce::Rectangle<float> filterArea,
        juce::Rectangle<float> fullArea,
        float y);

    std::optional<MatrixRegion> findMatrixRegion(
        juce::Point<float> point) const;
    std::optional<ParameterRegion> findParameterRegion(
        juce::Point<float> point) const;
    std::optional<SettingRegion> findSettingRegion(
        juce::Point<float> point) const;
    std::optional<RandomControlRegion> findRandomControlRegion(
        juce::Point<float> point) const;
    std::optional<SectionRegion> findSectionRegion(
        juce::Point<float> point) const;
    Eigen::Index cellRowAt(const MatrixRegion& region, float y) const;
    Eigen::Index cellColumnAt(const MatrixRegion& region, float x) const;
    double matrixValueAt(
        EditableMatrix matrix,
        std::size_t layerIndex,
        Eigen::Index row,
        Eigen::Index column) const;
    double parameterValueAt(
        EditableParameter parameter,
        std::size_t layerIndex,
        Eigen::Index generator) const;
    double settingValueAt(
        EditableSetting setting,
        std::size_t layerIndex) const;
    void sendRandomizeRequest(
        EditableMatrix matrix,
        std::size_t layerIndex);
    void editGenerationSettingsAsText(std::size_t layerIndex);
    bool isSectionExpanded(
        CollapsibleSection section,
        std::size_t layerIndex) const;
    void toggleSection(
        CollapsibleSection section,
        std::size_t layerIndex);
    juce::Rectangle<float> drawSectionToggle(
        juce::Graphics& g,
        CollapsibleSection section,
        std::size_t layerIndex,
        juce::Rectangle<float> headerBounds);
    void applyEditFromMouse(
        const MatrixRegion& region,
        Eigen::Index row,
        Eigen::Index column,
        double value);
    void applyParameterEditFromMouse(
        const ParameterRegion& region,
        double value);
    void applySettingEditFromMouse(
        const SettingRegion& region,
        double value);
};
