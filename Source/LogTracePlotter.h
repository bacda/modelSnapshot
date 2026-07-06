#pragma once

#include <JuceHeader.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class LogTracePlotter : public juce::Component,
                        private juce::Button::Listener,
                        private juce::ListBoxModel
{
public:
    LogTracePlotter();
    ~LogTracePlotter() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    bool loadFile(const juce::File& file);

private:
    struct SeriesSpec
    {
        juce::String path;
        juce::String displayName;
        int vectorLength = 1;
        bool enabled = false;
        bool probabilityLike = false;
        double minimum = 0.0;
        double maximum = 1.0;
        std::vector<std::vector<double>> values; // [time][cell]
    };

    struct TimeSlice
    {
        juce::String step;
        juce::String inputRow;
    };

    struct SeriesGroup
    {
        juce::String key;
        juce::String displayName;
        std::vector<int> seriesIndices;
        bool expanded = false;
    };

    struct VisibleSelectorRow
    {
        bool isGroup = true;
        int groupIndex = -1;
        int seriesIndex = -1;
    };

    class PlotCanvas : public juce::Component
    {
    public:
        explicit PlotCanvas(LogTracePlotter& ownerRef);

        void paint(juce::Graphics& g) override;
        std::pair<int, int> preferredSize() const;

    private:
        LogTracePlotter& owner;
    };

    class SeriesRowComponent : public juce::Component
    {
    public:
        explicit SeriesRowComponent(LogTracePlotter& ownerRef);

        void configureGroup(
            int rowIndex,
            const juce::String& text,
            bool checked,
            bool expanded);
        void configureSeries(
            int rowIndex,
            const juce::String& text,
            bool checked);
        void resized() override;

        juce::TextButton disclosure;
        juce::ToggleButton toggle;
        juce::Label label;
        int row = -1;
        bool group = false;

    private:
        LogTracePlotter& owner;
    };

    using RowSeries = std::map<std::string, std::vector<double>>;

    void loadLogFile();
    void exportCheckedFields();
    void saveStartingState();
    bool writeCheckedFieldExport(const juce::File& file);
    bool writeStartingStateFile(const juce::File& file);
    bool findInitialState(juce::var& stateOut) const;
    bool parseJsonLog(const juce::File& file);
    void clearData();
    void ingestStepObject(const juce::var& stepObject);
    void addSeriesIfNeeded(const juce::String& path, int vectorLength);
    int findSeriesIndex(const juce::String& path) const;
    void rebuildGroups();
    void rebuildVisibleRows();
    void recomputeSeriesRanges();
    void refreshCanvasSize();
    void adjustTimeZoom(int delta);
    void adjustCellZoom(int delta);

    static void collectNumericSeries(
        const juce::var& node,
        const juce::String& pathPrefix,
        RowSeries& rowSeries,
        juce::StringArray& discoveryOrder);

    static bool shouldEnableByDefault(const juce::String& path);
    static bool isProbabilityLikePath(const juce::String& path);
    static juce::String groupKeyForPath(const juce::String& path);
    static juce::String groupNameForKey(const juce::String& key);
    static juce::String prettyNameForPath(const juce::String& path);
    static juce::String compactNumber(double value);

    bool isGroupFullyEnabled(int groupIndex) const;
    void setGroupEnabled(int groupIndex, bool enabled);
    void toggleGroupExpanded(int groupIndex);

    double normalizedValue(const SeriesSpec& spec, double value) const;
    juce::Colour colourForValue(const SeriesSpec& spec, double value) const;

    int enabledSeriesCount() const;
    int plotHeight() const;
    int plotWidth() const;

    int getNumRows() override;
    void paintListBoxItem(
        int row,
        juce::Graphics& g,
        int width,
        int height,
        bool rowSelected) override;
    juce::Component* refreshComponentForRow(
        int row,
        bool rowSelected,
        juce::Component* existingComponentToUpdate) override;
    void buttonClicked(juce::Button* button) override;

    juce::TextButton loadButton { "Load Log" };
    juce::ToggleButton showValuesButton { "Values" };
    juce::TextButton timeZoomOutButton { "Time -" };
    juce::TextButton timeZoomInButton { "Time +" };
    juce::TextButton cellZoomOutButton { "Cells -" };
    juce::TextButton cellZoomInButton { "Cells +" };
    juce::TextButton exportButton { "Export Checked" };
    juce::TextButton saveStartStateButton { "Save Starting .state" };
    juce::Label fileLabel;
    juce::ListBox seriesList { "series", this };
    juce::Viewport viewport;
    PlotCanvas canvas;

    std::unique_ptr<juce::FileChooser> fileChooser;
    juce::File currentFile;
    std::vector<SeriesSpec> series;
    std::vector<SeriesGroup> groups;
    std::vector<VisibleSelectorRow> visibleRows;
    std::vector<TimeSlice> timeSlices;

    int leftPanelWidth = 330;
    int labelGutterWidth = 190;
    int columnWidth = 22;
    int cellSize = 13;
    int cellGap = 1;
    int seriesGap = 12;
    int headerHeight = 46;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LogTracePlotter)
};
