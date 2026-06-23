#include "NoisyORStateIO.h"

#include "NoisyORUtilities.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace noisy_or {
namespace {

struct Dimensions {
    Eigen::Index N = 0;
    Eigen::Index K = 0;
    Eigen::Index order = 0;
};

struct BetaSpecification {
    double shapeA = 0.05;
    double shapeB = 1.0;
    std::uint64_t seed = 0;
};

struct MatrixSource {
    enum class Kind { Matrix, Random };
    Kind kind = Kind::Matrix;
    Eigen::MatrixXd matrix;
    BetaSpecification beta;
};

struct LayerPatch {
    bool mentioned = false;
    std::optional<Dimensions> dimensions;
    std::optional<MatrixSource> predictions;
    std::optional<MatrixSource> filters;
    std::optional<Eigen::MatrixXd> leak;
    std::optional<Eigen::MatrixXd> parameters;
    std::optional<Eigen::MatrixXd> context;
    std::optional<Eigen::MatrixXd> topDownSupport;
    std::optional<CandidateSelectionPolicy> candidateSelection;
    std::optional<OnlineEMOptions> learning;
    std::optional<bool> learningEnabled;
};

struct FilePatch {
    std::optional<Eigen::MatrixXd> input;
    std::optional<bool> loopInput;
    std::optional<std::size_t> inputIndex;
    std::optional<std::size_t> layerCount;
    std::vector<LayerPatch> layers;
};

std::string parentPath(const std::string& path)
{
    const std::string::size_type separator = path.find_last_of("/\\");
    if (separator == std::string::npos) return {};
    if (separator == 0) return path.substr(0, 1);
    return path.substr(0, separator);
}

bool isAbsolutePath(const std::string& path)
{
    if (path.empty()) return false;
    if (path.front() == '/' || path.front() == '\\') return true;
    return path.size() >= 3
        && std::isalpha(static_cast<unsigned char>(path[0]))
        && path[1] == ':'
        && (path[2] == '/' || path[2] == '\\');
}

std::string joinPath(const std::string& baseDirectory, const std::string& path)
{
    if (baseDirectory.empty() || isAbsolutePath(path)) return path;
    const char last = baseDirectory.back();
    if (last == '/' || last == '\\') return baseDirectory + path;
    return baseDirectory + '/' + path;
}

std::vector<std::string> tokenize(const std::string& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "Could not open state file: " + path);
    }

    std::vector<std::string> tokens;
    std::string line;
    while (std::getline(input, line)) {
        std::string token;
        bool quoted = false;
        bool escaping = false;

        auto flush = [&]() {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        };

        for (const char character : line) {
            if (escaping) {
                token.push_back(character);
                escaping = false;
            } else if (quoted && character == '\\') {
                escaping = true;
            } else if (character == '"') {
                quoted = !quoted;
            } else if (!quoted && character == '#') {
                break;
            } else if (!quoted && std::isspace(
                       static_cast<unsigned char>(character))) {
                flush();
            } else {
                token.push_back(character);
            }
        }

        if (quoted) {
            throw std::runtime_error(
                "Unterminated quote in state file: " + path);
        }
        flush();
    }

    return tokens;
}

class TokenReader {
public:
    TokenReader(std::vector<std::string> tokens,
                std::string source)
        : tokens_(std::move(tokens)), source_(std::move(source)) {}

    bool empty() const noexcept { return position_ >= tokens_.size(); }

    const std::string& peek() const
    {
        if (empty()) fail("Unexpected end of file");
        return tokens_[position_];
    }

    std::string take()
    {
        const std::string value = peek();
        ++position_;
        return value;
    }

    double number(const char* name)
    {
        const std::string token = take();
        std::size_t consumed = 0;
        try {
            const double value = std::stod(token, &consumed);
            if (consumed != token.size() || !std::isfinite(value)) {
                fail(std::string("Invalid ") + name + ": " + token);
            }
            return value;
        } catch (const std::exception&) {
            fail(std::string("Invalid ") + name + ": " + token);
        }
    }

    std::size_t size(const char* name)
    {
        const std::string token = take();
        if (!token.empty() && token.front() == '-') {
            fail(std::string("Invalid ") + name + ": " + token);
        }
        std::size_t consumed = 0;
        try {
            const unsigned long long value = std::stoull(token, &consumed);
            if (consumed != token.size()) {
                fail(std::string("Invalid ") + name + ": " + token);
            }
            return static_cast<std::size_t>(value);
        } catch (const std::exception&) {
            fail(std::string("Invalid ") + name + ": " + token);
        }
    }

    bool boolean(const char* name)
    {
        const std::string token = take();
        if (token == "1" || token == "true" || token == "TRUE") return true;
        if (token == "0" || token == "false" || token == "FALSE") return false;
        fail(std::string("Invalid ") + name + ": " + token);
    }

    [[noreturn]] void fail(const std::string& message) const
    {
        throw std::runtime_error(
            source_ + " near token "
            + std::to_string(position_ + 1) + ": " + message);
    }

private:
    std::vector<std::string> tokens_;
    std::string source_;
    std::size_t position_ = 0;
};

Eigen::MatrixXd readInlineMatrix(TokenReader& reader)
{
    const std::size_t rows = reader.size("matrix row count");
    const std::size_t columns = reader.size("matrix column count");
    Eigen::MatrixXd matrix(
        static_cast<Eigen::Index>(rows),
        static_cast<Eigen::Index>(columns));

    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
            matrix(row, column) = reader.number("matrix value");
        }
    }
    return matrix;
}

Eigen::MatrixXd readExternalMatrix(
    const std::string& path,
    std::size_t rows,
    std::size_t columns)
{
    TokenReader reader(tokenize(path), path);
    Eigen::MatrixXd matrix(
        static_cast<Eigen::Index>(rows),
        static_cast<Eigen::Index>(columns));

    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
            matrix(row, column) = reader.number("external matrix value");
        }
    }
    if (!reader.empty()) {
        reader.fail("External matrix contains more values than declared");
    }
    return matrix;
}

Eigen::MatrixXd readMatrixValue(
    TokenReader& reader,
    const std::string& baseDirectory)
{
    const std::string mode = reader.take();
    if (mode == "INLINE") {
        return readInlineMatrix(reader);
    }
    if (mode == "FILE") {
        std::string path = joinPath(baseDirectory, reader.take());
        const std::size_t rows = reader.size("matrix row count");
        const std::size_t columns = reader.size("matrix column count");
        return readExternalMatrix(path, rows, columns);
    }
    reader.fail("Expected INLINE or FILE, received " + mode);
}

MatrixSource readParameterSource(
    TokenReader& reader,
    const std::string& baseDirectory)
{
    const std::string mode = reader.peek();
    if (mode == "INLINE" || mode == "FILE") {
        MatrixSource source;
        source.kind = MatrixSource::Kind::Matrix;
        source.matrix = readMatrixValue(reader, baseDirectory);
        return source;
    }
    if (mode == "RANDOM") {
        reader.take();
        MatrixSource source;
        source.kind = MatrixSource::Kind::Random;
        source.beta.shapeA = reader.number("Beta shapeA");
        source.beta.shapeB = reader.number("Beta shapeB");
        source.beta.seed = static_cast<std::uint64_t>(
            reader.size("random seed"));
        return source;
    }
    reader.fail("Expected INLINE, FILE, or RANDOM, received " + mode);
}

FilePatch parsePatch(const std::string& path)
{
    TokenReader reader(tokenize(path), path);
    FilePatch patch;
    std::optional<std::size_t> currentLayer;
    const std::string baseDirectory = parentPath(path);

    if (!reader.empty() && reader.peek() == "NOISY_OR_STATE") {
        reader.take();
        const std::size_t version = reader.size("format version");
        if (version != 1) reader.fail("Unsupported format version");
    }

    while (!reader.empty()) {
        const std::string directive = reader.take();

        if (directive == "INPUT") {
            patch.input = readMatrixValue(reader, baseDirectory);
        } else if (directive == "LOOP") {
            patch.loopInput = reader.boolean("LOOP value");
        } else if (directive == "INPUT_INDEX") {
            patch.inputIndex = reader.size("INPUT_INDEX value");
        } else if (directive == "LAYERS") {
            patch.layerCount = reader.size("layer count");
            patch.layers.resize(*patch.layerCount);
        } else if (directive == "LAYER") {
            currentLayer = reader.size("layer index");
            if (patch.layers.size() <= *currentLayer) {
                patch.layers.resize(*currentLayer + 1);
            }
            patch.layers[*currentLayer].mentioned = true;
        } else if (directive == "END_LAYER") {
            currentLayer.reset();
        } else {
            if (!currentLayer) {
                reader.fail("Layer directive outside a LAYER block: " + directive);
            }
            LayerPatch& layer = patch.layers[*currentLayer];

            if (directive == "DIMENSIONS") {
                Dimensions dimensions;
                dimensions.N = static_cast<Eigen::Index>(reader.size("N"));
                dimensions.K = static_cast<Eigen::Index>(reader.size("K"));
                dimensions.order = static_cast<Eigen::Index>(
                    reader.size("context order"));
                layer.dimensions = dimensions;
            } else if (directive == "R") {
                layer.predictions = readParameterSource(reader, baseDirectory);
            } else if (directive == "F") {
                layer.filters = readParameterSource(reader, baseDirectory);
            } else if (directive == "LEAK") {
                layer.leak = readMatrixValue(reader, baseDirectory);
            } else if (directive == "PARAMETERS") {
                layer.parameters = readMatrixValue(reader, baseDirectory);
            } else if (directive == "CONTEXT") {
                layer.context = readMatrixValue(reader, baseDirectory);
            } else if (directive == "TOP_DOWN") {
                layer.topDownSupport = readMatrixValue(reader, baseDirectory);
            } else if (directive == "CANDIDATE_SELECTION") {
                CandidateSelectionPolicy value;
                value.contextThreshold = reader.number("context threshold");
                value.topDownThreshold = reader.number("top-down threshold");
                value.observationThreshold = reader.number("observation threshold");
                value.activationThreshold = reader.number("activation threshold");
                value.useActivationSupport = reader.boolean("activation support flag");
                value.maximumSelectedGenerators =
                    reader.size("maximum selected generators");
                layer.candidateSelection = value;
            } else if (directive == "EM") {
                layer.learningEnabled = reader.boolean("EM enabled flag");
                OnlineEMOptions value;
                value.predictionLearningRate = reader.number("R learning rate");
                value.filterLearningRate = reader.number("F learning rate");
                value.baseRateLearningRate = reader.number("base learning rate");
                value.epsilon = reader.number("EM epsilon");
                value.binarizeObservation = reader.boolean("binarize flag");
                value.observationThreshold = reader.number("binarization threshold");
                layer.learning = value;
            } else {
                reader.fail("Unknown directive: " + directive);
            }
        }
    }

    return patch;
}

double sampleBeta(
    std::mt19937_64& generator,
    double shapeA,
    double shapeB)
{
    if (!(shapeA > 0.0) || !(shapeB > 0.0) ||
        !std::isfinite(shapeA) || !std::isfinite(shapeB)) {
        throw std::invalid_argument("Beta shapes must be finite and positive");
    }

    std::gamma_distribution<double> distributionA(shapeA, 1.0);
    std::gamma_distribution<double> distributionB(shapeB, 1.0);
    for (int attempt = 0; attempt < 100; ++attempt) {
        const double a = distributionA(generator);
        const double b = distributionB(generator);
        if (a + b > 0.0 && std::isfinite(a + b)) {
            return std::max(
                std::numeric_limits<double>::min(), a / (a + b));
        }
    }
    return shapeA / (shapeA + shapeB);
}

Eigen::MatrixXd randomPredictions(
    Eigen::Index K,
    Eigen::Index N,
    const BetaSpecification& specification)
{
    std::mt19937_64 generator(specification.seed);
    Eigen::MatrixXd result(K, N);
    for (Eigen::Index k = 0; k < K; ++k) {
        for (Eigen::Index i = 0; i < N; ++i) {
            result(k, i) = sampleBeta(
                generator, specification.shapeA, specification.shapeB);
        }
    }
    return result;
}

std::vector<Eigen::MatrixXd> randomFilters(
    Eigen::Index K,
    Eigen::Index N,
    Eigen::Index order,
    const BetaSpecification& specification)
{
    std::mt19937_64 generator(specification.seed);
    std::vector<Eigen::MatrixXd> result(static_cast<std::size_t>(K));

    for (Eigen::Index k = 0; k < K; ++k) {
        auto& filter = result[static_cast<std::size_t>(k)];
        filter.resize(N, order);
        for (Eigen::Index i = 0; i < N; ++i) {
            for (Eigen::Index lag = 0; lag < order; ++lag) {
                filter(i, lag) = sampleBeta(
                    generator, specification.shapeA, specification.shapeB);
            }
        }

        const double sum = filter.sum();
        if (sum <= probabilityEpsilon) {
            filter.setConstant(1.0 / static_cast<double>(N * order));
        } else {
            filter /= sum;
        }
    }
    return result;
}

std::vector<Eigen::MatrixXd> unpackFilters(
    const Eigen::MatrixXd& flattened,
    Eigen::Index K,
    Eigen::Index N,
    Eigen::Index order)
{
    if (flattened.rows() != K * N || flattened.cols() != order) {
        throw std::invalid_argument(
            "F must have flattened shape (K*N) x contextOrder");
    }

    std::vector<Eigen::MatrixXd> filters(static_cast<std::size_t>(K));
    for (Eigen::Index k = 0; k < K; ++k) {
        filters[static_cast<std::size_t>(k)] =
            flattened.middleRows(k * N, N);
    }
    return filters;
}

Eigen::VectorXd asVector(
    const Eigen::MatrixXd& matrix,
    Eigen::Index expectedSize,
    const char* name)
{
    if (matrix.size() != expectedSize ||
        (matrix.rows() != 1 && matrix.cols() != 1)) {
        throw std::invalid_argument(
            std::string(name) + " must be a row or column vector of length "
            + std::to_string(expectedSize));
    }
    return Eigen::Map<const Eigen::VectorXd>(matrix.data(), matrix.size());
}

GeneratorParameters unpackParameters(
    const Eigen::MatrixXd& matrix,
    Eigen::Index K)
{
    if (matrix.rows() != K || matrix.cols() != 4) {
        throw std::invalid_argument("PARAMETERS must have shape K x 4");
    }
    GeneratorParameters result;
    result.baseRate = matrix.col(0);
    result.bottomUpWeight = matrix.col(1);
    result.evidenceAmplitude = matrix.col(2);
    result.centering = matrix.col(3);
    return result;
}

bool compatibleFilters(
    const std::vector<Eigen::MatrixXd>& filters,
    Eigen::Index K,
    Eigen::Index N,
    Eigen::Index order)
{
    if (static_cast<Eigen::Index>(filters.size()) != K) return false;
    return std::all_of(filters.begin(), filters.end(), [&](const auto& filter) {
        return filter.rows() == N && filter.cols() == order;
    });
}

void warn(std::vector<std::string>& warnings, std::string message)
{
    warnings.push_back(std::move(message));
}

std::string layerPrefix(std::size_t index)
{
    return "Layer " + std::to_string(index) + ": ";
}

void validateLearning(const OnlineEMOptions& learning)
{
    if (!std::isfinite(learning.predictionLearningRate) ||
        !std::isfinite(learning.filterLearningRate) ||
        !std::isfinite(learning.baseRateLearningRate) ||
        learning.predictionLearningRate < 0.0 ||
        learning.filterLearningRate < 0.0 ||
        learning.baseRateLearningRate < 0.0) {
        throw std::invalid_argument(
            "EM learning rates must be finite and nonnegative");
    }
    if (!std::isfinite(learning.epsilon) ||
        learning.epsilon <= 0.0 || learning.epsilon >= 0.5) {
        throw std::invalid_argument("EM epsilon must lie in (0,0.5)");
    }
}

} // namespace

StateLoadResult loadModelState(
    const std::string& path,
    const ModelState* previous,
    const StateLoadDefaults& defaults)
{
    const FilePatch patch = parsePatch(path);
    StateLoadResult result;

    if (patch.input) {
        result.state.input = *patch.input;
    } else if (previous) {
        result.state.input = previous->input;
        warn(result.warnings, "INPUT omitted; retained previous input");
    } else {
        result.state.input.resize(0, 0);
        warn(result.warnings,
             "INPUT omitted; observations must be supplied externally");
    }

    if (patch.loopInput) {
        result.state.loopInput = *patch.loopInput;
    } else if (previous) {
        result.state.loopInput = previous->loopInput;
        warn(result.warnings, "LOOP omitted; retained previous value");
    } else {
        result.state.loopInput = true;
        warn(result.warnings, "LOOP omitted; defaulted to true");
    }

    if (patch.inputIndex) {
        result.state.inputIndex = *patch.inputIndex;
    } else if (previous) {
        result.state.inputIndex = previous->inputIndex;
        warn(result.warnings,
             "INPUT_INDEX omitted; retained previous value");
    } else {
        result.state.inputIndex = 0;
        warn(result.warnings, "INPUT_INDEX omitted; defaulted to zero");
    }

    const std::size_t inheritedLayerCount =
        previous && !previous->layers.empty()
            ? previous->layers.size()
            : std::max<std::size_t>(1, defaults.layerCount);
    const std::size_t layerCount = patch.layerCount
        ? *patch.layerCount
        : std::max(inheritedLayerCount, patch.layers.size());
    if (patch.layerCount && patch.layers.size() > layerCount) {
        throw std::invalid_argument(
            "A LAYER index exceeds the declared LAYERS count");
    }
    result.state.layers.resize(layerCount);

    if (!patch.layerCount) {
        warn(result.warnings, previous
            ? "LAYERS omitted; retained previous layer count"
            : "LAYERS omitted; used default layer count");
    }

    for (std::size_t index = 0; index < layerCount; ++index) {
        const LayerPatch emptyPatch;
        const LayerPatch& layerPatch = index < patch.layers.size()
            ? patch.layers[index] : emptyPatch;
        const LayerState* old = previous && index < previous->layers.size()
            ? &previous->layers[index] : nullptr;
        const std::string prefix = layerPrefix(index);

        Dimensions dimensions;
        if (layerPatch.dimensions) {
            dimensions = *layerPatch.dimensions;
        } else {
            if (layerPatch.predictions &&
                layerPatch.predictions->kind == MatrixSource::Kind::Matrix) {
                dimensions.K = layerPatch.predictions->matrix.rows();
                dimensions.N = layerPatch.predictions->matrix.cols();
            }
            if (layerPatch.context) {
                if (dimensions.N == 0) dimensions.N = layerPatch.context->rows();
                dimensions.order = layerPatch.context->cols();
            }
            if (layerPatch.leak && dimensions.N == 0) {
                dimensions.N = layerPatch.leak->size();
            }
            if (layerPatch.parameters && dimensions.K == 0) {
                dimensions.K = layerPatch.parameters->rows();
            }
            if (old) {
                if (dimensions.N == 0)
                    dimensions.N = old->configuration.predictions.cols();
                if (dimensions.K == 0)
                    dimensions.K = old->configuration.predictions.rows();
                if (dimensions.order == 0)
                    dimensions.order = old->configuration.initialContext.cols();
            }
            if (dimensions.N == 0) {
                if (index == 0 && result.state.input.cols() > 0) {
                    dimensions.N = result.state.input.cols();
                } else if (index > 0) {
                    dimensions.N = result.state.layers[index - 1]
                        .configuration.predictions.rows();
                }
            }
            if (dimensions.K == 0) dimensions.K = defaults.generatorCount;
            if (dimensions.order == 0) dimensions.order = defaults.contextOrder;
            warn(result.warnings, prefix
                + "DIMENSIONS omitted; inferred compatible dimensions/defaults");
        }

        if (dimensions.N <= 0 || dimensions.K <= 0 || dimensions.order <= 0) {
            throw std::invalid_argument(prefix
                + "Could not infer positive N, K, and context order");
        }

        LayerState layer;
        auto& configuration = layer.configuration;

        if (layerPatch.predictions) {
            if (layerPatch.predictions->kind == MatrixSource::Kind::Matrix) {
                configuration.predictions = layerPatch.predictions->matrix;
            } else {
                configuration.predictions = randomPredictions(
                    dimensions.K, dimensions.N, layerPatch.predictions->beta);
            }
        } else if (old &&
                   old->configuration.predictions.rows() == dimensions.K &&
                   old->configuration.predictions.cols() == dimensions.N) {
            configuration.predictions = old->configuration.predictions;
            warn(result.warnings, prefix + "R omitted; retained previous R");
        } else {
            BetaSpecification beta{
                defaults.betaShapeA,
                defaults.betaShapeB,
                defaults.randomSeed + 2 * index};
            configuration.predictions = randomPredictions(
                dimensions.K, dimensions.N, beta);
            warn(result.warnings, prefix
                + "R omitted; initialized from default Beta distribution");
        }

        if (layerPatch.filters) {
            if (layerPatch.filters->kind == MatrixSource::Kind::Matrix) {
                configuration.filters = unpackFilters(
                    layerPatch.filters->matrix,
                    dimensions.K, dimensions.N, dimensions.order);
            } else {
                configuration.filters = randomFilters(
                    dimensions.K, dimensions.N, dimensions.order,
                    layerPatch.filters->beta);
            }
        } else if (old && compatibleFilters(
                       old->configuration.filters,
                       dimensions.K, dimensions.N, dimensions.order)) {
            configuration.filters = old->configuration.filters;
            warn(result.warnings, prefix + "F omitted; retained previous F");
        } else {
            BetaSpecification beta{
                defaults.betaShapeA,
                defaults.betaShapeB,
                defaults.randomSeed + 2 * index + 1};
            configuration.filters = randomFilters(
                dimensions.K, dimensions.N, dimensions.order, beta);
            warn(result.warnings, prefix
                + "F omitted; initialized from default Beta distribution");
        }

        if (layerPatch.leak) {
            configuration.leak = asVector(
                *layerPatch.leak, dimensions.N, "LEAK");
        } else if (old && old->configuration.leak.size() == dimensions.N) {
            configuration.leak = old->configuration.leak;
            warn(result.warnings, prefix + "LEAK omitted; retained previous leak");
        } else {
            configuration.leak =
                Eigen::VectorXd::Constant(dimensions.N, defaults.leak);
            warn(result.warnings, prefix + "LEAK omitted; used default leak");
        }

        if (layerPatch.parameters) {
            configuration.parameters = unpackParameters(
                *layerPatch.parameters, dimensions.K);
        } else if (old &&
                   old->configuration.parameters.baseRate.size() == dimensions.K) {
            configuration.parameters = old->configuration.parameters;
            warn(result.warnings, prefix
                + "PARAMETERS omitted; retained previous parameters");
        } else {
            configuration.parameters.baseRate =
                Eigen::VectorXd::Constant(dimensions.K, defaults.baseRate);
            configuration.parameters.bottomUpWeight =
                Eigen::VectorXd::Constant(
                    dimensions.K, defaults.bottomUpWeight);
            configuration.parameters.evidenceAmplitude =
                Eigen::VectorXd::Constant(
                    dimensions.K, defaults.evidenceAmplitude);
            configuration.parameters.centering =
                Eigen::VectorXd::Constant(dimensions.K, defaults.centering);
            warn(result.warnings, prefix
                + "PARAMETERS omitted; used default generator parameters");
        }

        if (layerPatch.context) {
            configuration.initialContext = *layerPatch.context;
        } else if (old &&
                   old->configuration.initialContext.rows() == dimensions.N &&
                   old->configuration.initialContext.cols() == dimensions.order) {
            configuration.initialContext = old->configuration.initialContext;
            warn(result.warnings, prefix
                + "CONTEXT omitted; retained previous context");
        } else {
            configuration.initialContext =
                Eigen::MatrixXd::Zero(dimensions.N, dimensions.order);
            warn(result.warnings, prefix + "CONTEXT omitted; used zero context");
        }

        if (layerPatch.topDownSupport) {
            layer.initialTopDownSupport = asVector(
                *layerPatch.topDownSupport, dimensions.K, "TOP_DOWN");
        } else if (old && old->initialTopDownSupport.size() == dimensions.K) {
            layer.initialTopDownSupport = old->initialTopDownSupport;
            warn(result.warnings, prefix
                + "TOP_DOWN omitted; retained previous support");
        } else {
            layer.initialTopDownSupport =
                Eigen::VectorXd::Zero(dimensions.K);
            warn(result.warnings, prefix
                + "TOP_DOWN omitted; used zero support");
        }

        if (layerPatch.candidateSelection) {
            configuration.candidateSelection = *layerPatch.candidateSelection;
        } else if (old) {
            configuration.candidateSelection =
                old->configuration.candidateSelection;
            warn(result.warnings, prefix
                + "CANDIDATE_SELECTION omitted; retained previous policy");
        } else {
            configuration.candidateSelection = defaults.candidateSelection;
            warn(result.warnings, prefix
                + "CANDIDATE_SELECTION omitted; used default policy");
        }

        if (layerPatch.learning) {
            layer.learning = *layerPatch.learning;
            layer.learningEnabled = *layerPatch.learningEnabled;
        } else if (old) {
            layer.learning = old->learning;
            layer.learningEnabled = old->learningEnabled;
            warn(result.warnings, prefix
                + "EM omitted; retained previous learning settings");
        } else {
            layer.learning = defaults.learning;
            layer.learningEnabled = defaults.learningEnabled;
            warn(result.warnings, prefix
                + "EM omitted; used default learning settings");
        }

        result.state.layers[index] = std::move(layer);
    }

    validateModelState(result.state);
    return result;
}

void validateModelState(const ModelState& state)
{
    if (state.layers.empty()) {
        throw std::invalid_argument("Model state must contain at least one layer");
    }
    if (state.input.size() > 0) {
        if (!state.input.allFinite() ||
            (state.input.array() < 0.0).any() ||
            (state.input.array() > 1.0).any()) {
            throw std::invalid_argument(
                "INPUT must contain finite values in [0,1]");
        }
        if (state.input.cols() !=
            state.layers.front().configuration.predictions.cols()) {
            throw std::invalid_argument(
                "INPUT column count must equal bottom-layer N");
        }
        const std::size_t rowCount =
            static_cast<std::size_t>(state.input.rows());
        if ((state.loopInput && state.inputIndex >= rowCount) ||
            (!state.loopInput && state.inputIndex > rowCount)) {
            throw std::invalid_argument("INPUT_INDEX is outside INPUT rows");
        }
    } else if (state.inputIndex != 0) {
        throw std::invalid_argument(
            "INPUT_INDEX must be zero when INPUT is absent");
    }

    std::vector<LayerConfiguration> configurations;
    configurations.reserve(state.layers.size());

    for (std::size_t index = 0; index < state.layers.size(); ++index) {
        const auto& layer = state.layers[index];
        const auto& configuration = layer.configuration;

        for (const auto& filter : configuration.filters) {
            if ((filter.array() < 0.0).any() ||
                (filter.array() > 1.0).any()) {
                throw std::invalid_argument(
                    layerPrefix(index) + "F entries must lie in [0,1]");
            }
        }
        if (!configuration.initialContext.allFinite() ||
            (configuration.initialContext.array() < 0.0).any() ||
            (configuration.initialContext.array() > 1.0).any()) {
            throw std::invalid_argument(
                layerPrefix(index) + "CONTEXT entries must lie in [0,1]");
        }
        validateLearning(layer.learning);
        if (layer.initialTopDownSupport.size() != 0) {
            if (layer.initialTopDownSupport.size() !=
                    configuration.predictions.rows()) {
                throw std::invalid_argument(
                    layerPrefix(index) + "TOP_DOWN must have length K");
            }
            requireProbabilityVector(
                layer.initialTopDownSupport, "TOP_DOWN");
        }

        const auto& selection = configuration.candidateSelection;
        if (!std::isfinite(selection.contextThreshold) ||
            !std::isfinite(selection.topDownThreshold) ||
            !std::isfinite(selection.observationThreshold) ||
            !std::isfinite(selection.activationThreshold) ||
            selection.maximumSelectedGenerators >=
                std::numeric_limits<std::size_t>::digits) {
            throw std::invalid_argument(
                layerPrefix(index) + "Invalid candidate selection settings");
        }

        configurations.push_back(configuration);
    }

    // Reuse the statistical model's own shape and probability validation.
    NoisyORStack validated(std::move(configurations));
    (void) validated;
}

NoisyORStack makeInitializedStack(const ModelState& state)
{
    validateModelState(state);
    std::vector<LayerConfiguration> configurations;
    std::vector<Eigen::VectorXd> topDownSupport;
    configurations.reserve(state.layers.size());
    topDownSupport.reserve(state.layers.size());
    for (const auto& layer : state.layers) {
        configurations.push_back(layer.configuration);
        topDownSupport.push_back(layer.initialTopDownSupport.size() == 0
            ? Eigen::VectorXd::Zero(
                  layer.configuration.predictions.rows())
            : layer.initialTopDownSupport);
    }

    NoisyORStack stack(std::move(configurations));
    stack.initialize(topDownSupport);
    return stack;
}

ModelState snapshotModelState(
    const NoisyORStack& stack,
    const ModelState& state)
{
    if (stack.size() != state.layers.size()) {
        throw std::invalid_argument(
            "Stack and ModelState must contain the same number of layers");
    }

    ModelState snapshot = state;
    for (std::size_t index = 0; index < stack.size(); ++index) {
        const NoisyORLayer& source = stack.layer(index);
        LayerState& destination = snapshot.layers[index];
        destination.configuration = source.configuration();
        destination.configuration.initialContext = source.context();
        destination.initialTopDownSupport = source.topDownSupport();
    }
    validateModelState(snapshot);
    return snapshot;
}

bool readNextInput(ModelState& state, Eigen::VectorXd& observation)
{
    if (state.input.rows() == 0) {
        return false;
    }

    const std::size_t rowCount =
        static_cast<std::size_t>(state.input.rows());
    if (state.inputIndex >= rowCount) {
        if (!state.loopInput) {
            return false;
        }
        state.inputIndex = 0;
    }

    observation = state.input
        .row(static_cast<Eigen::Index>(state.inputIndex))
        .transpose();
    ++state.inputIndex;
    if (state.loopInput && state.inputIndex == rowCount) {
        state.inputIndex = 0;
    }
    return true;
}

} // namespace noisy_or
