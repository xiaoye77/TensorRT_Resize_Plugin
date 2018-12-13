#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <memory>
#include <string.h>
#include <cstdint>
#include <chrono>

#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include "common.h"

// Log errors and exit
#define RETURN_AND_LOG(ret, severity, message)                              \
do {                                                                        \
    std::string error_message = "cpp_inference: " + std::string(message);   \
    gLogger.log(nvinfer1::ILogger::Severity::k ## severity, error_message.c_str());   \
    return (ret);                                                           \
} while(0);

// Define the input dimensions for the model
static const int INPUT_H = 480;
static const int INPUT_W = 640;
// Define the max batch size and max workspace size
static const long long MAX_WORKSPACE_SIZE = 1 << 30;
static const int MAX_BATCH_SIZE = 1;

// Define input and output layers and uff model path
static const char* uff_model_filepath = "model.uff";
static const char* input = "prefix/my_input";
static const char* output = "prefix/my_end";
// Logger for logging info, warnings, and errors
static Logger gLogger;

// Calculate the volume of given Dims
// Needed to allocate adequate memory
inline int64_t volume(const nvinfer1::Dims& d) {
    int64_t v = 1;
    for (int64_t i = 0; i < d.nbDims; ++i) // All dims
        v *= d.d[i];
    return v;
}
// Calculate the size of the DataType
inline unsigned int elementSize(nvinfer1::DataType t) {
    switch(t) {
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    assert(0);
    return 0;
}
// Allocate memory in the GPU
// Raise error if not enough memory available
void* safeCudaMalloc(size_t memSize) {
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr) {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

std::vector<std::pair<int64_t, nvinfer1::DataType>>
calculateBindingBufferSizes(const nvinfer1::ICudaEngine& engine, int nbBindings, int batchSize) {
    std::vector<std::pair<int64_t, nvinfer1::DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine.getBindingDimensions(i);
        nvinfer1::DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }
    return sizes;
}

// Create the Cuda Buffer and input the data
void* createCudaBuffer(int64_t eltCount, nvinfer1::DataType dtype) {
    assert(eltCount == INPUT_H * INPUT_W * 3); // 3 for channels
    assert(elementSize(dtype) == sizeof(float));
    // Allocate memory equal to volume of Dims and size of type being used
    size_t memSize = eltCount * elementSize(dtype);
    float* inputs = new float[eltCount];

    // This is our stub input for now
    uint8_t fileData[INPUT_H * INPUT_W * 3];
    for (int i = 0; i < INPUT_H * INPUT_W * 3; ++i) {
        fileData[i] = uint8_t(i % 128);
    }

    //std::cout << "--- INPUT ---\n";
    //for (int i = 0; i < INPUT_W; ++i) {
    //    for (int j = 0; j < INPUT_H; ++j) {
    //        std::cout << int(fileData[i * INPUT_W + j]) << " ";
    //    }
    //    std::cout << "\n";
    //}
    std::cout << "Printing first 10 elements\n";
    for (int i = 0; i < 10; ++i)
        std::cout << int(fileData[i]) << " ";
    std::cout << std::endl;

    // Fix and/or normalize the input
    for (int i = 0; i < eltCount; ++i) {
        inputs[i] = float(fileData[i]);
    }

    // Allocate enough memory in the GPU and copy the input there
    void* deviceMem = safeCudaMalloc(memSize);
    CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));

    delete[] inputs;
    // Return the pointer pointing to our data
    return deviceMem;
}

void printOutput(int64_t eltCount, nvinfer1::DataType dtype, void* buffer) {
    // Copy the results of the operation from GPPU back to host
    // and print them
    //std::cout << eltCount << " eltCount\n" << std::endl;
    assert(elementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * elementSize(dtype);
    // Create memory for results
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    // Print results
    //std::cout << "--- OUTPUT ---" << std::endl;
    //for (int i = 0; i < 15; ++i) {
    //    for (int j = 0; j < 10; ++j) {
    //        std::cout << outputs[i * 15 + j] << " ";
    //    }
    //    std::cout << "\n";
    //}
    std::cout << "Printing first 10 elements\n";
    std::cout << "eltCount is " << eltCount << "\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << outputs[i] << " ";
    }
    std::cout << std::endl;
    delete[] outputs;
}

nvinfer1::ICudaEngine* loadModelAndCreateEngine(const char* uffFile,
                                                int maxBatchSize,
                                                nvuffparser::IUffParser* parser) {
    // Here we load the model from a uff file
    // Create builder and network definition
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    // Parse the uff file
    if(!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");

    builder->setMaxBatchSize(MAX_BATCH_SIZE);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);

    // Build the engine
    nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");
    // Free unnecessary memory
    network->destroy();
    builder->destroy();

    return engine;
}

void execute(nvinfer1::ICudaEngine& engine) {
    // Execute the engine
    nvinfer1::IExecutionContext* context = engine.createExecutionContext();
    int nbBindings = engine.getNbBindings();
    int batchSize = MAX_BATCH_SIZE;
    assert(nbBindings == 2);

    std::vector<void*> buffers(nbBindings);
    auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    int bindingIdxInput = 0;
    for (int i = 0; i < nbBindings; ++i) {
        if (engine.bindingIsInput(i)) {
            bindingIdxInput = i;
        } else {
            auto bufferSizesOutput = buffersSizes[i];
            buffers[i] = safeCudaMalloc(bufferSizesOutput.first *
                                        elementSize(bufferSizesOutput.second));
        }
    }

    auto bufferSizesInput = buffersSizes[bindingIdxInput];

    buffers[bindingIdxInput] = createCudaBuffer(bufferSizesInput.first,
                                                bufferSizesInput.second);
    context->execute(batchSize, &buffers[0]);
    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx) {
        if (engine.bindingIsInput(bindingIdx))
            continue;
        auto bufferSizesOutput = buffersSizes[bindingIdx];
        printOutput(bufferSizesOutput.first, bufferSizesOutput.second,
                    buffers[bindingIdx]);
    }
    CHECK(cudaFree(buffers[bindingIdxInput]));

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        if (!engine.bindingIsInput(bindingIdx))
            CHECK(cudaFree(buffers[bindingIdx]));
    context->destroy();
}

/* Nearest Neighbor Plugin for TRT  */
class ResizePlugin : public nvinfer1::IPlugin {
public:
    ResizePlugin(const nvinfer1::Weights* weights, int nbWeights);
    ResizePlugin(const void* data, size_t length);

    int getNbOutputs() const override { return 1; }
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                        int nbInputDims) override;
    void configure( const nvinfer1::Dims* inputs, int nbInputs,
                    const nvinfer1::Dims* outputs, int nbOutputs, int) override;
    int initialize() override;
    virtual void terminate() override;

    virtual size_t getWorkspaceSize(int) const override { return 0; }
    virtual int enqueue(int batchSize, const void*const *inputs, void** outputs,
                        void* workspace, cudaStream_t stream) override;
    virtual size_t getSerializationSize() override;
    virtual void serialize(void* buffer) override;
private:
    int _ndims;
    float _scale[nvinfer1::Dims::MAX_DIMS];
    nvinfer1::Dims _output_dims;
    int old_w, old_h, old_c; // old dimensions
    int new_w, new_h, new_c; // new dimensions
};

ResizePlugin::ResizePlugin(const nvinfer1::Weights* weights, int nbWeights) {
    return; // no weights are passed from our model

    assert(nbWeights == 1); // The new sizes
    assert(weights->count == 2); // Width and height must be both specified
    assert(weights->type == nvinfer1::DataType::kINT32); // Dims must be ints
    new_w = *((int*) weights->values + 0); // width
    new_h = *((int*) weights->values + 1); // height
}

ResizePlugin::ResizePlugin(const void* data, size_t length) {

}

nvinfer1::Dims ResizePlugin::getOutputDimensions(int index,
                            const nvinfer1::Dims* inputs, int nbInputDims) {
    std::cout << "Getting Output Dimensions...\n";
    assert(nbInputDims == 1);
    assert(inputs->nbDims == 3);
    assert(inputs->type[0] == nvinfer1::DimensionType::kCHANNEL);
    assert(inputs->type[1] == nvinfer1::DimensionType::kSPATIAL);
    assert(inputs->type[2] == nvinfer1::DimensionType::kSPATIAL);

    old_c = inputs->d[0]; // old channels
    old_w = inputs->d[1]; // old width
    old_h = inputs->d[2]; // old height

    new_w = old_w * 2;
    new_h = old_h * 2;

    std::cout << "old " << old_c << " " << old_w << " " << old_h << std::endl;
    std::cout << "new " << old_c << " " << new_w << " " << new_h << std::endl;

    nvinfer1::Dims output;
    output.d[0] = old_c; // preserve channel number
    output.d[1] = new_w; // new width
    output.d[2] = new_h; // new height
    
    output.nbDims = 3; // for 3 dimensions
    output.type[0] = nvinfer1::DimensionType::kCHANNEL;
    output.type[1] = nvinfer1::DimensionType::kSPATIAL;
    output.type[2] = nvinfer1::DimensionType::kSPATIAL;

    return output;
}

void ResizePlugin::configure(const nvinfer1::Dims* inputs, int nbInputs,
                            const nvinfer1::Dims* outputs, int nbOutputs, int) {
    std::cout << "Configuring...\n";
    std::cout << "d0 " << inputs->d[0] << "\n";
    std::cout << "d1 " << inputs->d[1] << "\n";
    std::cout << "d2 " << inputs->d[2] << "\n";
    std::cout << std::endl;
}

int ResizePlugin::initialize() {
    std::cout << "Initializing...\n";
    return 0;
}

void ResizePlugin::terminate() {
    std::cout << "Terminating...\n";
}

// CUDA kernel for nearest neighbor resizing
template <typename Data>
__global__
void resize_nearest_kernel_2d(int nbatch,
                          float2 scale,
                          int2 osize,
                          Data const* idata, int istride, int ibatchstride,
                          Data*       odata, int ostride, int obatchstride) {
    int x0 = threadIdx.x + blockIdx.x * blockDim.x;
    int y0 = threadIdx.y + blockIdx.y * blockDim.y;
    int z0 = blockIdx.z;
    for(int batch = z0; batch < nbatch; batch += gridDim.z) {
        for(int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y) {
            for(int ox = x0; ox<osize.x; ox += blockDim.x * gridDim.x) {
                int ix = int(ox / scale.x);
                int iy = int(oy / scale.y);
                odata[batch * obatchstride + oy * ostride + ox] =
                    idata[batch * ibatchstride + iy * istride + ix];
            }
        }
    }
}

int ResizePlugin::enqueue(int batchSize, const void*const *inputs,
                        void** outputs, void* workspace, cudaStream_t stream) {
    std::cout << "Enqueuing...\n";
    float scale_h = new_h / (old_h + 0.0); // height scaling factor
    float scale_w = new_w / (old_w + 0.0); // width scaling factor
    float2 scale = {scale_h, scale_w}; // reverse order
    int2 osize = {new_h, new_w};
    int istride = old_h;
    int ostride = new_h;
    int ibatchstride = old_w * istride;
    int obatchstride = new_w * ostride;
    dim3 block(32, 16);
    dim3 grid((osize.x - 1) / block.y + 1,
              (osize.y - 1) / block.y + 1,
              std::min(batchSize * old_c, 65535));
    // Call kernel
    resize_nearest_kernel_2d<<<grid, block, 0, stream>>>
        (batchSize * old_c, scale, osize,
         static_cast<float const*> (inputs[0]), istride, ibatchstride,
         static_cast<float*      >(outputs[0]), ostride, obatchstride);
    return cudaGetLastError() != cudaSuccess;
}

size_t ResizePlugin::getSerializationSize() {
    return 0;
}

void ResizePlugin::serialize(void* buffer) {

}

/* Create the factory for dealing with custom plugins and layers */
class MyPluginFactory : public nvinfer1::IPluginFactory, public nvuffparser::IPluginFactory {
public:
    // Create plugin function
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName,
                            const nvinfer1::Weights* weights,
                            int nbWeights,
                            const nvuffparser::FieldCollection fc) override;
    // Create plugin for serialized data
    nvinfer1::IPlugin* createPlugin(const char* layerName,
                            const void* serialData,
                            size_t serialLength) override;
    // Check if plugin is implemented
    bool isPlugin(const char* name) override;
    void destroyPlugin();

    // We need two resize pointers, since in our model
    // we have two resize layers
    // Assertion error if using only one
    std::unique_ptr<ResizePlugin> mResizePlugin_1{ nullptr };
    std::unique_ptr<ResizePlugin> mResizePlugin_2{ nullptr };
};

nvinfer1::IPlugin* MyPluginFactory::createPlugin(const char* layerName,
                            const nvinfer1::Weights* weights,
                            int nbWeights,
                            const nvuffparser::FieldCollection fc) {
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "_ResizeNN")) {
        if (mResizePlugin_1.get() == nullptr) {
            mResizePlugin_1 = std::unique_ptr<ResizePlugin>(new ResizePlugin(weights, nbWeights));
            return mResizePlugin_1.get();
        } else {
            mResizePlugin_2 = std::unique_ptr<ResizePlugin>(new ResizePlugin(weights, nbWeights));
            return mResizePlugin_2.get();
        }
    } else {
        assert(0);
        return nullptr;
    }
}

nvinfer1::IPlugin* MyPluginFactory::createPlugin(const char* layerName,
                        const void* serialData,
                        size_t serialLength) {
    std::cout << "create serial plugin\n";
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "_ResizeNN")) {
        assert(mResizePlugin_1.get() == nullptr);
        mResizePlugin_1 = std::unique_ptr<ResizePlugin>(new ResizePlugin(serialData, serialLength));
        return mResizePlugin_1.get();
    } else {
        assert(0);
        return nullptr;
    }
}

bool MyPluginFactory::isPlugin(const char* name) {
    return (!strcmp(name, "_ResizeNN"));
}

void MyPluginFactory::destroyPlugin() {
    mResizePlugin_1.reset();
}


int main() {
    MyPluginFactory parserPluginFactory;
    std::cout << "Using " << uff_model_filepath << std::endl;
    // Create uff parser
    nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
    // Register input, output layers, and plugin factory
    parser->registerInput(input, nvinfer1::DimsCHW(3, INPUT_H, INPUT_W),
                                 nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(output);
    parser->setPluginFactory(&parserPluginFactory);
    // Create engine
    nvinfer1::ICudaEngine* engine = loadModelAndCreateEngine(uff_model_filepath,
                                                            MAX_BATCH_SIZE,
                                                            parser);
    if (!engine)
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Unable to load engine");

    parser ->destroy();
    // Infer
    auto start = std::chrono::system_clock::now();
    int p = 0;
    while (p < 1) {
        ++p;
        execute(*engine);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> dif = end - start;
        if (dif.count() > 1) {
            std::cout << "FPS: " << p << std::endl;
            p = 0;
            start = end;
        }
    }

    engine->destroy();
    parserPluginFactory.destroyPlugin();
    return 0;
}
