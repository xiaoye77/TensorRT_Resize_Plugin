#ifndef _RESIZE_PLUGIN_H_
#define _RESIZE_PLUGIN_H_

#include <cassert>
#include <iostream>
#include <cudnn.h>
#include <cstring>
#include <stdio.h>
#include <cuda_fp16.h>

#include "NvInfer.h"
#include "NvUffParser.h"

#define CHECK(status)                               \
{                                                   \
    if (status != 0)                                \
    {                                               \
        std::cout << "Cuda failure: " << status;    \
        abort();                                    \
    }                                               \
}

class ResizePlugin : public nvinfer1::IPlugin {
public:
	ResizePlugin(const nvinfer1::Weights* weights, int nbWeights);
    ResizePlugin(const void* buffer, size_t size) {
        std::cout << "serial\n" ;
    }

    inline int getNbOutputs() const override { return 1; };
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    int initialize() override;
    inline void terminate() override { ; };

    inline size_t getWorkspaceSize(int) const override { return 0; };
    int enqueue(int batchSize,
                const void*const *inputs,
                void** outputs,
                void*,
                cudaStream_t stream) override;

    size_t getSerializationSize() override;
    void serialize(void* buffer) override;

    void configure(const nvinfer1::Dims*inputs,
                    int nbInputs,
                    const nvinfer1::Dims* outputs,
                    int nbOutputs,
                    int) override;
protected:
    float _scale[4];
    int _ndims = 2;
    nvinfer1::Dims _output_dims;
    int oc, ow, oh;
    int nc, nw, nh;
};

ResizePlugin::ResizePlugin(const nvinfer1::Weights* weights, int nbWeights) {
    std::cout << "Weight constructor\n";
    nc = 256;
    std::cout<<nbWeights<<" nbweights\n";
    return;
    nw = *((int*) weights->values + 0);
    nh = *((int*) weights->values + 1);
    std::cout<< "Setting new dimensions. nw = " << nw << ", nh = " << nh << "\n";
}

nvinfer1::Dims ResizePlugin::getOutputDimensions(int index,
                                        const nvinfer1::Dims* inputs,
                                        int nbInputDims) {
    nvinfer1::Dims const& input = inputs[0];
    nvinfer1::Dims output;
    output.nbDims = input.nbDims;
    int s = 0;

    std::cout<<nbInputDims<<" nbInputDims\n";

    oc = input.d[0];
    ow = input.d[1];
    oh = input.d[2];
    
    if (ow == 14) nw = 27;
    else nw = ow * 2;
    nh = oh * 2;

    std::cout<< "Setting old dimensions. ow = " << ow << ", oh = " << oh << "\n";

    _scale[0] = 1;
    _scale[1] = (nw / (ow + 0.0));
    _scale[2] = (nh / (oh + 0.0));
    std::cout<< "Setting scale. sw = " << _scale[1] << ", sh = " << _scale[2] << "\n";

    for(int d = 0; d < input.nbDims; ++d) {
        output.type[d] = input.type[d];
        if(input.type[d] == nvinfer1::DimensionType::kSPATIAL) {
            output.d[d] = int(input.d[d] * _scale[s++]);
        } else {
            output.d[d] = input.d[d];
            ++s;
        }
    }
    output.d[3] = 3;
    std::cout<<"scale "<<_scale[0]<<" "<<_scale[1]<<" "<<_scale[2]<<"\n";
    std::cout<<"input "<<input.d[0]<<" "<<input.d[1]<<" "<<input.d[2]<<"\n";
    std::cout<<"output "<<output.d[0]<<" "<<output.d[1]<<" "<<output.d[2]<<"\n";
    return output;
}

int ResizePlugin::initialize() {
    _output_dims = nvinfer1::Dims{nc, nw, nh};
    return 0;
}

template <typename Data>
__global__
void resize_nearest_kernel_2d(int nbatch,
                                float2 scale,
                                int2 osize,
                                Data const* idata, int istride, int ibatchstride,
                                Data* odata, int ostride, int obatchstride) {
    int x0 = threadIdx.x + blockIdx.x * blockDim.x;
    int y0 = threadIdx.y + blockIdx.y * blockDim.y;
    int z0 = blockIdx.z;
    for(int batch = z0; batch < nbatch; batch += gridDim.z) {
        for(int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y) {
            for(int ox = x0; ox < osize.x; ox += blockDim.x * gridDim.x) {
                int ix = int(ox / scale.x);
                int iy = int(oy / scale.y);
                odata[batch * obatchstride + oy * ostride + ox] =
                    idata[batch * ibatchstride + iy * istride + ix];
            }
        }
    }
}

int ResizePlugin::enqueue(int batchSize,
            const void*const *inputs,
            void** outputs,
            void*,
            cudaStream_t stream) {
    auto const& input_dims = nvinfer1::Dims{oc, ow, oh};
	int nchan = input_dims.d[0];
	switch( _ndims ) {
        case 2: {
            float2 scale = {_scale[1], _scale[0]};
            int2 osize = {_output_dims.d[2], _output_dims.d[1]};
            int istride =  input_dims.d[2];
            int ostride = _output_dims.d[2];
            int ibatchstride =  input_dims.d[1] * istride;
            int obatchstride = _output_dims.d[1] * ostride;
            dim3 block(32, 16);
            dim3 grid((osize.x - 1) / block.x + 1,
                      (osize.y - 1) / block.y + 1,
                      std::min(batchSize * nchan, 65535));
            //resize_nearest_kernel_2d<<<grid, block, 0, stream>>>
            resize_nearest_kernel_2d
                (batchSize * nchan, scale, osize,
                static_cast<float const*>( inputs[0]), istride, ibatchstride,
                static_cast<float*      >(outputs[0]), ostride, obatchstride);
            return cudaGetLastError() != cudaSuccess;
        }
        default: return -1;
    }
}

size_t ResizePlugin::getSerializationSize() {
    // buff
    std::cout << "getSerializationSize" << std::endl;
    return 0;
}

void ResizePlugin::serialize(void* buffer) {
    std::cout << "serialize" << std::endl;
    // buff
}

void ResizePlugin::configure(const nvinfer1::Dims*inputs,
                int nbInputs,
                const nvinfer1::Dims* outputs,
                int nbOutputs,
                int) {


    std::cout << "Old " << oc << " " << ow << " " << oh <<"\n";
    std::cout << "New " << nc << " " << nw << " " << nh <<"\n";

    std::cout << _scale[1] << " scale 1\n";
    std::cout << _scale[2] << " scale 2\n";
}

class ResizePluginFactory : public nvinfer1::IPluginFactory, public nvuffparser::IPluginFactory {
public:
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName,
                            const nvinfer1::Weights* weights,
                            int nbWeights,
                            const nvuffparser::FieldCollection fc) override;
    nvinfer1::IPlugin* createPlugin(const char* layerName,
                            const void* serialData,
                            size_t serialLength) override;
    bool isPlugin(const char* name) override;
    void destroyPlugin();

    ResizePlugin* mResizePlugin{ nullptr };
};

nvinfer1::IPlugin* ResizePluginFactory::createPlugin(const char* layerName,
                            const nvinfer1::Weights* weights,
                            int nbWeights,
                            const nvuffparser::FieldCollection fc) {
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "_ResizeNN")) {
        assert(mResizePlugin == nullptr);
        mResizePlugin = new ResizePlugin(weights, nbWeights);
        return mResizePlugin;
    } else {
        assert(0);
        return nullptr;
    }
}

nvinfer1::IPlugin* ResizePluginFactory::createPlugin(const char* layerName,
                        const void* serialData,
                        size_t serialLength) {
    std::cout << "create serial plugin\n";
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "_ResizeNN")) {
        assert(mResizePlugin == nullptr);
        mResizePlugin = new ResizePlugin(serialData, serialLength);
        return mResizePlugin;
    } else {
        assert(0);
        return nullptr;
    }
}

bool ResizePluginFactory::isPlugin(const char* name) {
    return (!strcmp(name, "_ResizeNN"));
}

void ResizePluginFactory::destroyPlugin() {
    delete mResizePlugin;
}

#endif
