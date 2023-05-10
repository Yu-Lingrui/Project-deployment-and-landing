/*
 * Copyright (c) 2021 ExtremeVision Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
* 本demo采用YOLOX目标检测器,检测车辆和车牌
* 首次运行时采用tensorrt先加载onnx模型,并保存trt模型,以便下次运行时直接加载trt模型,加快初始化
*
*/

#ifndef COMMON_DET_INFER_H
#define COMMON_DET_INFER_H
#include <memory>
#include <map>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
typedef struct Polygon
{
    vector<Point> points;
    string name;
} Polygon;

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};
class SampleDetector
{
public:
    SampleDetector();
    ~SampleDetector();
    bool Init(const std::string &strModelName, float thresh);
    bool UnInit();
    bool ProcessImage(cv::Mat &img, cv::Mat &mask, std::vector<Polygon> &detectedPolygons);
    cv::Mat generate_rgb(Mat frame, Mat mask);
private:
    void loadOnnx(const std::string strName);
    void loadTrt(const std::string strName);

private:
    std::vector<std::string> class_names;
    vector<Vec3b> class_colors;
    int num_class;
    nvinfer1::ICudaEngine *m_CudaEngine;
    nvinfer1::IRuntime *m_CudaRuntime;
    nvinfer1::IExecutionContext *m_CudaContext;
    cudaStream_t m_CudaStream;
    void *m_ArrayDevMemory[2]{0};
    void *m_ArrayHostMemory[2]{0};
    int m_ArraySize[2]{0};
    int m_iInputIndex;
    int m_iOutputIndex;
    int m_iClassNums;
    cv::Size m_InputSize;
    cv::Mat m_Resized;
    cv::Mat m_Normalized;
    std::vector<cv::Mat> m_InputWrappers{};
    cv::Mat ResizeImage(const cv::Mat &srcImg, int *newh, int *neww, int *top, int *left);

private:
    bool m_bUninit = false;
    float mThresh;
};

#endif