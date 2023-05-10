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

#ifndef JI_SAMPLEDETECTOR_HPP
#define JI_SAMPLEDETECTOR_HPP
#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

#include "ji.h"
#include "ji_utils.h"
#include "WKTParser.h"
#include "Configuration.hpp"


#define STATUS int
using namespace std;
using namespace cv;
using namespace Ort;

typedef struct Polygon
{
    vector<Point> points;
    string name;
} Polygon;

class SampleDetector
{

public:
    /*
     * @breif 检测器构造函数     
    */ 
    SampleDetector();
    
    /*
     * @breif 检测器析构函数     
    */ 
    ~SampleDetector();
    
    /*
     * @breif 初始化检测器相关的资源
     * @param strModelName 检测器加载的模型名称     
     * @param thresh 检测阈值
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */ 
    STATUS Init();

    /*
     * @breif 去初始化,释放模型检测器的资源     
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */ 
    STATUS UnInit();
    
    /*
     * @breif 根据送入的图片进行模型推理, 并返回检测结果
     * @param inFrame 输入图片
     * @param result 检测结果通过引用返回
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */
    STATUS ProcessImage(cv::Mat &image, Mat &mask, vector<Polygon> &detectedPolygons);
    cv::Mat generate_rgb(Mat frame, Mat mask);
public:
    // 接口的返回值定义
    static const int ERROR_BASE = 0x0200;
    static const int ERROR_INPUT = 0x0201;
    static const int ERROR_INIT = 0x0202;
    static const int ERROR_PROCESS = 0x0203;
    static const int STATUS_SUCCESS = 0x0000;   
private:    
    std::vector<std::string> class_names;
    vector<Vec3b> class_colors;
    int num_class ;
    
    const int inpWidth = 513;
    const int inpHeight = 513;
    const float mean_[3] = { 0.485, 0.456, 0.406 };
    const float std_[3] = { 0.229, 0.224, 0.225 };
    vector<float> input_image_;
    void Normalize(Mat img);

    Session *mSession;
    Env mEnv = Env(ORT_LOGGING_LEVEL_ERROR, "deeplabv3plus");
    SessionOptions mSessionOptions = SessionOptions();
    vector<char *> mInputNames;
    vector<char *> mOutputNames;
};

#endif //JI_SAMPLEDETECTOR_HPP
