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

#ifndef JI_SAMPLEALGORITHM_HPP
#define JI_SAMPLEALGORITHM_HPP
#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Configuration.hpp"
#include "SampleDetector.hpp"

#define STATUS int
using namespace std;
using namespace cv;

class SampleAlgorithm
{

public:
    
    SampleAlgorithm();
    ~SampleAlgorithm();

    /*
     * @breif 初始化算法运行资源
     * @return STATUS 返回调用结果,成功返回STATUS_SUCCESS
    */    
    STATUS Init();

    /*
     * @breif 去初始化，释放算法运行资源
     * @return STATUS 返回调用结果,成功返回STATUS_SUCCESS
    */    
    STATUS UnInit();

    /*
     * @breif 算法业务处理函数，输入分析图片，返回算法分析结果
     * @param inFrame 输入图片对象  
     * @param args 输入算法参数, json字符串
     * @param event 返回的分析结果结构体
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */    
    STATUS Process(const Mat &inFrame, const char *args, JiEvent &event);

    /*
     * @breif 更新算法实例的配置
     * @param args 输入算法参数, json字符串     
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */ 
    STATUS UpdateConfig(const char *args);

    /*
     * @breif 调用Process接口后,获取处理后的图像
     * @param out 返回处理后的图像结构体     
     * @param outCount 返回调用次数的计数
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */ 
    STATUS GetOutFrame(JiImageInfo **out, unsigned int &outCount);

private:
    cv::Mat mOutputFrame{0};    // 用于存储算法处理后的输出图像，根据ji.h的接口规范，接口实现需要负责释放该资源
    JiImageInfo mOutImage[1]; // 根据算法实际需求，确定输出图片数组大小,本demo每次仅分析处理一幅图
    unsigned int mOutCount = 1;//本demo每次仅分析处理一幅图    
    Configuration mConfig;     //跟配置相关的类

public:
    //接口的返回值的定义
    static const int ERROR_BASE = 0x0100;
    static const int ERROR_INPUT = 0x0101;
    static const int ERROR_INIT = 0x0102;
    static const int ERROR_PROCESS = 0x0103;
    static const int ERROR_CONFIG = 0x0104;
    static const int STATUS_SUCCESS = 0x0000;
       
private:
    std::string mStrLastArg;  //算法参数缓存,动态参数与缓存参数不一致时才会进行更新  
    std::string mStrOutJson;  //返回的json缓存,注意当算法实例销毁时,对应的通过算法接口获取的json字符串也将不在可用
    std::shared_ptr<SampleDetector> mDetector{nullptr}; //算法检测器实例
};

#endif //JI_SAMPLEALGORITHM_HPP
