#include <sys/stat.h>
#include <fstream>
#include <glog/logging.h>

#include "SampleDetector.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "ji_utils.h"
#include "./logging.h"
#define INPUT_NAME "input"
#define OUTPUT_NAME "output"
static const int INPUT_H = 512;
static const int INPUT_W = 512;
using namespace nvinfer1;
#define random(a,b) (rand()%(b-a)+a)
cv::Mat SampleDetector::ResizeImage(const cv::Mat &srcimg, int *newh, int *neww, int *top, int *left)
{
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = m_InputSize.height;
    *neww = m_InputSize.width;
    cv::Mat dstimg;
    if (srch != srcw)
    {
        float hw_scale = (float)srch / srcw;
        if (hw_scale > 1)
        {
            *newh = m_InputSize.height;
            *neww = int(m_InputSize.width / hw_scale);
            cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
            *left = int((m_InputSize.width - *neww) * 0.5);
            cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, m_InputSize.width - *neww - *left, cv::BORDER_CONSTANT, 0);
        }
        else
        {
            *newh = (int)m_InputSize.height * hw_scale;
            *neww = m_InputSize.width;
            cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
            *top = (int)(m_InputSize.height - *newh) * 0.5;
            cv::copyMakeBorder(dstimg, dstimg, *top, m_InputSize.height - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 0);
        }
    }
    else
    {
        cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
    }
    return dstimg;
}
static bool ifFileExists(const char *FileName)
{
    struct stat my_stat;
    return (stat(FileName, &my_stat) == 0);
}
void SampleDetector::loadOnnx(const std::string strModelName)
{
    Logger gLogger;
    //根据tensorrt pipeline 构建网络
    IBuilder *builder = createInferBuilder(gLogger);
    builder->setMaxBatchSize(1);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(strModelName.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
    IBuilderConfig *config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1ULL << 30);
    m_CudaEngine = builder->buildEngineWithConfig(*network, *config);

    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + ".trt";
    IHostMemory *gieModelStream = m_CudaEngine->serialize();
    std::string serialize_str;
    std::ofstream serialize_output_stream;
    serialize_str.resize(gieModelStream->size());
    memcpy((void *)serialize_str.data(), gieModelStream->data(), gieModelStream->size());
    serialize_output_stream.open(strTrtName.c_str());
    serialize_output_stream << serialize_str;
    serialize_output_stream.close();
    m_CudaContext = m_CudaEngine->createExecutionContext();
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
}
SampleDetector::SampleDetector()
{
}

void SampleDetector::loadTrt(const std::string strName)
{
    Logger gLogger;
    IRuntime *runtime = createInferRuntime(gLogger);
    std::ifstream fin(strName);
    std::string cached_engine = "";
    while (fin.peek() != EOF)
    {
        std::stringstream buffer;
        buffer << fin.rdbuf();
        cached_engine.append(buffer.str());
    }
    fin.close();
    m_CudaEngine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    m_CudaContext = m_CudaEngine->createExecutionContext();
    runtime->destroy();
}

bool SampleDetector::Init(const std::string &strModelName, float thresh)
{
    mThresh = thresh;
    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + ".trt";
    if (ifFileExists(strTrtName.c_str()))
    {
        loadTrt(strTrtName);
    }
    else
    {
        loadOnnx(strModelName);
    }
    // 分配输入输出的空间,DEVICE侧和HOST侧
    m_iInputIndex = m_CudaEngine->getBindingIndex(INPUT_NAME);
    m_iOutputIndex = m_CudaEngine->getBindingIndex(OUTPUT_NAME);
    m_InputSize = cv::Size(INPUT_H, INPUT_W);

    cudaMalloc(&m_ArrayDevMemory[m_iInputIndex], 3 * INPUT_H * INPUT_W * sizeof(float));
    m_ArrayHostMemory[m_iInputIndex] = malloc(3 * INPUT_H * INPUT_W * sizeof(float));
    //方便NHWC到NCHW的预处理
    m_InputWrappers.emplace_back(INPUT_H, INPUT_W, CV_32FC1, m_ArrayHostMemory[m_iInputIndex]);
    m_InputWrappers.emplace_back(INPUT_H, INPUT_W, CV_32FC1, m_ArrayHostMemory[m_iInputIndex]+ sizeof(float) * INPUT_H * INPUT_W);
    m_InputWrappers.emplace_back(INPUT_H, INPUT_W, CV_32FC1, m_ArrayHostMemory[m_iInputIndex]+ 2 * sizeof(float) * INPUT_H * INPUT_W);

    cudaMalloc(&m_ArrayDevMemory[m_iOutputIndex], 5 * INPUT_H * INPUT_W * sizeof(float));
    m_ArrayHostMemory[m_iOutputIndex] = malloc(5 * INPUT_H * INPUT_W * sizeof(float));
    cudaStreamCreate(&m_CudaStream);
    m_bUninit = false;
}

bool SampleDetector::UnInit()
{
    if (m_bUninit == true)
    {
        return false;
    }
    for (auto &p : m_ArrayDevMemory)
    {
        cudaFree(p);
        p = nullptr;
    }
    for (auto &p : m_ArrayHostMemory)
    {
        free(p);
        p = nullptr;
    }
    cudaStreamDestroy(m_CudaStream);
    m_CudaContext->destroy();
    m_CudaEngine->destroy();
    m_bUninit = true;
}

SampleDetector::~SampleDetector()
{
    UnInit();
}


bool SampleDetector::ProcessImage(cv::Mat &img, cv::Mat &mask, std::vector<Polygon> &detectedPolygons)
{
    int newh = 0, neww = 0, top = 0, left = 0;
    m_Resized = ResizeImage(img, &newh, &neww, &top, &left);
    m_Resized.convertTo(m_Normalized, CV_32FC3, 1 / 255.);
    cv::split(m_Normalized, m_InputWrappers);
    auto ret = cudaMemcpyAsync(m_ArrayDevMemory[m_iInputIndex], m_ArrayHostMemory[m_iInputIndex], 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
    auto ret1 = m_CudaContext->enqueue(1, m_ArrayDevMemory, m_CudaStream, nullptr);
    ret = cudaMemcpyAsync(m_ArrayHostMemory[m_iOutputIndex], m_ArrayDevMemory[m_iOutputIndex], 5 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyDeviceToHost, m_CudaStream);
    ret = cudaStreamSynchronize(m_CudaStream);
    Mat outimg = Mat::zeros(INPUT_H, INPUT_W, CV_8UC1);
    float *result = (float *)m_ArrayHostMemory[m_iOutputIndex];
    int n = 0, i = 0, j = 0, area = INPUT_H * INPUT_W;
    for(i = 0;i < INPUT_H; i++)
    {
        for(j = 0;j < INPUT_W; j++)
        {
            int max_id = 0;
            float max_prob = -1000;
            for (n = 0; n < 5; n++)
            {
                float pix_data = result[n*area+i*INPUT_W+j];
                if(pix_data > max_prob)
                {
                    max_prob = pix_data;
                    max_id = n;
                }
            }
            outimg.at<uchar>(i,j) = max_id;
        }
    }
    cv::Mat tmp = cv::Mat(cv::Size(INPUT_W - 2 * left, INPUT_H - 2 * top), CV_8UC1, cv::Scalar(0));
    if (top == 0 && left == 0)
    {
        resize(outimg, mask, cv::Size(img.cols, img.rows), INTER_NEAREST);
    }
    else if (left == 0)
    {
        memcpy(tmp.data, outimg.data + top * INPUT_W * 1, (INPUT_H - 2 * top) * INPUT_W * 1);
        resize(tmp, mask, cv::Size(img.cols, img.rows),INTER_NEAREST);
    }
    else if (top == 0)
    {
        for (int i = 0; i < INPUT_H; i++)
        {
            memcpy(tmp.data + i * (INPUT_W - 2 * left) * 1, outimg.data + i * INPUT_W * 1 + left * 1, (INPUT_W - 2 * left) * 1);
        }
        resize(tmp, mask, cv::Size(img.cols, img.rows),INTER_NEAREST);
    }
//     Mat binaryImage;
//     threshold(mask, binaryImage,0,255, THRESH_BINARY);

//     vector<vector<Point>> contours;
// 	findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));  ///查找轮廓，一个图片中可能有多个轮廓
//     vector<vector<Point>> contours_poly(contours.size());

//     vector<Point2f>center(contours.size());
//     vector<float>radius(contours.size());
// 	for (int i=0;i<contours.size();i++)
// 	{
// 		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);  //epsilon==3 , 求出第i个轮廓的外接多边形
//         minEnclosingCircle(contours_poly[i], center[i], radius[i]);   ///求出轮廓的最小包围圆形
//         int id = mask.at<uchar>(int(center[i].y), int(center[i].x));   ///根据轮廓的最小包围圆形的中心点，求出该轮廓的类别
//         Polygon poly{contours_poly[i], class_names[id]};
//         detectedPolygons.push_back(poly);
//     }
    return 0;
}

vector<cv::Mat> SampleDetector::generate_rgb(Mat frame, Mat mask,vector<int> &areas,vector<bool> &enable,vector<vector<int>>& colors)
{
    Mat bg_smoke = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
    Mat fire_smoke = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
    Mat black_smoke = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
    Mat white_smoke = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
    Mat yellow_smoke = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
    Mat fire = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
    Mat black = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
    Mat white = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
    Mat yellow = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
    vector<cv::Mat> res;
    res.push_back(fire_smoke);
    res.push_back(black_smoke);
    res.push_back(white_smoke);
    res.push_back(yellow_smoke);
    res.push_back(bg_smoke);
    res.push_back(fire);
    res.push_back(black);
    res.push_back(white);
    res.push_back(yellow);
    int i = 0, j = 0;
    for(i = 0;i < mask.rows; i++)
    {
        for(j = 0;j < mask.cols; j++)
        {
            int id = mask.at<uchar>(i,j);
            if(id>0 &&enable[id])
            {
                areas[id]++;
                res[id-1].at<Vec3b>(i, j)[0] = colors[id][0];
                res[id-1].at<Vec3b>(i, j)[1] = colors[id][1];
                res[id-1].at<Vec3b>(i, j)[2] = colors[id][2];
                res[id-1+5].at<Vec3b>(i, j)[0] = frame.at<Vec3b>(i, j)[0];
                res[id-1+5].at<Vec3b>(i, j)[1] = frame.at<Vec3b>(i, j)[1];
                res[id-1+5].at<Vec3b>(i, j)[2] = frame.at<Vec3b>(i, j)[2];
                
            }else{
                res[4].at<Vec3b>(i, j)[0] = frame.at<Vec3b>(i, j)[0];
                res[4].at<Vec3b>(i, j)[1] = frame.at<Vec3b>(i, j)[1];
                res[4].at<Vec3b>(i, j)[2] = frame.at<Vec3b>(i, j)[2];
            }
        }
    }
    
    return res;
}