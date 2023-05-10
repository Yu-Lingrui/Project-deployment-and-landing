#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
//jsoncpp 相关的头文件
#include "reader.h"
#include "writer.h"
#include "value.h"
#include "SampleDetector.hpp"
#define random(a,b) (rand()%(b-a)+a)
static const int INPUT_H = 512;
static const int INPUT_W = 512;

SampleDetector::SampleDetector()
{
    
}

SampleDetector::~SampleDetector()
{
    UnInit();
}
cv::Mat SampleDetector::ResizeImage(cv::Mat &srcimg, int *newh, int *neww, int *top, int *left)
{
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = INPUT_H;
    *neww = INPUT_W;
    cv::Mat dstimg;
    if (srch != srcw)
    {
        float hw_scale = (float)srch / srcw;
        if (hw_scale > 1)
        {
            *newh = INPUT_H;
            *neww = int(INPUT_W/ hw_scale);
            cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
            *left = int((INPUT_W - *neww) * 0.5);
            cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, INPUT_W - *neww - *left, cv::BORDER_CONSTANT, 0);
        }
        else
        {
            *newh = (int)INPUT_H * hw_scale;
            *neww = INPUT_W;
            cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
            *top = (int)(INPUT_H - *newh) * 0.5;
            cv::copyMakeBorder(dstimg, dstimg, *top, INPUT_H - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 0);
        }
    }
    else
    {
        cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
    }
    return dstimg;
}
STATUS SampleDetector::Init()
{        
    std::string labelPath = "/usr/local/ev_sdk/model_1/label.txt";
    // std::string labelPath = "/project/train/models/label.txt";
    std::ifstream ifs(labelPath);
    if (!ifs)
    {
        LOG(ERROR) << labelPath << " not found!";
        return ERROR_INIT;
    }

    srand((int)time(0));  // 产生随机种子
    std::string line;
    while (std::getline(ifs, line))
    {
        class_names.push_back(line);
        Vec3b rgb;
        int n = random(0, 255);
        rgb[0] = n;
        n = random(0, 255);
        rgb[1] = n;
        n = random(0, 255);
        rgb[2] = n;
        class_colors.push_back(rgb);
    }
    
    this->num_class  = class_names.size();

    string modelPath = "/usr/local/ev_sdk/model/model.onnx";
    // string modelPath = "/project/train/models/best.onnx";
    
    mSessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    mSession = new Session(mEnv, modelPath.c_str(), mSessionOptions);
    size_t numInputNodes = mSession->GetInputCount();
    size_t numOutputNodes = mSession->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < numInputNodes; i++)
    {
        mInputNames.push_back(mSession->GetInputName(i, allocator));
    }
    for (int i = 0; i < numOutputNodes; i++)
    {
        mOutputNames.push_back(mSession->GetOutputName(i, allocator));
    }
    LOG(INFO) << "Init Done.";
    return 0;
}

STATUS SampleDetector::UnInit()
{
    if( mSession != nullptr)
    {
        delete mSession;
        mSession = nullptr;
    }
}

void SampleDetector::Normalize(Mat img)
{
//    img.convertTo(img, CV_32F);
    int row = img.rows;
    int col = img.cols;
    this->input_image_.resize(row * col * img.channels());
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                float pix = img.ptr<uchar>(i)[j * 3 + c];
                this->input_image_[c * row * col + i * col + j] = pix/255.0;
            }
        }
    }
}


STATUS SampleDetector::ProcessImage(cv::Mat &inFrame, Mat &mask, vector<Polygon> &detectedPolygons)
{
    if (inFrame.empty())
    {
        SDKLOG(ERROR) << "Invalid input!";
        return -1;
    }

//     Mat dstimg;
//     cvtColor(inFrame, dstimg, COLOR_BGR2RGB);
//     resize(dstimg, dstimg, Size(this->inpWidth, this->inpHeight), INTER_LINEAR);

//     this->Normalize(dstimg);
    
    int newh = 0, neww = 0, top = 0, left = 0;
    m_Resized = ResizeImage(inFrame, &newh, &neww, &top, &left);
//     m_Resized.convertTo(m_Normalized, CV_32FC3, 1 / 255.);
    this->Normalize(m_Resized);
    array<int64_t, 4> inputShape{1, 3, this->inpHeight, this->inpWidth};

    auto allocateInfo = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Value inputTensor = Value::CreateTensor<float>(allocateInfo, input_image_.data(), input_image_.size(), inputShape.data(), inputShape.size());

    // 开始推理
    vector<Value> ortOutputs = mSession->Run(RunOptions{nullptr}, &mInputNames[0], &inputTensor, 1, mOutputNames.data(), mOutputNames.size());
    const float *outs = ortOutputs[0].GetTensorMutableData<float>();

    Mat outimg = Mat::zeros(this->inpHeight, this->inpWidth, CV_8UC1);
    int n = 0, i = 0, j = 0, area = this->inpHeight * this->inpWidth;
    for(i = 0;i < this->inpHeight; i++)
    {
        for(j = 0;j < this->inpWidth; j++)
        {
            int max_id = 0;
            float max_prob = -1000;
            for (n = 0; n < this->num_class; n++)
            {
                const float pix_data = outs[n*area+i*this->inpWidth+j];
                if(pix_data > max_prob)
                {
                    max_prob = pix_data;
                    max_id = n;
                }
            }
            outimg.at<uchar>(i,j) = max_id;
        }
    }
//     resize(outimg, mask, Size(inFrame.cols, inFrame.rows), INTER_NEAREST);
    cv::Mat tmp = cv::Mat(cv::Size(INPUT_W - 2 * left, INPUT_H - 2 * top), CV_8UC1, cv::Scalar(0));
    if (top == 0 && left == 0)
    {
        resize(outimg, mask, cv::Size(inFrame.cols, inFrame.rows), INTER_NEAREST);
    }
    else if (left == 0)
    {
        memcpy(tmp.data, outimg.data + top * INPUT_W * 1, (INPUT_H - 2 * top) * INPUT_W * 1);
        resize(tmp, mask, cv::Size(inFrame.cols, inFrame.rows),INTER_NEAREST);
    }
    else if (top == 0)
    {
        for (int i = 0; i < INPUT_H; i++)
        {
            memcpy(tmp.data + i * (INPUT_W - 2 * left) * 1, outimg.data + i * INPUT_W * 1 + left * 1, (INPUT_W - 2 * left) * 1);
        }
        resize(tmp, mask, cv::Size(inFrame.cols, inFrame.rows),INTER_NEAREST);
    }
    Mat binaryImage;
    threshold(mask, binaryImage,0,255, THRESH_BINARY);

    vector<vector<Point>> contours;
	findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));  ///查找轮廓，一个图片中可能有多个轮廓
    vector<vector<Point>> contours_poly(contours.size());

    vector<Point2f>center(contours.size());
    vector<float>radius(contours.size());
    //Mat drawing(inFrame.size(), CV_8UC3, Scalar::all(0));
	for (int i=0;i<contours.size();i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);  //epsilon==3 , 求出第i个轮廓的外接多边形
        minEnclosingCircle(contours_poly[i], center[i], radius[i]);   ///求出轮廓的最小包围圆形
        int id = mask.at<uchar>(int(center[i].y), int(center[i].x));   ///根据轮廓的最小包围圆形的中心点，求出该轮廓的类别
        Polygon poly{contours_poly[i], class_names[id]};
        detectedPolygons.push_back(poly);
        
		/*drawContours(drawing, contours_poly, i, Scalar(230, 130, 255), 1);
        circle(drawing, center[i], 4, Scalar(0, 0, 255), -1);
        putText(drawing, poly.name, center[i], FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
        for(int j=0;j<poly.points.size();j++)
        {
            circle(drawing, poly.points[j], 3, Scalar(255, 0, 0), -1);
        }*/
    }
    //imwrite("/project/drawing.jpg", drawing);
    return 0;
}

cv::Mat SampleDetector::generate_rgb(Mat frame, Mat mask)
{
    Mat outimg = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
    int i = 0, j = 0;
    for(i = 0;i < mask.rows; i++)
    {
        for(j = 0;j < mask.cols; j++)
        {
            int id = mask.at<uchar>(i,j);
            if(id>0)
            {
                outimg.at<Vec3b>(i, j)[0] = this->class_colors[id][0];
                outimg.at<Vec3b>(i, j)[1] = this->class_colors[id][1];
                outimg.at<Vec3b>(i, j)[2] = this->class_colors[id][2];
            }
            else
            {
                outimg.at<Vec3b>(i, j)[0] = frame.at<Vec3b>(i, j)[0];
                outimg.at<Vec3b>(i, j)[1] = frame.at<Vec3b>(i, j)[1];
                outimg.at<Vec3b>(i, j)[2] = frame.at<Vec3b>(i, j)[2];
            }
        }
    }
    return outimg;
}
