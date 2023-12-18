#include <sys/stat.h>
#include <fstream>
#include <glog/logging.h>

#include "sample_detector.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "ji_utils.h"


using namespace ev;

SampleDetector::SampleDetector()
{
}

bool SampleDetector::Init(float thresh, const std::string &uuid)
{
    m_thresh = thresh;//传入后处理阈值
    m_uuid = uuid;//传入的模型uuid
}

bool SampleDetector::UnInit()
{
}

SampleDetector::~SampleDetector()
{
    UnInit();
}

EVStatus SampleDetector::Run(cv::Mat &in_mat, cv::Mat &out_mat, int out_width, int out_height, bool normalize=true, bool bgr2rgb=true)
{
        REC_TIME(t0);
        if (out_width < 10)
        {
            EVLOG(ERROR) << "input size too small:" << out_width;
            return EV_FAIL;
        }
        float m_scale = std::min(out_height / static_cast<float>(in_mat.rows), out_width / static_cast<float>(in_mat.cols));
        cv::Size new_size = cv::Size{in_mat.cols * m_scale, in_mat.rows * m_scale};
        cv::Mat resized_mat;
        cv::resize(in_mat, resized_mat, new_size);

        cv::Mat board = cv::Mat(cv::Size(out_width, out_height), CV_8UC3, cv::Scalar(114, 114, 114));

        resized_mat.copyTo(board(cv::Rect{0, 0, resized_mat.cols, resized_mat.rows}));

        if (bgr2rgb)
        {
            cv::cvtColor(board, board, cv::COLOR_BGR2RGB);
        }

        if (normalize)
        {
            board.convertTo(board, CV_32F, 1 / 255.);
        }
        else
        {
            board.convertTo(board, CV_32F);
        }

        out_mat = board.clone();
        std::vector<cv::Mat> chw_wrappers;
        chw_wrappers.emplace_back(out_width, out_height, CV_32FC1, out_mat.data);
        chw_wrappers.emplace_back(out_width, out_height, CV_32FC1, out_mat.data + sizeof(float) * out_height * out_width);
        chw_wrappers.emplace_back(out_width, out_height, CV_32FC1, out_mat.data + 2 * sizeof(float) * out_height * out_width);
        cv::split(board, chw_wrappers);

        REC_TIME(t1);
        EVLOG(INFO) << "YOLOv5Preprocessor run time(ms):" << RUN_TIME(t1 - t0);

        return EV_SUCCESS;
}

bool SampleDetector::ProcessImage(const cv::Mat &img, std::vector<ev::vision::BoxInfo> &det_results)
{
    det_results.clear();
    cv::Mat cv_in_mat1;
    //前处理
    SampleDetector::Run(const_cast<cv::Mat&>(img), cv_in_mat1, 320, 192);

    //准备输入数据
    EVModelData in;
    EVModelData out;
    EVMatData in_mat;

    in.desc = NULL;
    in.mat = &in_mat;

    in.mat_num = 1; // 输入图像数量,也可以是多张;如果是多张,则in.mat为数组指针

    in_mat.data = cv_in_mat1.data;
    in_mat.data_size = cv_in_mat1.cols * cv_in_mat1.rows * 3 * 4;
    in_mat.width = cv_in_mat1.cols;
    in_mat.height = cv_in_mat1.rows;
    in_mat.aligned_width = cv_in_mat1.cols;
    in_mat.aligned_height = cv_in_mat1.rows;
    in_mat.channel = 3;
    in_mat.loc = EV_DATA_HOST;
    in_mat.type = EV_UINT8;

    //执行推理
    std::lock_guard<std::mutex> lock_guard(m_mutex);//用于多线程时,线程安全
    EVDeploy::GetModel().RunInfer(m_uuid, &in, &out);
    SDKLOG(INFO) << "RunInfer done";
    // 输出的数量由out.mat_num指示,输出的数据封装在out.mat中,如果是多个输出,则out.mat为指向多个输出的指针,
    // 每一个输出的维度信息由out.mat[i]->dims指示
    // 每一个输出的名称信息由out.mat[i]->desc指示
    for (int j = 0; j < out.mat_num; ++j)
    {
        SDKLOG(INFO) << "output name: " << out.mat[j].desc;
        for (int k = 0; k < out.mat[j].dims.size(); ++k)
        {
            SDKLOG(INFO) << "dims " << k << ":" << out.mat[j].dims[k];
        }
    }

    //后处理   
    float scale = m_preprocessor.GetScale();
    m_postprocessor.Run(out.mat, det_results, scale, m_thresh, img.cols, img.rows);

    // 注意释放out.mat,否则会有内存泄露!!!!
    if (out.mat)
    {
        delete[] out.mat;
    }

    return true;
}
