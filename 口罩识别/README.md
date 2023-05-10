# 口罩识别
## 1、背景描述
 
项目背景：厨师口罩识别算法就是应用于厨房监控视频中通过机器视觉算法识别是否带了口罩，可对进入作业区域的人员进行自动识别：若检测到人员未佩戴口罩，可立即报警，报警信号同步推送至管理人员。该算法极大地提升了明厨亮灶项目更好的落地，也提高了消费者对厨房重地的信任。


## 2、训练模型
实战项目选的是口罩识别，模型选择yolov5s。首先，在训练之前，我们将训练集进行划分训练集：测试集为8:2。数据集保存在/home/data中，我们将VOC格式的数据集转成txt格式给yolov5调用。

提升精度：使用kmeans聚类，尝试将anchor大小更贴近小目标(yolov5自带autoanchor，效果不大)。将backbone冻结，用yolov5自带超参数搜索(遗传)，最后用搜索出的yaml文件进行训练，效果略微提升。后面发现训练过程中明显欠拟合，改用yolo5m文件配置进行训练，性能轻松上0.7。

测试的时候有两个类别不平衡(其实影响不大)，可选取 --image-weights(数据加载权重分配),增加fl_gamma的值(不建议处理不平衡，会影响其他主要类别的精度和召回)。

训练img_ize=480,防止cuda内存不够(默认640)

平台T4功率较低，训练轮次太少，可以保存好训练的权重文件，下次继续预加载训练。


## 3、转换 onnx模型
yolov5仓库地址（下载v5_6.1版本）： https://github.com/ultralytics/yolov5
### 1) 配置环境
onnx>=1.9.0  
onnx-simplifier>=0.4.1  

### 2) export.py 导出 onnx
yolov5官方只支持tensorrt为8的推理，而平台为7，需要修改导出代码，参照3rd/export.py
python export.py --data ./data/cvmart.yaml --weights /project/train/models/exp/weights/best.pt --simplify --include onnx
由于tensorrt和opencv不支持INT32，需要利用转换工具将其转成INT16
convert仓库地址： https://github.com/aadhithya/onnx-typecast (仓库用的python2，py3注释掉logger)
python ../convert.py /project/train/models/exp/weights/best.onnx /project/train/models/exp/weights/best.onnx

**注意挂载模型的路径**

### 3) 可视化onnx 
工具网址： https://netron.app

输出最后维度： box（x_center，y_center，width，height） + box_score + 类别信息



## 4、下载封装代码并修改
	gitee仓库地址：https://gitee.com/cvmart/ev_sdk_demo4.0_pedestrian_intrusion
	cp -r ev_sdk_demo4.0_pedestrian_intrusion-master/* ./ev_sdk/

### 1）修改配置文件
 - config/algo_config.json
	"mark_text_en": ["person"],
    	"mark_text_zh": ["行人"], 
 - src/Configuration.hpp 
    std::map<std::string, std::vector<std::string> > targetRectTextMap = { {"en",{"vehicle", "plate"}}, {"zh", {"车辆","车牌"}}};// 检测目标框顶部文字
 -   // 修改，定义报警类型    
   std::vector<int> alarmType = {1,2,3,8,9,10};//根据实际业务需求
    int target_count;
   
### 2）修改模型路径
 - src/SampleAlgorithm.cpp

### 3）修改模型推理
 - src/SampleDetector.cpp

 #### yolov5的话改变通道顺序变为RGB
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex]);
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex] + sizeof(float) * dims_i.d[2] * dims_i.d[3] );
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex] + 2 * sizeof(float) * dims_i.d[2] * dims_i.d[3]);
 #### 修改bug，height对应rows
    float r = std::min(m_InputSize.height / static_cast<float>(img.rows), m_InputSize.width / static_cast<float>(img.cols));
 #### 除以255
    m_Resized.convertTo(m_Normalized, CV_32FC3, 1.0/255);

    src/SampleAlgorithm.cpp 修改 ProcessImage 报警逻辑
    mConfig.target_count = 0;//业务输出需要
    for (auto &obj : detectedObjects)
    {
        for (auto &roiPolygon : mConfig.currentROIOrigPolygons)
        {   
            auto iter = find(mConfig.alarmType.begin(), mConfig.alarmType.end(), obj.label);
            if(iter != mConfig.alarmType.end())
            {
                mConfig.target_count++;
            }
            if(iter == mConfig.alarmType.end())
            {
                continue;
            }
        }
    }
        

如果下载的车牌识别sdk，按如下修改:
 #### 在头文件里SampleDetector类增加变量 m_iBoxNums——>模型输出box数
	int m_iBoxNums;
 #### 在SampleDetector::Init 初始化m_iBoxNums——>模型输出box数
	m_iBoxNums = dims_i.d[1];
 #### 修改后处理部分 generate_yolox_proposals方法
 ```
void SampleDetector::generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<BoxInfo>& objects)
{
    float *data = (float *)feat_blob;
    int dimensions = m_iClassNums + 5;
    for (int i = 0; i < m_iBoxNums; ++i) {

    float confidence = data[4];
    if (confidence >= prob_threshold) {
        float * classes_scores = data + 5;
        cv::Mat scores(1, m_iClassNums, CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
        if (max_class_score*confidence > prob_threshold) {
            BoxInfo box;
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];
            box.score = confidence;
            box.label = class_id.x;
            box.x1 = x - 0.5 * w;
            box.y1 = y - 0.5 * h;
            box.x2 = x + 0.5 * w;
            box.y2 = y + 0.5 * h;
            objects.push_back(box);
        }
    }
    data += dimensions;
    }
    runNms(objects, 0.3);
    // printf("objects num:%d\n", objects.size());
}
```

## 5、编译测试

### 1）编译
  - 编译SDK库
    mkdir -p /usr/local/ev_sdk/build
    cd /usr/local/ev_sdk/build
    cmake ..
    make install 
  - 编译测试工具
    mkdir -p /usr/local/ev_sdk/test/build
    cd /usr/local/ev_sdk/test/build
    cmake ..
    make install 

### 2）调试
  - 输入单张图片(视频)，需要指定输入输出文件
    /usr/local/ev_sdk/bin/test-ji-api -f 1 -i /project/inputs/kouzhao1.mp4 -o /project/outputs/result.mp4

## 6、提交封装测试
改好模型目录和阈值
`/project/train/models/exp/weights/best.onnx`
需要重新make！！！

**注意：yolov5中backbone的C3模块检测较慢，且yolov5m参数较大，算法测试fps会相对很低，所以总的算法测试分数可能还没有yolov5s高；对检测速度要求高的部署可直接选取轻量化模型(fastestnet)，或者对yolo主干部分轻量化处理(shuffle,mobile,深度可分离等)(如果tensorrt版本不支持一些结构处理，可能要分离backbone做单独处理)，tensorrt可选取(INT_8,效果一般)**

### 代码目录结构
```
ev_sdk
|-- 3rd             # 第三方源码或库目录，发布时请删除
|   |-- wkt_parser          # 针对使用WKT格式编写的字符串的解析器
|   |-- jsoncpp_simple         # jsoncpp库，简单易用
|   `-- fonts         # 支持中文画图的字体库
|-- bin               # test_api
|-- build             # 编译文件
|-- CMakeLists.txt          # 本项目的cmake构建文件
|-- README.md       # 本说明文件
|-- model           # 模型数据存放文件夹
|-- config          # 程序配置目录
|   |-- README.md   # algo_config.json文件各个参数的说明和配置方法
|   `-- algo_config.json    # 程序配置文件
|-- doc
|-- include         # 库头文件目录
|   `-- ji.h        # libji.so的头文件，理论上仅有唯一一个头文件
|-- lib             # 本项目编译并安装之后，默认会将依赖的库放在该目录，包括libji.so
|-- src             # 实现ji.cpp的代码
`-- test            # 针对ji.h中所定义接口的测试代码，请勿修改！！！
```
## 7、本地云服务器测试
由于真实项目数据集的保护私密性，不导出训练真实模型。本次只利用yolo官方预训练模型利用flask完成本地服务器测试