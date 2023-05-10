ev_detect 是一个方便开发者在极市平台上,便于开发检测类任务的训练工具套件.

## 训练功能使用方式:

1.在极市平台上选取对应训练套件镜像(比如开发套件-训练镜像), 点击安装按钮, ev_detect代码将自动拷贝到/project/train/src_repo/ev_detect路径下

2.修改对应的文件:

2-1. 可以修改splitData.py文件中的变量**train_split_rate**的数值,  划分训练集和验证集的划分比例, 默认为数据集内80%为训练集, 20%为验证集.其它可直接保持默认

    train_split_rate = 0.8

2-2. 修改**data/EVDATA.yaml**下的类别数量(nc参数)和类别名(names参数), 其它可直接保持默认.例如:

    nc: 6 # number of classes
    names: [ "mobile_phone","person_on_phone","hand","back_head","side_head", "front_head"]  # class names

2-3. 选取相应的模型结构, 该套件已为大家提供了对应的训练脚本, 例如选择用yolov5来训练, 就可在极市平台上直接发起训练任务, 执行

    bash /project/train/src_repo/ev_detect/train_yolov5.sh
    
    train_yolov5.sh 训练yolov5s模型
    train_yolov7.sh 训练yolov7模型
    train_yolox.sh 训练yolox-s模型

2-4. 训练提供的可供修改的参数说明(在train_yolov5.sh等脚本内, 已提供默认train_det.py的参数, 如需修改可参考下面的参数说明)

    python train_det.py
                         --weights 初始化的模型权重(yolov5s和yolov7的coco预训练权重已提供在套件中)
                         --cfg 模型结构的配置文件(如modelYaml/yolov5s.yaml)
                         --hyp 数据增强的配置文件(如data/hyps/hyp.scratch.yaml)
                         --data 数据类型的配置文件(如data/EVDATA.yaml)
                         --exit-ok 不递增exp的名字, 默认使用exp这个文件夹
                         --resume 会去读取最新的exp文件夹下的last.pt文件
                         --batch-size batch_size数量(默认为16)
                         --epochs 训练的epochs数量(默认为100)
                         --imgsz 训练图像的大小(默认为640)
                         --multi-scale 多尺度训练(默认是关闭的)
                         --optimizer 可选择优化器类型(默认为'SGD', 可供选择的为'SGD', 'Adam', 'AdamW')

备注: 模型训练参数和数据增强的配置参数均在**data/hyps/hyp.scratch.yaml**处修改, 再通过--hyp传入训练当中.

2-5. 转换模型可供修改的参数说明(在train_yolov5.sh等脚本内, 已提供默认export.py的参数, 如需修改可参考下面的参数说明)

```
python export.py
                     --weights 模型权重所在路径
                     --imgsz 输入图像大小(默认640)
                     --half 是否启用float16精度(默认不开启)
                     --include 输出的模型格式(默认输出onnx)
PS:如需转atlas模型, 可执行python export.py --include onnx, atlas
```

## 模型测试功能:

1. 复制./tests/ji.py到模型测试指定路径中(/usr/local/ev_sdk/src/ji.py), 接口代码已写好,在ji.py中,通过tensorrt来进行模型推理, 并且同一套代码兼容yolov5, yolov7, yolox三个模型结构.

2. 创建/usr/local/ev_sdk/src/labels.txt文件,并写入类别名称(可参考./tests/labels.txt的格式), 一个类别一行, 请与训练中执行的EVDATA.yaml中的类别顺序保持一致.例如:
   
       mobile_phone
       person_on_phone
       hand
       back_head
       side_head
       front_head

3. 修改ji.py中init函数里的的模型路径和类别路径,如
   
        name_path = '/usr/local/ev_sdk/src/labels.txt'
        model_path = "/project/train/models/exp/weights/last.onnx"

4. 可以在极市平台上发起模型测试, 选择相应的last.onnx, 即可完成模型测试
