CURRENT_DIR=$(cd $(dirname $0); pwd)
echo "${CURRENT_DIR}"
cd "${CURRENT_DIR}"
export YOLOV5_CONFIG_DIR="${CURRENT_DIR}/configs"

#把数据分为训练和测试, 可以修改相应比例
python splitData.py
# 需要修改data/EVDATA.yaml的类别, modelYaml/yolov5.yaml中nc的数量
python train_det.py --mode yolov5 --data data/EVDATA.yaml --exist-ok --cfg modelYaml/yolov5s.yaml --weights yolov5s.pt --batch-size 64 --project /project/train/models --epochs 30
# 导出export
# python tools/export.py --mode yolov5 --weights /project/train/models/exp/weights/last.pt --img 640
