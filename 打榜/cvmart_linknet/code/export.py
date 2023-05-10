from linkNet import LinkNet
import torch
path="/project/train/models/model.pth"
onnx_path="/project/train/models/model.onnx"
net=LinkNet(5)
net.load_state_dict(torch.load(path, map_location='cpu'))
test_arr = torch.randn(1,3,512,512)
input_names = ['input']
output_names = ['output']
torch.onnx.export(
    net,
    test_arr,
    onnx_path,
    verbose=False,
    opset_version=11,
    input_names=input_names,
    output_names=output_names,
    # dynamic_axes={"input":{3:"width"}}            #动态推理W纬度，若需其他动态纬度可以自行修改，不需要动态推理的话可以注释这行
)
print('->>模型转换成功！')