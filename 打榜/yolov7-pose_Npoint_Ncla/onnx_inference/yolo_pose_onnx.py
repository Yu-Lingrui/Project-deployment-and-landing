import onnxruntime
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression
from utils.plots import output_to_target, plot_skeleton_kpts

device = torch.device("cuda:0")

image = cv2.imread('../1.jpg')
image = cv2.resize(image, (640, 640))
image = letterbox(image, 640, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]), dtype=torch.float16)

if torch.cuda.is_available():
    image = image.to(device)

print(image.shape)
session = onnxruntime.InferenceSession('../best.onnx', None)
input_name = session.get_inputs()[0].name
output = session.run([], {input_name: image.detach().cpu().numpy()})
for i in range(len(output)):
    output[i] = np.array(output[i]).reshape(-1, 62)
output = np.concatenate((output[:]), axis=0)
output = np.expand_dims(output, axis=0)
output = torch.from_numpy(output)

output = non_max_suppression(output, 0.05, 0.45, multi_label=True, kpt_label=True, nc=6, nkpt=17)
output = output_to_target(output)
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
for idx in range(output.shape[0]):
    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

# matplotlib inline
cv2.imwrite("tmp.png", nimg)

