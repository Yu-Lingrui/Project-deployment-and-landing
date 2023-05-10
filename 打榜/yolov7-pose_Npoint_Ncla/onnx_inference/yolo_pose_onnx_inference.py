import numpy as np
import cv2
import argparse
import onnxruntime
import torch
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="../best.onnx")

args = parser.parse_args()

device = torch.device("cuda:0")

_CLASS_COLOR_MAP = [
    (0, 0, 255),  # Person (blue).
    (255, 0, 0),  # Bear (red).
    (0, 255, 0),  # Tree (lime).
    (255, 0, 255),  # Bird (fuchsia).
    (0, 255, 255),  # Sky (aqua).
    (255, 255, 0),  # Cat (yellow).
]

palette = np.array([[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255],
                    [255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51],
                    ])

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette
radius = 3


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def read_img():
    image = cv2.imread('./1.jpg')
    # image = cv2.resize(image, (640, 640))
    image = letterbox(image, 320, stride=32, auto=False)[0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    #image = torch.tensor(np.array([image.numpy()]), dtype=torch.float16)

    if torch.cuda.is_available():
        image = image.to(device)

    return image


def model_inference(model_path=None, input=None):
    # onnx_model = onnx.load(args.model_path)
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: input.detach().cpu().numpy()})
    return output


def model_inference_image(model_path):
    input = read_img()
    output = model_inference(model_path, input)
    post_process(output[0], score_threshold=0.15)


def post_process(output, score_threshold=0.3):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    nc = 6
    det_bboxes, det_scores, det_labels, kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 5+nc:]
    img = cv2.imread("./1.jpg")
    # To generate color based on det_label, to look into the codebase of Tensorflow object detection api.
    # f = open(dst_txt_file, 'wt')
    for idx in range(len(det_bboxes)):
        det_bbox = det_bboxes[idx]
        kpt = kpts[idx]
        ######################################
        kpt[2::3] = np.round(kpt[2::3])
        kpt[0::3] *= img.shape[1] / 320
        kpt[1::3] *= img.shape[0] / 320
        det_bbox[0::2] *= img.shape[1] / 320
        det_bbox[1::2] *= img.shape[0] / 320
        ######################################
        # if det_scores[idx] > 0:
        #     f.write("{:8.0f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f}\n".format(det_labels[idx], det_scores[idx],
        #                                                                        det_bbox[1], det_bbox[0], det_bbox[3],
        #                                                                        det_bbox[2]))
        if det_scores[idx] > score_threshold:
            color_map = _CLASS_COLOR_MAP[int(det_labels[idx])]
            img = cv2.rectangle(img, (int(det_bbox[0]), int(det_bbox[1])), (int(det_bbox[2]), int(det_bbox[3])),
                                color_map[::-1], 2)
            cv2.putText(img, "id:{}".format(int(det_labels[idx])), (int(det_bbox[0] + 5), int(det_bbox[1]) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            cv2.putText(img, "score:{:2.1f}".format(det_scores[idx]), (int(det_bbox[0] + 5), int(det_bbox[1]) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            plot_skeleton_kpts(img, kpt)
    cv2.imwrite("tmp.jpg", img)
    # f.close()


def plot_skeleton_kpts(im, kpts, steps=3):
    num_kpts = len(kpts) // steps
    # plot keypoints
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5:  # Confidence of a keypoint has to be greater than 0.5
            # 如果点无穷大 直接绘制在原点
            try:
                cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
            except:
                cv2.circle(im, (int(0), int(0)), radius, (int(r), int(g), int(b)), -1)
    # plot skeleton
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
        pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
        conf1 = kpts[(sk[0] - 1) * steps + 2]
        conf2 = kpts[(sk[1] - 1) * steps + 2]
        if conf1 > 0.5 and conf2 > 0.5:  # For a limb, both the keypoint confidence must be greater than 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


def main():
    model_inference_image(model_path=args.model_path)


if __name__ == "__main__":
    main()
