import torch
import torchvision
import cv2
import random
import time

from torchvision.io import ImageReadMode, read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

import argparse

kinds = ['background', 'person']  # 类别


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return b, g, r


@torch.inference_mode()
def predict(args):
    img = read_image(args.img_path, ImageReadMode.RGB)
    ori_img = img.clone()
    img = img.float() / 255.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    print(model.load_state_dict(torch.load(args.weights_path, map_location="cpu")))
    model.to(device)

    model.eval()
    for _ in range(10):
        _ = model([img.to(device)])

    t_start = time_synchronized()

    predictions = model([img.to(device)])[0]
    boxes = predictions['boxes'].cpu()
    labels = predictions['labels'].cpu()
    scores = predictions['scores'].cpu()
    masks = predictions['masks'].cpu()

    inds = torch.where(scores >= args.box_conf)[0]
    for idx in inds:
        color = random_color()
        mask = masks[idx, 0] > args.mask_conf
        label = labels[idx].item()
        box = boxes[idx].view(-1, 4)
        ori_img = draw_segmentation_masks(ori_img, mask, alpha=0.5, colors=color)
        ori_img = draw_bounding_boxes(ori_img, box, labels=[kinds[label] + ': ' + str(scores[idx].item())[:4]], colors=color)

    cv2.imwrite('./res.jpg', ori_img.numpy().transpose(1, 2, 0)[:, :, ::-1])
    t_end = time_synchronized()
    print('inference time {}'.format(t_end - t_start))


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="利用torchvision集成工具生成预测结果")
    parse.add_argument('--weights-path', type=str, default='MaskRCNN_9.pth')
    parse.add_argument('--img-path', type=str, default='./1.jpg')
    parse.add_argument('--box-conf', type=float, default=0.7)
    parse.add_argument('--mask-conf', type=float, default=0.5)
    args = parse.parse_args()
    predict(args)
