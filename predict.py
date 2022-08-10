import torch
import torchvision
import time
import numpy as np
import argparse
from PIL import Image

from torchvision import transforms
import matplotlib.pyplot as plt

from detection.draw_box_utils import draw_objs

indices = {"1": "person", }


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


@torch.inference_mode()
def predict(args):
    ori_img = Image.open(args.img_path).convert('RGB')
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(ori_img)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    print(model.load_state_dict(torch.load(args.weights_path, map_location="cpu")))
    model.to(device)

    model.eval()
    for _ in range(10):
        _ = model([img.to(device)])

    t_start = time_synchronized()

    predictions = model([img.to(device)])[0]

    predict_boxes = predictions["boxes"].to("cpu").numpy()
    predict_classes = predictions["labels"].to("cpu").numpy()
    predict_scores = predictions["scores"].to("cpu").numpy()
    predict_mask = predictions["masks"].to("cpu").numpy()
    predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

    if len(predict_boxes) == 0:
        print("没有检测到任何目标!")
        return

    plot_img = draw_objs(ori_img,
                         boxes=predict_boxes,
                         classes=predict_classes,
                         scores=predict_scores,
                         masks=predict_mask,
                         category_index=indices,
                         box_thresh=args.box_conf,
                         mask_thresh=args.mask_conf,
                         line_thickness=3,
                         font='arial.ttf',
                         font_size=20)
    plt.imshow(plot_img)
    plt.show()
    # 保存预测的图片结果
    plot_img.save("test_result.jpg")

    t_end = time_synchronized()
    print('inference time {}'.format(t_end - t_start))


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="预测")
    parse.add_argument('--weights-path', type=str, default='MaskRCNN_9.pth')
    parse.add_argument('--img-path', type=str, default='./1.jpg')
    parse.add_argument('--box-conf', type=float, default=0.7)
    parse.add_argument('--mask-conf', type=float, default=0.5)
    args = parse.parse_args()
    predict(args)
