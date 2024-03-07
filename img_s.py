import cv2
import numpy as np
import torch
import pandas as pd
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms
import PIL.Image
import supervision as sv

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

class_names = ("car")
def coco(bboxes, classes, frame):
    with PIL.Image.open(frame) as img:
        img_size = img.size

    coco_formats = []
    for bbox, cls in zip(bboxes, classes):
        # Compute normalized coordinates
        x, y, w, h = bbox
        normalized_x = x / img_size[0]
        normalized_y = y / img_size[1]
        normalized_w = w / img_size[0]
        normalized_h = h / img_size[1]

        # Get class ID
        class_id = cls.item() if torch.is_tensor(cls) else cls

        # Combine into COCO format string
        coco_format = f"{class_id} {normalized_x:.6f} {normalized_y:.6f} {normalized_w:.6f} {normalized_h:.6f}"
        # print(coco_format)
        coco_formats.append(coco_format)

    return coco_formats

def xyxy_to_xywh(boxes):
    x_center = (boxes[:, 0] + boxes[:, 2]) / 2
    y_center = (boxes[:, 1] + boxes[:, 3]) / 2
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    return torch.stack((x_center, y_center, width, height), dim=1)
def run_image(
        runner,
        input_image,
        max_num_boxes=100,
        score_thr=0.05,
        nms_thr=0.5,
        output_image="output.png",
):
    texts = [[t.strip()] for t in class_names.split(",")] + [[" "]]
    data_info = runner.pipeline(dict(img_id=0, img_path=input_image,
                                     texts=texts))

    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        runner.model.class_names = texts
        pred_instances = output.pred_instances

    keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep_idxs]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    print("pred_instances")
    print(pred_instances)
    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )
    print("detections.xyxy")
    print(detections.xyxy)
    px = pd.DataFrame(detections.xyxy).astype("float")
    class_id = pd.DataFrame(detections.class_id).astype("int")
    confidence = pd.DataFrame(detections.confidence).astype("float")
    px["4"] = confidence
    px["5"] = class_id
    print(px)
    boxes = torch.tensor(detections.xyxy)
    class_id = torch.tensor(detections.class_id)
    confidence = torch.tensor(detections.confidence)
    xywh = xyxy_to_xywh(boxes)

    coco_format = coco(xywh, class_id, input_image)
    for format in coco_format:
        print(format)


if __name__ == "__main__":
    # load config
    cfg = Config.fromfile(
        "./configs/pretrain/cfg.py"
    )
    cfg.work_dir = "."
    cfg.load_from = "yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pth"
    runner = Runner.from_cfg(cfg)
    runner.call_hook("before_run")
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    runner.model.eval()
    run_image(runner, "1car.jpg")
    # sv.plot_image(img)

