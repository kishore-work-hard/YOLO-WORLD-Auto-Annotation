import os
import shutil
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import threading
from PIL import Image, ImageTk
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

def read_classes_from_file(file_path):
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file.readlines()]
    return classes

def coco(bboxes, classes, frame):
    with PIL.Image.open(frame) as img:
        img_size = img.size

    coco_formats = []
    for bbox, cls in zip(bboxes, classes):
        x, y, w, h = bbox
        normalized_x = x / img_size[0]
        normalized_y = y / img_size[1]
        normalized_w = w / img_size[0]
        normalized_h = h / img_size[1]

        class_id = cls.item() if torch.is_tensor(cls) else cls

        coco_format = f"{class_id} {normalized_x:.6f} {normalized_y:.6f} {normalized_w:.6f} {normalized_h:.6f}"
        coco_formats.append(coco_format)

    return coco_formats

def run_image(
        runner,
        input_image,
        class_names,
        max_num_boxes=100,
        score_thr=0.05,
        nms_thr=0.5,
        output_image="output.png",
):
    texts = [[t.strip()] for t in class_names] + [[" "]]
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
    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )

    px = pd.DataFrame(detections.xyxy).astype("float")
    class_id = pd.DataFrame(detections.class_id).astype("int")
    confidence = pd.DataFrame(detections.confidence).astype("float")
    px["4"] = confidence
    px["5"] = class_id
    print(px)

    boxes = torch.tensor(detections.xyxy)
    class_id = torch.tensor(detections.class_id)
    confidence = torch.tensor(detections.confidence)
    xywh = torch.stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
                        boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]), dim=1)

    coco_format = coco(xywh, class_id, input_image)
    return coco_format



def process_images_in_folder(runner, input_folder, output_folder, progress_var, status_var):
    total_files = len([filename for filename in os.listdir(input_folder) if filename.endswith(('.jpg', '.jpeg', '.png'))])
    processed_files = 0
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_image = os.path.join(input_folder, filename)
            output_image = os.path.join(output_folder, filename)
            output_txt = os.path.splitext(output_image)[0] + ".txt"

            # Process the image
            coco_format = run_image(runner, input_image, class_names)


            # Write COCO format to a .txt file
            with open(output_txt, "w") as txt_file:
                for format in coco_format:
                    txt_file.write(format + "\n")

            # Move the image and .txt file to the output folder
            shutil.move(input_image, output_image)
            processed_files += 1
            progress_var.set(int(processed_files / total_files * 100))  # Update progress
            status_var.set(f"Processed {processed_files}/{total_files} files")  # Update status
    status_var.set("Done")

# Create Tkinter GUI
root = tk.Tk()
root.title("Image Processing GUI")

# Select input folder
def select_input_folder():
    input_folder = filedialog.askdirectory()
    input_folder_entry.delete(0, tk.END)
    input_folder_entry.insert(0, input_folder)

# Select output folder
def select_output_folder():
    output_folder = filedialog.askdirectory()
    output_folder_entry.delete(0, tk.END)
    output_folder_entry.insert(0, output_folder)

# Process images button
def process_images_thread():
    input_folder = input_folder_entry.get()
    output_folder = output_folder_entry.get()
    progress_var.set(0)  # Reset progress
    status_var.set("Processing...")
    threading.Thread(target=process_images_in_folder, args=(runner, input_folder, output_folder, progress_var, status_var)).start()

# Input folder label and entry
input_folder_label = tk.Label(root, text="Input Folder:")
input_folder_label.grid(row=0, column=0)
input_folder_entry = tk.Entry(root, width=50)
input_folder_entry.grid(row=0, column=1)
input_folder_button = tk.Button(root, text="Browse", command=select_input_folder)
input_folder_button.grid(row=0, column=2)

# Output folder label and entry
output_folder_label = tk.Label(root, text="Output Folder:")
output_folder_label.grid(row=1, column=0)
output_folder_entry = tk.Entry(root, width=50)
output_folder_entry.grid(row=1, column=1)
output_folder_button = tk.Button(root, text="Browse", command=select_output_folder)
output_folder_button.grid(row=1, column=2)

# Load class names from file
class_names = read_classes_from_file("classes.txt")

cfg = Config.fromfile("./configs/pretrain/cfg.py")
cfg.work_dir = "."
cfg.load_from = "yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pth"
runner = Runner.from_cfg(cfg)
runner.call_hook("before_run")
runner.load_or_resume()
pipeline = cfg.test_dataloader.dataset.pipeline
runner.pipeline = Compose(pipeline)
runner.model.eval()

# Process images button
process_button = tk.Button(root, text="Process Images", command=process_images_thread)
process_button.grid(row=2, column=1)

# Progress bar
progress_var = tk.IntVar()
progress = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", variable=progress_var)
progress.grid(row=3, column=0, columnspan=3, pady=10)

# Status label
status_var = tk.StringVar()
status_label = tk.Label(root, textvariable=status_var)
status_label.grid(row=4, column=0, columnspan=3)

root.mainloop()

