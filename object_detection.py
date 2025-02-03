from datasets import load_dataset, Dataset
import numpy as np
from PIL import Image, ImageDraw
import albumentations as A
from transformers import AutoImageProcessor
from transformers import AutoModelForObjectDetection
from huggingface_hub import login
from transformers import TrainingArguments
from transformers import Trainer
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.nn.functional import softmax

import torch
batch_metrics = []

wandb.login(key="9534200ecfc167e1056d46e7fd8bc852064e3e87")

login(token="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")
id2label = {
    0: "credential",
}
label2id = {v: k for k, v in id2label.items()}

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]

    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels

    return batch

checkpoint = "facebook/detr-resnet-50-dc5"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

train_transform = A.Compose(
    [
        A.LongestMaxSize(500),
        A.PadIfNeeded(500, 500, border_mode=0, value=(0, 0, 0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.GaussianBlur(p=0.5),
        A.GaussNoise(p=0.5),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category"]
    ),
)





def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations

def transform_aug_ann(examples, transform):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        height, width = image.shape[:2]
        
        # Convert COCO format (x,y,w,h) to VOC format (xmin,ymin,xmax,ymax) and normalize
        voc_bboxes = []
        valid_categories = []
        for bbox, cat in zip(objects["bbox"], objects["category"]):
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                # Convert to VOC and normalize
                xmin = x / width
                ymin = y / height
                xmax = min((x + w) / width, 1.0)  # Clip to 1.0
                ymax = min((y + h) / height, 1.0)  # Clip to 1.0
                
                # Validate normalized coordinates
                if 0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1:
                    voc_bboxes.append([xmin, ymin, xmax, ymax])
                    valid_categories.append(cat)
        
        if voc_bboxes:
            out = transform(image=image, bboxes=voc_bboxes, category=valid_categories)
            
            # Denormalize back to pixel coordinates and convert to COCO format
            coco_bboxes = [
                [
                    box[0] * width,  # x
                    box[1] * height, # y
                    (box[2] - box[0]) * width,  # width
                    (box[3] - box[1]) * height  # height
                ] 
                for box in out["bboxes"]
            ]
            
            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(coco_bboxes)
            categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


def convert_voc_to_coco(bbox):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    return [xmin, ymin, width, height]

def transform_train(examples):
    return transform_aug_ann(examples, transform=train_transform)






def filter_invalid_bboxes(example):
    valid_bboxes = []
    valid_bbox_ids = []
    valid_categories = []
    valid_areas = []

    for i, bbox in enumerate(example['objects']['bbox']):
        x_min, y_min, width, height = bbox[:4]
        x_max = x_min + width
        y_max = y_min + height
        if x_min < x_max and y_min < y_max:
            valid_bboxes.append(bbox)
            valid_bbox_ids.append(example['objects']['id'][i])
            valid_categories.append(example['objects']['category'][i])
            valid_areas.append(example['objects']['area'][i])
        else:
            print(f"Image with invalid bbox: {example['image_id']} Invalid bbox detected and discarded: {bbox} - id: {example['objects']['id'][i]} - category: {example['objects']['category'][i]}")

    example['objects']['bbox'] = valid_bboxes
    example['objects']['id'] = valid_bbox_ids
    example['objects']['category'] = valid_categories
    example['objects']['area'] = valid_areas

    return example
def denormalize_boxes(boxes, width, height):
    boxes = boxes.clone()
    boxes[:, 0] *= width  # xmin
    boxes[:, 1] *= height  # ymin
    boxes[:, 2] *= width  # xmax
    boxes[:, 3] *= height  # ymax
    return boxes

# def compute_metrics(eval_pred, compute_result):
#     global batch_metrics

#     (loss_dict, scores, pred_boxes, last_hidden_state, encoder_last_hidden_state), labels = eval_pred

#     image_sizes = []
#     target = []
#     for label in labels:

#         image_sizes.append(label['orig_size'])
#         width, height = label['orig_size']
#         denormalized_boxes = denormalize_boxes(label["boxes"], width, height)
#         target.append(
#             {
#                 "boxes": denormalized_boxes,
#                 "labels": label["class_labels"],
#             }
#         )
#     predictions = []
#     for score, box, target_sizes in zip(scores, pred_boxes, image_sizes):
#         # Extract the bounding boxes, labels, and scores from the model's output
#         pred_scores = score[:, :-1]  # Exclude the no-object class
#         pred_scores = softmax(pred_scores, dim=-1)
#         width, height = target_sizes
#         pred_boxes = denormalize_boxes(box, width, height)
#         pred_labels = torch.argmax(pred_scores, dim=-1)

#         # Get the scores corresponding to the predicted labels
#         pred_scores_for_labels = torch.gather(pred_scores, 1, pred_labels.unsqueeze(-1)).squeeze(-1)
#         predictions.append(
#             {
#                 "boxes": pred_boxes,
#                 "scores": pred_scores_for_labels,
#                 "labels": pred_labels,
#             }
#         )

#     metric = MeanAveragePrecision(box_format='xywh', class_metrics=True)

#     if not compute_result:
#         # Accumulate batch-level metrics
#         batch_metrics.append({"preds": predictions, "target": target})
#         return {}
#     else:
#         # Compute final aggregated metrics
#         # Aggregate batch-level metrics (this should be done based on your metric library's needs)
#         all_preds = []
#         all_targets = []
#         for batch in batch_metrics:
#             all_preds.extend(batch["preds"])
#             all_targets.extend(batch["target"])

#         # Update metric with all accumulated predictions and targets
#         metric.update(preds=all_preds, target=all_targets)
#         metrics = metric.compute()

#         # Convert and format metrics as needed
#         classes = metrics.pop("classes")
#         map_per_class = metrics.pop("map_per_class")
#         mar_100_per_class = metrics.pop("mar_100_per_class")

#         for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
#             class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
#             metrics[f"map_{class_name}"] = class_map
#             metrics[f"mar_100_{class_name}"] = class_mar

#         # Round metrics for cleaner output
#         metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

#         # Clear batch metrics for next evaluation
#         batch_metrics = []

#         return metrics

def main():
 dataset = load_dataset('Pravallika6/credentials')
 train_dataset = dataset['train']
 length= len(train_dataset)
 train_dataset = train_dataset.filter(filter_invalid_bboxes)
 length_after = len(train_dataset)
 print(f"Length of dataset before filtering: {length}")
 print(f"Length of dataset after filtering: {length_after}")
 
 train_dataset_transformed = train_dataset.with_transform(transform_train)
 model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
 output_dir = "detr-resnet-50-finetuned-credential"
 login(token="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")
 training_args = TrainingArguments(
    
    output_dir=output_dir,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    fp16=True,
    save_steps=200,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
    
)
 wandb.init(
    project="detr-resnet-50-dc5-finetuned-credential", # change this
    name="detr-resnet-50-dc5-finetuned-credential", # change this
    config=training_args,
)
 trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset_transformed,
    tokenizer=image_processor, 

)
 trainer.train()
 trainer.push_to_hub("Pravallika6/detr-resnet-50-finetuned-credential")
 

 


if __name__ == "__main__":
    main()
