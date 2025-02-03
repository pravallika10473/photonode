from transformers import AutoImageProcessor
from datasets import load_dataset
import albumentations
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer
from huggingface_hub import login

# Load the image processor
checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# Load your merged dataset
credentials = load_dataset("pravallika6/credentials")

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
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

# Add this function to clip bounding boxes to valid range
def clip_bboxes(bboxes):
    """Clip bounding box coordinates to [0, 1] range"""
    return [[
        max(0.0, min(1.0, coord)) 
        for coord in box
    ] for box in bboxes]

# transforming a batch
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        
        # Clip bounding boxes before transformation
        clipped_bboxes = clip_bboxes(objects["bbox"])
        out = transform(image=image, bboxes=clipped_bboxes, category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")

credentials["train"] = credentials["train"].with_transform(transform_aug_ann)

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    ignore_mismatched_sizes=True,
)

# Add this before training - you'll need your HF token from huggingface.co/settings/tokens
login(token="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")  # Replace with your actual token

training_args = TrainingArguments(
    output_dir="detr-resnet-50_finetuned_credentials",
    per_device_train_batch_size=8,
    num_train_epochs=20,
    fp16=True,
    save_steps=200,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=credentials["train"],
    tokenizer=image_processor,
)

trainer.train()
#save the model
model.save_pretrained("detr-resnet-50_finetuned_credentials")
trainer.push_to_hub("pravallika6/detr-resnet-50_finetuned_credentials")
print("Model successfully trained")



