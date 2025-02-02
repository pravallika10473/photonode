#!/usr/bin/env python3
"""
Training script for DETR object detection model on CPPE-5 dataset.
"""

import torch
import numpy as np
import albumentations
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)
from transformers import DetrConfig, DetrForObjectDetection


def format_annotations(image_id, category, area, bbox):
    """Format annotations according to DETR requirements."""
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


def create_transform():
    """Create albumentations transform pipeline."""
    return albumentations.Compose(
        [
            albumentations.Resize(480, 480),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.2),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
        bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
    )


def create_transform_function(image_processor, transform):
    """Create function to transform and augment annotations."""
    def transform_aug_ann(examples):
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        
        for image, objects in zip(examples["image"], examples["objects"]):
            # Convert to numpy array and normalize to [0, 1]
            image = np.array(image.convert("RGB"))
            image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": format_annotations(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        return image_processor(images=images, annotations=targets, return_tensors="pt")
    
    return transform_aug_ann


def create_data_collator(image_processor):
    """Create collate function for data loader."""
    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        
        # Use the correct method for padding
        encoding = image_processor(
            pixel_values,
            return_tensors="pt"
        )
        
        labels = [item["labels"] for item in batch]
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        # Create pixel mask based on non-zero values
        batch["pixel_mask"] = (encoding["pixel_values"].sum(dim=1) != 0).float()
        batch["labels"] = labels
        return batch
    
    return collate_fn


def prepare_dataset(dataset):
    """Prepare and clean the dataset."""
    # Create label mappings
    categories = dataset["train"].features["objects"].feature["category"].names
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}
    
    # Remove problematic examples
    remove_idx = [590, 821, 822, 875, 876, 878, 879]
    keep = [i for i in range(len(dataset["train"])) if i not in remove_idx]
    dataset["train"] = dataset["train"].select(keep)
    
    return dataset, id2label, label2id


def main():
    # Load and prepare dataset
    cppe5 = load_dataset("cppe-5")
    cppe5, id2label, label2id = prepare_dataset(cppe5)
    
    # Initialize image processor with correct size parameters
    checkpoint = "facebook/detr-resnet-50"
    image_processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        do_resize=True,
        size={
            "shortest_edge": 480,
            "longest_edge": 480
        },
        do_pad=True,
        do_rescale=True,
        rescale_factor=1/255,
        return_tensors="pt",
        use_fast=True  # Enable fast mode
    )
    
    # Set up transformations
    transform = create_transform()
    transform_func = create_transform_function(image_processor, transform)
    cppe5["train"] = cppe5["train"].with_transform(transform_func)
    
    # Initialize model with custom configuration
    config = DetrConfig(
        num_labels=5,  # CPPE-5 has 5 classes
        id2label=id2label,
        label2id=label2id,
        backbone="resnet50",
        use_pretrained_backbone=True,
        use_timm_backbone=True,
        num_channels=3,
        num_queries=100,
    )
    
    # Create a new model with our config
    model = DetrForObjectDetection(config)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="/scratch/general/vast/u1475870/photonode/model_output",
        per_device_train_batch_size=8,
        num_train_epochs=10,
        fp16=True,
        save_steps=200,
        logging_steps=50,
        learning_rate=1e-5,
        weight_decay=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
    )
    
    # Initialize trainer with processing_class instead of tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=create_data_collator(image_processor),
        train_dataset=cppe5["train"],
        processing_class=image_processor,  # Updated from tokenizer
    )
    
    # Train model
    trainer.train()


if __name__ == "__main__":
    main()