import requests
from transformers import pipeline
import numpy as np
from PIL import Image, ImageDraw
from torchvision.ops import box_convert
# Use a pipeline as a high-level helper
# Load model directly
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from object_detection import val_dataset
import torch
import os
import matplotlib.pyplot as plt
import tqdm

# Set up model and processor
processor = AutoImageProcessor.from_pretrained("Pravallika6/detr-finetuned-credentials")
model = AutoModelForObjectDetection.from_pretrained("Pravallika6/detr-finetuned-credentials")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

id2label = {
    0: "credential",
}

def plot_results(pil_img, scores, labels, boxes, output_path):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    
    colors = COLORS * 100
    for score, label, box, c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        # Draw the bounding box
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                           fill=False, color=c, linewidth=3)
        ax.add_patch(rect)
        
        # Add label and score
        text = f'{id2label[label]}: {score:0.2f}'
        ax.text(x1, y1 - 5, text, fontsize=15,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=c))
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def process_image(pixel_values, image, image_id):
    # Move to device and add batch dimension
    pixel_values = pixel_values.unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=None)

    # Get probabilities from logits
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.65  # Adjust threshold as needed

    # Get predictions
    pred_boxes = outputs.pred_boxes[0, keep]
    scores = probas[keep].max(-1).values
    labels = probas[keep].argmax(-1)

    # Convert boxes to image size
    h, w = image.size[1], image.size[0]
    pred_boxes = pred_boxes.cpu() * torch.Tensor([w, h, w, h])
    pred_boxes = box_convert(pred_boxes, in_fmt='cxcywh', out_fmt='xyxy')

    # Create output directory if it doesn't exist
    output_dir = '/uufs/chpc.utah.edu/common/home/u1475870/photonode/predictions'
    os.makedirs(output_dir, exist_ok=True)

    # Save the visualization
    output_path = os.path.join(output_dir, f'pred_{image_id}.png')
    plot_results(image, scores, labels, pred_boxes, output_path)

    return {
        'image_id': image_id,
        'num_detections': len(scores),
        'scores': scores.tolist(),
        'boxes': pred_boxes.tolist()
    }

def main():
    # Process all validation images
    results = []
    for idx in tqdm.tqdm(range(len(val_dataset))):
        # Get image and target
        pixel_values, target = val_dataset[idx]
        image_id = target['image_id'].item()
        
        # Load original image
        image_info = val_dataset.coco.loadImgs(image_id)[0]
        image_path = os.path.join('/uufs/chpc.utah.edu/common/home/u1475870/photonode/combined_dataset/val', image_info['file_name'])
        image = Image.open(image_path)
        
        # Process image
        result = process_image(pixel_values, image, image_id)
        results.append(result)
        
        print(f"\nProcessed image {image_id}:")
        print(f"Number of detections: {result['num_detections']}")
        print(f"Confidence scores: {[f'{s:.2f}' for s in result['scores']]}")
        print("-" * 50)

if __name__ == "__main__":
    main()

