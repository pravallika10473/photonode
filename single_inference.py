import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops import box_convert
import os

# Set up model and processor
processor = AutoImageProcessor.from_pretrained("Pravallika6/detr-finetuned-credentials")
model = AutoModelForObjectDetection.from_pretrained("Pravallika6/detr-finetuned-credentials")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set model to evaluation mode
model.eval()

def plot_results(pil_img, scores, labels, boxes, output_path):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    
    for score, label, box in zip(scores, labels, boxes):
        box = box.tolist()
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 fill=False, color='red', linewidth=3))
        text = f'credential: {score:0.2f}'
        ax.text(box[0], box[1], text, fontsize=15,
                bbox=dict(facecolor='white', alpha=0.8))

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_single_image(image_path):
    # Load and process image
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert outputs to numpy
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.65  # Confidence threshold

    # Convert boxes to image coordinates
    boxes = outputs.pred_boxes[0, keep]
    probas = probas[keep]
    
    # Convert boxes to [x0, y0, x1, y1] format
    boxes = box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
    
    # Scale boxes to image size
    w, h = image.size
    boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32).to(device)

    return {
        'image': image,
        'scores': probas.max(-1).values,
        'labels': probas.argmax(-1),
        'boxes': boxes
    }

def main():
    # Single image path
    image_path = "/uufs/chpc.utah.edu/common/home/u1475870/photonode/combined_dataset/val/images/1bdd4e82-IMG_3727.jpg"  # Update this with your image path
    output_dir = '/uufs/chpc.utah.edu/common/home/u1475870/photonode/outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Process image
    print(f"Processing image: {image_path}")
    result = process_single_image(image_path)
    
    # Save visualization
    output_path = os.path.join(output_dir, f'pred_single_{os.path.basename(image_path)}')
    plot_results(
        result['image'], 
        result['scores'], 
        result['labels'], 
        result['boxes'], 
        output_path
    )
    
    print("\nResults:")
    print(f"Number of detections: {len(result['scores'])}")
    print(f"Confidence scores: {[f'{s:.2f}' for s in result['scores']]}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
