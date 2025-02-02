import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import os
import glob

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_model_path():
    """Get the model path based on environment"""
    scratch_dir = os.getenv('SCRATCH_DIR', '/scratch/general/vast/u1475870/photonode/')
    model_dir = os.path.join(scratch_dir, "model_output")
    
    # Look for latest checkpoint
    checkpoint_pattern = os.path.join(model_dir, "checkpoint-*")
    checkpoints = sorted(glob.glob(checkpoint_pattern))
    
    print(f"Looking for checkpoints in: {model_dir}")
    print(f"Found checkpoints: {checkpoints}")
    
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"Using latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    raise FileNotFoundError(f"No checkpoints found in {model_dir}. Please train the model first.")

def run_inference(image_path):
    """Run object detection inference on an image file"""
    try:
        # Load image from file
        print(f"Loading image from {image_path}")
        image = Image.open(image_path).convert('RGB')
        print(f"Image loaded successfully: size={image.size}, mode={image.mode}")
        
        # Get model path and load model
        model_path = get_model_path()
        print(f"Loading model from {model_path}")
        
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        print("Image processor loaded successfully")
        
        model = AutoModelForObjectDetection.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully and moved to device:", device)
        
        # Process image
        print("Running inference...")
        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([image.size[::-1]]).to(device)
            results = image_processor.post_process_object_detection(
                outputs, 
                threshold=0.5,
                target_sizes=target_sizes
            )[0]
        
        # Draw results
        print("Drawing results...")
        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes)
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = tuple(box)
            
            # Draw box
            draw.rectangle((x, y, x2, y2), outline="red", width=2)
            
            # Draw label
            label_text = f"{model.config.id2label[label.item()]}: {score:.2f}"
            draw.text((x, y-10), label_text, fill="red")
            
            print(f"Detected {model.config.id2label[label.item()]} with confidence {score:.3f} at location {box}")
        
        # Save results
        output_dir = os.path.join(os.getenv('SCRATCH_DIR', '/scratch/general/vast/u1475870/photonode/'), 
                                 "inference_results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "detection_result.jpg")
        image_with_boxes.save(output_path)
        print(f"Results saved to {output_path}")
        
        return results
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        print("Full error:")
        print(traceback.format_exc())
        raise

def main():
    # Use a local image file
    image_path = "/uufs/chpc.utah.edu/common/home/u1475870/photonode/test_images/image.png"
    print(f"Starting inference on {image_path}")
    results = run_inference(image_path)
    print("Inference completed successfully")

if __name__ == "__main__":
    main()