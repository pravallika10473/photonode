import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection

def main():
    try:
        # Load image
        image_path = "/uufs/chpc.utah.edu/common/home/u1475870/photonode/test_images/detection_result.jpg"
        image = Image.open(image_path)
        if not image:
            raise ValueError(f"Failed to load image from {image_path}")

        # Load model and processor
        model_name = "Pravallika6/detr-resnet-50_finetuned_cppe5"
        image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModelForObjectDetection.from_pretrained(model_name)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Process image
        with torch.no_grad():
            # Prepare inputs
            inputs = image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get predictions
            outputs = model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image.size[::-1]]).to(device)
            results = image_processor.post_process_object_detection(
                outputs, 
                threshold=0.5,
                target_sizes=target_sizes
            )[0]

        # Print results
        if len(results["scores"]) == 0:
            print("No objects detected.")
        else:
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
        
        print("Processing completed successfully")
        draw = ImageDraw.Draw(image)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = tuple(box)
            draw.rectangle((x, y, x2, y2), outline="red", width=1)
            draw.text((x, y), model.config.id2label[label.item()], fill="white")

        image.save("/uufs/chpc.utah.edu/common/home/u1475870/photonode/test_images/inference_result.jpg")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
