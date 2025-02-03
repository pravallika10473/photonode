import requests
from transformers import pipeline
import numpy as np
from PIL import Image, ImageDraw



image = Image.open("/uufs/chpc.utah.edu/common/home/u1475870/photonode/test_images/test.jpg")

obj_detector = pipeline(
    "object-detection", model="Pravallika6/detr-resnet-50-finetuned-credential", device="cuda:0", use_fast=True

)

def plot_results(image, results, threshold=0.6):
    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for result in results:
        score = result['score']
        label = result['label']
        box = list(result['box'].values())

        if score > threshold:
            x1, y1, x2, y2 = tuple(box)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1 + 5, y1 - 10), label, fill="white")
            draw.text((x1 + 5, y1 + 10), f'{score:.2f}', fill='green' if score > 0.7 else 'red')

    return image

results = obj_detector(image)
print(results)

image = plot_results(image, results)
image.save("/uufs/chpc.utah.edu/common/home/u1475870/photonode/test_images/test_annotated.JPG")

