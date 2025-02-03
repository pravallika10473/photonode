from transformers import AutoModelForObjectDetection

# Load your model
model = AutoModelForObjectDetection.from_pretrained("your_model_name")

# Method 1: Check the config
print("Number of labels:", model.config.num_labels)
print("\nLabel mapping (id2label):")
print(model.config.id2label)

# Method 2: Get all unique labels
print("\nAll labels:")
for i in range(model.config.num_labels):
    if i in model.config.id2label:
        print(f"{i}: {model.config.id2label[i]}")

# Method 3: Get label2id mapping
print("\nLabel to ID mapping:")
print(model.config.label2id)
