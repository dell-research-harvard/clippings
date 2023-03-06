from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

print(processor)
print(processor.feature_extractor)



url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

text_features=(processor.tokenizer(["a photo of a cat","a photo of a dog"], return_tensors="pt", padding=True))

print(text_features.keys())
model_output=model.forward(input_ids=text_features["input_ids"],pixel_values=image,
                                    attention_mask=text_features["attention_mask"])

print(model_output)
# outputs = model(**inputs)
print(model_output.keys())
logits_per_image = model_output.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

##Check which probability is higher (argmax) 
print(probs.argmax(dim=1))

