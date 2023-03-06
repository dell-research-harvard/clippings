from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

print(processor)
print(processor.feature_extractor)
CLIP_BASE_TRANSFORM_CENTER= T.Compose([  
        T.Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),
        T.ToTensor(),
        ##CEnter crop
        T.CenterCrop((224, 224)),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711),)
    ])


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

image_features = CLIP_BASE_TRANSFORM_CENTER(image)

##add batch dimension
image_features=image_features.unsqueeze(0)
text_features=(processor.tokenizer(["a photo of a cat","a photo of a dog"], return_tensors="pt", padding=True,max_length=77,truncation=True))

print(text_features.keys())
model_output=model.forward(input_ids=text_features["input_ids"],pixel_values=image_features,attention_mask=text_features["attention_mask"])

print(model_output)
# outputs = model(**inputs)
print(model_output.keys())
logits_per_image = model_output.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

##Check which probability is higher (argmax) 
print(probs.argmax(dim=1))


print(processor.tokenizer)