import glob  
import os
import clip
import torch
from PIL import Image

class_folders = ['airplane', 'alarm_clock', 'ant', 'ape', 'apple', 'armor', 'axe', 'banana', 'bat', 'bear', 'bee', 
    'beetle', 'bell', 'bench', 'bicycle', 'blimp', 'bread', 'butterfly', 'cabin', 'camel', 'candle', 'cannon', 'car_(sedan)',
    'castle', 'cat', 'chair', 'chicken', 'church', 'couch', 'cow', 'crab', 'crocodilian', 'cup', 'deer', 'dog', 'dolphin', 'door',
    'duck', 'elephant', 'eyeglasses', 'fan', 'fish', 'flower', 'frog', 'geyser', 'giraffe', 'guitar', 'hamburger', 'hammer', 'harp', 'hat',
    'hedgehog', 'helicopter', 'hermit_crab', 'horse', 'hot-air_balloon', 'hotdog', 'hourglass', 'jack-o-lantern', 'jellyfish', 'kangaroo',
    'knife', 'lion', 'lizard', 'lobster', 'motorcycle', 'mouse', 'mushroom', 'owl', 'parrot', 'pear', 'penguin', 'piano', 'pickup_truck',
    'pig', 'pineapple', 'pistol', 'pizza', 'pretzel', 'rabbit', 'raccoon', 'racket', 'ray', 'rhinoceros', 'rifle', 'rocket', 'sailboat', 
    'saw', 'saxophone', 'scissors', 'scorpion', 'sea_turtle', 'seagull', 'seal', 'shark', 'sheep', 'shoe', 'skyscraper', 'snail', 'snake', 
    'songbird', 'spider', 'spoon', 'squirrel', 'starfish', 'strawberry', 'swan', 'sword', 'table', 'tank', 'teapot', 'teddy_bear', 'tiger', 
    'tree', 'trumpet', 'turtle', 'umbrella', 'violin', 'volcano', 'wading_bird', 'wheelchair', 'windmill', 'window', 'wine_bottle', 'zebra'
]

def readFileImages(strFolderName):
    print(strFolderName)
    image_list = []
    st = os.path.join(strFolderName, "*.png")

    for filename in glob.glob(st): 
        image_list.append(filename)

    return image_list    

folder_path = os.path.join(os.getcwd(), "airplane")
image_list = readFileImages(folder_path)
print("List of images:", image_list)

# Check if the image_list is not empty before accessing its elements
if image_list:
    image = image_list[1]
    print("Selected image:", image)
else:
    print("No images found in the folder.")

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Prepare the inputs
image_path = image_list[3]
image = Image.open(image_path)

image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_folders]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{class_folders[index]:>16s}: {100 * value.item():.2f}%")

