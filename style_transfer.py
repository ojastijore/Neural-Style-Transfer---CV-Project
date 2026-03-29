import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load image with normalization
def load_image(path):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# user input
content_path = input("Enter content image path: ").strip().replace('"', '')
style_path = input("Enter style image path: ").strip().replace('"', '')

# fixed output directory
output_dir = r"C:\Users\asus\Documents\computer_vision_project\output"
os.makedirs(output_dir, exist_ok=True)

# output filename
file_name = input("Enter output file name: ").strip()
if not file_name.endswith(('.jpg', '.png')):
    file_name += ".jpg"

output_path = os.path.join(output_dir, file_name)

# load images
content = load_image(content_path)
style = load_image(style_path)

# pretrained model (VGG19)
model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# layers
content_layer = '21'
style_layers = ['0', '5', '10', '19', '28']

# feature extraction
def get_features(x):
    features = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name == content_layer:
            features['content'] = x
        if name in style_layers:
            features[name] = x
    return features

# gram matrix
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# extract features
content_features = {k: v.detach() for k, v in get_features(content).items()}
style_features = get_features(style)

style_grams = {layer: gram_matrix(style_features[layer]).detach() for layer in style_layers}

# target image
target = content.clone().requires_grad_(True).to(device)

optimizer = optim.Adam([target], lr=0.003)

# tuned weights
style_weight = 3e4
content_weight = 1

# training loop
for step in range(200):
    target_features = get_features(target)

    content_loss = torch.mean((target_features['content'] - content_features['content'])**2)

    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]

        _, c, h, w = target_feature.shape
        style_loss += torch.mean((target_gram - style_gram)**2) / (c * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss: {total_loss.item()}")

# ✅ FINAL SAFE IMAGE CONVERSION (FIXES YOUR ERROR)

output = target.clone().detach().cpu()

# remove batch dimension
output = output.squeeze(0)

# force RGB (first 3 channels only)
output = output[:3, :, :]

# unnormalize properly
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

output = output * std + mean
output = output.clamp(0, 1)

# convert to image
output = transforms.ToPILImage()(output)

# save output
output.save(output_path)

print("Output saved at:", output_path)