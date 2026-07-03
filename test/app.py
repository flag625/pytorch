from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import sys

app = Flask(__name__)

model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

IMAGENET_CLASSES = ['class_%d' % i for i in range(1000)]

@app.route('/')
def home():
    print('Python %s on %s' % (sys.version, sys.platform))
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded'
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)
    class_name = IMAGENET_CLASSES[predicted_idx.item()]
    return f'Predicted: {class_name}'

if __name__ == '__main__':
    app.run(debug=True)