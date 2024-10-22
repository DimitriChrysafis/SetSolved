import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn


def loadModel(modelPath, numClasses):
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 16 * 16, 512)
            self.fc2 = nn.Linear(512, numClasses)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv3(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(-1, 64 * 16 * 16)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleCNN()
    model.load_state_dict(torch.load(modelPath, weights_only=True))
    model.eval()
    return model


def loadClassNames(classNamesPath):
    with open(classNamesPath, 'r') as f:
        classNames = [line.strip() for line in f]
    return classNames


def preprocessImage(imagePath):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(imagePath).convert('RGB')  # Convert to RGB for consistency
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def convertToBlackAndWhite(imagePath):
    with Image.open(imagePath) as img:
        bwImage = img.convert('L')  # Convert to grayscale
        bwImage = bwImage.convert('RGB')  # Convert to 3-channel grayscale
        bwImage.save(imagePath)  # Overwrite the original image


def predictCard(imagePath, modelPath, classNamesPath):
    model = loadModel(modelPath, len(loadClassNames(classNamesPath)))
    image = preprocessImage(imagePath)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    classNames = loadClassNames(classNamesPath)
    predictedClass = classNames[predicted.item()]

    return predictedClass


if __name__ == '__main__':
    imagePath = '/Users/dimitrichrysafis/Desktop/card.png'
    modelPath = 'cardClassifier.pth'
    classNamesPath = 'classNames.txt'

    convertToBlackAndWhite(imagePath)
    predictedClass = predictCard(imagePath, modelPath, classNamesPath)
    print(f'The predicted card is: {predictedClass}')
