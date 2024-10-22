import os
import torch
import json
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from itertools import combinations
import cv2
import numpy as np
import os
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import intocolor
from skimage import io, color, feature
from skimage.morphology import skeletonize


sourceImagePath = 'image1.png'
processedCardsDir = 'processedcardscolorfilter'
boundedCardsDir = 'boundedcards'
cardsDir = 'cards'

for directory in [processedCardsDir, boundedCardsDir, cardsDir]:
    if not os.path.exists(directory):
        os.makedirs(directory)


def DetectCardEdges(imagePath, whiteRegionOutputPath, edgeOutputPath, justEdgesOutputPath, cardsFolder):
    image = io.imread(imagePath)

    if image.shape[2] == 4:
        image = image[:, :, :3]

    if not os.path.exists(cardsFolder):
        os.makedirs(cardsFolder)

    gray = color.rgb2gray(image)
    binary = gray > 0.7
    binary = binary.astype(np.uint8) * 255

    blurred = cv2.GaussianBlur(binary, (5, 5), 0)

    edges = feature.canny(blurred, sigma=1)

    skeleton = skeletonize(edges)
    skeleton = (skeleton * 255).astype(np.uint8)
    dilatedEdges = cv2.dilate(skeleton, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dilatedEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contourImage = image.copy()
    cardContours = []
    justEdges = np.zeros_like(gray)
    cardCount = 0

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > 1000:
                cardContours.append(approx)
                points = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = points.sum(axis=1)
                rect[0] = points[np.argmin(s)]
                rect[2] = points[np.argmax(s)]
                diff = np.diff(points, axis=1)
                rect[1] = points[np.argmin(diff)]
                rect[3] = points[np.argmax(diff)]

                (tl, tr, br, bl) = rect
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))

                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

                if maxWidth < maxHeight:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

                cardCount += 1
                cardOutputPath = os.path.join(cardsFolder, f'card_{cardCount}.png')
                cv2.imwrite(cardOutputPath, warped)

    cv2.drawContours(contourImage, cardContours, -1, (0, 255, 0), 2)
    cv2.drawContours(justEdges, cardContours, -1, 255, 2)

    cv2.imwrite(whiteRegionOutputPath, blurred)
    cv2.imwrite(edgeOutputPath, contourImage)
    cv2.imwrite(justEdgesOutputPath, justEdges)


def removeWhiteBackground(image):
    img = image.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    return img


def applyColorFilter(image, color):
    img = image.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        r, g, b, a = item

        if color == 'red' and r > g and r > b:
            newData.append(item)
        elif color == 'green' and g > r and g > b:
            newData.append(item)
        elif color == 'purple' and b > r and b > g:
            newData.append(item)
        else:
            newData.append((0, 0, 0, 0))

    img.putdata(newData)
    return img


def cropBorders(image, borderSize):
    width, height = image.size
    croppedImage = image.crop((borderSize, borderSize, width - borderSize, height - borderSize))
    return croppedImage


def compareImages(original, filtered):
    originalData = original.getdata()
    filteredData = filtered.getdata()

    originalNonWhitePixels = 0
    filteredNonTransparentPixels = 0

    for origPixel, filtPixel in zip(originalData, filteredData):
        if origPixel[3] > 0:
            originalNonWhitePixels += 1
        if filtPixel[3] > 0:
            filteredNonTransparentPixels += 1

    if originalNonWhitePixels == 0:
        return 0
    return filteredNonTransparentPixels / originalNonWhitePixels



def getBoundingBox(image):
    img = image.convert("RGBA")
    datas = img.getdata()
    width, height = image.size

    left = width
    right = 0
    top = height
    bottom = 0

    for y in range(height):
        for x in range(width):
            pixel = datas[y * width + x]
            if pixel[3] > 0:
                if x < left:
                    left = x
                if x > right:
                    right = x
                if y < top:
                    top = y
                if y > bottom:
                    bottom = y

    if left < right and top < bottom:
        return (left, top, right + 1, bottom + 1)
    else:
        return None


def processBoundingBoxImages():
    processed_images = set()

    for imageName in os.listdir(processedCardsDir):
        if imageName.endswith(".png"):
            baseName, ext = os.path.splitext(imageName)
            if any(color in imageName for color in ['red', 'green', 'purple']):
                imagePath = os.path.join(processedCardsDir, imageName)

                try:
                    img = Image.open(imagePath)
                    bbox = getBoundingBox(img)
                    if bbox:
                        croppedImg = img.crop(bbox)
                        croppedImageName = f"{baseName}.png"
                        croppedImagePath = os.path.join(boundedCardsDir, croppedImageName)

                        if croppedImageName not in processed_images:
                            croppedImg.save(croppedImagePath, "PNG")
                            processed_images.add(croppedImageName)

                            print(f"Processed and saved: {croppedImagePath}")
                    else:
                        print(f"No bounding box found for {imageName}")

                except Exception as e:
                    print(f"Error processing {imageName}: {e}")




def processAllImages():
    imagePath = 'image1.png'
    whiteRegionOutputPath = 'white_regions_blurred.png'
    edgeOutputPath = 'edges_filtered.png'
    justEdgesOutputPath = 'just_edges.png'
    cardsFolder = 'cards'

    DetectCardEdges(imagePath, whiteRegionOutputPath, edgeOutputPath, justEdgesOutputPath, cardsFolder)

    processBoundingBoxImages()


processAllImages()


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


def preprocessImage(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = image.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def parseClassName(className):
    # Split the class name into components based on the structure of the prediction
    # Example: "OneSquiggleFull" -> "One", "Squiggle", "Full"
    numbers = ["One", "Two", "Three"]
    shapes = ["Diamond", "Oval", "Squiggle"]
    fills = ["Full", "Partial", "Empty"]

    number = next((n for n in numbers if n in className), None)

    shape = next((s for s in shapes if s in className), None)

    pattern = next((f for f in fills if f in className), None)

    return number, shape, pattern


def predictCard(image, modelPath, classNamesPath):
    model = loadModel(modelPath, len(loadClassNames(classNamesPath)))
    image = preprocessImage(image)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    classNames = loadClassNames(classNamesPath)
    predictedClass = classNames[predicted.item()]

    number, shape, pattern = parseClassName(predictedClass)

    return number, shape, pattern, predictedClass


def updateJsonFile(jsonPath, number, shape, pattern):
    if os.path.exists(jsonPath):
        with open(jsonPath, 'r') as f:
            data = json.load(f)

        data['number'] = number
        data['shape'] = shape
        data['pattern'] = pattern

        with open(jsonPath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f'Updated {jsonPath}')
    else:
        print(f'JSON file not found: {jsonPath}')


def processImages(directoryPath, modelPath, classNamesPath):
    for filename in os.listdir(directoryPath):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            imagePath = os.path.join(directoryPath, filename)
            jsonPath = os.path.join(directoryPath, f"{filename.split('.')[0]}.json")

            with Image.open(imagePath) as img:
                # Convert to grayscale temporarily for prediction
                number, shape, pattern, predictedClass = predictCard(img, modelPath, classNamesPath)

                # Print each characteristic
                print(f'{filename}:')
                print(f'  Predicted Class: {predictedClass}')
                print(f'  Number: {number}')
                print(f'  Shape: {shape}')
                print(f'  Pattern (Fill): {pattern}')

                # Update the corresponding JSON file
                updateJsonFile(jsonPath, number, shape, pattern)

def loadJsonData():
    jsonDataList = []
    for filename in os.listdir(cardsDir):
        if filename.endswith('.json'):
            jsonPath = os.path.join(cardsDir, filename)
            with open(jsonPath, 'r') as f:
                jsonData = json.load(f)
                jsonData['filename'] = filename  # Include filename in the data
                jsonDataList.append(jsonData)
    return jsonDataList


cardsDir = 'cards'
setsDir = 'sets'

def jsonToTuple(jsonData):
    numberMap = {"One": 1, "Two": 2, "Three": 3}
    shapeMap = {"Diamond": 1, "Oval": 2, "Squiggle": 3}
    patternMap = {"Full": 1, "Partial": 2, "Empty": 3}
    colorMap = {"red": 1, "green": 2, "purple": 3}

    number = numberMap[jsonData["number"]]
    shape = shapeMap[jsonData["shape"]]
    pattern = patternMap[jsonData["pattern"]]
    color = colorMap[jsonData["color"]]

    return (number, shape, pattern, color)

def isSet(cards):
    for i in range(4):
        values = {card[i] for card in cards}
        if len(values) != 1 and len(values) != 3:
            return False
    return True

def combineImages(imagePaths, outputPath):
    images = [Image.open(path) for path in imagePaths]

    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths)
    total_height = sum(heights)

    combinedImage = Image.new('RGB', (total_width, total_height))

    y_offset = 0
    for image in images:
        combinedImage.paste(image, (0, y_offset))
        y_offset += image.height

    combinedImage.save(outputPath)

def process_set(combo):
    cardCombo, filenames = zip(*combo)
    if isSet(cardCombo):
        print("Found a set:")
        for filename in filenames:
            print(f"Card: {filename}")

        imagePaths = [os.path.join(cardsDir, filename.replace('.json', '.png')) for filename in filenames]
        outputPath = os.path.join(setsDir, '_'.join(filenames).replace('.json', '.png'))

        combineImages(imagePaths, outputPath)
        print(f"Combined into {outputPath}")

def findSets():
    if not os.path.exists(setsDir):
        os.makedirs(setsDir)

    jsonDataList = loadJsonData()
    cardTuples = [(jsonToTuple(data), data['filename']) for data in jsonDataList]

    allcombos = list(combinations(cardTuples, 3))

    with ThreadPoolExecutor() as executor:
        future_to_combo = {executor.submit(process_set, combo): combo for combo in allcombos}
        for future in as_completed(future_to_combo):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing set: {e}")

def main():
    print("running multi-threaded processing...")
    intocolor.processImagesMultiThreaded()

if __name__ == '__main__':

    directoryPath = 'cards'
    modelPath = 'cardClassifier.pth'
    classNamesPath = 'classNames.txt'
    main()

    processImages(directoryPath, modelPath, classNamesPath)
    findSets()

