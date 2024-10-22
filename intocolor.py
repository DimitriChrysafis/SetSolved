import os
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

sourceDir = './cards'
destinationDir = './cards'

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

        if color == 'red':
            if r > 150 and r > 1.2 * g and r > 1.2 * b:
                newData.append(item)
            else:
                newData.append((0, 0, 0, 0))
        elif color == 'green':
            if g > 150 and g > 1.1 * r and g > 1.2 * b:
                newData.append(item)
            else:
                newData.append((0, 0, 0, 0))
        elif color == 'purple':
            if r > 100 and b > 100 and b > 1.2 * g and r > g and abs(r - b) < 80:
                newData.append(item)
            else:
                newData.append((0, 0, 0, 0))
        else:
            newData.append((0, 0, 0, 0))

    img.putdata(newData)
    return img

def cropBorders(image, borderSize):
    width, height = image.size
    return image.crop((borderSize, borderSize, width - borderSize, height - borderSize))

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

def saveColorToJson(imageName, color):
    jsonFileName = os.path.join(destinationDir, f"{imageName.split('.')[0]}.json")

    if os.path.exists(jsonFileName):
        with open(jsonFileName, 'r') as jsonFile:
            data = json.load(jsonFile)
    else:
        data = {}

    data["color"] = color

    with open(jsonFileName, 'w') as jsonFile:
        json.dump(data, jsonFile, indent=4)

def processImage(imageName, borderTrim=10):
    imagePath = os.path.join(sourceDir, imageName)

    try:
        img = Image.open(imagePath)
        img = removeWhiteBackground(img)
        imgCropped = cropBorders(img, borderTrim)

        colorProportions = {}

        for color in ['red', 'green', 'purple']:
            filteredImg = applyColorFilter(imgCropped, color)
            similarity = compareImages(imgCropped, filteredImg)
            colorProportions[color] = similarity

        mostProminentColor = max(colorProportions, key=colorProportions.get)
        saveColorToJson(imageName, mostProminentColor)

        print(f"{imageName}: Most prominent color is {mostProminentColor}")

    except Exception as e:
        print(f"Error processing {imageName}: {e}")

def processImagesMultiThreaded(borderTrim=10):
    imageNames = [imageName for imageName in os.listdir(sourceDir) if imageName.endswith(".png")]
    with ThreadPoolExecutor() as executor:
        executor.map(lambda imageName: processImage(imageName, borderTrim), imageNames)
