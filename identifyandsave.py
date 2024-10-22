import cv2
import numpy as np
import os
from skimage import io, color, feature
from skimage.morphology import skeletonize

def DetectCardEdges(imagePath, whiteRegionOutputPath, edgeOutputPath, justEdgesOutputPath, cardsFolder):
    image = io.imread(imagePath)
    if image is None:
        raise ValueError("Image not found or could not be loaded.")

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

if __name__ == "__main__":
    imagePath = 'image1.png'
    whiteRegionOutputPath = 'white_regions_blurred.png'
    edgeOutputPath = 'edges_filtered.png'
    justEdgesOutputPath = 'just_edges.png'
    cardsFolder = 'cards'

    DetectCardEdges(imagePath, whiteRegionOutputPath, edgeOutputPath, justEdgesOutputPath, cardsFolder)
