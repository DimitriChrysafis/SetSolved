import cv2
import numpy as np
import os

def detectCardEdges(imagePath, whiteRegionOutputPath, edgeOutputPath, justEdgesOutputPath, cardsFolder):
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError("Image not found or could not be loaded.")

    if not os.path.exists(cardsFolder):
        os.makedirs(cardsFolder)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilatedEdges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
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

                transparentImg = warped

                cardCount += 1
                cardOutputPath = os.path.join(cardsFolder, f'card_{cardCount}.png')
                cv2.imwrite(cardOutputPath, transparentImg)
                print(f"Card {cardCount} saved to: {cardOutputPath}")

    cv2.drawContours(contourImage, cardContours, -1, (0, 255, 0), 2)
    cv2.drawContours(justEdges, cardContours, -1, 255, 2)

    cv2.imwrite(whiteRegionOutputPath, blurred)
    cv2.imwrite(edgeOutputPath, contourImage)
    cv2.imwrite(justEdgesOutputPath, justEdges)

    print(f"White regions extracted and saved to: {whiteRegionOutputPath}")
    print(f"Card edges detected and saved to: {edgeOutputPath}")
    print(f"Image with just edges saved to: {justEdgesOutputPath}")
    print(f"All cards saved to folder: {cardsFolder}")

if __name__ == "__main__":
    imagePath = 'image.jpg'
    whiteRegionOutputPath = 'white_regions_blurred.png'
    edgeOutputPath = 'edges_filtered.png'
    justEdgesOutputPath = 'just_edges.png'
    cardsFolder = 'cards'

    detectCardEdges(imagePath, whiteRegionOutputPath, edgeOutputPath, justEdgesOutputPath, cardsFolder)
