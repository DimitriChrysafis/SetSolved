
import cv2
import numpy as np
import os

def detect_card_edges(image_path, white_region_output_path, edge_output_path, just_edges_output_path, cards_folder):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or could not be loaded.")

    if not os.path.exists(cards_folder):
        os.makedirs(cards_folder)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = image.copy()
    card_contours = []
    just_edges = np.zeros_like(gray)
    card_count = 0

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > 1000:
                card_contours.append(approx)
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

                transparent_img = warped

                card_count += 1
                card_output_path = os.path.join(cards_folder, f'card_{card_count}.png')
                cv2.imwrite(card_output_path, transparent_img)
                print(f"Card {card_count} saved to: {card_output_path}")

    cv2.drawContours(contour_image, card_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(just_edges, card_contours, -1, 255, 2)

    cv2.imwrite(white_region_output_path, blurred)
    cv2.imwrite(edge_output_path, contour_image)
    cv2.imwrite(just_edges_output_path, just_edges)

    print(f"White regions extracted and saved to: {white_region_output_path}")
    print(f"Card edges detected and saved to: {edge_output_path}")
    print(f"Image with just edges saved to: {just_edges_output_path}")
    print(f"All cards saved to folder: {cards_folder}")

if __name__ == "__main__":
    image_path = 'image.jpg'
    white_region_output_path = 'white_regions_blurred.png'
    edge_output_path = 'edges_filtered.png'
    just_edges_output_path = 'just_edges.png'
    cards_folder = 'cards'

    detect_card_edges(image_path, white_region_output_path, edge_output_path, just_edges_output_path, cards_folder)
