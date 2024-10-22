import os
import torch
import json
import torchvision.transforms as T
from PIL import Image
from torch import nn
from itertools import combinations
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

s = 'image1.png'
p = 'processedcardscolorfilter'
b = 'boundedcards'
c = 'cards'

for d in [p, b, c]:
    if not os.path.exists(d):
        os.makedirs(d)


def d(i, f, g):
    img = cv2.imread(i)
    if img is None:
        raise ValueError("Image not found or could not be loaded.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ci = img.copy()
    cc = []
    count = 0

    for cnt in contours:
        eps = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)

        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > 1000:
                cc.append(approx)
                pts = approx.reshape(4, 2)
                r = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                r[0] = pts[np.argmin(s)]
                r[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                r[1] = pts[np.argmin(diff)]
                r[3] = pts[np.argmax(diff)]
                (tl, tr, br, bl) = r
                wa = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                wb = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                mw = max(int(wa), int(wb))
                ha = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                hb = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                mh = max(int(ha), int(hb))
                dst = np.array([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(r, dst)
                warped = cv2.warpPerspective(img, M, (mw, mh), flags=cv2.INTER_LINEAR)

                count += 1
                op = os.path.join(f, f'card_{count}.png')
                cv2.imwrite(op, warped)

                op2 = os.path.join(g, f'card_{count}.png')
                cv2.imwrite(op2, warped)

                shape = "unknown"
                if len(approx) == 3:
                    shape = "triangle"
                elif len(approx) == 4:
                    shape = "quadrilateral"
                elif len(approx) > 4:
                    shape = "circle"

                color = "unknown"
                m = {
                    "number": "",
                    "color": color,
                    "shape": shape,
                    "pattern": "",
                    "specialnum": ""
                }
                jp = os.path.join(g, f'card_{count}.json')
                with open(jp, 'w') as f:
                    json.dump(m, f, indent=4)

    cv2.drawContours(ci, cc, -1, (0, 255, 0), 2)
    justEdges = np.zeros_like(gray)
    cv2.drawContours(justEdges, cc, -1, 255, 2)

    cv2.imwrite('white_regions_blurred.png', blurred)
    cv2.imwrite('edges_filtered.png', ci)
    cv2.imwrite('just_edges.png', justEdges)
    print(f"White regions extracted and saved to: white_regions_blurred.png")
    print(f"Card edges detected and saved to: edges_filtered.png")
    print(f"Image with just edges saved to: just_edges.png")
    print(f"All cards saved to folder: {f}")
    print(f"Individual cards saved to folder: {g}")


def r(i):
    img = i.convert("RGBA")
    data = img.getdata()
    newData = []
    for item in data:
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img


def a(i, c):
    img = i.convert("RGBA")
    data = img.getdata()
    newData = []
    for item in data:
        r, g, b, a = item
        if c == 'red' and r > g and r > b:
            newData.append(item)
        elif c == 'green' and g > r and g > b:
            newData.append(item)
        elif c == 'purple' and b > r and b > g:
            newData.append(item)
        else:
            newData.append((0, 0, 0, 0))
    img.putdata(newData)
    return img


def c(i, s):
    w, h = i.size
    return i.crop((s, s, w - s, h - s))


def co(o, f):
    origData = o.getdata()
    filtData = f.getdata()
    origNonWhite = 0
    filtNonTransparent = 0
    for origPixel, filtPixel in zip(origData, filtData):
        if origPixel[3] > 0:
            origNonWhite += 1
        if filtPixel[3] > 0:
            filtNonTransparent += 1
    if origNonWhite == 0:
        return 0
    return filtNonTransparent / origNonWhite


def p(b=10):
    for n in os.listdir(p):
        if n.endswith(".png"):
            ip = os.path.join(p, n)
            try:
                img = Image.open(ip)
                img = r(img)
                imgC = c(img, b)
                cp = {}
                fp = []
                for color in ['red', 'green', 'purple']:
                    fImg = a(imgC, color)
                    fP = os.path.join(p, f"{n.split('.')[0]}_{color}.png")
                    fImg.save(fP, "PNG")
                    fp.append(fP)
                    sim = co(imgC, fImg)
                    cp[color] = sim
                mc = max(cp, key=cp.get)
                jp = os.path.join(c, f"card_{n.split('.')[0].split('_')[1]}.json")
                if os.path.exists(jp):
                    with open(jp, 'r') as f:
                        m = json.load(f)
                    m['color'] = mc
                    with open(jp, 'w') as f:
                        json.dump(m, f, indent=4)
            except Exception as e:
                print(f"Error processing {n}: {e}")


def g(i):
    img = i.convert("RGBA")
    data = img.getdata()
    w, h = img.size
    l, r, t, b = w, 0, h, 0
    for y in range(h):
        for x in range(w):
            pixel = data[y * w + x]
            if pixel[3] > 0:
                if x < l: l = x
                if x > r: r = x
                if y < t: t = y
                if y > b: b = y
    if l < r and t < b:
        return (l, t, r + 1, b + 1)
    else:
        return None


def pb():
    pi = set()
    for n in os.listdir(p):
        if n.endswith(".png"):
            base, ext = os.path.splitext(n)
            if any(color in n for color in ['red', 'green', 'purple']):
                ip = os.path.join(p, n)
                try:
                    img = Image.open(ip)
                    bbox = g(img)
                    if bbox:
                        croppedImg = img.crop(bbox)
                        croppedImageName = f"{base}.png"
                        croppedImagePath = os.path.join(b, croppedImageName)
                        if croppedImageName not in pi:
                            croppedImg.save(croppedImagePath, "PNG")
                            pi.add(croppedImageName)
                            print(f"Processed and saved: {croppedImagePath}")
                    else:
                        print(f"No bounding box found for {n}")
                except Exception as e:
                    print(f"Error processing {n}: {e}")


def pa():
    with ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(d, os.path.join(s), c, b): os.path.join(s)}
        for future in as_completed(future_to_image):
            imgPath = future_to_image[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {imgPath}: {e}")


if __name__ == '__main__':
    pa()
    p()
    pb()
