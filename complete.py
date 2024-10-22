import cv2
import numpy as np
import os
import torch
import torchvision.transforms as T
from PIL import Image

def d(i, w, e, j, c):
    img = cv2.imread(i)

    if not os.path.exists(c):
        os.makedirs(c)

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 180, 255, cv2.THRESH_BINARY)
    bl = cv2.GaussianBlur(b, (5, 5), 0)
    ed = cv2.Canny(bl, 50, 150)
    de = cv2.dilate(ed, np.ones((5, 5), np.uint8), iterations=1)
    ct, _ = cv2.findContours(de, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ci = img.copy()
    cc = []
    je = np.zeros_like(g)
    n = 0

    for c in ct:
        e = 0.02 * cv2.arcLength(c, True)
        a = cv2.approxPolyDP(c, e, True)

        if len(a) == 4:
            area = cv2.contourArea(c)
            if area > 1000:
                cc.append(a)
                p = a.reshape(4, 2)
                r = np.zeros((4, 2), dtype="float32")
                s = p.sum(axis=1)
                r[0] = p[np.argmin(s)]
                r[2] = p[np.argmax(s)]
                d = np.diff(p, axis=1)
                r[1] = p[np.argmin(d)]
                r[3] = p[np.argmax(d)]
                (tl, tr, br, bl) = r
                wA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                wB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                mw = max(int(wA), int(wB))
                hA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                hB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                mh = max(int(hA), int(hB))
                dst = np.array([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(r, dst)
                w = cv2.warpPerspective(img, M, (mw, mh), flags=cv2.INTER_LINEAR)

                n += 1
                cp = os.path.join(c, f'card_{n}.png')
                cv2.imwrite(cp, w)
                print(f"Card {n} saved to: {cp}")

    cv2.drawContours(ci, cc, -1, (0, 255, 0), 2)
    cv2.drawContours(je, cc, -1, 255, 2)

    cv2.imwrite(w, bl)
    cv2.imwrite(e, ci)
    cv2.imwrite(j, je)

    print(f"White regions extracted and saved to: {w}")
    print(f"Card edges detected and saved to: {e}")
    print(f"Image with just edges saved to: {j}")
    print(f"All cards saved to folder: {c}")

class C(torch.nn.Module):
    def __init__(self):
        super(C, self).__init__()
        self.c1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.f1 = torch.nn.Linear(64 * 16 * 16, 512)
        self.f2 = torch.nn.Linear(512, 81)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.c2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.c3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.f1(x))
        x = self.f2(x)
        return x

def l(f):
    with open(f, 'r') as fi:
        return [line.strip() for line in fi]

def p(i, m, cn, d):
    tr = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(i).convert('RGB')
    img = tr(img).unsqueeze(0).to(d)
    with torch.no_grad():
        o = m(img)
        _, pr = torch.max(o, 1)
    return cn[pr.item()]

def main():
    i = 'image1.png'
    w = 'white_regions_blurred.png'
    e = 'edges_filtered.png'
    j = 'just_edges.png'
    c = 'cards'

    d(i, w, e, j, c)

    d = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    m = C().to(d)
    m.load_state_dict(torch.load('card_classifier.pth', map_location=d, weights_only=True))
    m.eval()
    cn = l('class_names.txt')

    for cf in os.listdir(c):
        if cf.endswith('.png'):
            cp = os.path.join(c, cf)
            pc = p(cp, m, cn, d)
            print(f"Card {cf} is classified as: {pc}")

if __name__ == "__main__":
    main()
