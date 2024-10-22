import os
from PIL import Image

s = '/Users/dimitrichrysafis/PycharmProjects/setting/processedcardscolorfilter'
d = '/Users/dimitrichrysafis/PycharmProjects/setting/boundedcards'

if not os.path.exists(d):
    os.makedirs(d)


def g(i):
    img = i.convert("RGBA")
    data = img.getdata()
    w, h = i.size
    l, r, t, b = w, 0, h, 0

    for y in range(h):
        for x in range(w):
            p = data[y * w + x]
            if p[3] > 0:  # Look for non-transparent pixels
                if x < l: l = x
                if x > r: r = x
                if y < t: t = y
                if y > b: b = y

    return (l, t, r + 1, b + 1) if l < r and t < b else None


def p():
    for n in os.listdir(s):
        if n.endswith(".png"):
            p_path = os.path.join(s, n)
            img = Image.open(p_path)
            b = g(img)
            if b:
                c_img = img.crop(b)
                c_path = os.path.join(d, n)
                c_img.save(c_path, "PNG")
                print(f"Processed and saved: {c_path}")
            else:
                print(f"No bounding box found for {n}")


p()
