import os
import torch
import torchvision
import torchvision.transforms as t
from torch import nn, optim
from torch.utils.data import DataLoader

def m():
    d = '/Users/dimitrichrysafis/Desktop/sorted_dataset/'

    tr = t.Compose([
        t.Resize((128, 128)),
        t.ToTensor(),
        t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ds = torchvision.datasets.ImageFolder(root=d, transform=tr)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)

    class S(nn.Module):
        def __init__(self):
            super(S, self).__init__()
            self.c1 = nn.Conv2d(3, 16, 3, padding=1)
            self.c2 = nn.Conv2d(16, 32, 3, padding=1)
            self.c3 = nn.Conv2d(32, 64, 3, padding=1)
            self.f1 = nn.Linear(64 * 16 * 16, 512)
            self.f2 = nn.Linear(512, 81)

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

    e = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    m = S().to(e)
    c = nn.CrossEntropyLoss()
    o = optim.Adam(m.parameters(), lr=0.001)

    ne = 30
    for ep in range(ne):
        m.train()
        r = 0.0
        for img, lab in dl:
            img, lab = img.to(e), lab.to(e)
            o.zero_grad()
            out = m(img)
            l = c(out, lab)
            l.backward()
            o.step()
            r += l.item()
        print(f"Epoch {ep+1}/{ne}, Loss: {r/len(dl)}")

    torch.save(m.state_dict(), 'card_classifier.pth')

    with open('class_names.txt', 'w') as f:
        for n in ds.classes:
            f.write(f"{n}\n")

    m.eval()
    c = 0
    t = 0
    with torch.no_grad():
        for img, lab in dl:
            img, lab = img.to(e), lab.to(e)
            out = m(img)
            _, p = torch.max(out.data, 1)
            t += lab.size(0)
            c += (p == lab).sum().item()

    print(f'Accuracy: {100 * c / t}%')

if __name__ == '__main__':
    m()
