"""Aquatic Inpainting: predictor directo de códigos VQ-DINOv3 por patch.

Idea: la métrica compara el ID discreto (1024 prototipos) de cada patch oculto.
En vez de generar píxeles, entrenamos DINOv3 (últimos bloques) + cabeza para
predecir directamente el ID de cada patch 8x8 a partir de la imagen enmascarada.
Las etiquetas se obtienen pasando las imágenes ORIGINALES por el propio dino_vq.py
(con un target.csv que lista los 64 patches), así no hay que replicar su pipeline.

Subcomandos:
  labels   -> genera labels_train.csv usando dino_vq.py sobre train/
  train    -> entrena y guarda checkpoint
  predict  -> escribe submission.csv a partir de test/ + target.csv
"""
import argparse, glob, os, random, subprocess, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def list_images(folder):
    exts = ("png", "jpg", "jpeg", "webp", "bmp")
    files = sorted(f for e in exts for f in glob.glob(os.path.join(folder, f"*.{e}")))
    return {os.path.splitext(os.path.basename(f))[0]: f for f in files}


def load_img(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).permute(2, 0, 1)  # uint8 CHW


def patch_mask_to_pixels(m):  # m: (B,64) bool -> (B,1,256,256)
    B = m.shape[0]
    return F.interpolate(m.view(B, 1, 8, 8).float(), scale_factor=32, mode="nearest")


def random_mask(B, lo=8, hi=20, device="cpu"):
    m = torch.zeros(B, 64, dtype=torch.bool, device=device)
    for i in range(B):
        k = random.randint(lo, hi)
        m[i, torch.randperm(64, device=device)[:k]] = True
    return m


class Net(nn.Module):
    """Réplica de dino_vq.py (capa 10, pool 2x2, L2, coseno vs codebook) pero sobre la
    imagen ENMASCARADA; las últimas `unfreeze` capas (<=10) se afinan y una cabeza
    residual corrige el embedding antes del coseno con el codebook."""
    LAYER = 10

    def __init__(self, codebook, unfreeze=4):
        super().__init__()
        from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(MODEL_ID)
        layers = self.bb.layer if hasattr(self.bb, "layer") else self.bb.encoder.layer
        keep = nn.ModuleList(list(layers)[: self.LAYER])  # descarta capas 11-12 (no se usan)
        if hasattr(self.bb, "layer"): self.bb.layer = keep
        else: self.bb.encoder.layer = keep
        for p in self.bb.parameters(): p.requires_grad = False
        for blk in keep[-unfreeze:]:
            for p in blk.parameters(): p.requires_grad = True
        d = self.bb.config.hidden_size
        self.nreg = int(getattr(self.bb.config, "num_register_tokens", 0) or 0)
        self.norm = nn.LayerNorm(d)
        self.ctx = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 8, d * 2, batch_first=True, norm_first=True), 2)
        self.pos = nn.Parameter(torch.zeros(1, 64, d))
        self.proj = nn.Linear(d, d); nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.cb = nn.Parameter(F.normalize(torch.as_tensor(codebook).float(), dim=1))
        self.scale = nn.Parameter(torch.tensor(30.0))

    def forward(self, x):  # x normalizado (B,3,256,256)
        h = self.bb(pixel_values=x, output_hidden_states=True).hidden_states[-1]  # = hidden_states[10]
        h = h[:, -256:]  # tokens espaciales (descarta CLS + registers)
        B, _, D = h.shape
        h = h.reshape(B, 8, 2, 8, 2, D).mean((2, 4)).reshape(B, 64, D).float()
        h = F.normalize(h, dim=-1)
        z = h + self.proj(self.ctx(self.norm(h) + self.pos))
        return self.scale * F.normalize(z, dim=-1) @ F.normalize(self.cb, dim=-1).T  # (B,64,1024)


def prep(u8, mask):  # u8: (B,3,H,W) uint8 ; mask (B,64) bool
    x = u8.float() / 255.0
    x = x * (1 - patch_mask_to_pixels(mask).to(x.device))  # negro en lo oculto
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# ------------------------------------------------------------------ comandos
def cmd_labels(a):
    imgs = list_images(a.train_dir)
    ids = list(imgs)
    pd.DataFrame({"ID": ids, "target": " ".join(map(str, range(64)))}).to_csv(a.out_target, index=False)
    cmd = [sys.executable, a.dino_vq, "--images-root", a.train_dir, "--target-csv", a.out_target,
           "--codebook-path", a.codebook, "--output-csv", a.out_labels]
    print(" ".join(cmd)); subprocess.check_call(cmd)


def load_labels(path, ids):
    df = pd.read_csv(path)
    img, p = df["Id"].str.rsplit("_", n=1).str[0], df["Id"].str.rsplit("_", n=1).str[1].astype(int)
    lab = {i: np.full(64, -1, np.int64) for i in ids}
    for i, pp, t in zip(img, p, df["Target"]):
        if i in lab: lab[i][pp] = t
    return lab


class DS(torch.utils.data.Dataset):
    def __init__(self, ids, imgs, lab, train):
        self.ids, self.imgs, self.lab, self.train = ids, imgs, lab, train
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        x = load_img(self.imgs[self.ids[i]])
        return x, torch.from_numpy(self.lab[self.ids[i]])


def cmd_train(a):
    dev = "cuda"
    imgs = list_images(a.train_dir); ids = sorted(imgs)
    lab = load_labels(a.labels, ids); ids = [i for i in ids if (lab[i] >= 0).all()]
    random.seed(0); random.shuffle(ids)
    nv = max(200, len(ids) // 20); va, tr = ids[:nv], ids[nv:]
    print(f"train {len(tr)} val {len(va)}")
    cb = np.load(a.codebook)
    net = Net(cb, a.unfreeze).to(dev)
    if a.resume and os.path.exists(a.ckpt):
        net.load_state_dict(torch.load(a.ckpt, map_location=dev)); print("reanudando desde", a.ckpt)
    bbp = [p for p in net.bb.parameters() if p.requires_grad]
    hp = [p for n, p in net.named_parameters() if not n.startswith("bb.")]
    opt = torch.optim.AdamW([{"params": bbp, "lr": a.lr * 0.1}, {"params": hp, "lr": a.lr}], weight_decay=0.05)
    dl = torch.utils.data.DataLoader(DS(tr, imgs, lab, True), a.bs, shuffle=True, num_workers=2, drop_last=True)
    dv = torch.utils.data.DataLoader(DS(va, imgs, lab, False), a.bs, num_workers=2)
    total = a.epochs * len(dl); sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, [a.lr * 0.1, a.lr], total_steps=total, pct_start=0.05)
    best = 0
    for ep in range(a.epochs):
        net.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev); m = random_mask(len(x), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(prep(x, m))
            loss = F.cross_entropy(lg[m].float(), y[m], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        net.eval(); c = n = 0
        g = torch.Generator().manual_seed(1); random.seed(1)
        with torch.no_grad():
            for x, y in dv:
                x, y = x.to(dev), y.to(dev); m = random_mask(len(x), device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = net(prep(x, m))
                c += (lg.argmax(-1)[m] == y[m]).sum().item(); n += m.sum().item()
        acc = c / n; print(f"ep {ep} loss {loss.item():.3f} val_acc {acc:.4f}", flush=True)
        if acc > best:
            best = acc; torch.save({k: v for k, v in net.state_dict().items()}, a.ckpt)
    print("best", best)


def cmd_predict(a):
    dev = "cuda"; net = Net(np.load(a.codebook), a.unfreeze).to(dev)
    net.load_state_dict(torch.load(a.ckpt, map_location=dev)); net.eval()
    tg = pd.read_csv(a.target_csv); imgs = list_images(a.test_dir); rows = []
    with torch.no_grad():
        for i, t in zip(tg["ID"].astype(str), tg["target"].astype(str)):
            ps = [int(p) for p in t.split()]
            x = load_img(imgs[i])[None].to(dev)
            m = torch.zeros(1, 64, dtype=torch.bool, device=dev); m[0, ps] = True
            # el test ya trae negro; usamos la imagen tal cual (prep re-aplica negro igual)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(prep(x, m))
            pred = lg[0].argmax(-1).cpu().numpy()
            rows += [(f"{i}_{p}", int(pred[p])) for p in ps]
    pd.DataFrame(rows, columns=["Id", "Target"]).to_csv(a.out, index=False)
    print("escrito", a.out, len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("labels", "train", "predict"):
        p = sp.add_parser(name)
        p.add_argument("--train-dir", default="data/train"); p.add_argument("--codebook", default="data/codebook.npy")
        p.add_argument("--dino-vq", default="data/dino_vq.py"); p.add_argument("--out-target", default="data/target_all64.csv")
        p.add_argument("--out-labels", default="data/labels_train.csv"); p.add_argument("--labels", default="data/labels_train.csv")
        p.add_argument("--ckpt", default="outputs/best.pt"); p.add_argument("--epochs", type=int, default=15)
        p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--unfreeze", type=int, default=4); p.add_argument("--test-dir", default="data/test")
        p.add_argument("--resume", action="store_true"); p.add_argument("--target-csv", default="data/target.csv"); p.add_argument("--out", default="outputs/submission.csv")
    a = ap.parse_args(); {"labels": cmd_labels, "train": cmd_train, "predict": cmd_predict}[a.cmd](a)
