"""Entrenamiento v2: maestro DINOv3 en linea + aumentos + validacion por clusters.

- Maestro: copia congelada de DINOv3 que calcula, sobre CADA imagen aumentada y sin mascara, el embedding
  exacto de la capa 10 (igual que dino_vq.py). De ahi salen la etiqueta (argmax coseno con el codebook) y
  el embedding objetivo => no hace falta generar etiquetas y se pueden usar aumentos sin coste.
- Alumno: Net (de aquatic.py) ve la imagen aumentada con 8-20 parches negros. Perdida = CE + lam*(1-coseno).
- Validacion: split por CLUSTERS de escenas (no aleatorio), para que val_acc se parezca mas al test.
Uso: python src/v2.py train --epochs 30 --unfreeze 10 --seed 0 --ckpt outputs/v2_best_s0.pt [--persist DIR]
"""
import argparse, json, os, random, shutil, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import aquatic
from aquatic import MODEL_ID, MEAN, STD, Net, list_images, load_img, random_mask, atomic_save, patch_mask_to_pixels


def get_dev(): return "cuda" if torch.cuda.is_available() else "cpu"


def amp_dtype(dev):
    return torch.bfloat16 if (dev == "cpu" or torch.cuda.is_bf16_supported()) else torch.float16


def get_teacher(dev):
    from transformers import AutoModel
    tdt = torch.float16 if dev == "cuda" else torch.float32  # fp16 = mismo dtype que dino_vq.py en GPU
    t = AutoModel.from_pretrained(MODEL_ID, torch_dtype=tdt).to(dev).eval()
    for p in t.parameters(): p.requires_grad = False
    return t, tdt


@torch.no_grad()
def teacher_embed(t, tdt, xf):
    """xf: float [0,1] (B,3,256,256) -> embedding por parche (B,64,D) normalizado, descriptor global (B,D)."""
    x = ((xf - MEAN.to(xf.device)) / STD.to(xf.device)).to(tdt)
    h = t(pixel_values=x, output_hidden_states=True).hidden_states[Net.LAYER]
    sp = h[:, -256:]; B, _, D = sp.shape
    pooled = sp.reshape(B, 8, 2, 8, 2, D).mean((2, 4)).reshape(B, 64, D)  # igual que dino_vq.py
    return F.normalize(pooled.float(), dim=-1), sp.float().mean(1)


def prep_f(xf, mask):  # xf float [0,1]; mask (B,64) -> entrada normalizada con parches negros
    xf = xf * (1 - patch_mask_to_pixels(mask).to(xf.device))
    return (xf - MEAN.to(xf.device)) / STD.to(xf.device)


_PW = np.array([[0.16, 0.16, 0.16, 0.19, 0.18, 0.18, 0.2 , 0.16],
          [0.16, 0.2 , 0.2 , 0.2 , 0.22, 0.19, 0.19, 0.16],
          [0.2 , 0.21, 0.23, 0.23, 0.23, 0.22, 0.2 , 0.18],
          [0.24, 0.25, 0.25, 0.25, 0.26, 0.24, 0.21, 0.2 ],
          [0.22, 0.23, 0.23, 0.27, 0.27, 0.27, 0.22, 0.21],
          [0.23, 0.24, 0.26, 0.28, 0.26, 0.27, 0.25, 0.2 ],
          [0.21, 0.25, 0.24, 0.25, 0.25, 0.26, 0.25, 0.21],
          [0.22, 0.23, 0.24, 0.25, 0.23, 0.23, 0.21, 0.2 ]]).ravel()  # frecuencia por celda de las mascaras del test (sesgo hacia el centro)
_PW = _PW / _PW.sum()
_NB = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def walk_masks(B, lo=8, hi=20, device="cpu", pc_lo=0.1, pc_hi=0.35):
    """Mascaras con la misma estadistica que el test: agrupadas y algo centradas (parches vecinos 0.48,
    componente mayor ~5.3). Cada parche nuevo se agrega junto a uno existente con prob. pc, o al azar si no."""
    rs = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
    out = np.zeros((B, 64), bool)
    for b in range(B):
        n, pc, m = int(rs.integers(lo, hi + 1)), rs.uniform(pc_lo, pc_hi), out[b]
        m[rs.choice(64, p=_PW)] = True; cnt = 1
        while cnt < n:
            if rs.random() < pc:
                idx = np.flatnonzero(m); y, x = divmod(int(idx[rs.integers(len(idx))]), 8)
                dy, dx = _NB[rs.integers(4)]; y, x = y + dy, x + dx
                j = y * 8 + x if (0 <= y < 8 and 0 <= x < 8) else -1
            else:
                j = int(rs.choice(64, p=_PW))
            if j >= 0 and not m[j]: m[j] = True; cnt += 1
    return torch.as_tensor(out, device=device)


def make_mask(a, B, device):
    return walk_masks(B, device=device) if a.mask == "walk" else random_mask(B, device=device)


def augment(xf, level=1):
    """level 1: volteo horizontal + brillo/contraste suave. level 2: ademas recorte/zoom aleatorio y ruido leve
    (el maestro etiqueta la imagen ya aumentada, asi que las etiquetas siguen siendo consistentes)."""
    B = xf.shape[0]
    if level >= 2:
        sc = torch.where(torch.rand(B, device=xf.device) < 0.5, torch.empty(B, device=xf.device).uniform_(0.75, 1.0),
                         torch.ones(B, device=xf.device))
        tx, ty = [(torch.rand(B, device=xf.device) * 2 - 1) * (1 - sc) for _ in range(2)]
        th = torch.zeros(B, 2, 3, device=xf.device); th[:, 0, 0] = sc; th[:, 1, 1] = sc; th[:, 0, 2] = tx; th[:, 1, 2] = ty
        xf = F.grid_sample(xf, F.affine_grid(th, xf.shape, align_corners=False), mode="bilinear",
                           padding_mode="border", align_corners=False)
    flip = torch.rand(B, device=xf.device) < 0.5
    xf = torch.where(flip[:, None, None, None], xf.flip(3), xf)
    jit = (torch.rand(B, device=xf.device) < 0.5)[:, None, None, None]
    b = 1 + (torch.rand(B, 1, 1, 1, device=xf.device) - 0.5) * 0.2
    c = 1 + (torch.rand(B, 1, 1, 1, device=xf.device) - 0.5) * 0.2
    mu = xf.mean((1, 2, 3), keepdim=True)
    xf = torch.where(jit, (((xf - mu) * c + mu) * b).clamp(0, 1), xf)
    if level >= 2:
        nz = (torch.rand(B, device=xf.device) < 0.3)[:, None, None, None]
        xf = torch.where(nz, (xf + torch.randn_like(xf) * 0.02).clamp(0, 1), xf)
    return xf


class DS(torch.utils.data.Dataset):
    def __init__(self, ids, imgs): self.ids, self.imgs = ids, imgs
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return load_img(self.imgs[self.ids[i]])


def kmeans_cos(x, k, iters=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k]].clone()
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        for j in range(k):
            m = a == j
            if m.any(): c[j] = F.normalize(x[m].mean(0), dim=0)
    return (x @ c.T).argmax(1)


def make_split(ids, desc, holdout, k, seed=0):
    """Reserva clusters completos de escenas (~holdout del total) como validacion."""
    a = kmeans_cos(F.normalize(torch.as_tensor(desc).float(), dim=1), min(k, len(ids)), seed=seed).numpy()
    order = sorted(set(a.tolist())); random.Random(seed).shuffle(order)
    val_clusters, n = set(), 0
    for cl in order:
        if n >= holdout * len(ids): break
        val_clusters.add(cl); n += int((a == cl).sum())
    va = [i for i, c in zip(ids, a) if c in val_clusters]; tr = [i for i, c in zip(ids, a) if c not in val_clusters]
    return tr, va


def get_split(a, ids, imgs, teacher, tdt, dev):
    cache_dirs = [d for d in (os.path.dirname(a.ckpt) or ".", a.persist) if d]
    for d in cache_dirs:
        f = os.path.join(d, "split_v2.json")
        if os.path.exists(f):
            sp = json.load(open(f))
            if set(sp["train"] + sp["val"]) == set(ids):
                print("split por clusters cargado de", f); return sp["train"], sp["val"]
    dl = torch.utils.data.DataLoader(DS(ids, imgs), 64, num_workers=a.workers)
    desc = torch.cat([teacher_embed(teacher, tdt, x.to(dev).float() / 255)[1].cpu() for x in dl]).numpy()
    tr, va = make_split(ids, desc, a.holdout, a.k)
    for d in cache_dirs:
        os.makedirs(d, exist_ok=True); json.dump(dict(train=tr, val=va), open(os.path.join(d, "split_v2.json"), "w"))
    print(f"split por clusters: train {len(tr)} val {len(va)}")
    return tr, va


def cmd_train(a):
    dev = get_dev(); t0 = time.time(); dt = amp_dtype(dev)
    random.seed(a.seed); torch.manual_seed(a.seed)
    imgs = list_images(a.train_dir); ids = sorted(imgs)
    if a.limit_imgs: ids = ids[:a.limit_imgs]
    cb = np.load(a.codebook); cbn = F.normalize(torch.as_tensor(cb).float(), dim=1).to(dev)
    teacher, tdt = get_teacher(dev)
    tr, va = get_split(a, ids, imgs, teacher, tdt, dev)
    if a.full: tr = tr + va  # entrenamiento final con todos los datos (val queda solo como monitor, ya no es independiente)
    print(f"v2 | train {len(tr)} val {len(va)} amp {dt} aug {a.aug} lam {a.lam}", flush=True)
    net = Net(cb, a.unfreeze).to(dev)
    bbp = [p for p in net.bb.parameters() if p.requires_grad]
    hp = [p for n, p in net.named_parameters() if not n.startswith("bb.")]
    opt = torch.optim.AdamW([{"params": bbp, "lr": a.lr * 0.1}, {"params": hp, "lr": a.lr}], weight_decay=a.wd)
    dl = torch.utils.data.DataLoader(DS(tr, imgs), a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                                     persistent_workers=a.workers > 0)
    dv = torch.utils.data.DataLoader(DS(va, imgs), a.bs, num_workers=a.workers)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr * 0.1, a.lr], total_steps=a.epochs * len(dl), pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=(dt == torch.float16 and dev == "cuda"))
    state_path = a.ckpt.replace(".pt", "_state.pt")
    cfg = dict(v=2, epochs=a.epochs, unfreeze=a.unfreeze, bs=a.bs, lr=a.lr, seed=a.seed, lam=a.lam, aug=a.aug, full=a.full, ls=a.ls, wd=a.wd, mask=a.mask)
    best, ep_t, start = 0.0, 0, 0
    if os.path.exists(state_path) and not a.fresh:
        try:
            st = torch.load(state_path, map_location=dev, weights_only=False)
            if st["cfg"] == cfg:
                net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
                scaler.load_state_dict(st["scaler"]); start, best = st["epoch"] + 1, st["best"]
                print(f"[v2 seed {a.seed}] reanudando en la epoca {start + 1}/{a.epochs} (mejor val_acc {best:.4f})", flush=True)
            else:
                print("estado previo con otra configuracion: se ignora")
        except Exception as e:
            print("!! no se pudo leer el estado previo:", type(e).__name__, e)
    finished = start >= a.epochs
    for ep in range(start, a.epochs):
        te = time.time(); net.train()
        for bi, x in enumerate(dl):
            if a.max_steps and bi >= a.max_steps: break
            xf = x.to(dev, non_blocking=True).float() / 255
            if a.aug: xf = augment(xf, a.aug)
            T, _ = teacher_embed(teacher, tdt, xf)
            lab = (T @ cbn.T).argmax(-1)
            m = make_mask(a, len(xf), dev)
            with torch.autocast(dev, dtype=dt):
                lg, z = net(prep_f(xf, m), return_emb=True)
            ce = F.cross_entropy(lg[m].float(), lab[m], label_smoothing=a.ls)
            cs = (1 - (z[m].float() * T[m]).sum(-1)).mean()
            loss = ce + a.lam * cs
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        net.eval(); c = n = 0
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(1234)  # mismas mascaras de validacion cada epoca, sin alterar el RNG de entrenamiento
            for vi, x in enumerate(dv):
                if a.max_steps and vi >= a.max_steps: break
                xf = x.to(dev).float() / 255
                T, _ = teacher_embed(teacher, tdt, xf); lab = (T @ cbn.T).argmax(-1)
                m = make_mask(a, len(xf), dev)
                with torch.autocast(dev, dtype=dt):
                    lg = net(prep_f(xf, m))
                c += (lg.argmax(-1)[m] == lab[m]).sum().item(); n += m.sum().item()
        acc = c / max(n, 1); ep_t = time.time() - te
        print(f"[v2 seed {a.seed}] ep {ep+1}/{a.epochs} loss {loss.item():.3f} (ce {ce.item():.3f} cos {cs.item():.3f}) "
              f"val_acc {acc:.4f} ({ep_t/60:.1f} min)", flush=True)
        if acc >= best or a.full:  # con --full se guarda la ultima epoca
            best = acc; torch.save(net.state_dict(), a.ckpt)
            if a.persist:
                os.makedirs(a.persist, exist_ok=True); shutil.copy(a.ckpt, a.persist)
        if not a.max_steps:
            st = dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                      epoch=ep, best=best, cfg=cfg)
            try: atomic_save(st, state_path, a.persist if (ep + 1) % 2 == 0 else "")
            except Exception as e: print("aviso: no se pudo guardar el estado:", type(e).__name__, e)
        finished = ep == a.epochs - 1
        if not finished and a.time_budget and (time.time() - t0 + ep_t) > a.time_budget * 60:
            print("presupuesto de tiempo agotado: se detiene (se reanuda exacto en la proxima corrida)"); break
    print(f"best val_acc {best:.4f}" + ("  [COMPLETO]" if finished else "  [INCOMPLETO]"))
    with open(a.ckpt + ".val", "w") as f: f.write(str(best))
    if finished and not a.max_steps:
        open(a.ckpt + ".finished", "w").close()
        for f in (state_path, os.path.join(a.persist, os.path.basename(state_path)) if a.persist else None):
            if f and os.path.exists(f): os.remove(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True); p = sp.add_parser("train")
    p.add_argument("--train-dir", default="data/train"); p.add_argument("--codebook", default="data/codebook.npy")
    p.add_argument("--ckpt", default="outputs/v2_best_s0.pt"); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--unfreeze", type=int, default=10); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--time-budget", type=float, default=0)
    p.add_argument("--persist", default=""); p.add_argument("--fresh", action="store_true")
    p.add_argument("--lam", type=float, default=1.0); p.add_argument("--aug", type=int, default=1)
    p.add_argument("--holdout", type=float, default=0.05); p.add_argument("--k", type=int, default=200)
    p.add_argument("--full", action="store_true"); p.add_argument("--mask", choices=["uniform", "walk"], default="uniform"); p.add_argument("--ls", type=float, default=0.05); p.add_argument("--wd", type=float, default=0.05); p.add_argument("--max-steps", type=int, default=0); p.add_argument("--limit-imgs", type=int, default=0)
    a = ap.parse_args(); cmd_train(a)
