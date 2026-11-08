"""Adaptacion transductiva al test (sin etiquetas reales).

Idea: las imagenes del test son ENTRADAS dadas. Sobre ellas se puede seguir afinando el modelo con auto-supervision:
  1) el maestro DINOv3 (congelado) calcula codigos aproximados de los parches VISIBLES de cada imagen (la ve con los negros);
  2) se esconden unos parches visibles extra y el alumno aprende a predecirlos desde el resto;
  3) despues se predicen los parches realmente ocultos con el alumno ya adaptado a esas escenas.
No usa ninguna etiqueta del test. En validacion se puede medir el efecto con etiquetas reales (ver diag.py, metodo 'adapt').
Uso (test real): python src/adapt.py predict --ckpts a.pt,b.pt --epochs 8 --out outputs/submission.csv
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aquatic, v2
from aquatic import Net, list_images, load_img, prep, patch_mask_to_pixels


@torch.no_grad()
def pseudo_codes(teacher, tdt, cbn, X, M, dev, bs=32):
    """Codigos aproximados (N,64) del maestro sobre la imagen TAL COMO ESTA (con negros)."""
    out = []
    for i in range(0, len(X), bs):
        xf = X[i:i + bs].to(dev).float() / 255 * (1 - patch_mask_to_pixels(M[i:i + bs].to(dev)))
        T, _ = v2.teacher_embed(teacher, tdt, xf); out.append((T @ cbn.T).argmax(-1).cpu())
    return torch.cat(out)


def adapt_net(net, X, M, Tc, dev, epochs=8, bs=16, lr=2e-4, bb_lr=2e-5, kmin=3, kmax=6, seed=0, log=print):
    """Afina `net` (en sitio) prediciendo parches visibles extra ocultados. X uint8, M (N,64) bool oculto real, Tc codigos aprox."""
    g = torch.Generator().manual_seed(seed); net.train()
    bbp = [p for p in net.bb.parameters() if p.requires_grad]; hp = [p for n, p in net.named_parameters() if not n.startswith("bb.")]
    opt = torch.optim.AdamW([{"params": bbp, "lr": bb_lr}, {"params": hp, "lr": lr}], weight_decay=0.05)
    N = len(X); steps = epochs * ((N + bs - 1) // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, [bb_lr, lr], total_steps=steps, pct_start=0.1)
    for ep in range(epochs):
        perm = torch.randperm(N, generator=g); tot = n = 0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]; x = X[idx].to(dev); m = M[idx].to(dev); t = Tc[idx].to(dev)
            r = torch.rand(len(idx), 64, generator=g).to(dev); r[m] = 2.0                  # solo parches visibles
            k = torch.randint(kmin, kmax + 1, (len(idx),), generator=g).to(dev)
            E = r.argsort(1).argsort(1) < k[:, None]                                          # k parches visibles al azar
            with torch.autocast(dev, dtype=torch.bfloat16): lg = net(prep(x, m | E))
            loss = F.cross_entropy(lg[E].float(), t[E])
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item() * len(idx); n += len(idx)
        log(f"   adaptacion ep {ep + 1}/{epochs} loss {tot / n:.3f}")
    net.eval(); return net


def load_nets(paths, cb, unfreeze, dev):
    nets = []
    for p in paths:
        n = Net(cb, unfreeze).to(dev); n.load_state_dict(torch.load(p, map_location=dev)); nets.append(n.eval())
    return nets


@torch.no_grad()
def ensemble_probs(nets, X, M, dev, bs=16):
    P = []
    for i in range(0, len(X), bs):
        x, m = X[i:i + bs].to(dev), M[i:i + bs].to(dev); pr = 0
        for n in nets:
            with torch.autocast(dev, dtype=torch.bfloat16): pr = pr + n(prep(x, m)).float().softmax(-1)
        P.append((pr / len(nets)).cpu())
    return torch.cat(P)


def cmd_predict(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"; cb = np.load(a.codebook); cbn = F.normalize(torch.as_tensor(cb).float(), dim=1).to(dev)
    tg = pd.read_csv(a.target_csv); imgs = list_images(a.test_dir)
    X = torch.stack([load_img(imgs[str(k)]) for k in tg["ID"]]); M = torch.zeros(len(tg), 64, dtype=torch.bool)
    for j, t in enumerate(tg["target"].astype(str)): M[j, [int(p) for p in t.split()]] = True
    teacher, tdt = v2.get_teacher(dev); Tc = pseudo_codes(teacher, tdt, cbn, X, M, dev); del teacher; torch.cuda.empty_cache()
    paths = [p for p in a.ckpts.split(",") if p]; nets = load_nets(paths, cb, a.unfreeze, dev); t0 = time.time()
    print(f"adaptando {len(nets)} modelos a {len(X)} imagenes de test (sin etiquetas)", flush=True)
    for k, n in enumerate(nets):
        print(f"modelo {k + 1}/{len(nets)}", flush=True); adapt_net(n, X, M, Tc, dev, epochs=a.epochs, lr=a.lr, bb_lr=a.bb_lr, seed=k)
    P = ensemble_probs(nets, X, M, dev); pred = P.argmax(-1).numpy(); rows = []
    for j, (i, t) in enumerate(zip(tg["ID"].astype(str), tg["target"].astype(str))):
        rows += [(f"{i}_{p}", int(pred[j, int(p)])) for p in t.split()]
    pd.DataFrame(rows, columns=["Id", "Target"]).to_csv(a.out, index=False)
    print("escrito", a.out, len(rows), f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True); p = sp.add_parser("predict")
    p.add_argument("--test-dir", default="data/test"); p.add_argument("--target-csv", default="data/target.csv"); p.add_argument("--codebook", default="data/codebook.npy")
    p.add_argument("--ckpts", required=True); p.add_argument("--unfreeze", type=int, default=10); p.add_argument("--out", default="outputs/submission.csv")
    p.add_argument("--epochs", type=int, default=8); p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--bb-lr", type=float, default=2e-5)
    cmd_predict(ap.parse_args())
