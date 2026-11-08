"""Diagnostico (GPU): ¿que metodo acierta mas codigos en imagenes de validacion con mascaras tipo test?

Compara, sobre N imagenes de train que los modelos v2 NO vieron (clusters retenidos):
  student : nuestros modelos v2 (promedio de softmax)
  telea   : inpainting clasico de OpenCV + maestro DINOv3 (igual que dino_vq.py)
  sd      : Stable Diffusion inpainting + maestro DINOv3 (pixeles visibles originales intactos)
  fusion  : student + w * votos de sd
Uso: python src/diag.py --n 120 --methods student,telea,sd --persist DIR
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aquatic, v2
from aquatic import Net, list_images, load_img, prep, patch_mask_to_pixels

PROMPT = "underwater photograph, seabed, rocks, corals, sand, marine life, natural light, sharp, detailed"
NEG = "text, watermark, frame, border, black region, blurry, cartoon"


def codes_of(teacher, tdt, cbn, x_u8_or_f, dev):
    xf = x_u8_or_f.to(dev).float()
    if xf.max() > 1.5: xf = xf / 255
    T, _ = v2.teacher_embed(teacher, tdt, xf)
    return (T @ cbn.T).argmax(-1)  # (B,64)


def acc(pred, lab, m): return float((pred[m] == lab[m]).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="data/train"); ap.add_argument("--codebook", default="data/codebook.npy")
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--methods", default="student,telea,sd")
    ap.add_argument("--ckpts", default="outputs/v2_best_s0.pt,outputs/v2_best_s1.pt"); ap.add_argument("--unfreeze", type=int, default=10)
    ap.add_argument("--persist", default=""); ap.add_argument("--split", default="")
    ap.add_argument("--sd-model", default="sd2-community/stable-diffusion-2-inpainting")
    ap.add_argument("--sd-steps", type=int, default=30); ap.add_argument("--sd-samples", type=int, default=1)
    ap.add_argument("--sd-guidance", type=float, default=7.5); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default="outputs/diag.json"); ap.add_argument("--test-dir", default="data/test"); ap.add_argument("--target-csv", default="data/target.csv")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; methods = a.methods.split(","); res = {}
    cb = np.load(a.codebook); cbn = F.normalize(torch.as_tensor(cb).float(), dim=1).to(dev)
    imgs = list_images(a.train_dir)
    spf = a.split or next((f for f in (os.path.join(a.persist, "split_v2.json"), "outputs/split_v2.json") if f and os.path.exists(f)), "")
    assert spf, "no hay split_v2.json (corre antes el entrenamiento v2)"
    val = sorted(json.load(open(spf))["val"]); ids = val[::max(1, len(val) // a.n)][:a.n]
    print(f"diag: {len(ids)} imagenes de validacion (clusters que los modelos v2 no vieron)", flush=True)
    X = torch.stack([load_img(imgs[i]) for i in ids])                      # uint8 (N,3,256,256)
    torch.manual_seed(0); M = v2.walk_masks(len(ids))                        # mascaras tipo test, fijas
    teacher, tdt = v2.get_teacher(dev)
    lab = torch.cat([codes_of(teacher, tdt, cbn, X[i:i + 16], dev) for i in range(0, len(ids), 16)]).cpu()
    mask_f = patch_mask_to_pixels(M).bool()                                  # (N,1,256,256)
    Pst = None
    if {"student", "fusion", "sd", "conf", "shift"} & set(methods):
        ps = [p for p in a.ckpts.split(",") if os.path.exists(p)]; assert ps, f"no existen {a.ckpts}"
        nets = []
        for p in ps:
            n_ = Net(cb, a.unfreeze).to(dev); n_.load_state_dict(torch.load(p, map_location=dev)); nets.append(n_.eval())
        P = []
        with torch.no_grad():
            for i in range(0, len(ids), 16):
                x, m = X[i:i + 16].to(dev), M[i:i + 16].to(dev); pr = 0
                for n_ in nets:
                    with torch.autocast(dev, dtype=torch.bfloat16): pr = pr + n_(prep(x, m)).float().softmax(-1)
                P.append((pr / len(nets)).cpu())
        Pst = torch.cat(P); nets_c = nets; torch.cuda.empty_cache()
        res["student"] = acc(Pst.argmax(-1), lab, M); print(f"student ({len(ps)} modelos): {res['student']:.4f}", flush=True)
        res["mask_media"] = float(M.sum(1).float().mean())
    if "shift" in methods:
        # ¿el test se parece menos al train que nuestra validacion? (descriptor global del maestro, coseno al vecino mas cercano)
        tr_ids = sorted(json.load(open(spf))["train"])

        @torch.no_grad()
        def desc(u8, masks=None):
            out = []
            for i in range(0, len(u8), 64):
                xf = u8[i:i + 64].to(dev).float() / 255
                if masks is not None: xf = xf * (1 - patch_mask_to_pixels(masks[i:i + 64].to(dev)))
                out.append(F.normalize(v2.teacher_embed(teacher, tdt, xf)[1], dim=1))
            return torch.cat(out)
        Dtr = torch.cat([desc(torch.stack([load_img(imgs[j]) for j in tr_ids[i:i + 256]])) for i in range(0, len(tr_ids), 256)])
        timgs = list_images(a.test_dir); Xt = torch.stack([load_img(timgs[k]) for k in sorted(timgs)])
        sim = lambda D: (D @ Dtr.T).max(1).values.cpu()
        s_val_m, s_val_u, s_test = sim(desc(X, M)), sim(desc(X)), sim(desc(Xt))
        q = lambda t: [round(float(t.quantile(x)), 3) for x in (0.1, 0.5, 0.9)]
        res["shift"] = dict(val_enmascarada=q(s_val_m), val_completa=q(s_val_u), test=q(s_test))
        print(f"similitud al vecino mas cercano del train (p10, p50, p90): VAL enmascarada {q(s_val_m)} | VAL completa {q(s_val_u)} | TEST {q(s_test)}", flush=True)
        if Pst is not None:
            per = torch.stack([(Pst[i].argmax(-1)[M[i]] == lab[i][M[i]]).float().mean() for i in range(len(ids))])
            order = torch.argsort(s_val_m); parts = torch.tensor_split(order, 4)
            res["acc_por_cuartil_similitud"] = [(round(float(s_val_m[g].mean()), 3), round(float(per[g].mean()), 4)) for g in parts]
            print("acierto de nuestros modelos por cuartil de similitud (de menos a mas parecido al train):", res["acc_por_cuartil_similitud"], flush=True)
            lo = s_val_m <= s_test.median()
            if lo.sum() > 5: res["acc_val_tan_poco_parecida_como_test"] = float(per[lo].mean()); print(f"acierto en validacion con similitud <= mediana del test ({int(lo.sum())} imgs): {res['acc_val_tan_poco_parecida_como_test']:.4f}")
    if "conf" in methods:
        # confianza de los modelos sobre los parches reales del test: si estan calibrados, estima el acierto sin etiquetas
        import pandas as pd
        tg = pd.read_csv(a.target_csv); timgs = list_images(a.test_dir); Pt, mt = [], []
        with torch.no_grad():
            for i in range(0, len(tg), 16):
                rows = tg.iloc[i:i + 16]; x = torch.stack([load_img(timgs[str(k)]) for k in rows["ID"]]).to(dev)
                m = torch.zeros(len(rows), 64, dtype=torch.bool, device=dev)
                for j, t in enumerate(rows["target"].astype(str)): m[j, [int(p) for p in t.split()]] = True
                pr = 0
                for n_ in nets_c:
                    with torch.autocast(dev, dtype=torch.bfloat16): pr = pr + n_(prep(x, m)).float().softmax(-1)
                Pt.append((pr / len(nets_c)).cpu()); mt.append(m.cpu())
        Pt, mt = torch.cat(Pt), torch.cat(mt)
        cv_, cp_ = Pst.max(-1).values[M], (Pst.argmax(-1) == lab)[M].float()     # confianza y acierto en val
        ct_ = Pt.max(-1).values[mt]
        edges = torch.quantile(cv_, torch.linspace(0, 1, 11)); b = lambda c: torch.bucketize(c, edges[1:-1])
        tabla = torch.stack([cp_[b(cv_) == k].mean() for k in range(10)])
        est = float(tabla[b(ct_)].mean())
        res["conf"] = dict(conf_media_val=float(cv_.mean()), acc_val=float(cp_.mean()), conf_media_test=float(ct_.mean()), acierto_estimado_test=est)
        print(f"confianza media: VAL {float(cv_.mean()):.3f} (acierto real {float(cp_.mean()):.3f}) | TEST {float(ct_.mean()):.3f} -> acierto estimado en test {est:.3f}", flush=True)
    if "dup" in methods:
        # ¿hay imagenes del test (o partes) identicas a las del train? hash exacto de parches visibles de 32x32
        import pandas as pd, collections
        tr_all = sorted(imgs); H = {}
        for j in tr_all:
            p = load_img(imgs[j]).permute(1, 2, 0).numpy().reshape(8, 32, 8, 32, 3).transpose(0, 2, 1, 3, 4).reshape(64, -1)
            for k in range(64): H.setdefault(hash(p[k].tobytes()), (j, k))
        tg = pd.read_csv(a.target_csv); timgs = list_images(a.test_dir); n_img = n_ge = n_full = 0; cnt_list = []; donors = {}
        for i_, t in zip(tg["ID"].astype(str), tg["target"].astype(str)):
            hid = {int(p) for p in t.split()}
            p = load_img(timgs[i_]).permute(1, 2, 0).numpy().reshape(8, 32, 8, 32, 3).transpose(0, 2, 1, 3, 4).reshape(64, -1)
            c = collections.Counter(); vis = 0
            for k in range(64):
                if k in hid: continue
                vis += 1; h = H.get(hash(p[k].tobytes()))
                if h and p[k].max() > 0: c[h[0]] += 1
            top = c.most_common(1)[0] if c else (None, 0); cnt_list.append(top[1]); n_img += 1
            n_ge += top[1] >= 8; n_full += top[1] >= 0.8 * vis
            if top[1] >= 8: donors[i_] = top[0]
        res["dup"] = dict(imagenes_test=n_img, con_8_o_mas_parches_identicos=n_ge, casi_completas=n_full)
        print(f"duplicados: de {n_img} imagenes de test, {n_ge} comparten >= 8 parches visibles idénticos con UNA imagen del train; {n_full} casi completas (>=80% de lo visible)", flush=True)
        json.dump(donors, open(a.out.replace(".json", "_donantes.json"), "w"))
    if "telea" in methods:
        import cv2
        out = []
        for i in range(len(ids)):
            im = X[i].permute(1, 2, 0).numpy()[:, :, ::-1].copy(); mk = mask_f[i, 0].numpy().astype(np.uint8) * 255
            r = cv2.inpaint(im, mk, 5, cv2.INPAINT_TELEA)[:, :, ::-1].copy()
            out.append(torch.from_numpy(r).permute(2, 0, 1))
        R = torch.stack(out)
        pr = torch.cat([codes_of(teacher, tdt, cbn, R[i:i + 16], dev) for i in range(0, len(ids), 16)]).cpu()
        res["telea"] = acc(pr, lab, M); print(f"telea + maestro: {res['telea']:.4f}", flush=True)
    if "sd" in methods:
        from diffusers import StableDiffusionInpaintPipeline
        from PIL import Image
        pipe = StableDiffusionInpaintPipeline.from_pretrained(a.sd_model, torch_dtype=torch.float16, safety_checker=None).to(dev)
        pipe.set_progress_bar_config(disable=True)
        K = a.sd_samples; votes = torch.zeros(len(ids), 64, 1024); first = None; t0 = time.time()
        for k in range(K):
            g = torch.Generator("cpu").manual_seed(1000 + k); outs = []
            for i in range(0, len(ids), a.bs):
                sl = range(i, min(i + a.bs, len(ids)))
                ims = [Image.fromarray(X[j].permute(1, 2, 0).numpy()).resize((512, 512), Image.BICUBIC) for j in sl]
                mks = [Image.fromarray((mask_f[j, 0].numpy().astype(np.uint8) * 255)).resize((512, 512), Image.NEAREST) for j in sl]
                gen = pipe(prompt=[PROMPT] * len(ims), negative_prompt=[NEG] * len(ims), image=ims, mask_image=mks,
                           num_inference_steps=a.sd_steps, guidance_scale=a.sd_guidance, generator=g).images
                for j, im in zip(sl, gen):
                    r = torch.from_numpy(np.array(im.resize((256, 256), Image.LANCZOS))).permute(2, 0, 1)
                    outs.append(torch.where(mask_f[j], r, X[j]))              # los pixeles visibles quedan intactos
                if i == 0 and k == 0: print(f"  primer lote SD: {time.time() - t0:.0f}s", flush=True)
            R = torch.stack(outs)
            pr = torch.cat([codes_of(teacher, tdt, cbn, R[i:i + 16], dev) for i in range(0, len(ids), 16)]).cpu()
            votes.scatter_add_(2, pr[..., None], torch.ones(len(ids), 64, 1))
            if first is None: first = pr; res["sd_1muestra"] = acc(pr, lab, M); print(f"sd 1 muestra + maestro: {res['sd_1muestra']:.4f}", flush=True)
        if K > 1: res[f"sd_voto{K}"] = acc(votes.argmax(-1), lab, M); print(f"sd voto de {K}: {res[f'sd_voto{K}']:.4f}")
        if Pst is not None:
            for w in (0.1, 0.2, 0.3, 0.5, 0.8, 1.5):
                res[f"fusion_w{w}"] = acc((Pst + w * votes / K).argmax(-1), lab, M); print(f"fusion student + {w} * sd: {res[f'fusion_w{w}']:.4f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    if a.persist: os.makedirs(a.persist, exist_ok=True); json.dump(res, open(os.path.join(a.persist, "diag.json"), "w"), indent=1)
    print("\nRESUMEN", json.dumps(res))


if __name__ == "__main__": main()
