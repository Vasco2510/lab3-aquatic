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
    ap.add_argument("--sd-model", default="stabilityai/stable-diffusion-2-inpainting")
    ap.add_argument("--sd-steps", type=int, default=30); ap.add_argument("--sd-samples", type=int, default=1)
    ap.add_argument("--sd-guidance", type=float, default=7.5); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default="outputs/diag.json")
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
    if "student" in methods or "fusion" in methods or "sd" in methods:
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
        Pst = torch.cat(P); del nets; torch.cuda.empty_cache()
        res["student"] = acc(Pst.argmax(-1), lab, M); print(f"student ({len(ps)} modelos): {res['student']:.4f}", flush=True)
        res["mask_media"] = float(M.sum(1).float().mean())
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
    print("\nRESUMEN", json.dumps({k: round(v, 4) for k, v in res.items()}))


if __name__ == "__main__": main()
