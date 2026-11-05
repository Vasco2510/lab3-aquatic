"""Orquestador de punta a punta (mismo comando en tu PC y en Colab).

  python src/pipeline.py                      # data -> labels -> train -> predict -> check
  python src/pipeline.py --submit             # ...y sube a Kaggle
  python src/pipeline.py --stages train,predict --resume
  python src/pipeline.py --persist /content/drive/MyDrive/lab3_aquatic   # copia ckpt/submission a Drive

Secretos (HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY), en este orden de prioridad:
  1) variables de entorno  2) Colab Secrets  3) archivo .env en la raíz
"""
import argparse, os, pathlib, shutil, subprocess, sys, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
COMPETITION = "2026-ii-aquatic-inpainting"
DRIVE_IDS = {  # carpeta pública del curso
    "train.zip": "1fX_LiPIqm5s5S1pYt0vfVc_61rPBNeo9",
    "test.zip": "1FBww-BFkITeoDU5uWP11lSIbbraBIU5h",
    "target.csv": "1GNR3Lj3O1MEndpCQf79sguR3Rc-ZPiDY",
    "codebook.npy": "1Hhjxp23rHuciFSItJiLBP2wfA_AK2Nyw",
    "dino_vq.py": "1SjiA9eAcG8LrW-I6KOtiraJjbjwag8h0",
}
STAGES = ["doctor", "data", "labels", "preflight", "train", "predict", "check", "submit"]


def load_secrets(names=("HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"), extra=None):
    env_file = {}
    for f in (ROOT / ".env", extra):  # .env del repo y .env guardado en Drive (--persist)
        if f and pathlib.Path(f).exists():
            for line in pathlib.Path(f).read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1); env_file.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    try:
        from google.colab import userdata
    except Exception:
        userdata = None
    for n in names:
        if os.environ.get(n):
            continue
        v = None
        if userdata is not None:
            try: v = userdata.get(n)
            except Exception: v = None
        v = v or env_file.get(n)
        if v: os.environ[n] = v
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])


def sh(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call(list(map(str, cmd)), cwd=ROOT)


def stage_doctor(a):
    """Verifica en segundos todo lo que puede fallar por configuracion, antes de gastar GPU."""
    bad = []
    def chk(name, cond, hint=""):
        print(("  OK    " if cond else "  FALLA ") + name + ("" if cond else f"  -> {hint}"))
        if not cond: bad.append(name)
    needs_gpu = bool({"labels", "preflight", "train", "predict"} & set(a.stages.split(",")))
    if needs_gpu:
        import torch
        chk("GPU disponible", torch.cuda.is_available(), "Runtime con GPU (cuenta Colab Pro)")
    tok = os.environ.get("HF_TOKEN")
    chk("HF_TOKEN presente", bool(tok), "Secrets, PERSIST/.env o prompt")
    if tok:
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download("facebook/dinov3-vitb16-pretrain-lvd1689m", "config.json", token=tok); ok = True
        except Exception as e:
            ok = False; print("     detalle:", type(e).__name__)
        chk("acceso al modelo DINOv3", ok, "acepta la licencia en huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m")
    if a.persist:
        d = pathlib.Path(a.persist)
        chk(f"carpeta persist existe ({a.persist})", d.is_dir(), "Drive no montado o ruta mal escrita")
        if d.is_dir():
            try: (d / ".w").write_text("x"); (d / ".w").unlink(); w = True
            except Exception: w = False
            chk("persist escribible", w, "permisos de Drive")
            print("  info  .env en persist:", (d / ".env").exists())
    if a.submit:
        chk("KAGGLE_USERNAME/KEY", bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")), "faltan en .env")
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            r = subprocess.run([sys.executable, "-m", "kaggle", "competitions", "submissions", "-c", COMPETITION],
                               capture_output=True, text=True, cwd=ROOT)
            chk(f"Kaggle: credenciales validas y unido a '{COMPETITION}'", r.returncode == 0,
                (r.stderr or r.stdout).strip()[:200])
    if "data" not in a.stages.split(","):
        for f in ("target.csv", "codebook.npy", "dino_vq.py"):
            chk(f"data/{f}", (DATA / f).exists(), "ejecuta la etapa data")
    if bad: sys.exit("DOCTOR: %d problema(s). Corrige y vuelve a ejecutar." % len(bad))
    print("DOCTOR: todo en orden")


def stage_data(a):
    import gdown
    DATA.mkdir(exist_ok=True)
    for name, fid in DRIVE_IDS.items():
        dst = DATA / name
        if not dst.exists():
            gdown.download(id=fid, output=str(dst), quiet=False)
        if name.endswith(".zip") and not (DATA / name[:-4]).exists():
            zipfile.ZipFile(dst).extractall(DATA)
    print("train:", len(list((DATA / "train").glob("*"))), "test:", len(list((DATA / "test").glob("*"))))


def stage_labels(a):
    if (DATA / "labels_train.csv").exists() and not a.force:
        print("labels ya existen, se omite"); return
    sh([sys.executable, "src/aquatic.py", "labels", "--train-dir", "data/train"])


PRESETS = {  # --preset; cualquier flag explicito lo sobreescribe
    "smoke": dict(epochs=1, seeds=1, unfreeze=2),            # ~minutos: comprueba que todo corre
    "full":  dict(epochs=15, seeds=1, unfreeze=4),
    "night": dict(epochs=20, seeds=3, unfreeze=6, hours=8),  # ensamble de 3 semillas, con tope de tiempo
}


def stage_preflight(a):
    """Prueba de humo automatica (segundos): 5 pasos de train + prediccion de 8 imagenes.
    Si algo esta roto, aborta ANTES de gastar horas de GPU."""
    OUT.mkdir(exist_ok=True)
    sh([sys.executable, "src/aquatic.py", "train", "--epochs", 1, "--max-steps", 5, "--bs", 8, "--unfreeze", a.unfreeze,
        "--ckpt", "outputs/smoke.pt", "--workers", 2])
    sh([sys.executable, "src/aquatic.py", "predict", "--ckpts", "outputs/smoke.pt", "--limit", 8,
        "--out", "outputs/smoke_submission.csv", "--unfreeze", a.unfreeze])
    for f in OUT.glob("smoke.pt*"): f.unlink()
    print("preflight OK: el pipeline funciona de punta a punta")


def ckpt_name(seed): return f"best_s{seed}.pt"


def stage_train(a):
    OUT.mkdir(exist_ok=True)
    pdir = pathlib.Path(a.persist) if a.persist else None
    per_seed = ((a.hours * 60 - 15) / a.seeds) if a.hours else 0  # 15 min de reserva para predecir/enviar
    for seed in range(a.seed0, a.seed0 + a.seeds):
        name = ckpt_name(seed)
        if pdir and (pdir / f"done_s{seed}").exists() and not a.force:
            print(f"seed {seed} ya completada en {pdir}; se omite")
            shutil.copy(pdir / name, OUT / name); continue
        if pdir and (pdir / name).exists() and not (OUT / name).exists():
            shutil.copy(pdir / name, OUT / name)  # reanudar tras desconexion
        cmd = [sys.executable, "src/aquatic.py", "train", "--epochs", a.epochs, "--bs", a.bs, "--unfreeze", a.unfreeze,
               "--lr", a.lr, "--seed", seed, "--ckpt", f"outputs/{name}", "--workers", a.workers]
        if per_seed: cmd += ["--time-budget", int(per_seed)]
        if a.persist: cmd += ["--persist", a.persist]
        if a.resume or (OUT / name).exists(): cmd.append("--resume")
        sh(cmd)
        if pdir:
            pdir.mkdir(parents=True, exist_ok=True); shutil.copy(OUT / name, pdir / name); (pdir / f"done_s{seed}").touch()


def stage_predict(a):
    OUT.mkdir(exist_ok=True)
    ck = sorted(str(p.relative_to(ROOT)) for p in OUT.glob("best_s*.pt"))
    if not ck: sys.exit("No hay checkpoints en outputs/ (entrena primero)")
    sh([sys.executable, "src/aquatic.py", "predict", "--ckpts", ",".join(ck)])
    persist(a, "submission.csv")


def stage_check(a):
    import pandas as pd
    sub, tg = pd.read_csv(OUT / "submission.csv"), pd.read_csv(DATA / "target.csv")
    exp = sum(len(str(t).split()) for t in tg["target"])
    assert len(sub) == exp, f"filas {len(sub)} != esperadas {exp}"
    assert not sub["Id"].duplicated().any(), "Ids duplicados"
    assert sub["Target"].between(0, 1023).all(), "Target fuera de 0..1023"
    print(f"submission OK: {len(sub)} filas; clases distintas usadas: {sub['Target'].nunique()}")


def stage_submit(a):
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        sys.exit("Faltan KAGGLE_USERNAME / KAGGLE_KEY")
    sh([sys.executable, "-m", "kaggle", "competitions", "submit", "-c", COMPETITION,
        "-f", "outputs/submission.csv", "-m", a.message])


def persist(a, name):
    if a.persist and (OUT / name).exists():
        pathlib.Path(a.persist).mkdir(parents=True, exist_ok=True)
        shutil.copy(OUT / name, pathlib.Path(a.persist) / name); print("copiado a", a.persist)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="data,labels,train,predict,check")
    ap.add_argument("--submit", action="store_true", help="añade la etapa de envío a Kaggle")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--epochs", type=int); ap.add_argument("--unfreeze", type=int); ap.add_argument("--seeds", type=int)
    ap.add_argument("--hours", type=float, help="tope total de entrenamiento (horas); reparte entre semillas")
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--persist", default=None, help="carpeta (Drive) donde copiar best.pt y submission.csv")
    ap.add_argument("--message", default="auto")
    a = ap.parse_args()
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k, None) is None: setattr(a, k, v)
    load_secrets(extra=(pathlib.Path(a.persist) / ".env") if a.persist else None)
    try: print('codigo en ejecucion, commit:', subprocess.check_output(['git','rev-parse','--short','HEAD'], cwd=ROOT, text=True).strip())
    except Exception: pass
    st = a.stages.split(",") + (["submit"] if a.submit else [])
    if "train" in st and not a.no_preflight: st.append("preflight")
    if "doctor" not in st: st.append("doctor")
    for s in st:
        assert s in STAGES, f"etapa desconocida {s}"
    for s in [x for x in STAGES if x in st]:
        print(f"\n===== {s} =====", flush=True); globals()["stage_" + s](a)
