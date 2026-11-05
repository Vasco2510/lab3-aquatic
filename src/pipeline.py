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
STAGES = ["data", "labels", "train", "predict", "check", "submit"]


def load_secrets(names=("HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")):
    env_file = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1); env_file[k.strip()] = v.strip().strip('"').strip("'")
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


def stage_train(a):
    OUT.mkdir(exist_ok=True)
    if a.persist and (pathlib.Path(a.persist) / "best.pt").exists() and not (OUT / "best.pt").exists():
        shutil.copy(pathlib.Path(a.persist) / "best.pt", OUT / "best.pt")  # reanudar desde Drive
    cmd = [sys.executable, "src/aquatic.py", "train", "--epochs", a.epochs, "--bs", a.bs,
           "--unfreeze", a.unfreeze, "--lr", a.lr]
    if a.resume: cmd.append("--resume")
    sh(cmd)
    persist(a, "best.pt")


def stage_predict(a):
    OUT.mkdir(exist_ok=True)
    sh([sys.executable, "src/aquatic.py", "predict"])
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
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--unfreeze", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--persist", default=None, help="carpeta (Drive) donde copiar best.pt y submission.csv")
    ap.add_argument("--message", default="auto")
    a = ap.parse_args()
    load_secrets()
    st = a.stages.split(",") + (["submit"] if a.submit else [])
    for s in st:
        assert s in STAGES, f"etapa desconocida {s}"
    for s in [x for x in STAGES if x in st]:
        print(f"\n===== {s} =====", flush=True); globals()["stage_" + s](a)
