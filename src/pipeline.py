"""Orquestador de punta a punta (mismo comando en tu PC y en Colab).

  python src/pipeline.py                      # data -> labels -> train -> predict -> check
  python src/pipeline.py --submit             # ...y sube a Kaggle
  python src/pipeline.py --stages train,predict --resume
  python src/pipeline.py --persist /content/drive/MyDrive/lab3_aquatic   # copia ckpt/submission a Drive

Secretos (HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY), en este orden de prioridad:
  1) variables de entorno  2) Colab Secrets  3) archivo .env en la raíz
"""
import argparse, os, pathlib, shutil, subprocess, sys, time, zipfile

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
STAGES = ["doctor", "data", "labels", "diag", "preflight", "train", "predict", "check", "submit"]


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
    k = os.environ.get("KAGGLE_KEY", "")
    if k.startswith("KGAT_"):  # tokens nuevos de Kaggle: se pasan por KAGGLE_API_TOKEN
        os.environ.setdefault("KAGGLE_API_TOKEN", k)


def kaggle_cmd():
    """`python -m kaggle` ya no existe en versiones nuevas: usa el ejecutable `kaggle`."""
    exe = shutil.which("kaggle")
    return [exe] if exe else [sys.executable, "-c", "from kaggle.cli import main; main()"]


def sh(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call(list(map(str, cmd)), cwd=ROOT)


def stage_doctor(a):
    """Verifica en segundos todo lo que puede fallar por configuracion, antes de gastar GPU."""
    bad = []
    def chk(name, cond, hint=""):
        print(("  OK    " if cond else "  FALLA ") + name + ("" if cond else f"  -> {hint}"))
        if not cond: bad.append(name)
    needs_gpu = bool({"labels", "diag", "preflight", "train", "predict"} & set(a.stages.split(",")))
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
            r = subprocess.run(kaggle_cmd() + ["competitions", "submissions", "-c", COMPETITION],
                               capture_output=True, text=True, cwd=ROOT)
            k = os.environ.get("KAGGLE_KEY", "")
            print(f"  info  KAGGLE_USERNAME={os.environ['KAGGLE_USERNAME']!r} (debe ser el usuario exacto de tu perfil); "
                  f"KAGGLE_KEY: {len(k)} caracteres, formato {'KGAT_ (token nuevo)' if k.startswith('KGAT_') else 'clasico'}")
            chk(f"Kaggle: credenciales validas y unido a '{COMPETITION}'", r.returncode == 0,
                (r.stderr or r.stdout).strip()[:200])
    if "data" not in a.stages.split(","):
        for f in ("target.csv", "codebook.npy", "dino_vq.py"):
            chk(f"data/{f}", (DATA / f).exists(), "ejecuta la etapa data")
    if bad: sys.exit("DOCTOR: %d problema(s). Corrige y vuelve a ejecutar." % len(bad))
    print("DOCTOR: todo en orden")


def stage_data(a):
    """Descarga de la carpeta publica del curso UNA vez y la cachea en TU Drive (--persist/data):
    las siguientes corridas copian desde tu Drive y no gastan la cuota de descarga del profesor."""
    import gdown
    DATA.mkdir(exist_ok=True)
    cache = pathlib.Path(a.persist) / "data" if a.persist else None
    roots = [pathlib.Path(a.persist), cache] if a.persist else []  # raiz de persist (copia manual) y persist/data
    for name, fid in DRIVE_IDS.items():
        dst = DATA / name
        if not dst.exists():
            hit = next((r / name for r in roots if (r / name).exists()), None)
            if hit:
                print(f"{name}: copiando desde tu Drive ({hit})"); shutil.copy(hit, dst)
            else:
                gdown.download(id=fid, output=str(dst), quiet=False)
                if cache:
                    cache.mkdir(parents=True, exist_ok=True); shutil.copy(dst, cache / name)
                    print(f"{name}: guardado en cache {cache}")
        if name.endswith(".zip") and not (DATA / name[:-4]).exists():
            zipfile.ZipFile(dst).extractall(DATA)
    print("train:", len(list((DATA / "train").glob("*"))), "test:", len(list((DATA / "test").glob("*"))))


def stage_labels(a):
    if a.mode in ("v2", "v3"): print("v2/v3 calculan las etiquetas en linea: se omite"); return
    if (DATA / "labels_train.csv").exists() and not a.force:
        print("labels ya existen, se omite"); return
    sh([sys.executable, "src/aquatic.py", "labels", "--train-dir", "data/train"])


PRESETS = {  # --preset; cualquier flag explicito lo sobreescribe
    "smoke": dict(epochs=1, seeds=1, unfreeze=2),            # ~minutos: comprueba que todo corre
    "full":  dict(epochs=15, seeds=1, unfreeze=4),
    "night": dict(epochs=20, seeds=3, unfreeze=6, hours=8),  # ensamble de 3 semillas, con tope de tiempo
    # v2: maestro en linea + aumentos + validacion por clusters (src/v2.py)
    "v2smoke": dict(mode="v2", epochs=1, seeds=1, unfreeze=10),
    "v2":      dict(mode="v2", epochs=30, seeds=2, unfreeze=10, hours=6),
    # v2b: receta distinta para diversificar el ensamble (aumentos con zoom/ruido, mas suavizado, mas peso al coseno)
    # v3: igual que v2 pero con mascaras como las del test (agrupadas); es el arreglo principal
    "v3smoke": dict(mode="v3", epochs=1, seeds=1, unfreeze=10),
    "v3":      dict(mode="v3", epochs=36, seeds=3, unfreeze=10, hours=6),
    "v2b":     dict(mode="v2", epochs=30, seeds=3, unfreeze=10, hours=5, aug=2, ls=0.1, lam=2.0, lr=7e-4, wd=0.1),
}


def stage_preflight(a):
    """Prueba de humo automatica (segundos): 5 pasos de train + prediccion de 8 imagenes.
    Si algo esta roto, aborta ANTES de gastar horas de GPU."""
    OUT.mkdir(exist_ok=True)
    if a.mode in ("v2", "v3"):
        sh([sys.executable, "src/v2.py", "train", "--epochs", 1, "--max-steps", 3, "--bs", 8, "--unfreeze", a.unfreeze,
            "--ckpt", "outputs/smoke.pt", "--workers", 2, "--limit-imgs", 200, "--k", 10, "--holdout", 0.2,
            "--aug", a.aug if a.aug is not None else 1, "--mask", "walk" if a.mode == "v3" else "uniform"])
        (OUT / "split_v2.json").unlink(missing_ok=True)
    else:
        sh([sys.executable, "src/aquatic.py", "train", "--epochs", 1, "--max-steps", 5, "--bs", 8, "--unfreeze", a.unfreeze,
            "--ckpt", "outputs/smoke.pt", "--workers", 2])
    sh([sys.executable, "src/aquatic.py", "predict", "--ckpts", "outputs/smoke.pt", "--limit", 8,
        "--out", "outputs/smoke_submission.csv", "--unfreeze", a.unfreeze])
    for f in OUT.glob("smoke.pt*"): f.unlink()
    print("preflight OK: el pipeline funciona de punta a punta")


def ckpt_name(seed, mode="v1"): return f"best_s{seed}.pt" if mode == "v1" else f"{mode}_best_s{seed}.pt"


def stage_train(a):
    OUT.mkdir(exist_ok=True)
    pdir = pathlib.Path(a.persist) if a.persist else None
    per_seed = ((a.hours * 60 - 15) / a.seeds) if a.hours else 0  # 15 min de reserva para predecir/enviar
    for seed in range(a.seed0, a.seed0 + a.seeds):
        name = ckpt_name(seed, a.mode); state = name.replace(".pt", "_state.pt")
        done = f"done_s{seed}" if a.mode == "v1" else f"done_{a.mode}_s{seed}"  # marcador propio por version
        if pdir and (pdir / done).exists() and (pdir / name).exists() and not a.force:
            print(f"seed {seed} ya completada en {pdir}; se omite")
            shutil.copy(pdir / name, OUT / name); continue
        if pdir and not a.force:  # reanudar tras desconexion: ultimo estado completo + mejor checkpoint
            for f in (name, state):
                if (pdir / f).exists() and not (OUT / f).exists(): shutil.copy(pdir / f, OUT / f)
        script = "src/v2.py" if a.mode in ("v2", "v3") else "src/aquatic.py"
        cmd = [sys.executable, script, "train", "--epochs", a.epochs, "--bs", a.bs, "--unfreeze", a.unfreeze,
               "--lr", a.lr, "--seed", seed, "--ckpt", f"outputs/{name}", "--workers", a.workers]
        if a.mode in ("v2", "v3"):
            if a.mode == "v3": cmd += ["--mask", "walk"]
            if a.full: cmd.append("--full")
            for k in ("aug", "ls", "lam", "wd"):
                if getattr(a, k) is not None: cmd += [f"--{k}", getattr(a, k)]
        if per_seed: cmd += ["--time-budget", int(per_seed)]
        if a.persist: cmd += ["--persist", a.persist]
        if a.force: cmd.append("--fresh")
        elif a.mode == "v1" and (a.resume or (OUT / name).exists()): cmd.append("--resume")
        sh(cmd)
        if pdir:
            pdir.mkdir(parents=True, exist_ok=True); shutil.copy(OUT / name, pdir / name)
            if (OUT / (name + ".finished")).exists(): (pdir / done).touch()
            else: print(f"seed {seed} INCOMPLETA (tope de tiempo): la proxima corrida la continua desde su ultima epoca")


def stage_predict(a):
    OUT.mkdir(exist_ok=True)
    pats = [f"{('' if m == 'v1' else m + '_')}best_s*.pt" for m in
            (a.use.split(",") if a.use else (["v1", "v2", "v3"] if a.all_models else [a.mode]))]
    if a.persist:  # traer de Drive los checkpoints que falten (p. ej. runtime nuevo)
        for pat in pats:
            for f in pathlib.Path(a.persist).glob(pat):
                if "_state" not in f.name and not (OUT / f.name).exists(): shutil.copy(f, OUT / f.name)
    ck = sorted(str(p.relative_to(ROOT)) for pat in pats for p in OUT.glob(pat) if "_state" not in p.name)
    if not ck: sys.exit("No hay checkpoints en outputs/ (entrena primero)")
    sh([sys.executable, "src/aquatic.py", "predict", "--ckpts", ",".join(ck)])
    persist(a, "submission.csv")
    # registro: cada envio guarda una copia con nombre propio (nunca se sobreescribe)
    import datetime, re
    seeds = "-".join(re.search(r"s(\d+)", pathlib.Path(c).stem).group(1) for c in ck)
    kinds = a.use.replace(",", "") if a.use else ("v1v2v3" if a.all_models else a.mode)
    a.sub_tag = f"{datetime.datetime.now():%Y%m%d-%H%M}_{kinds}_n{len(ck)}_s{seeds}" + ("_full" if a.full else "")
    sdir = OUT / "submissions"; sdir.mkdir(exist_ok=True)
    shutil.copy(OUT / "submission.csv", sdir / f"{a.sub_tag}.csv")
    line = f"{a.sub_tag},{kinds},{len(ck)},{';'.join(pathlib.Path(c).name for c in ck)},{'full' if a.full else ''}\n"
    with open(sdir / "registro.csv", "a", encoding="utf-8") as f: f.write(line)
    if a.persist:
        pd_ = pathlib.Path(a.persist) / "submissions"; pd_.mkdir(parents=True, exist_ok=True)
        shutil.copy(sdir / f"{a.sub_tag}.csv", pd_ / f"{a.sub_tag}.csv")
        with open(pd_ / "registro.csv", "a", encoding="utf-8") as f: f.write(line)
    a.sub_name = f"{a.sub_tag}.csv"
    print(f"registro: submissions/{a.sub_tag}.csv")


def stage_diag(a):
    """Experimento corto: student vs inpainting clasico vs Stable Diffusion + maestro, en validacion con mascaras tipo test."""
    OUT.mkdir(exist_ok=True)
    if "sd" in a.diag_methods.split(","): sh([sys.executable, "-m", "pip", "install", "-q", "diffusers>=0.30", "accelerate", "opencv-python-headless"])
    else: sh([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"])
    names = [n for n in a.diag_ckpts.split(",") if n]
    if a.persist:
        for n in names + ["split_v2.json"]:
            f = pathlib.Path(a.persist) / n
            if f.exists() and not (OUT / n).exists(): shutil.copy(f, OUT / n)
    sh([sys.executable, "src/diag.py", "--n", a.diag_n, "--methods", a.diag_methods, "--sd-samples", a.diag_samples,
        "--ckpts", ",".join(f"outputs/{n}" for n in names)] + (["--persist", a.persist] if a.persist else []))


def stage_check(a):
    import pandas as pd
    if a.submit_file and a.persist:  # revisar un CSV guardado en Drive (el mismo que luego se envia)
        shutil.copy(pathlib.Path(a.persist) / "submissions" / a.submit_file, OUT / "submission.csv"); a.sub_name = a.submit_file
    sub, tg = pd.read_csv(OUT / "submission.csv"), pd.read_csv(DATA / "target.csv")
    exp = sum(len(str(t).split()) for t in tg["target"])
    assert len(sub) == exp, f"filas {len(sub)} != esperadas {exp}"
    assert not sub["Id"].duplicated().any(), "Ids duplicados"
    assert sub["Target"].between(0, 1023).all(), "Target fuera de 0..1023"
    print(f"submission OK: {len(sub)} filas; clases distintas usadas: {sub['Target'].nunique()}")
    # guarda anti-bugs: coincidencia con el ultimo envio guardado (dos modelos sanos coinciden en bastante mas del 30%)
    prev = sorted((pathlib.Path(a.persist) / "submissions").glob("*.csv")) if a.persist else []
    prev = [f for f in prev if f.name != getattr(a, "sub_name", "")]
    if prev:
        ref = pd.read_csv(prev[-1]).set_index("Id")["Target"]; cur = sub.set_index("Id")["Target"]
        ag = float((cur == ref.reindex(cur.index)).mean()); print(f"coincidencia con {prev[-1].name}: {ag:.1%}")
        a.agreement = ag
        if ag < 0.30 and a.submit and not a.force_submit:
            sys.exit("GUARDA: este CSV casi no coincide con el envio anterior (posible bug). No se envia. Usa --force-submit si es intencional.")


def stage_submit(a):
    if a.submit_file: a.sub_tag = pathlib.Path(a.submit_file).stem  # el CSV ya fue copiado y revisado en la etapa check
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        sys.exit("Faltan KAGGLE_USERNAME / KAGGLE_KEY")
    sh(kaggle_cmd() + ["competitions", "submit", "-c", COMPETITION,
        "-f", "outputs/submission.csv", "-m", (getattr(a, "sub_tag", None) or f"{a.preset} {a.mode}") if a.message == "auto" else a.message])


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
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float)
    ap.add_argument("--aug", type=int); ap.add_argument("--ls", type=float); ap.add_argument("--lam", type=float); ap.add_argument("--wd", type=float)
    ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mode", choices=["v1", "v2", "v3"], help="v1 = original; v2 = maestro en linea + aumentos; v3 = v2 + mascaras tipo test")
    ap.add_argument("--use", help="modelos para el ensamble, p. ej. v3 o v3,v2 (reemplaza a --all-models)")
    ap.add_argument("--all-models", action="store_true", help="el ensamble incluye modelos v1 y v2")
    ap.add_argument("--full", action="store_true", help="v2: entrenar con todos los datos (sin retener clusters)")
    ap.add_argument("--diag-n", type=int, default=120); ap.add_argument("--diag-methods", default="student,telea,sd")
    ap.add_argument("--diag-samples", type=int, default=1); ap.add_argument("--diag-ckpts", default="v2_best_s0.pt,v2_best_s1.pt")
    ap.add_argument("--submit-file", help="nombre de un CSV de persist/submissions para enviar tal cual (ya revisado)")
    ap.add_argument("--force-submit", action="store_true")
    ap.add_argument("--loop-hours", type=float, default=0, help="modo agente: repite ciclos (entrenar semillas nuevas, ensamblar, enviar) durante N horas")
    ap.add_argument("--cycle-seeds", type=int, default=2, help="semillas nuevas por ciclo del agente")
    ap.add_argument("--keep-submits", type=int, default=3, help="envios de Kaggle que el agente deja sin usar cada dia (limite 10)")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--persist", default=None, help="carpeta (Drive) donde copiar best.pt y submission.csv")
    ap.add_argument("--message", default="auto")
    a = ap.parse_args()
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k, None) is None: setattr(a, k, v)
    if a.mode is None: a.mode = "v1"
    if a.lr is None: a.lr = 1e-3
    load_secrets(extra=(pathlib.Path(a.persist) / ".env") if a.persist else None)
    try: print('codigo en ejecucion, commit:', subprocess.check_output(['git','rev-parse','--short','HEAD'], cwd=ROOT, text=True).strip())
    except Exception: pass
    def run_stages(st):
        for s_ in [x for x in STAGES if x in st]:
            print(f"\n===== {s_} =====", flush=True); globals()["stage_" + s_](a)

    def kaggle_rows():
        import csv, io
        r = subprocess.run(kaggle_cmd() + ["competitions", "submissions", "-c", COMPETITION, "-v"], cwd=ROOT, capture_output=True, text=True)
        try: return list(csv.DictReader(io.StringIO(r.stdout)))
        except Exception: return []

    def next_seed():
        import re
        seeds = [int(re.search(r"_s(\d+)\.pt$", f.name).group(1)) for d in (OUT, pathlib.Path(a.persist) if a.persist else OUT)
                 for f in d.glob(f"{a.mode}_best_s*.pt") if "_state" not in f.name]
        return max(seeds) + 1 if seeds else a.seed0

    if a.loop_hours:
        import datetime
        t_end = time.time() + a.loop_hours * 3600; cyc = fails = 0; log = (pathlib.Path(a.persist) / "loop_log.txt") if a.persist else (OUT / "loop_log.txt")
        def note(msg):
            line = f"{datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC | {msg}"; print(line, flush=True)
            with open(log, "a", encoding="utf-8") as f: f.write(line + "\n")
        note(f"agente iniciado: {a.loop_hours} h, {a.cycle_seeds} semillas por ciclo, modo {a.mode}, ensamble {a.use or a.mode}")
        while True:
            remain = (t_end - time.time()) / 3600; need = a.cycle_seeds * a.epochs * 2.8 / 60 + 0.35
            if remain < need: note(f"quedan {remain:.1f} h (< {need:.1f} h de un ciclo): el agente termina"); break
            if fails >= 3: note("3 ciclos seguidos con error: el agente se detiene para no gastar unidades"); break
            a.seed0, a.seeds, a.hours, a.stages = next_seed(), a.cycle_seeds, need * 1.25, "data,train,predict,check"
            today = f"{datetime.datetime.utcnow():%Y-%m-%d}"; used = sum(1 for r in kaggle_rows() if str(r.get("date", "")).startswith(today))
            can_submit = used <= 10 - a.keep_submits - 1
            st = a.stages.split(",") + (["submit"] if can_submit else []) + (["preflight"] if cyc == 0 and not a.no_preflight else [])
            if "doctor" not in st: st.append("doctor")
            cyc += 1; note(f"ciclo {cyc}: semillas {a.seed0}..{a.seed0 + a.seeds - 1}; envios hoy {used}/10 -> {'enviara' if can_submit else 'NO enviara (reserva)'}")
            try:
                run_stages(st); fails = 0
                tag = getattr(a, "sub_tag", "?"); score = "sin enviar"
                if can_submit:
                    for _ in range(6):
                        time.sleep(30)
                        row = next((r for r in kaggle_rows() if r.get("description") == tag), None)
                        if row and row.get("publicScore") not in (None, "", "None"): score = row["publicScore"]; break
                note(f"ciclo {cyc} listo: {tag} -> puntaje publico {score}")
            except (SystemExit, subprocess.CalledProcessError) as e:
                fails += 1; note(f"ciclo {cyc} fallo ({type(e).__name__}: {e}); reintento en 2 min"); time.sleep(120)
        sys.exit(0)

    st = a.stages.split(",") + (["submit"] if a.submit else [])
    if "train" in st and not a.no_preflight: st.append("preflight")
    if "doctor" not in st: st.append("doctor")
    for s in st:
        assert s in STAGES, f"etapa desconocida {s}"
    for s in [x for x in STAGES if x in st]:
        print(f"\n===== {s} =====", flush=True); globals()["stage_" + s](a)
