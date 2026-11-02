# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "scikit-learn",
#     "pandas",
#     "pillow",
#     "psutil",
#     "matplotlib",
#     "transformers",
#     "huggingface_hub",
# ]
# ///
# Knowledge as an Instrument (audit v6), marimo notebook for molab. Run all cells top to bottom.
# Set SMOKE = True in the config cell for a quick synthetic test. torch comes from molab's own CUDA image.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Knowledge as an Instrument — correcting, detecting and *auditing* foundation-model symbols with task labels

    **The shift.** Neuro-symbolic (NeSy) vision treats knowledge as a *teacher*: the program's labels are supposed to train the
    symbols. That premise provably fails in general — labels determine symbols only up to the program's symmetries (reasoning
    shortcuts; Marconato et al. 2023, Bortolotti et al. 2025), and no statistic of the labels can tell a shortcut from the truth.
    v6 gives knowledge a different job. A frozen foundation model (CLIP ViT-L/14) is the **symbol reader** — a noisy channel.
    The program plus each image's task label is a **code** on the reader's output, and a code does three things to a channel:

    1. **Correct** — decode inside the fibre of the label and retrain the reader on decoded values (KDR, v5; exact state
       evolution in the Gaussian model, G2).
    2. **Detect** — every label the reader fails to reproduce is a certain error; a reader that reproduces all labels is exactly
       right wherever the label *forces* the concept (shortcut ceiling).
    3. **Audit** — *measure the reader's per-concept accuracy without a single concept label.* This is the new result:
       the forced values are exact ground truth on a known sub-population, and three nested assumptions turn them into
       population claims — A0 (none: a distribution-free bracket of width P(unforced)), A1 (locality: point identification
       **iff the code forces the concept both ways**, a sharp interval if one way, nothing if never), A2 (multi-object codes such
       as MNIST-Add: full confusion matrix identified, by construction).

    One data-free property of the code — its **forcing type per concept (two-sided / one-sided / none)** — therefore decides
    three things at once: whether decode-and-retrain converges from chance (state evolution), whether the concept's accuracy
    is auditable from task labels (A1), and how much a shortcut can corrupt it (A0). Reasoning shortcuts are the unforced mass.
    Every claim in the notebook lives in a **ledger** with the weakest assumption that supports it.

    | ID | Kind | Statement | P(FAIL) if the code is correct |
    |---|---|---|---|
    | G1 | guarantee | decoded concept vectors satisfy the knowledge for their label | 0 |
    | G2 | guarantee | state evolution = fresh-sample simulation of cross-fitted KDR (max gap ≤ .025) | ≈ 0 |
    | G3 | guarantee | Learn-then-Test risk control of claims on unseen families (calibration labels), held-out check | ≤ .06 |
    | G4 | guarantee | class-calibrated conformal world elimination on unseen classes | ≈ .02 |
    | **G5** | guarantee | planted data: A0 CI covers always (even with cross-talk); A1 CI covers under locality; A2 exact on pairs | ≈ 0 |
    | **G6** | guarantee | real data: the A0 bracket contains every reader's true error on every sample (identity, knowledge exact) | 0 |
    | F1–F3 | exact fact | class-code vs per-image labels; forcing type of every code; BDD-OIA forces no concept TRUE | — |
    | P1–P3 | prediction | (replicated from pre-registration 3) KDR on identifying codes; state-evolution ranking; MNIST-Add | as stated |
    | **P4–P8** | prediction | label-free audit on real images (CUB-Logic, MNIST-Add): accuracy of the audit, model selection by audit | risky |
    | M | measured | locality tests, BDD-OIA one-sided intervals, coverage at guaranteed risk, shortcut gaps | no claim |

    **Data are real**: CUB-200-2011 (per-image attributes, certainty ≥ "probably"), MNIST, BDD-OIA (official release). Concept
    labels are used **only to score** methods and audits (and for the oracle / G3 calibration). `SMOKE = True` swaps in
    synthetic stand-ins with the same formats **only to test the plumbing**.

    **Run:** run all cells. Every stage checkpoints to `WORK_DIR` and is mirrored to
    `hf://buckets/<HF_BUCKET_NESY>/<run>` (optional, set HF_TOKEN_NESY and HF_BUCKET_NESY); a new session pulls the bucket and skips finished stages.
    Budget ≈ 20 GB VRAM, < 14 GB RAM, ≈ 2.5–3.5 h on an RTX 6000-class GPU. BDD-OIA: if the Google Drive download is blocked,
    set `BDDOIA_ZIP=/path/to/zip` (the rest of the notebook runs without it).
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pre-registration 4 — Knowledge as an Instrument (written 2026-10-18 before any real-image result; synthetic tests only)
    **Hypothesis.** Task labels plus knowledge cannot *teach* symbols beyond the code's symmetries, but they can *measure* a
    foundation-model reader: on concepts the code forces both ways, the reader's per-concept error is identified from task
    labels alone (under locality), and that measurement is accurate enough on real images to rank and select readers.

    **Why the dominant frame predicts otherwise.** In the teacher frame, task accuracy and label consistency are the only
    label-free signals, and both are blind to shortcuts; concept accuracy is assumed to require concept annotations.

    **Predictions** (thresholds fixed now; cells defined in §7c / §8):
    * **P4** CUB-Logic, zero-shot reader, audit codes design1 and design12 (two-sided), E1te and E3, non-degenerate cells:
      |A1 estimate − true error| ≤ .05 in ≥ 70 % of concept cells. Kill: < 50 %.
    * **P5** MNIST-Add, every reader (zero-shot, NeSy, NeSy-zs-init, KDR variants) on test pairs: |A2 estimated digit accuracy −
      true| ≤ .02. Kill: any reader off by > .05.
    * **P6** CUB-Logic, all readers × training codes × splits, independent audit code design2: Spearman(A1 estimate, true error)
      ≥ .70 over non-degenerate two-sided cells. Kill: < .40.
    * **P7** Label-free model selection, CUB-Logic E3: per (triple, training code, concept) pick the reader with the smallest
      design2-A1 estimate. Pooled true error of the selected reader ≤ the zero-shot error and ≤ the median candidate's error,
      and within .03 of the best candidate's (oracle-selected) error.
    * **P8** MNIST-Add: the KDR round chosen by A2-estimated accuracy has true accuracy within .01 of the best round.

    **Measured, no claim:** locality-test rejection rates; A0 width vs forced mass; BDD-OIA (rsbench rules, one-sided):
    A0/A1 intervals and how often they contain the true error, and how often the rsbench knowledge itself is wrong (forced values
    vs reason labels); MNIST-SumParity A2 estimates (non-identified: expected wrong).

    **Honest pre-statement.** CLIP's errors on a concept may depend on co-occurring attributes (binding / cross-talk), which
    violates locality; then A1 is biased (the planted cross-talk case in G5 shows by how much) and the locality test is the
    alarm. A0 does not depend on locality. Trained readers can drift toward a symmetry of their training code on the unforced
    mass, where locality is untestable; this is why P6/P7 use an independent audit code.
    **Analysis:** no metric or threshold changes after seeing results; verdicts print PASS/FAIL as written.
    """)
    return



@app.cell
def _():
    # ---- 1. SETUP: env vars BEFORE torch, imports, budgets, helpers ------------------------------
    import os, sys, gc, ctypes, json, time, math, tarfile, shutil, subprocess, urllib.request, itertools, re
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


    def ensure_pkg(pkg, import_name=None):
        try:
            __import__(import_name or pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


    for _p, _m in [("numpy", None), ("scipy", None), ("scikit-learn", "sklearn"), ("pandas", None),
                   ("pillow", "PIL"), ("psutil", None), ("matplotlib", None)]:
        ensure_pkg(_p, _m)

    import numpy as np
    import torch
    import psutil
    from PIL import Image
    from scipy.stats import spearmanr, rankdata

    RAM_CAP_GB, RAM_SOFT_GB, VRAM_CAP_GB = 30.0, 26.0, 70.0


    def rss_gb():
        p = psutil.Process(); t = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                t += c.memory_info().rss
            except psutil.Error:
                pass
        return t / 2**30


    def ram_guard(extra_gb, what="allocation"):
        used = rss_gb()
        if used + extra_gb > RAM_SOFT_GB:
            raise MemoryError(f"{what}: need ~{extra_gb:.1f} GB on top of {used:.1f} GB; soft cap {RAM_SOFT_GB} GB")


    def trim_ram():
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
        return rss_gb()


    def gpu_setup():
        if not torch.cuda.is_available():
            return {"device": "cpu", "torch": torch.__version__}
        free, total = torch.cuda.mem_get_info(); total_gb = total / 2**30
        cap = min(VRAM_CAP_GB, total_gb * 0.95)
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total_gb))
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        return {"device": "cuda", "name": torch.cuda.get_device_name(0), "total_gb": round(total_gb, 1),
                "cap_gb": round(cap, 1), "torch": torch.__version__}


    def gpu_free():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.empty_cache()


    def vram_report():
        if not torch.cuda.is_available():
            return {}
        return {"alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


    def atomic_write(path, writer):
        base, ext = os.path.splitext(path); tmp = f"{base}.tmp{ext}"
        writer(tmp); os.replace(tmp, path)


    def _json_default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)


    def save_json(path, obj):
        def _w(tmp):
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=1, default=_json_default)
        atomic_write(path, _w)


    def load_json(path):
        with open(path) as f:
            return json.load(f)


    HW = gpu_setup(); DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_DTYPE = torch.float16 if DEV.type == "cuda" else torch.float32


    # Encoders. transformers 5 changed get_image_features / get_text_features to return an output object instead of a
    # tensor, and renamed torch_dtype to dtype. These helpers call the vision/text towers and projections directly, which
    # gives the same vectors on transformers 4.x and 5.x, and cast after loading so no dtype keyword is needed.
    def load_clip(name, text=True):
        ensure_pkg("transformers")
        from transformers import CLIPModel, CLIPTokenizer
        model = CLIPModel.from_pretrained(name).to(device=DEV, dtype=MODEL_DTYPE).eval()
        return model, (CLIPTokenizer.from_pretrained(name) if text else None)


    def clip_image_embed(model, x):
        """Projected CLIP image embedding of normalised pixels x (n, 3, 224, 224), returned as float32."""
        return model.visual_projection(model.vision_model(pixel_values=x.to(MODEL_DTYPE)).pooler_output).float()


    def clip_text_embed(model, tok, prompts):
        """Projected CLIP text embedding of a list of prompts, returned as float32."""
        t = tok(prompts, padding=True, return_tensors="pt")
        out = model.text_model(input_ids=t["input_ids"].to(DEV), attention_mask=t["attention_mask"].to(DEV))
        return model.text_projection(out.pooler_output).float()
    print("hardware:", HW, "| RAM in use: %.2f GB" % rss_gb())
    return (DEV, HW, Image, MODEL_DTYPE, atomic_write, clip_image_embed, clip_text_embed, ensure_pkg, gpu_free, itertools, json, load_clip, load_json, np, os, ram_guard, rankdata, re, rss_gb, save_json, spearmanr, subprocess, sys, tarfile, time, torch, trim_ram, urllib, vram_report)



@app.cell
def _(ensure_pkg, np, os, subprocess, sys, time, torch):
    # ---- 2. CONFIG -------------------------------------------------------------------------------
    SMOKE = os.environ.get("SMOKE", "0") == "1"      # True = synthetic plumbing test only (no downloads, not results)
    # SMOKE = True
    FORCE_RECOMPUTE = False
    RUN_NAME = os.environ.get("FAG_RUN", "smoke_v6" if SMOKE else "audit_v6_clipL14")
    CFG = dict(
        seed=int(os.environ.get("FAG_SEED", 0)),          # confirmatory run: a NEW seed and a NEW run name (FAG_RUN), so nothing resumes
        work_dir=os.environ.get("FAG_WORK", "fag_work_" + RUN_NAME),
        cub_urls=["https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
                  "https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/"],
        encoder="openai/clip-vit-large-patch14",
        encoder_nt="facebook/dinov2-large",          # no-text control
        encode_dino=False,                            # DINOv2 control not used in v5
        img_batch=16 if SMOKE else 128,
        min_classes_per_attr=3 if SMOKE else 10,      # Koh-style 112-attribute filter
        reliable_certainty=3,                         # instance label trusted if certainty in {3 probably, 4 definitely}
        family_split=(0.6, 0.2, 0.2),                 # E1 train families / E2 refutation families / E3 unseen families
        n_seeds=2 if SMOKE else 5,
        pca_ranks=[4, 8, 16, 32, 64] if SMOKE else [16, 32, 64, 128, 256, 768],
        # ---- CUB-Logic codes and KDR ----
        n_triples=3 if SMOKE else 8,
        kdr_rounds=4 if SMOKE else 8, kdr_folds=5, zs_temp=1.5,
        n_ens=4 if SMOKE else 10, knn_k=10, nesy_epochs=200 if SMOKE else 400,
        se_mc=60000 if SMOKE else 300000,
        # ---- v6 audit ----
        audit_alpha=0.05, audit_planted_reps=40 if SMOKE else 200, ds_boot=10 if SMOKE else 50,
        # ---- MNIST ----
        mnist_train=3000 if SMOKE else 30000, mnist_test=1000 if SMOKE else 10000,
        # ---- BDD-OIA ----
        bdd_train_cap=1500 if SMOKE else 12000, bdd_test_cap=600 if SMOKE else 4000,
        # ---- guarantees ----
        risk_delta=0.10, risk_deltas=[0.05, 0.10, 0.20], risk_alpha=0.05, risk_check=0.01,
        # ---- world elimination ----
        alpha=0.1, world_m=8 if SMOKE else 40, world_gamma=0.12, world_eps_max=0.15, n_world_splits=2 if SMOKE else 5,
        deadline_h=11.0,
    )
    os.makedirs(CFG["work_dir"], exist_ok=True)
    T0 = time.time()
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])


    def ckpt(name):
        return os.path.join(CFG["work_dir"], name)


    def have(name):
        return (not FORCE_RECOMPUTE) and os.path.exists(ckpt(name))


    def check_deadline():
        if (time.time() - T0) / 3600 > CFG["deadline_h"]:
            hf_push(RUN_NAME, force=True)
            raise TimeoutError("session deadline reached; re-run to resume from checkpoints / HF bucket")


    print("SMOKE =", SMOKE, "| run:", RUN_NAME, "| work dir:", os.path.abspath(CFG["work_dir"]))


    # ---- 1b. HF BUCKET SYNC: every checkpoint/result is mirrored to the bucket; a new session resumes from it ------
    # Optional: mirror checkpoints to your own Hugging Face bucket. Set HF_TOKEN_NESY and HF_BUCKET_NESY; sync is off without a token.
    HF_TOKEN = os.environ.get("HF_TOKEN_NESY", "")
    HF_BUCKET = os.environ.get("HF_BUCKET_NESY", "your-hf-name/your-bucket")
    HF_URI = f"hf://buckets/{HF_BUCKET}"
    HF_SYNC = os.environ.get("HF_SYNC", "1" if HF_TOKEN else "0") == "1"
    os.environ["HF_TOKEN"] = HF_TOKEN
    SYNC_EXCLUDE = ["data/*", "*.tgz", "*.part", "*.tmp*"]   # raw dataset never uploaded; encoding chunks ARE synced (resume)
    HF_STATE = {"ok": None, "last_push": 0.0, "errors": []}


    def _hf():
        ensure_pkg("huggingface_hub>=1.32", "huggingface_hub")
        import importlib, huggingface_hub as h
        if not hasattr(h, "sync_bucket"):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub"])
            h = importlib.reload(h)
        return h


    def hf_init(run_name):
        """Create the bucket if needed and pull this run's previous checkpoints (cross-session resume)."""
        if not HF_SYNC:
            print("HF sync disabled"); HF_STATE["ok"] = False; return
        try:
            h = _hf()
            try:
                h.create_bucket(HF_BUCKET, exist_ok=True, token=HF_TOKEN)
            except Exception as e:          # bucket may already exist under an org we cannot create in
                print("create_bucket:", repr(e)[:160])
            HF_STATE["ok"] = True
            try:
                h.sync_bucket(f"{HF_URI}/{run_name}", CFG["work_dir"], token=HF_TOKEN, quiet=True)
                print(f"HF bucket ready: {HF_URI}/{run_name} (pulled existing checkpoints, if any)")
            except Exception as e:          # first run of this name: nothing to pull yet
                print(f"HF bucket ready: {HF_URI}/{run_name} (nothing pulled: {repr(e)[:120]})")
        except Exception as e:
            HF_STATE["ok"] = False; HF_STATE["errors"].append(repr(e)[:300])
            print("WARNING: HF bucket unavailable, continuing locally only:", repr(e)[:300])


    def hf_push(run_name, force=False, min_interval_s=60):
        """Mirror the work dir to the bucket (raw images excluded). Never raises."""
        if not HF_STATE["ok"]:
            return False
        if not force and time.time() - HF_STATE["last_push"] < min_interval_s:
            return False
        try:
            _hf().sync_bucket(CFG["work_dir"], f"{HF_URI}/{run_name}", exclude=SYNC_EXCLUDE, token=HF_TOKEN, quiet=True)
            HF_STATE["last_push"] = time.time(); return True
        except Exception as e:
            HF_STATE["errors"].append(repr(e)[:300]); print("WARNING: HF push failed:", repr(e)[:200]); return False


    hf_init(RUN_NAME)
    return (CFG, HF_STATE, HF_URI, RUN_NAME, SMOKE, T0, check_deadline, ckpt, have, hf_push)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Memory plan (computed before anything is allocated)
    | object | size | budget |
    |---|---|---|
    | CUB-200 CLIP ViT-L/14 features, 11,788 × 768 float32 | 0.04 GB (RAM + GPU copy) | RAM 26 soft / VRAM 70 |
    | CUB per-image attributes, 11,788 × 312 int8 (×2) | 0.007 GB | |
    | CLIP ViT-L/14 fp16 weights (encoding only, freed after) | ≈ 0.9 GB VRAM | |
    | image batch 128 × 3 × 224² fp16 + activations | ≈ 6–10 GB VRAM peak | |
    | MNIST 30k + 10k CLIP features × 768 float32 | 0.12 GB | |
    | BDD-OIA ≈ 22k frames × 768 float32 | 0.07 GB | |
    | decode tensors, MNIST pairs (n × 10 × 10 float64, n ≤ 15k) | ≤ 0.012 GB | |
    | BDD block decoding, 2^9 worlds × 12k frames float64 | ≈ 0.05 GB per copy | |

    Predicted peak: < 14 GB RAM (dominated by the CUB tarball extraction and PIL decoding threads), ≈ 20 GB VRAM. Images are
    streamed in 1,024-image chunks that are checkpointed, so a restart resumes mid-encoding.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Data: CUB-200-2011 (synthetic CUB-format stand-in only when SMOKE)
    Per-image attributes carry a certainty id (1 not visible, 2 guessing, 3 probably, 4 definitely); ground truth for every
    grounding metric = per-image labels with certainty ≥ 3. Knowledge = class-attribute percentages (`class_attribute_labels_continuous.txt`),
    and Koh-style class signatures for the class-code baselines. Classes are split **by family** (last word of the class name)
    into E1 (training families), E2 (refutation families, never trained on) and E3 (unseen deployment families).
    """)
    return



@app.cell
def _(CFG, Image, SMOKE, np, os, ram_guard, tarfile, urllib):
    def make_synthetic_cub(root, n_classes=24, imgs_per_class=30, n_attr=24, seed=0):
        """SMOKE ONLY. Attribute j paints a coloured patch at a fixed place; instance attributes = class rate draws;
        this plants a (near-)linear ground truth so certificate errors are measurable."""
        rng = np.random.default_rng(seed)
        base = os.path.join(root, "CUB_200_2011")
        os.makedirs(os.path.join(base, "images"), exist_ok=True); os.makedirs(os.path.join(base, "attributes"), exist_ok=True)
        rate = np.clip(rng.beta(0.6, 0.6, size=(n_classes, n_attr)), 0.02, 0.98)
        loc = rng.integers(0, 52, size=(n_attr, 2)); col = rng.integers(0, 255, size=(n_attr, 3)); bg = rng.integers(0, 255, size=(n_classes, 3))
        with open(os.path.join(root, "attributes.txt"), "w") as f:
            for j in range(n_attr):
                f.write(f"{j + 1} has_part{j // 4}_color::c{j % 4}\n")
        names = [f"{c + 1:03d}.Bird_{c}_Fam{c % 6}" for c in range(n_classes)]
        with open(os.path.join(base, "classes.txt"), "w") as f:
            for c, nm in enumerate(names):
                f.write(f"{c + 1} {nm}\n")
        L_img, L_lab, L_split, L_attr = [], [], [], []
        iid = 0; inst = np.zeros((n_classes, imgs_per_class, n_attr))
        for c in range(n_classes):
            d = os.path.join(base, "images", names[c]); os.makedirs(d, exist_ok=True)
            for k in range(imgs_per_class):
                iid += 1
                a = (rng.random(n_attr) < rate[c]).astype(int); inst[c, k] = a
                img = (np.zeros((64, 64, 3)) + bg[c] // 2 + rng.integers(0, 30, (64, 64, 3))).clip(0, 255).astype(np.uint8)
                for j in np.where(a == 1)[0]:
                    y0, x0 = loc[j]; img[y0:y0 + 10, x0:x0 + 10] = col[j]
                rel = f"{names[c]}/img_{iid}.png"; Image.fromarray(img).save(os.path.join(base, "images", rel))
                L_img.append(f"{iid} {rel}"); L_lab.append(f"{iid} {c + 1}"); L_split.append(f"{iid} {int(k < imgs_per_class // 2)}")
                for j in range(n_attr):
                    L_attr.append(f"{iid} {j + 1} {a[j]} {int(rng.choice([2, 3, 4], p=[.15, .35, .5]))} 1.0")
        for fn, L in [("images.txt", L_img), ("image_class_labels.txt", L_lab), ("train_test_split.txt", L_split)]:
            with open(os.path.join(base, fn), "w") as f:
                f.write("\n".join(L) + "\n")
        with open(os.path.join(base, "attributes", "image_attribute_labels.txt"), "w") as f:
            f.write("\n".join(L_attr) + "\n")
        np.savetxt(os.path.join(base, "attributes", "class_attribute_labels_continuous.txt"), inst.mean(1) * 100, fmt="%.4f")
        return base


    def download_cub(root):
        """CUB-200-2011 original archive. Caltech's server often answers 403, so the fast.ai S3 copy of the
        same file goes first; the md5 is the official one, so a wrong or truncated file is rejected."""
        import hashlib
        base = os.path.join(root, "CUB_200_2011")
        if os.path.exists(os.path.join(base, "images.txt")):
            return base
        tgz = os.path.join(root, "CUB_200_2011.tgz")
        md5_ok = "97eceeb196236b17998738112f37df78"
        s3 = "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz"
        urls = [s3, s3, s3] + CFG["cub_urls"]                 # retries: a dropped connection can end the stream early
        if not os.path.exists(tgz):
            last = None
            for url in urls:
                try:
                    print("downloading", url); tmp = tgz + ".part"; h = hashlib.md5()
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                        while True:
                            b = r.read(1 << 22)
                            if not b:
                                break
                            f.write(b); h.update(b)
                    if h.hexdigest() != md5_ok:
                        raise IOError(f"incomplete or wrong file ({os.path.getsize(tmp) / 2**30:.2f} GB, md5 {h.hexdigest()})")
                    os.replace(tmp, tgz); break
                except Exception as e:
                    last = e; print("  failed:", repr(e)[:200])
            else:
                raise RuntimeError(f"Could not download CUB; put CUB_200_2011.tgz in {root}. Last error: {last}")
        print("extracting ..."); ram_guard(1.0, "tar extract")
        with tarfile.open(tgz) as t:
            try:
                t.extractall(root, filter="data")          # Python >= 3.12 (and patched 3.10/3.11): safe extraction, no warning
            except TypeError:
                t.extractall(root)
        return base


    DATA_ROOT = os.path.join(CFG["work_dir"], "data"); os.makedirs(DATA_ROOT, exist_ok=True)
    if SMOKE:
        CUB = os.path.join(DATA_ROOT, "CUB_200_2011")
        if not os.path.exists(os.path.join(CUB, "images.txt")):
            CUB = make_synthetic_cub(DATA_ROOT)
    else:
        CUB = download_cub(DATA_ROOT)
    print("CUB root:", CUB)
    return (CUB,)



@app.cell
def _(CUB, atomic_write, ckpt, have, load_json, np, os, ram_guard, save_json, trim_ram):
    def parse_cub(base):
        def rd(fn):
            with open(os.path.join(base, fn)) as f:
                return [l.split() for l in f if l.strip()]
        imgs = rd("images.txt"); ids = np.array([int(r[0]) for r in imgs]); paths = [r[1] for r in imgs]
        lab = dict((int(r[0]), int(r[1]) - 1) for r in rd("image_class_labels.txt"))
        spl = dict((int(r[0]), int(r[1])) for r in rd("train_test_split.txt"))
        labels = np.array([lab[i] for i in ids]); is_train = np.array([spl[i] == 1 for i in ids])
        with open(os.path.join(base, "classes.txt")) as f:
            cnames = [l.split(None, 1)[1].strip() for l in f if l.strip()]
        an_path = os.path.join(os.path.dirname(base), "attributes.txt")
        if not os.path.exists(an_path):
            an_path = os.path.join(base, "attributes", "attributes.txt")
        with open(an_path) as f:
            anames = [l.split(None, 1)[1].strip() for l in f if l.strip()]
        idx = {i: n for n, i in enumerate(ids)}; N, M = len(ids), len(anames)
        ram_guard(2 * N * M / 2**30 + 0.5, "attribute matrices")
        A = np.zeros((N, M), np.int8); Cert = np.zeros((N, M), np.int8)
        with open(os.path.join(base, "attributes", "image_attribute_labels.txt")) as f:
            for line in f:
                p = line.split()
                if len(p) < 4:
                    continue
                i, j, pres, cert = int(p[0]), int(p[1]) - 1, int(p[2]), int(p[3])
                r = idx.get(i)
                if r is not None and 0 <= j < M:
                    A[r, j] = pres; Cert[r, j] = cert
        cont = np.loadtxt(os.path.join(base, "attributes", "class_attribute_labels_continuous.txt")) / 100.0
        return paths, labels, is_train, anames, cnames, A, Cert, cont


    if have("meta.npz"):
        _z = np.load(ckpt("meta.npz"), allow_pickle=False)
        LABELS, IS_TRAIN, A_INST, CERT, PCT = _z["labels"], _z["is_train"], _z["A"], _z["cert"], _z["pct"]
        _m = load_json(ckpt("meta_names.json")); PATHS, ANAMES, CLASSNAMES = _m["paths"], _m["anames"], _m["cnames"]
    else:
        PATHS, LABELS, IS_TRAIN, ANAMES, CLASSNAMES, A_INST, CERT, PCT = parse_cub(CUB)
        atomic_write(ckpt("meta.npz"), lambda t: np.savez(t, labels=LABELS, is_train=IS_TRAIN, A=A_INST, cert=CERT, pct=PCT))
        save_json(ckpt("meta_names.json"), {"paths": PATHS, "anames": ANAMES, "cnames": CLASSNAMES})
    N_IMG, N_CLS = len(LABELS), int(LABELS.max()) + 1
    print(f"{N_IMG} images, {N_CLS} classes, {len(ANAMES)} attributes | RAM {trim_ram():.2f} GB")
    return (ANAMES, A_INST, CERT, CLASSNAMES, IS_TRAIN, LABELS, N_CLS, N_IMG, PATHS, PCT)



@app.cell
def _(ANAMES, A_INST, CERT, CFG, CLASSNAMES, IS_TRAIN, LABELS, N_CLS, PCT, np, re):
    # ---- knowledge, ground truth, family split ----------------------------------------------------
    FAMILY = [re.split(r"[_ ]", n.split(".", 1)[-1])[-1].lower() for n in CLASSNAMES]
    _fams = sorted(set(FAMILY)); _rf = np.random.default_rng(CFG["seed"]); _rf.shuffle(_fams)
    _sizes = {f: sum(1 for x in FAMILY if x == f) for f in _fams}
    _tot = N_CLS; _acc = 0; GROUP_OF_FAM = {}
    for _f in _fams:                                   # greedy fill E1 -> E2 -> E3 by class count
        frac = _acc / _tot
        GROUP_OF_FAM[_f] = 1 if frac < CFG["family_split"][0] else (2 if frac < sum(CFG["family_split"][:2]) else 3)
        _acc += _sizes[_f]
    CLS_GROUP = np.array([GROUP_OF_FAM[f] for f in FAMILY])
    IMG_GROUP = CLS_GROUP[LABELS]
    E1_TR = np.where((IMG_GROUP == 1) & IS_TRAIN)[0]; E1_TE = np.where((IMG_GROUP == 1) & ~IS_TRAIN)[0]
    E2 = np.where(IMG_GROUP == 2)[0]; E3 = np.where(IMG_GROUP == 3)[0]
    # Koh-style class signatures (computed on E1-train images only for E1 classes; class-code baseline)
    _keep_rate = np.stack([A_INST[(LABELS == c) & IS_TRAIN].mean(0) for c in range(N_CLS)])
    KEEP = np.where((_keep_rate > 0.5).sum(0) >= CFG["min_classes_per_attr"])[0]
    K_CON = len(KEEP); CNAMES = [ANAMES[j] for j in KEEP]
    Y_INST = A_INST[:, KEEP].astype(np.int8); RELIABLE = CERT[:, KEEP] >= CFG["reliable_certainty"]
    SIG = (_keep_rate[:, KEEP] > 0.5).astype(np.int8); Y_CLASS = SIG[LABELS]
    PCT_K = PCT[:, KEEP]
    print(f"K={K_CON} concepts | families {len(_fams)} | classes E1/E2/E3 = {[(CLS_GROUP == g).sum() for g in (1, 2, 3)]} | "
          f"images E1tr {len(E1_TR)} E1te {len(E1_TE)} E2 {len(E2)} E3 {len(E3)} | reliable {RELIABLE.mean():.3f} | "
          f"instance != class-code on reliable labels: {((Y_INST != Y_CLASS) & RELIABLE).sum() / RELIABLE.sum():.3f}")
    return (CNAMES, E1_TE, E1_TR, E2, E3, FAMILY, GROUP_OF_FAM, K_CON, RELIABLE, SIG, Y_INST)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Frozen encoders: CLIP ViT-L/14 (image + concept text, prompt ensemble, random-word control) and DINOv2-L (no text)
    Chunked and resumable; each chunk is mirrored to the HF bucket. In SMOKE mode a fixed random pixel projection stands in
    for both encoders and the "text" directions are a noisy version of the planted directions (flagged, smoke only).
    """)
    return



@app.cell
def _(CFG, CNAMES, CUB, DEV, E1_TR, Image, K_CON, MODEL_DTYPE, N_IMG, PATHS, RUN_NAME, SMOKE, Y_INST, atomic_write, check_deadline, ckpt, clip_image_embed, clip_text_embed, gpu_free, have, hf_push, load_clip, np, os, rss_gb, torch, trim_ram, vram_report):
    def _load_img(rel, size=224):
        im = Image.open(os.path.join(CUB, "images", rel)).convert("RGB")
        w, h = im.size; s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        w, h = im.size; l, t = (w - size) // 2, (h - size) // 2
        return np.asarray(im.crop((l, t, l + size, t + size)), dtype=np.float32) / 255.0


    TEMPLATES = ["a photo of a bird whose {part} is {val}.", "a close-up photo of a bird; its {part} is {val}.",
                 "a bird with {part}: {val}."]
    _RAND_WORDS = ["lantern", "quorum", "basalt", "meridian", "tundra", "velvet", "cobalt", "harbor", "pivot", "saffron",
                   "orbit", "granite", "willow", "cipher", "ember", "glacier", "marble", "nectar", "prism", "thistle"]


    def prompt_for(name, tmpl=TEMPLATES[0]):
        part, val = name.split("::") if "::" in name else (name, "")
        part = part.replace("has_", "").replace("_", " ").strip()
        val = val.replace("_", " ").replace("(", "").replace(")", "").strip()
        return tmpl.format(part=part, val=val)


    def random_prompts(k, seed=0):
        r = np.random.default_rng(seed)
        return [f"a photo of a bird with {' '.join(r.choice(_RAND_WORDS, 3))}." for _ in range(k)]


    CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32); CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32); IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


    def _encode_images(tag, fwd, mean, std):
        chunk_dir = ckpt(f"img_chunks_{tag}"); os.makedirs(chunk_dir, exist_ok=True)
        bs, CH = CFG["img_batch"], 1024
        mu, sd = torch.tensor(mean, device=DEV).view(1, 3, 1, 1), torch.tensor(std, device=DEV).view(1, 3, 1, 1)
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(8)
        for c0 in range(0, N_IMG, CH):
            cf = os.path.join(chunk_dir, f"c{c0:06d}.npy")
            if os.path.exists(cf):
                continue
            check_deadline(); res = []
            for b0 in range(c0, min(c0 + CH, N_IMG), bs):
                arr = np.stack(list(pool.map(_load_img, PATHS[b0:min(b0 + bs, c0 + CH, N_IMG)])))
                x = (torch.from_numpy(arr).permute(0, 3, 1, 2).to(DEV) - mu) / sd
                with torch.inference_mode():
                    f = fwd(x).float()
                res.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
            atomic_write(cf, lambda t: np.save(t, np.concatenate(res)))
            print(f"  [{tag}] encoded {min(c0 + CH, N_IMG)}/{N_IMG}  VRAM {vram_report()}  RAM {rss_gb():.2f} GB", flush=True)
            hf_push(RUN_NAME)
        return np.concatenate([np.load(os.path.join(chunk_dir, f)) for f in sorted(os.listdir(chunk_dir)) if f.endswith(".npy")])


    def encode_all():
        if have("clip_img.npy") and have("clip_txt.npy") and have("clip_rand.npy") and (have("dino_img.npy") or not CFG["encode_dino"]):
            out = [np.load(ckpt("clip_img.npy")), np.load(ckpt("clip_txt.npy")), np.load(ckpt("clip_rand.npy"))]
            out.append(np.load(ckpt("dino_img.npy")) if have("dino_img.npy") else None)
            return out
        if SMOKE:
            rng = np.random.default_rng(1)
            P = rng.normal(size=(32 * 32 * 3, 96)).astype(np.float32) / 50
            feats = np.stack([(_load_img(p, 32) - 0.5).reshape(-1) @ P for p in PATHS]).astype(np.float32)
            feats /= np.linalg.norm(feats, axis=1, keepdims=True)
            tr = E1_TR[: max(40, len(E1_TR) // 3)]       # SMOKE-ONLY stand-in for text: noisy planted directions
            txt = np.stack([feats[tr][Y_INST[tr, j] == 1].mean(0) - feats[tr][Y_INST[tr, j] == 0].mean(0) for j in range(K_CON)])
            txt = (txt / np.linalg.norm(txt, axis=1, keepdims=True) + rng.normal(size=txt.shape) * 0.08).astype(np.float32)
            rnd = rng.normal(size=txt.shape).astype(np.float32)
            for nm, arr in [("clip_img.npy", feats), ("clip_txt.npy", txt), ("clip_rand.npy", rnd)]:
                np.save(ckpt(nm), arr)
            return feats, txt, rnd, None
        model, tok = load_clip(CFG["encoder"])
        feats = _encode_images("clip", lambda x: clip_image_embed(model, x), CLIP_MEAN, CLIP_STD)

        def _txt(prompts):
            with torch.inference_mode():
                return torch.nn.functional.normalize(clip_text_embed(model, tok, prompts), dim=1)
        txt = torch.nn.functional.normalize(sum(_txt([prompt_for(n, tm) for n in CNAMES]) for tm in TEMPLATES), dim=1).cpu().numpy()
        rnd = _txt(random_prompts(K_CON)).cpu().numpy()
        print("example prompt:", prompt_for(CNAMES[0]))
        for nm, arr in [("clip_img.npy", feats), ("clip_txt.npy", txt), ("clip_rand.npy", rnd)]:
            atomic_write(ckpt(nm), lambda t_, a=arr: np.save(t_, a))
        del model; gpu_free(); hf_push(RUN_NAME, force=True)
        dino = None
        if CFG["encode_dino"]:
            try:
                from transformers import AutoModel
                dm = AutoModel.from_pretrained(CFG["encoder_nt"]).to(device=DEV, dtype=MODEL_DTYPE).eval()
                dino = _encode_images("dino", lambda x: dm(pixel_values=x.to(MODEL_DTYPE)).pooler_output.float(), IMNET_MEAN, IMNET_STD)
                atomic_write(ckpt("dino_img.npy"), lambda t_: np.save(t_, dino)); del dm; gpu_free(); hf_push(RUN_NAME, force=True)
            except Exception as e:
                print("DINOv2 control skipped:", repr(e)[:200])
        return feats, txt, rnd, dino


    FEATS, TXT, TXT_RAND, DINO = encode_all()
    D_FEAT = FEATS.shape[1]
    Xall = torch.from_numpy(FEATS).float().to(DEV)
    print("CLIP features", FEATS.shape, "| text", TXT.shape, "| DINOv2", None if DINO is None else DINO.shape, "| RAM %.2f GB" % trim_ram())
    return (CLIP_MEAN, CLIP_STD, FEATS, TXT, TXT_RAND, Xall)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Statistical helpers (logistic / softmax probes, AUROC, bootstrap weights, centred text directions)
    """)
    return



@app.cell
def _(DEV, TXT, TXT_RAND, np, rankdata, torch):
    from sklearn.metrics import roc_auc_score


    def eta2(values, groups, mask=None):
        """Correlation ratio Var(E[v|g]) / Var(v) over entries where mask is True."""
        v = np.asarray(values, float); g = np.asarray(groups)
        if mask is not None:
            v, g = v[mask], g[mask]
        if len(v) < 3 or v.var() == 0:
            return float("nan")
        mu = v.mean(); ss_b = 0.0
        for c in np.unique(g):
            vc = v[g == c]; ss_b += len(vc) * (vc.mean() - mu) ** 2
        return float(ss_b / (len(v) * v.var()))


    def safe_auc(y, s):
        try:
            return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")
        except ValueError:
            return float("nan")


    def fit_logreg(X, Y, M=None, l2=1e-4, iters=300, w=None):
        """Full-batch (masked) logistic regression, L-BFGS. X (n,d) torch; Y (n,K) {0,1}; M (n,K) mask; w (n,) weights."""
        n, d = X.shape; K = Y.shape[1]
        lin = torch.nn.Linear(d, K).to(X.device)
        Mt = torch.ones_like(Y, dtype=torch.float32) if M is None else M.float()
        if w is not None:
            Mt = Mt * w[:, None]
        opt = torch.optim.LBFGS(lin.parameters(), lr=0.5, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            l = torch.nn.functional.binary_cross_entropy_with_logits(lin(X), Y.float(), reduction="none")
            loss = (l * Mt).sum() / Mt.sum().clamp(min=1) + l2 * (lin.weight ** 2).sum()
            loss.backward(); return loss
        opt.step(closure)
        return lin


    def fit_softmax(X, y, l2=1e-4, iters=300, w=None, C=None):
        n, d = X.shape; C = C or int(y.max().item()) + 1
        lin = torch.nn.Linear(d, C).to(X.device)
        wt = torch.ones(n, device=X.device) if w is None else w
        opt = torch.optim.LBFGS(lin.parameters(), lr=0.5, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = (torch.nn.functional.cross_entropy(lin(X), y, reduction="none") * wt).sum() / wt.sum() + l2 * (lin.weight ** 2).sum()
            loss.backward(); return loss
        opt.step(closure)
        return lin


    def fit_softmax_soft(X, Q, l2=1e-4, iters=300):
        """Softmax readout fit to soft targets Q (n,C) (cross-entropy to a distribution, i.e. the M-step of EM)."""
        lin = torch.nn.Linear(X.shape[1], Q.shape[1]).to(X.device)
        opt = torch.optim.LBFGS(lin.parameters(), lr=0.5, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = -(Q * torch.log_softmax(lin(X), 1)).sum(1).mean() + l2 * (lin.weight ** 2).sum()
            loss.backward(); return loss
        opt.step(closure)
        return lin


    def boot_weights(n, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.bincount(torch.randint(0, n, (n,), generator=g), minlength=n).float().to(DEV)


    def tci(x):
        x = np.asarray([v for v in x if v == v], dtype=float)
        if len(x) < 2:
            return (float(x.mean()) if len(x) else float("nan"), float("nan"))
        from scipy.stats import t as _t
        return float(x.mean()), float(_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


    def partial_spearman(x, y, z):
        """Spearman of x,y after linearly removing rank(z) from both ranks."""
        rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
        def res(a):
            A = np.vstack([rz, np.ones_like(rz)]).T
            return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
        return float(np.corrcoef(res(rx), res(ry))[0, 1])


    def text_dirs_centered(T):
        T = torch.from_numpy(T).float().to(DEV)
        return torch.nn.functional.normalize(T - T.mean(0, keepdim=True), dim=1)


    TXT_C = text_dirs_centered(TXT); TXT_RC = text_dirs_centered(TXT_RAND)
    return (TXT_C, boot_weights, fit_logreg, fit_softmax, fit_softmax_soft, safe_auc, text_dirs_centered)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6b. Programs as blocks, and the baselines (shared by CUB-Logic and BDD-OIA)
    A program is a list of independent **blocks**. A block = concept indices + a world→label consistency matrix
    WL (2^kb × n_labels): WL[v, o] = 1 iff world v can produce block-label o (several labels for an ambiguous world, none for an
    invalid one). Everything below (exact NeSy likelihood, forced values, conformal label sets) is computed from WL exactly.
    Baselines, each giving a per-(image, concept) **score** and **prediction**, compared to the ledger at *matched coverage*:
    **NeSy-conf** (DeepProbLog-style exact likelihood), **Ensemble** (10 bootstraps, vote margin), **k-NN** (forced evidence),
    **BEARS-style** (our reimplementation of Marconato et al. UAI 2024: members pushed to disagree on concepts via a KL term,
    score = 1 − entropy of the mean), **COCOCO-style** (our reimplementation of arXiv 2605.18202's idea: per-concept split
    conformal sets + conformal label set, pruned by deduction; *uses concept labels on a calibration split*),
    **LF-CBM-style** (Oikarinen et al. ICLR 2023 concept layer, cos³ match to CLIP similarities) and **zero-shot CLIP**.
    """)
    return



@app.cell
def _(DEV, boot_weights, gpu_free, itertools, np, torch):
    def worlds_of(kb):
        return np.array(list(itertools.product([0, 1], repeat=kb)), dtype=np.int64)


    def make_block(ix, fn):
        """fn(world tuple) -> list of label tuples the world can produce ([] = invalid world). Returns spec dict."""
        V = worlds_of(len(ix)); outs = [fn(tuple(int(x) for x in v)) for v in V]
        labels = sorted({o for os_ in outs for o in os_})
        lid = {o: i for i, o in enumerate(labels)}
        WL = np.zeros((len(V), len(labels)), bool)
        for wi, os_ in enumerate(outs):
            for o in os_:
                WL[wi, lid[o]] = True
        return {"ix": list(ix), "V": V, "WL": WL, "labels": labels, "lid": lid}


    def label_ids(spec, label_tuples):
        return np.array([spec["lid"].get(tuple(int(x) for x in t), -1) for t in label_tuples])


    def cons_mask(spec, lab_id):
        """(n, 2^kb) bool: worlds consistent with each image's block label (all False if the label is impossible)."""
        WL = spec["WL"]; m = np.zeros((len(lab_id), WL.shape[0]), bool); ok = lab_id >= 0
        m[ok] = WL[:, lab_id[ok]].T
        return m


    def forced_from_cons(cons, kb):
        V = worlds_of(kb); F = np.full((cons.shape[0], kb), -1, np.int64); any_ = cons.any(1)
        for j in range(kb):
            h1 = (cons & (V[:, j] == 1)[None]).any(1); h0 = (cons & (V[:, j] == 0)[None]).any(1)
            F[any_ & h1 & ~h0, j] = 1; F[any_ & h0 & ~h1, j] = 0
        return F


    def nesy_train(X, specs, cons_list, K, seed=0, epochs=400, w=None, kl_ref=None, lam=0.0, lr=0.02, W0=None, b0=None):
        """Exact marginal likelihood: sum_blocks -log sum_{v consistent} p(v|x); images with an impossible label skipped.
        kl_ref (n,K): BEARS-style diversity term  -lam * mean KL(Bern(p) || Bern(ref))."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = ((torch.randn(K, X.shape[1], generator=g) * 0.01).to(DEV) if W0 is None else W0.clone().float()).requires_grad_(True)
        b = (torch.zeros(K, device=DEV) if b0 is None else b0.clone().float()).requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=lr)
        wt = torch.ones(X.shape[0], device=DEV) if w is None else w
        Vs = [torch.tensor(s["V"], dtype=torch.float32, device=DEV) for s in specs]
        Cs = [torch.tensor(c, device=DEV) for c in cons_list]
        for ep in range(epochs):
            lo = X @ W.T + b; loss = 0.0
            for s, V, C in zip(specs, Vs, Cs):
                ix = s["ix"]
                lp = torch.nn.functional.logsigmoid(lo[:, ix]) @ V.T + torch.nn.functional.logsigmoid(-lo[:, ix]) @ (1 - V).T
                ll = torch.logsumexp(lp.masked_fill(~C, -1e9), 1); ok = C.any(1).float()
                loss = loss - (ll * ok * wt).sum() / (ok * wt).sum().clamp(min=1)
            if kl_ref is not None and lam > 0:
                p = torch.sigmoid(lo).clamp(1e-3, 1 - 1e-3); q = kl_ref.clamp(1e-3, 1 - 1e-3)
                loss = loss - lam * (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return W.detach(), b.detach()


    def lfcbm_projection(Xtr, T, epochs=600, lr=1e-3, seed=0):
        P = Xtr @ T.T; P = (P - P.mean(0)) / (P.std(0) + 1e-6)
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = (T.clone() + 0.01 * torch.randn(T.shape, generator=g).to(DEV)).requires_grad_(True)
        opt = torch.optim.Adam([W], lr=lr)
        for _ in range(epochs):
            A = Xtr @ W.T; A = (A - A.mean(0)) / (A.std(0) + 1e-6)
            loss = -(((A * P).mean(0)).clamp(min=0) ** 3).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return W.detach()


    def world_post(p, V):
        lw = np.log(np.clip(p, 1e-6, 1)) @ V.T + np.log(np.clip(1 - p, 1e-6, 1)) @ (1 - V).T
        pw = np.exp(lw - lw.max(1, keepdims=True)); return pw / pw.sum(1, keepdims=True)


    def concept_baselines(Xtr, specs, cons_tr, K, Xev, F_tr, T_dirs=None, cal=None, n_ens=10, knn_k=10, alpha=0.1):
        """Return {name: (score (n_ev,K), pred (n_ev,K) bool)}.  cal = (Xcal, Ccal (n,K) 0/1, Mcal (n,K) reliable mask,
        lab_ids_cal list per block of true block-label ids)  -> enables COCOCO-style."""
        out = {}
        heads = [nesy_train(Xtr, specs, cons_tr, K, seed=s, w=(boot_weights(Xtr.shape[0], 50 + s) if s else None)) for s in range(n_ens)]
        with torch.no_grad():
            P = torch.stack([torch.sigmoid(Xev @ W.T + b) for W, b in heads]).cpu().numpy()
        out["nesy_conf"] = (np.abs(P[0] - .5), P[0] > .5)
        vote = (P > .5).mean(0)
        out["ensemble"] = (np.abs(vote - .5) + 1e-3 * np.abs(P.mean(0) - .5), vote > .5)
        ref, Pb = None, []
        for s in range(min(5, n_ens)):
            W, b = nesy_train(Xtr, specs, cons_tr, K, seed=100 + s, kl_ref=ref, lam=0.1 if s else 0.0)
            with torch.no_grad():
                Pb.append(torch.sigmoid(Xev @ W.T + b)); pt = torch.sigmoid(Xtr @ W.T + b)
            ref = pt if ref is None else (ref * s + pt) / (s + 1)
        pb = torch.stack(Pb).mean(0).clamp(1e-6, 1 - 1e-6).cpu().numpy()
        out["bears_style"] = (1 + pb * np.log2(pb) + (1 - pb) * np.log2(1 - pb), pb > .5)
        sim = torch.nn.functional.normalize(Xev, dim=1) @ torch.nn.functional.normalize(Xtr, dim=1).T
        nn_idx = sim.topk(min(knn_k * 4, Xtr.shape[0]), dim=1).indices.cpu().numpy(); del sim
        km = np.full((Xev.shape[0], K), .5)
        for j in range(K):
            fj = F_tr[:, j]; vals = fj[nn_idx]                            # (n_ev, 4k)
            for i in range(len(vals)):
                v = vals[i][vals[i] >= 0][:knn_k]
                if len(v):
                    km[i, j] = v.mean()
        out["knn"] = (np.abs(km - .5), km > .5)
        if T_dirs is not None:
            Wl = lfcbm_projection(Xtr, T_dirs)
            for nm, Wp in (("lfcbm_style", Wl), ("zeroshot_clip", T_dirs)):
                with torch.no_grad():
                    At, Ae = Xtr @ Wp.T, Xev @ Wp.T
                    z = ((Ae - At.mean(0)) / (At.std(0) + 1e-6)).cpu().numpy()
                out[nm] = (np.abs(z), z > 0)
        if cal is not None:
            Xc, Cc, Mc, labc = cal
            with torch.no_grad():
                pc = torch.sigmoid(Xc @ heads[0][0].T + heads[0][1]).cpu().numpy()
            pe = P[0]; dec = np.zeros(pe.shape, bool); val = pe > .5
            for s, lc in zip(specs, labc):
                ix, V, WL = s["ix"], s["V"], s["WL"]; kb = len(ix); a_c = alpha / 2 / kb
                ok_set = np.ones((pe.shape[0], len(V)), bool)
                for jj, j in enumerate(ix):                              # per-concept split-conformal sets
                    m = Mc[:, j]
                    if m.sum() < 10:
                        continue
                    sc = 1 - np.where(Cc[m, j] == 1, pc[m, j], 1 - pc[m, j])
                    lvl = min(1.0, np.ceil((len(sc) + 1) * (1 - a_c)) / len(sc))
                    q = np.quantile(sc, lvl, method="higher")
                    keep1 = (1 - pe[:, j]) <= q; keep0 = pe[:, j] <= q
                    ok_set &= np.where(V[:, jj][None] == 1, keep1[:, None], keep0[:, None])
                plab_c = world_post(pc[:, ix], V) @ WL                   # label scores on calibration images
                good = lc >= 0
                if good.sum() >= 10:
                    scl = 1 - plab_c[good, lc[good]]
                    lvl = min(1.0, np.ceil((good.sum() + 1) * (1 - alpha / 2)) / good.sum())
                    ql = np.quantile(scl, lvl, method="higher")
                    Lset = (1 - world_post(pe[:, ix], V) @ WL) <= ql      # conformal label set (n_ev, n_labels)
                    ok_set &= (Lset.astype(np.int64) @ WL.T.astype(np.int64)) > 0   # deduction: world must produce an allowed label
                for jj, j in enumerate(ix):
                    a1 = (ok_set & (V[:, jj] == 1)[None]).any(1); a0 = (ok_set & (V[:, jj] == 0)[None]).any(1)
                    dec[:, j] = a1 ^ a0; val[:, j] = a1 & ~a0
            out["cococo_style"] = (dec.astype(float) + 1e-3 * np.abs(pe - .5), val)
        del heads; gpu_free()
        return out


    BASELINES = ["nesy_conf", "ensemble", "knn", "bears_style", "cococo_style", "lfcbm_style", "zeroshot_clip"]
    return (concept_baselines, cons_mask, forced_from_cons, label_ids, make_block, nesy_train)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6c. The audit: task labels + knowledge as a *measuring instrument* for a symbol reader (new in v6)
    Notation: concept j, reader ĉ_j(x) ∈ {0,1} (any fixed function of the image: zero-shot CLIP, KDR, a NeSy head, …),
    label y = f(c). The label **forces** c_j = v on the event F_v = {every concept vector in the fibre of y has c_j = v}
    (exact, from the program). F = F_0 ∪ F_1, U = the rest (c_j free). Forcing type of j: **two-sided** (both F_0, F_1 possible),
    **one-sided**, or **none** — a data-free fact of the code (F2).

    * **A0 — assumption-free bracket (Theorem A0).** If the knowledge is correct, {forced & ĉ_j ≠ forced value} ⊆ {ĉ_j wrong} ⊆
      {forced & wrong} ∪ U. Both outer events are observable without concept labels, so
      P(F, wrong_F) ≤ err_j ≤ P(F, wrong_F) + P(U), as an *identity on every sample* and, with Clopper–Pearson bounds on the
      two observable proportions, as a distribution-free confidence interval. Its width is the unforced mass P(U):
      **code design is audit design.** Corollary (shortcut ceiling): a reader that reproduces every label is exactly correct on F,
      so a reasoning shortcut can only err on U.
    * **A1 — locality (Theorem A1).** Assume the reader's error on j depends on the image only through c_j:
      ε_v = P(ĉ_j ≠ v | c_j = v, c_−j) for all c_−j. Then ε_1 = P(ĉ_j = 0 | F_1) and ε_0 = P(ĉ_j = 1 | F_0) exactly, and with
      q = P(ĉ_j = 1) = π(1 − ε_1) + (1 − π)ε_0: err_j = π ε_1 + (1 − π) ε_0, π = (q − ε_0)/(1 − ε_0 − ε_1).
      **Two-sided forcing ⇒ err_j point-identified; one-sided ⇒ a sharp closed-form interval; none ⇒ [0, 1].**
      (Sharpness and the "none" case are checked against an exact LP over all joint laws in the write-up.)
      Locality has a testable implication: the error must be equal across all labels that force the same value (χ² test,
      reported as `loc_p`). It is untestable on U — exactly where shortcuts live — so for *trained* readers we audit with an
      **independent audit code** (a different program on held-out images, design2 below): drift toward a symmetry σ of the
      training code is visible iff σ moves c_j on the audit code's forced set.
    * **A2 — multi-object codes (Theorem A2).** When the concepts of one label come from *different images* (MNIST-Add), locality
      and cross-image conditional independence hold by construction for any per-image reader evaluated out of fold, and the
      reader's full 10×10 confusion matrix is identified from (ĉ_a, ĉ_b, a + b) alone: y = 0 forces (0, 0) and gives row 0;
      y = s gives row s from rows < s (triangular recursion). Estimated by code-constrained Dawid–Skene EM, bootstrap CI over pairs.
      Sum parity forces nothing and has 20 label-preserving digit transpositions: not identified (reported as such).

    **G5** checks A0/A1 on planted data in this cell (theorem check: a FAIL is a bug).
    """)
    return



@app.cell
def _(CFG, RUN_NAME, ckpt, cons_mask, forced_from_cons, have, hf_push, label_ids, load_json, make_block, np, save_json):
    from scipy.stats import beta as _beta, chi2_contingency


    def cp_lo(k, n, a):
        """One-sided Clopper-Pearson lower bound at level a (exact)."""
        return float(_beta.ppf(a, k, n - k + 1)) if (k > 0 and n > 0) else 0.0


    def cp_hi(k, n, a):
        return float(_beta.ppf(1 - a, k + 1, n - k)) if (k < n and n > 0) else 1.0


    def _a1_point(e1, e0, q, pF1, pF0):
        den = 1 - e0 - e1
        if den <= 1e-6:
            return float("nan")
        pi = min(max((q - e0) / den, pF1), 1 - pF0)          # logical range: P(F1) <= pi <= 1 - P(F0)
        return pi * e1 + (1 - pi) * e0


    def _a1_one_sided(e0, q, pF1, pF0):
        """eps0 identified (forced FALSE exists), eps1 free. err = pi(1 - 2 e0) - q + 2 e0 over the feasible pi range.
        Feasible pi: pi(1 - e1) = q - (1 - pi) e0 must lie in [0, pi]; plus P(F1) <= pi <= 1 - P(F0)."""
        lo = max(pF1, 1 - q / e0 if e0 > 0 else 0.0, (q - e0) / (1 - e0) if e0 < 1 else 0.0)
        hi = 1 - pF0
        if lo > hi + 1e-12:
            return float("nan"), float("nan")
        f = lambda p: p * (1 - 2 * e0) - q + 2 * e0
        return min(f(lo), f(hi)), max(f(lo), f(hi))


    def audit_concept(chat, f, strata, alpha=0.05, truth=None, n_grid=7):
        """Label-free audit of one binary concept. chat (n,) reader output {0,1}; f (n,) forced value {-1,0,1} from the label;
        strata (n,) label key (locality test); truth (n,) optional, EVALUATION ONLY (never used in the estimate)."""
        chat = np.asarray(chat).astype(int); f = np.asarray(f); n = len(chat)
        nF1, nF0 = int((f == 1).sum()), int((f == 0).sum()); nU = n - nF1 - nF0
        k1, k0 = int(((f == 1) & (chat == 0)).sum()), int(((f == 0) & (chat == 1)).sum())
        kq = int(chat.sum()); wf = k1 + k0
        o = {"n": n, "type": ("two" if nF1 and nF0 else "one_T" if nF1 else "one_F" if nF0 else "none"),
             "forced_frac": (nF1 + nF0) / max(n, 1),
             "A0_sample": [wf / max(n, 1), (wf + nU) / max(n, 1)],
             "A0_ci": [cp_lo(wf, n, alpha / 2), cp_hi(wf + nU, n, alpha / 2)]}
        e1 = k1 / nF1 if nF1 else float("nan"); e0 = k0 / nF0 if nF0 else float("nan"); q = kq / max(n, 1)
        pF1, pF0 = nF1 / max(n, 1), nF0 / max(n, 1)
        a3 = alpha / 3
        if o["type"] == "two":
            o["A1"] = _a1_point(e1, e0, q, pF1, pF0)
            vals = [_a1_point(a, b, c, pF1, pF0) for a in np.linspace(cp_lo(k1, nF1, a3 / 2), cp_hi(k1, nF1, a3 / 2), n_grid)
                    for b in np.linspace(cp_lo(k0, nF0, a3 / 2), cp_hi(k0, nF0, a3 / 2), n_grid)
                    for c in np.linspace(cp_lo(kq, n, a3 / 2), cp_hi(kq, n, a3 / 2), n_grid)]
            vals = [v for v in vals if v == v]
            o["A1_ci"] = [min(vals), max(vals)] if vals else [0.0, 1.0]
            o["A1_degenerate"] = bool(1 - e0 - e1 <= 0.05)            # reader near/below chance on F: formula ill-conditioned
        elif o["type"] in ("one_F", "one_T"):
            if o["type"] == "one_F":
                ek, nk, kk, qq, pa, pb = e0, nF0, k0, q, pF1, pF0
            else:                                                      # relabel c -> 1 - c
                ek, nk, kk, qq, pa, pb = e1, nF1, k1, 1 - q, pF0, pF1
            o["A1_interval"] = list(_a1_one_sided(ek, qq, pa, pb))
            los, his = [], []
            for a in np.linspace(cp_lo(kk, nk, alpha / 4), cp_hi(kk, nk, alpha / 4), n_grid):
                kq_ = kq if o["type"] == "one_F" else n - kq
                for c in np.linspace(cp_lo(kq_, n, alpha / 4), cp_hi(kq_, n, alpha / 4), n_grid):
                    lo, hi = _a1_one_sided(a, c, pa, pb)
                    if lo == lo:
                        los.append(lo); his.append(hi)
            o["A1_ci"] = [min(los), max(his)] if los else [0.0, 1.0]
        else:
            o["A1_interval"] = [0.0, 1.0]; o["A1_ci"] = [0.0, 1.0]
        # locality test: error equal across labels forcing the same value (Bonferroni over values)
        pv = []
        for v in (0, 1):
            m = f == v
            keys = [s for s in np.unique(strata[m]) if (m & (strata == s)).sum() >= 5]
            if len(keys) >= 2:
                tab = np.array([[((strata == s) & m & (chat != v)).sum(), ((strata == s) & m & (chat == v)).sum()] for s in keys])
                if (tab.sum(0) > 0).all():
                    pv.append(float(chi2_contingency(tab)[1]))
        o["loc_p"] = min(1.0, min(pv) * len(pv)) if pv else float("nan")
        if truth is not None:
            o["true_err"] = float((chat != np.asarray(truth)).mean())
        return o


    def audit_reader(chat, F, strata, alpha=0.05, truth=None):
        """chat, F (n,K); strata (n,K) or (n,); truth optional (evaluation only)."""
        S = strata if np.ndim(strata) == 2 else np.repeat(np.asarray(strata)[:, None], chat.shape[1], 1)
        return [audit_concept(chat[:, j], F[:, j], S[:, j], alpha, None if truth is None else truth[:, j]) for j in range(chat.shape[1])]


    def ds_pairs(h1, h2, y, M, iters=200, init=None, diag0=0.6):
        """Code-constrained Dawid-Skene EM for a per-image reader on pairs: latent digits (a, b) inside the fibre M[y];
        shared confusion C[a, h] and prior pi. Label-free (uses reader outputs and task labels only)."""
        K = M.shape[1]
        if init is None:
            C = np.full((K, K), (1 - diag0) / (K - 1)); np.fill_diagonal(C, diag0); pi = np.full(K, 1.0 / K)
        else:
            C, pi = init[0].copy(), init[1].copy()
        Mm = M[y]
        for _ in range(iters):
            J = pi[None, :, None] * pi[None, None, :] * C[:, h1].T[:, :, None] * C[:, h2].T[:, None, :] * Mm
            J /= J.sum((1, 2), keepdims=True) + 1e-300
            A, B = J.sum(2), J.sum(1)
            pi = A.sum(0) + B.sum(0); pi /= pi.sum()
            Cn = np.zeros((K, K)); np.add.at(Cn.T, h1, A); np.add.at(Cn.T, h2, B)
            C = (Cn + 1e-6) / (Cn + 1e-6).sum(1, keepdims=True)
        return C, pi


    def ds_audit(h1, h2, y, M, n_boot=50, seed=0, alpha=0.05):
        """Label-free accuracy estimate pi . diag(C) of a digit reader, with a percentile bootstrap over pairs."""
        C, pi = ds_pairs(h1, h2, y, M)
        est = float(pi @ np.diag(C)); r = np.random.default_rng(seed); bs = []
        for _ in range(n_boot):
            i = r.integers(0, len(y), len(y))
            Cb, pb = ds_pairs(h1[i], h2[i], y[i], M, iters=30, init=(C, pi)); bs.append(float(pb @ np.diag(Cb)))
        return {"est_acc": est, "ci": [float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))] if bs else [0.0, 1.0],
                "confusion": np.round(C, 4).tolist(), "prior": np.round(pi, 4).tolist()}


    # ---- G5: theorem check of A0 / A1 on planted data (population truth known exactly) ---------------------------------
    def _planted_codes():
        out = {}
        for nm, fn in {"design1": lambda v: (v[0] & v[1], v[1] & (1 - v[2]), int(v[0] + v[1] + v[2] >= 2)),
                       "or3": lambda v: (int(v[0] or v[1] or v[2]),)}.items():
            out[nm] = (make_block([0, 1, 2], lambda v, fn=fn: [tuple(fn(v))]), fn)
        return out


    if have("s6c_audit_check.json"):
        G5 = load_json(ckpt("s6c_audit_check.json"))
    else:
        G5 = {"cases": []}; _r = np.random.default_rng(CFG["seed"] + 66)
        for _nm, (_s_, fn_) in _planted_codes().items():
            for xt in (0.0, 0.15):
                cov0 = cov1 = cnt = rej = 0; abserr = []
                for rep in range(CFG["audit_planted_reps"]):
                    _n = 3000; pc = _r.uniform(0.25, 0.75, 3); e0, e1 = _r.uniform(0.05, 0.35, 3), _r.uniform(0.05, 0.35, 3)
                    C = (_r.random((_n, 3)) < pc).astype(int)
                    _lab = label_ids(_s_, [tuple(fn_(tuple(int(x) for x in c))) for c in C])
                    F = forced_from_cons(cons_mask(_s_, _lab), 3)
                    pflip = np.where(C == 1, e1, e0) + xt * np.roll(C, -1, 1)
                    H = np.where(_r.random((_n, 3)) < pflip, 1 - C, C)
                    pop = pc * e1 + (1 - pc) * e0 + xt * np.roll(pc, -1)        # exact population error
                    for _j, _a in enumerate(audit_reader(H, F, _lab, CFG["audit_alpha"])):
                        cnt += 1; cov0 += _a["A0_ci"][0] <= pop[_j] <= _a["A0_ci"][1]
                        cov1 += _a["A1_ci"][0] <= pop[_j] <= _a["A1_ci"][1]
                        if _a["type"] == "two":
                            abserr.append(abs(_a["A1"] - pop[_j]))
                        rej += (_a["loc_p"] < 0.05) if _a["loc_p"] == _a["loc_p"] else 0
                G5["cases"].append({"code": _nm, "crosstalk": xt, "cells": cnt, "A0_cov": cov0 / cnt, "A1_cov": cov1 / cnt,
                                    "A1_mae": float(np.mean(abserr)) if abserr else None, "locality_rejected": rej / cnt})
                print(f"G5 planted {_nm:8s} cross-talk {xt}: A0 coverage {cov0 / cnt:.3f} | A1 coverage {cov1 / cnt:.3f} | "
                      f"A1 |err| {np.mean(abserr) if abserr else float('nan'):.4f} | locality rejected {rej / cnt:.3f}", flush=True)
        # A2: MNIST-Add-style pairs, planted structured confusion (theorem: exact identification)
        def _mask(task):
            M = np.zeros((19 if task == "add" else 2, 10, 10), bool)
            for a_ in range(10):
                for b_ in range(10):
                    M[(a_ + b_) if task == "add" else (a_ + b_) % 2, a_, b_] = True
            return M
        G5["A2"] = []
        for acc in (0.45, 0.8):
            Cp = np.zeros((10, 10))
            for a_ in range(10):
                Cp[a_, a_] = acc + _r.uniform(-.1, .1); oth = _r.choice([x for x in range(10) if x != a_], 2, replace=False)
                Cp[a_, oth] = (1 - Cp[a_, a_]) * np.array([.6, .3]); Cp[a_] += (1 - Cp[a_].sum()) / 10
            _n = 8000; A_, B_ = _r.integers(0, 10, _n), _r.integers(0, 10, _n)
            cum = Cp.cumsum(1); h1 = (_r.random(_n)[:, None] > cum[A_]).sum(1); h2 = (_r.random(_n)[:, None] > cum[B_]).sum(1)
            Ch, ph = ds_pairs(h1, h2, A_ + B_, _mask("add"))
            G5["A2"].append({"acc_planted": float(np.mean(np.diag(Cp))), "true_acc": float(((h1 == A_).mean() + (h2 == B_).mean()) / 2),
                             "est_acc": float(ph @ np.diag(Ch)), "max_abs_C_err": float(np.abs(Ch - Cp).max())})
            print("G5 planted A2 (pairs, sum code):", {k: round(v, 4) for k, v in G5["A2"][-1].items()}, flush=True)
        save_json(ckpt("s6c_audit_check.json"), G5); hf_push(RUN_NAME)
    G5_OK = (all(c["A0_cov"] >= 1 - CFG["audit_alpha"] - 0.02 for c in G5["cases"]) and
             all(c["A1_cov"] >= 1 - CFG["audit_alpha"] - 0.02 for c in G5["cases"] if c["crosstalk"] == 0) and
             all(abs(x["est_acc"] - x["true_acc"]) <= 0.02 for x in G5["A2"]))
    print("G5 (A0 always valid; A1 valid under locality; A2 exact on pairs):", "PASS" if G5_OK else "FAIL")
    return (G5, G5_OK, audit_reader, ds_audit, ds_pairs)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. State evolution of decode-and-retrain (theory, verified here before it is used as a prediction)
    Gaussian model: z ~ N(0, I_2K); concept c_j = 1[z_j > τ_j] (prevalence π_j = 1 − Φ(τ_j)); readout j has unit-norm
    direction v_j in span(e_1..e_2K) (coordinate K+j carries its own noise). One round of cross-fitted KDR =
    (i) calibrated per-concept posteriors P(c_j=1 | s_j) = Φ((ρ_j s_j − τ_j)/√(1−ρ_j²)), ρ_j = ⟨v_j, e_j⟩;
    (ii) MAP decoding inside the fibre of the label; (iii) least-squares refit on the decoded ±1 values, m_j = E[z t_j],
    plus fresh estimation noise of squared norm q = (d − 2K)/n outside the span (cross-fitting makes it independent of z).
    Because (ii) depends on z only through 2K coordinates, the population update is a closed map on (K × 2K) matrices —
    the state evolution. Its error is evaluated with the refit's own intercept. **Verified below against a fresh-sample
    simulation (G2).**
    """)
    return



@app.cell
def _(CFG, RUN_NAME, ckpt, have, hf_push, itertools, load_json, np, safe_auc, save_json):
    from scipy.stats import norm as norm_dist


    def _code_tables(fn, K):
        W = np.array(list(itertools.product([0, 1], repeat=K)), dtype=np.int64)
        Y = [fn(tuple(int(x) for x in w)) for w in W]
        keys = sorted(set(Y), key=str); lab = np.array([keys.index(y) for y in Y])
        M = np.zeros((len(keys), len(W)), bool); M[lab, np.arange(len(W))] = True
        return W, lab, M


    def _widx(C):
        return (C * (2 ** np.arange(C.shape[1])[::-1])).sum(1)


    def _decode(P, W, M, ylab, soft=True):
        """soft=True: posterior marginals inside the fibre (EM); soft=False: MAP concept vector (hard EM)."""
        P = np.clip(P, 1e-6, 1 - 1e-6); L = np.where(M[ylab], np.log(P) @ W.T + np.log(1 - P) @ (1 - W).T, -np.inf)
        if not soft:
            return W[L.argmax(1)].astype(float)
        L = L - L.max(1, keepdims=True); Pw = np.exp(L); Pw /= Pw.sum(1, keepdims=True); return Pw @ W


    def se_run(fn, K, taus, rho0, q, rounds, nmc, seed=0, fresh_sim=None, soft=True):
        """State evolution (fresh_sim=None) or a fresh-sample simulation with d - 2K = fresh_sim[0] noise dims and
        n = fresh_sim[1] samples per round. Returns per-round per-concept error of the LS readout (with intercept)."""
        rng = np.random.default_rng(seed); W, lab, M = _code_tables(fn, K); taus = np.asarray(taus, float)
        V = np.zeros((K, 2 * K)); V[np.arange(K), np.arange(K)] = rho0; V[np.arange(K), K + np.arange(K)] = np.sqrt(1 - np.asarray(rho0) ** 2)
        B = np.zeros(K)                                              # readout intercepts (round 0: zero-shot threshold at the mean)
        def err_of(V, B, z, C):
            return ((z @ V.T + B > 0).astype(int) != C).mean(0)
        out = []
        for r in range(rounds + 1):
            if fresh_sim is None:
                z = rng.normal(size=(nmc, 2 * K)); extra = None
            else:
                z = rng.normal(size=(fresh_sim[1], 2 * K)); extra = rng.normal(size=(fresh_sim[1], fresh_sim[0]))
            C = (z[:, :K] > taus).astype(np.int64); ylab = lab[_widx(C)]
            if fresh_sim is None:
                out.append(err_of(V, B, z, C))
            else:
                Vfull = np.concatenate([V, VX], 1) if r else np.concatenate([V, np.zeros((K, fresh_sim[0]))], 1)
                zf = np.concatenate([z, extra], 1); out.append(((zf @ Vfull.T + B > 0).astype(int) != C).mean(0))
            if r == rounds:
                break
            if fresh_sim is None:
                nv = np.linalg.norm(V, axis=1); S = z @ V.T / nv; rho = np.clip(V[np.arange(K), np.arange(K)] / nv, -.9999, .9999)
            else:
                nv = np.linalg.norm(Vfull, axis=1); S = zf @ Vfull.T / nv; rho = np.clip(V[np.arange(K), np.arange(K)] / nv, -.9999, .9999)
            P = norm_dist.cdf((rho * S - taus) / np.sqrt(1 - rho ** 2)); T = 2 * _decode(P, W, M, ylab, soft) - 1
            if fresh_sim is None:
                m = (z[:, :, None] * T[:, None, :]).mean(0).T - 0                   # (K, 2K); E[z] = 0
                B = T.mean(0); nm2 = (m ** 2).sum(1)
                V = m.copy(); V[np.arange(K), K + np.arange(K)] = np.sign(m[np.arange(K), K + np.arange(K)] + 1e-12) * \
                    np.sqrt(m[np.arange(K), K + np.arange(K)] ** 2 + q * (T.var(0) + 0.0))
            else:
                Tc = T - T.mean(0); mf = (zf.T @ Tc / len(zf)).T; B = T.mean(0) - 0.0
                V, VX = mf[:, :2 * K], mf[:, 2 * K:]
        return out


    def gauss_auc(rho, tau, nmc=200000, seed=1):
        z = np.random.default_rng(seed).normal(size=(nmc, 2)); s = rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]
        return safe_auc((z[:, 0] > tau).astype(int), s)


    _RHO_GRID = np.linspace(0.0, 0.995, 60)
    def rho_from_auc(auc, tau):
        """Invert the Gaussian-model AUROC of a readout at correlation rho for a concept with threshold tau."""
        a = np.array([gauss_auc(r, tau, 40000) for r in _RHO_GRID])
        return float(np.interp(np.clip(auc, 0.5, a.max()), a, _RHO_GRID))


    def se_floor(q, taus, nmc):
        """Clean-label least-squares floor of the readout error (labels = the concepts themselves) for noise level q."""
        K = len(taus)
        return float(se_run(lambda v: tuple(v), K, taus, [0.3] * K, q, 1, nmc)[1].mean())


    def calibrate_q(target_err, taus, nmc):
        lo, hi = 1e-4, 5.0
        for _ in range(18):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if se_floor(mid, taus, nmc) < target_err else (lo, mid)
        return float(np.sqrt(lo * hi))


    # ---- G2: the state evolution must reproduce a fresh-sample simulation of the same algorithm (theory check) ----
    if have("s6_se_check.json"):
        SE_CHECK = load_json(ckpt("s6_se_check.json"))
    else:
        SE_CHECK = {"cases": [], "max_gap": 0.0}
        _D1 = lambda v: (v[0] & v[1], v[1] & (1 - v[2]), int(v[0] + v[1] + v[2] >= 2)); _PAR = lambda v: (v[0] ^ v[1] ^ v[2],)
        for _nm, _fn in [("design1", _D1), ("parity", _PAR)]:
            for _taus in [(0.0, 0.0, 0.0), (0.5, -0.3, 0.8)]:
                for p0 in [0.25, 0.40]:
                    for soft in (True, False):
                        _rho0 = np.cos(np.pi * p0) * np.ones(3)
                        se = np.array([e.mean() for e in se_run(_fn, 3, _taus, _rho0, 58 / 3000, 6, CFG["se_mc"] * 2, soft=soft)])
                        fr = np.mean([[e.mean() for e in se_run(_fn, 3, _taus, _rho0, None, 6, None, seed=s, fresh_sim=(58, 3000), soft=soft)] for s in range(4)], 0)
                        _gap = float(np.max(np.abs(se - fr)))
                        SE_CHECK["cases"].append({"code": _nm, "taus": _taus, "p0": p0, "soft": soft, "se": se.tolist(), "sim": fr.tolist(), "gap": _gap})
                        SE_CHECK["max_gap"] = max(SE_CHECK["max_gap"], _gap)
                        print(f"G2 {_nm:8s} tau={_taus} p0={p0} {'soft' if soft else 'hard'}: SE {' '.join(f'{x:.3f}' for x in se)} | simulation {' '.join(f'{x:.3f}' for x in fr)} | gap {_gap:.3f}")
        save_json(ckpt("s6_se_check.json"), SE_CHECK); hf_push(RUN_NAME)
    print("G2 max |state evolution - simulation| =", round(SE_CHECK["max_gap"], 4))
    return (SE_CHECK, calibrate_q, norm_dist, rho_from_auc, se_run)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Decode-and-retrain (KDR) and the CUB-Logic experiment
    **KDR.** Round 0: zero-shot logits from the frozen encoder, z_j(x) = T·(⟨x, t_j⟩ − μ_j)/σ_j (t_j = centred CLIP text
    embedding of the concept's name; μ, σ over the unlabeled training pool). Each round: readout posteriors → decoding of every
    training image inside the fibre of its task label (soft = posterior marginals, i.e. one EM step; hard = MAP vector) →
    for each of 5 folds a logistic readout is fit on the decoded values of the OTHER folds and produces this fold's logits
    (cross-fitting). Evaluation logits = mean of the fold readouts. **Ledger rule:** KDR values are issued only for concepts
    the code identifies (it can force them both TRUE and FALSE: an exact, data-free fact); every other concept keeps its
    zero-shot value (provenance "prior"). Reason (lemma): any statistic computed from task labels — e.g. how many labels the
    predicted concepts reproduce — is invariant under the code's symmetry group, so no label-based rule can detect drift
    toward a symmetric shortcut; consistency-selected KDR is reported only as a measurement. No concept label is
    used anywhere; concept labels only evaluate.

    **CUB-Logic codes** over triples (a, b, c) of reliable, balanced, weakly correlated attributes (chosen on E1-train by a
    fixed rule): *design1* y = (a∧b, b∧¬c, maj); *parity* y = a⊕b⊕c (a pure-symmetry code: it forces nothing);
    *design12* = design1 plus (a∧c, c∧¬a, b∨c). Training: E1-train images; evaluation: E1-test (seen families) and **E3
    (unseen families)**. Baselines get the same features and the same labels: NeSy (DeepProbLog-style exact likelihood,
    random init), ensemble, k-NN on forced values, BEARS-style, LF-CBM-style, zero-shot CLIP; plus DeepProbLog initialised at
    the zero-shot readout (*nesy_zsinit*, the closest relative of KDR) and an instance-supervised oracle probe (upper bound).
    """)
    return



@app.cell
def _(CFG, CNAMES, DEV, E1_TE, E1_TR, E3, K_CON, RELIABLE, RUN_NAME, TXT_C, Xall, Y_INST, audit_reader, check_deadline, ckpt, concept_baselines, cons_mask, fit_logreg, forced_from_cons, gpu_free, have, hf_push, label_ids, load_json, make_block, nesy_train, norm_dist, np, rho_from_auc, safe_auc, save_json, torch):
    def zs_logits(X, T, ref):
        """Zero-shot logits; standardisation uses the unlabeled reference pool `ref` only."""
        with torch.no_grad():
            s_ref = ref @ T.T; mu, sd = s_ref.mean(0), s_ref.std(0) + 1e-6
            return (CFG["zs_temp"] * ((X @ T.T) - mu) / sd).cpu().numpy(), (CFG["zs_temp"] * T / sd[:, None], -CFG["zs_temp"] * mu / sd)


    def decode_blocks(P, specs, lab_list, soft=True):
        """Inside the fibre of each block label: posterior marginals (soft, = one EM step) or the MAP vector (hard).
        Returns targets (n,K) in [0,1] and the satisfiable mask (n,). Unsatisfiable labels keep the readout's own values."""
        D = P.copy() if soft else (P > 0.5).astype(float); sat = np.ones(len(P), bool)
        Pc = np.clip(P, 1e-6, 1 - 1e-6)
        for s_, lab in zip(specs, lab_list):
            ix, V, WL = s_["ix"], s_["V"], s_["WL"]
            L = np.log(Pc[:, ix]) @ V.T + np.log(1 - Pc[:, ix]) @ (1 - V).T
            ok = lab >= 0; C = np.zeros(L.shape, bool); C[ok] = WL[:, lab[ok]].T
            L = np.where(C, L, -np.inf); rows = np.where(ok)[0]
            if soft:
                Lr = L[rows] - L[rows].max(1, keepdims=True); Pw = np.exp(Lr); Pw /= Pw.sum(1, keepdims=True)
                D[np.ix_(rows, ix)] = Pw @ V
            else:
                D[np.ix_(rows, ix)] = V[L[rows].argmax(1)]
            sat &= ok
        return D, sat


    def consistency(logit, specs, lab_list):
        """Label-free model selection statistic: fraction of images whose predicted (hard) concepts reproduce their label."""
        Cp = (logit > 0).astype(np.int64); ok = np.ones(len(Cp), bool)
        for s_, lab in zip(specs, lab_list):
            w = (Cp[:, s_["ix"]] * (2 ** np.arange(len(s_["ix"]))[::-1])).sum(1)
            ok &= (lab >= 0) & s_["WL"][w, np.maximum(lab, 0)]
        return float(ok.mean())


    def kdr(Xtr, specs, lab_tr, Z0_tr, X_evals, rounds, folds=5, seed=0, soft=True):
        """Cross-fitted decode-and-retrain: the readout that produces fold f's logits is fit on all OTHER folds, so decoding of
        fold f never sees its own estimation noise (the condition under which the state evolution is exact).
        Returns eval logits per round, out-of-fold training logits per round (round 0 = zero-shot), consistency per round."""
        n = Xtr.shape[0]; fold = np.random.default_rng(seed).permutation(n) % folds
        logit_tr = Z0_tr.copy(); hist_ev, hist_tr = [], [logit_tr.copy()]; cons = [consistency(logit_tr, specs, lab_tr)]
        for r in range(rounds):
            D, sat = decode_blocks(1 / (1 + np.exp(-logit_tr)), specs, lab_tr, soft)
            new_tr = np.zeros_like(logit_tr); ev = [np.zeros((Xe.shape[0], D.shape[1])) for Xe in X_evals]
            for f in range(folds):
                trn = np.where((fold != f) & sat)[0]; tgt = np.where(fold == f)[0]
                probe = fit_logreg(Xtr[torch.from_numpy(trn).to(DEV)], torch.from_numpy(D[trn]).float().to(DEV))
                with torch.no_grad():
                    new_tr[tgt] = probe(Xtr[torch.from_numpy(tgt).to(DEV)]).cpu().numpy()
                    for k, Xe in enumerate(X_evals):
                        ev[k] += probe(Xe).cpu().numpy() / folds
                del probe
            logit_tr = new_tr; hist_tr.append(logit_tr.copy()); hist_ev.append(ev); cons.append(consistency(logit_tr, specs, lab_tr))
        return hist_ev, hist_tr, cons


    def kdr_select(Xtr, specs, lab_tr, Z0_tr, X_evals, Z0_evals, rounds, folds, seed):
        """Run soft and hard KDR; choose (mode, round) by out-of-fold label consistency, round 0 = zero-shot (no harm by
        construction of the choice set). Uses task labels only."""
        runs = {m: kdr(Xtr, specs, lab_tr, Z0_tr, X_evals, rounds, folds, seed, soft=(m == "soft")) for m in ("soft", "hard")}
        best = ("zeroshot", 0, runs["soft"][2][0])
        for m, (_, _, cons) in runs.items():
            for r in range(1, rounds + 1):
                if cons[r] > best[2] + 1e-9:
                    best = (m, r, cons[r])
        chosen = Z0_evals if best[0] == "zeroshot" else runs[best[0]][0][best[1] - 1]
        return chosen, runs, {"mode": best[0], "round": best[1], "consistency": best[2]}


    DESIGN1 = lambda v: (v[0] & v[1], v[1] & (1 - v[2]), int(v[0] + v[1] + v[2] >= 2))
    DESIGN12 = lambda v: DESIGN1(v) + (v[0] & v[2], v[2] & (1 - v[0]), v[1] | v[2])
    PARITY = lambda v: (v[0] ^ v[1] ^ v[2],)
    CODES = {"design1": DESIGN1, "parity": PARITY, "design12": DESIGN12}


    def code_facts(fn):
        """Exact (data-free): per concept, can some label force it TRUE / FALSE?"""
        s_ = make_block([0, 1, 2], lambda v: [tuple(fn(v))]); out = []
        for jj in range(3):
            t = f = False
            for o in range(s_["WL"].shape[1]):
                Vw = s_["V"][s_["WL"][:, o]]
                t |= bool(Vw[:, jj].min() == 1); f |= bool(Vw[:, jj].max() == 0)
            out.append({"force_true": t, "force_false": f})
        return s_, out


    CODE_SPECS = {k: code_facts(fn) for k, fn in CODES.items()}
    DESIGN2 = lambda v: (v[0] & v[2], v[2] & (1 - v[0]), v[1] | v[2])          # independent AUDIT code (never trained on)
    AUDIT_SPECS = {k: (code_facts(fn)[0], fn) for k, fn in {"design1": DESIGN1, "design2": DESIGN2, "design12": DESIGN12}.items()}
    print("audit codes forcing (exact):", {k: ["".join(["T" if x["force_true"] else "-", "F" if x["force_false"] else "-"]) for x in code_facts(fn)[1]] for k, (_, fn) in AUDIT_SPECS.items()})
    for _k, (_, _fx) in CODE_SPECS.items():
        print(f"code {_k:9s} forcing (exact): " + ", ".join(f"c{j}: {'T' if x['force_true'] else '-'}{'F' if x['force_false'] else '-'}" for j, x in enumerate(_fx)))


    def risk_at_cov(score, correct, cov, seed=0):
        if len(score) == 0 or cov <= 0:
            return float("nan")
        k = max(1, int(round(cov * len(score))))
        jit = np.random.default_rng(seed).random(len(score)) * 1e-9
        return float(1 - correct[np.argsort(-(score + jit))[:k]].mean())


    def pick_triples(n_trip):
        rel = RELIABLE[E1_TR].mean(0)
        prev = np.array([Y_INST[E1_TR, j][RELIABLE[E1_TR, j]].mean() if RELIABLE[E1_TR, j].any() else 0 for j in range(K_CON)])
        cand = [j for j in np.argsort(-rel) if 0.2 < prev[j] < 0.8 and rel[j] >= np.quantile(rel, 0.4)]
        used, trips = set(), []
        for j0 in cand:
            if j0 in used:
                continue
            t = [j0]
            for j in cand:
                if j in used or j in t:
                    continue
                m = RELIABLE[E1_TR][:, t + [j]].all(1)
                if m.sum() < 30:
                    continue
                Cm = Y_INST[E1_TR][m][:, t + [j]].astype(float)
                if np.all(np.abs(np.corrcoef(Cm.T)[-1, :-1]) < 0.3):
                    t.append(j)
                if len(t) == 3:
                    break
            if len(t) == 3:
                trips.append(t); used.update(t)
            if len(trips) == n_trip:
                break
        return trips


    TRIPLES = pick_triples(CFG["n_triples"])
    print("CUB-Logic triples:", [[CNAMES[j] for j in t] for t in TRIPLES])
    EVAL_SETS = {"E1te": E1_TE, "E3": E3}
    XN = torch.nn.functional.normalize(Xall, dim=1)


    def concept_metrics(logit, idx, j_list):
        """Per concept: accuracy and AUROC on reliable per-image labels; plus task accuracy vs concept accuracy later."""
        out = []
        for jj, j in enumerate(j_list):
            rel = RELIABLE[idx, j]; y = Y_INST[idx, j][rel]; s = logit[rel, jj]
            out.append({"acc": float(((s > 0).astype(int) == y).mean()) if rel.any() else float("nan"), "auc": safe_auc(y, s), "n": int(rel.sum())})
        return out


    def task_acc(logit, idx, j_list, fn):
        rel = RELIABLE[idx][:, j_list].all(1)
        if not rel.any():
            return float("nan")
        Ct = Y_INST[idx][rel][:, j_list].astype(int); Cp = (logit[rel] > 0).astype(int)
        return float(np.mean([fn(tuple(a)) == fn(tuple(b)) for a, b in zip(Ct, Cp)]))


    for _ti, trip in enumerate(TRIPLES):
        _tag = f"s7k_trip{_ti}.json"
        if have(_tag):
            continue
        check_deadline()
        m1 = RELIABLE[E1_TR][:, trip].all(1); tr_idx = E1_TR[m1]; C1 = Y_INST[tr_idx][:, trip].astype(int)
        Xtr = XN[torch.from_numpy(tr_idx).to(DEV)]; T = TXT_C[trip]
        X_ev = [XN[torch.from_numpy(EVAL_SETS[k]).to(DEV)] for k in EVAL_SETS]
        Z0_tr, (_W0, _b0) = zs_logits(Xtr, T, Xtr)
        Z0_ev = [zs_logits(Xe, T, Xtr)[0] for Xe in X_ev]
        prev = C1.mean(0); _taus = norm_dist.ppf(1 - np.clip(prev, 0.02, 0.98))
        out = {"triple": [CNAMES[j] for j in trip], "n_train": int(len(tr_idx)), "prevalence": prev.tolist(), "codes": {}}
        # oracle (instance-supervised) probe: an upper bound, NOT a competitor
        _orc = fit_logreg(Xtr, torch.from_numpy(C1).float().to(DEV))
        with torch.no_grad():
            orc_ev = [_orc(Xe).cpu().numpy() for Xe in X_ev]
        zs_auc_tr = [safe_auc(C1[:, jj], Z0_tr[:, jj]) for jj in range(3)]
        out["zs_auc_train"] = zs_auc_tr
        for _cname, _fn in CODES.items():
            spec = CODE_SPECS[_cname][0]
            _lab = label_ids(spec, [tuple(_fn(tuple(c))) for c in C1]); _cons = cons_mask(spec, _lab); _Ftr = forced_from_cons(_cons, 3)
            _res = {"forced_frac": (_Ftr >= 0).mean(0).tolist(), "methods": {}, "kdr_rounds": {}}
            _chosen, _runs, _sel = kdr_select(Xtr, [spec], [_lab], Z0_tr, X_ev, Z0_ev, CFG["kdr_rounds"], CFG["kdr_folds"], CFG["seed"] + _ti)
            _hist_ev, hist_tr, cons_soft = _runs["soft"]
            _res["selection"] = _sel; _res["consistency"] = {m: _runs[m][2] for m in _runs}
            _res["G1_violations"] = 0
            for _m in _runs:                                                  # G1: decoded targets lie inside the fibre
                Dh, sat_h = decode_blocks(1 / (1 + np.exp(-_runs[_m][1][-1])), [spec], [_lab], soft=False)
                _res["G1_violations"] += int(sum(tuple(_fn(tuple(d))) != tuple(_fn(tuple(c))) for d, c, s_ in zip(Dh.astype(int), C1, sat_h) if s_))
            _ident = np.array([x["force_true"] and x["force_false"] for x in CODE_SPECS[_cname][1]])     # exact, data-free (F2)
            ledger = [np.where(_ident[None, :], _hist_ev[-1][k], Z0_ev[k]) for k in range(len(X_ev))]
            _res["identified"] = _ident.tolist()
            _meth = {"zeroshot": Z0_ev, "kdr": ledger, "kdr_soft_all_rounds": _hist_ev[-1], "kdr_hard_all_rounds": _runs["hard"][0][-1],
                    "kdr_consistency_selected": _chosen, "oracle": orc_ev}
            Xcat = torch.cat(X_ev, 0); cuts = np.cumsum([0] + [x.shape[0] for x in X_ev])
            _base = concept_baselines(Xtr, [spec], [_cons], 3, Xcat, _Ftr, T_dirs=T, cal=None, n_ens=CFG["n_ens"], knn_k=CFG["knn_k"])
            for _bn, (_sc, _pr) in _base.items():
                if _bn == "zeroshot_clip":
                    continue
                _lg = np.where(_pr, _sc + 1e-3, -_sc - 1e-3)                    # signed confidence as a logit surrogate
                _meth[_bn] = [_lg[cuts[k]:cuts[k + 1]] for k in range(len(X_ev))]
            _Wz, _bz = nesy_train(Xtr, [spec], [_cons], 3, seed=7, epochs=CFG["nesy_epochs"], W0=_W0, b0=_b0)
            with torch.no_grad():
                _meth["nesy_zsinit"] = [(Xe @ _Wz.T + _bz).cpu().numpy() for Xe in X_ev]
            for _mn, lgs in _meth.items():
                _res["methods"][_mn] = {k: {"concepts": concept_metrics(lgs[i], EVAL_SETS[k], trip), "task_acc": task_acc(lgs[i], EVAL_SETS[k], trip, _fn)}
                                      for i, k in enumerate(EVAL_SETS)}
                _res["methods"][_mn]["E3_logits"] = np.round(lgs[1], 4).tolist()
            for _r, ev in enumerate(_hist_ev):
                _res["kdr_rounds"][_r + 1] = {k: [c["acc"] for c in concept_metrics(ev[i], EVAL_SETS[k], trip)] for i, k in enumerate(EVAL_SETS)}
            _res["kdr_rounds"][0] = {k: [c["acc"] for c in concept_metrics(Z0_ev[i], EVAL_SETS[k], trip)] for i, k in enumerate(EVAL_SETS)}
            # ---- v6 audit: every reader audited LABEL-FREE on seen (E1te) and unseen (E3) families, with the training code
            #      itself and with the audit codes; per-image truth is passed only to score the audit afterwards ----
            _res["audit"] = {}
            for _i, _k in enumerate(EVAL_SETS):
                idx = EVAL_SETS[_k]; r3 = RELIABLE[idx][:, trip].all(1); Ct = Y_INST[idx][r3][:, trip].astype(int)
                _res["audit"][_k] = {}
                for _acn, (aspec, afn) in AUDIT_SPECS.items():
                    alab = label_ids(aspec, [tuple(afn(tuple(int(x) for x in c))) for c in Ct]); aF = forced_from_cons(cons_mask(aspec, alab), 3)
                    _res["audit"][_k][_acn] = {mn: audit_reader((np.asarray(lgs[_i])[r3] > 0).astype(int), aF, alab, CFG["audit_alpha"], truth=Ct)
                                            for mn, lgs in _meth.items()}
            _za = _res["audit"]["E3"]["design2"]["zeroshot"]
            print(f"   audit (E3, code design2) zero-shot: A1 " + " ".join(f"{a.get('A1', float('nan')):.3f}/true {a['true_err']:.3f}" for a in _za), flush=True)
            # train-pool trajectory (in-distribution; what the state evolution models)
            _res["kdr_train_err"] = {m: [float(((h > 0).astype(int) != C1).mean()) for h in _runs[m][1]] for m in _runs}
            # state-evolution prediction from (code, prevalence, zero-shot AUROC on the pool, q calibrated to the oracle floor)
            _rho0 = [rho_from_auc(a if a == a else 0.5, t) for a, t in zip(zs_auc_tr, _taus)]
            _res["se_inputs"] = {"rho0": _rho0, "taus": _taus.tolist()}
            out["codes"][_cname] = _res
            e3 = _res["methods"]
            print(f"trip{_ti} {_cname:9s} E3 concept acc | " + " ".join(f"{mn} {np.nanmean([c['acc'] for c in e3[mn]['E3']['concepts']]):.3f}" for mn in e3) +
                  f" | selected {_sel['mode']}@{_sel['round']} | soft out-of-fold err {' '.join(f'{x:.3f}' for x in _res['kdr_train_err']['soft'])} | G1 violations {_res['G1_violations']}", flush=True)
            gpu_free()
        # the single noise parameter q of the state evolution is calibrated on the ORACLE's out-of-sample error (E1-test)
        out["oracle_E1te_err"] = float(np.nanmean([1 - c["acc"] for c in out["codes"]["design1"]["methods"]["oracle"]["E1te"]["concepts"]]))
        save_json(ckpt(_tag), out); hf_push(RUN_NAME, force=True)
        del _orc; gpu_free()
    S7 = [load_json(ckpt(f"s7k_trip{ti}.json")) for ti in range(len(TRIPLES))]
    return (AUDIT_SPECS, CODES, CODE_SPECS, S7, kdr_select, risk_at_cov, zs_logits)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7b. P2 — does the state evolution, fed only the code and the zero-shot channel quality, predict KDR's errors?
    Inputs per (triple, code): prevalences τ_j, zero-shot readout correlation ρ_j (from its AUROC on the training pool via the
    Gaussian-model inversion) and ONE noise parameter q calibrated so that the clean-label floor equals the oracle probe's
    error on seen families. Output: predicted per-concept error after the same number of rounds. Compared with KDR's measured
    error on seen-family test images (E1te; the theory models no distribution shift) and on unseen families (E3, reported).
    Concept labels enter only as evaluation (AUROC of the zero-shot readout, oracle floor): this is a test of the THEORY.
    """)
    return



@app.cell
def _(CFG, CODES, RUN_NAME, S7, calibrate_q, ckpt, have, hf_push, load_json, np, save_json, se_run, spearmanr):
    if have("s7b_se_pred.json"):
        SEP = load_json(ckpt("s7b_se_pred.json"))
    else:
        SEP = {"cells": []}
        for _ti, _tr in enumerate(S7):
            _taus = np.array(_tr["codes"]["design1"]["se_inputs"]["taus"]); _rho0 = _tr["codes"]["design1"]["se_inputs"]["rho0"]
            q = calibrate_q(_tr["oracle_E1te_err"], _taus, CFG["se_mc"] // 2)
            for _cname, _fn in CODES.items():
                pred = se_run(_fn, 3, _taus, _rho0, q, CFG["kdr_rounds"], CFG["se_mc"])
                meas_e1 = [1 - a for a in _tr["codes"][_cname]["kdr_rounds"][str(CFG["kdr_rounds"])]["E1te"]]
                meas_e3 = [1 - a for a in _tr["codes"][_cname]["kdr_rounds"][str(CFG["kdr_rounds"])]["E3"]]
                for _jj in range(3):
                    SEP["cells"].append({"triple": _ti, "code": _cname, "concept": _tr["triple"][_jj], "q": q, "rho0": _rho0[_jj],
                                         "pred_final": float(pred[-1][_jj]), "pred_round1": float(pred[1][_jj]),
                                         "meas_E1te": meas_e1[_jj], "meas_E3": meas_e3[_jj]})
        save_json(ckpt("s7b_se_pred.json"), SEP); hf_push(RUN_NAME)
    SEP_ROWS = [c for c in SEP["cells"] if c["meas_E1te"] == c["meas_E1te"]]
    P2_SP = spearmanr([c["pred_final"] for c in SEP_ROWS], [c["meas_E1te"] for c in SEP_ROWS]).correlation if len(SEP_ROWS) >= 5 else float("nan")
    P2_SP3 = spearmanr([c["pred_final"] for c in SEP_ROWS], [c["meas_E3"] for c in SEP_ROWS]).correlation if len(SEP_ROWS) >= 5 else float("nan")
    P2_MAE = float(np.mean([abs(c["pred_final"] - c["meas_E1te"]) for c in SEP_ROWS])) if SEP_ROWS else float("nan")
    print(f"P2: Spearman(predicted, measured KDR error) seen families {P2_SP:.3f} | unseen families {P2_SP3:.3f} | mean |pred - meas| {P2_MAE:.3f} over {len(SEP_ROWS)} cells")
    for _cname in CODES:
        cc = [c for c in SEP_ROWS if c["code"] == _cname]
        if cc:
            print(f"   {_cname:9s} predicted {np.mean([c['pred_final'] for c in cc]):.3f} vs measured {np.mean([c['meas_E1te'] for c in cc]):.3f} (E1te), {np.mean([c['meas_E3'] for c in cc]):.3f} (E3)")
    return (P2_MAE, P2_SP, P2_SP3, SEP, SEP_ROWS)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7c. The audit on real images — G6, P4, P6, P7
    Every reader of §7 (zero-shot, KDR variants, label-only NeSy baselines, NeSy-zs-init, oracle) was audited from its hard
    outputs and the task labels of an audit code only. Here the audit is scored against the per-image reliable labels.
    G6 is an identity (it fails only if the knowledge is wrong or the code has a bug). P4 tests A1 on a code-independent
    reader; P6/P7 use the independent audit code design2 (never trained on), the regime where trained readers' drift is visible.
    """)
    return



@app.cell
def _(AUDIT_SPECS, CODES, RUN_NAME, S7, ckpt, hf_push, np, save_json, spearmanr):
    # ---- 7c. Audit analysis on real CUB-Logic images: G6, P4, P6, P7 (truth = per-image reliable labels, scoring only) ----
    AUD_ROWS = []
    for _ti, _tr in enumerate(S7):
        for _cname, _res in _tr["codes"].items():
            for split, by_ac in _res["audit"].items():
                for _acn, by_m in by_ac.items():
                    for _mn, cl in by_m.items():
                        for _jj, _a in enumerate(cl):
                            AUD_ROWS.append({"triple": _ti, "code": _cname, "split": split, "audit_code": _acn, "method": _mn, "concept": _jj, **_a})
    G6_VIOL = sum(1 for r in AUD_ROWS if not (r["A0_sample"][0] - 1e-12 <= r["true_err"] <= r["A0_sample"][1] + 1e-12))
    print(f"G6 A0 sandwich identity (knowledge exact on CUB-Logic): {G6_VIOL} violations over {len(AUD_ROWS)} cells")


    def _ok2(r):
        return r["type"] == "two" and r.get("A1") == r.get("A1") and not r.get("A1_degenerate", False)


    P4_ROWS = [r for r in AUD_ROWS if r["method"] == "zeroshot" and r["code"] == "design1" and r["audit_code"] in ("design1", "design12") and _ok2(r)]
    P4_FRAC = float(np.mean([abs(r["A1"] - r["true_err"]) <= 0.05 for r in P4_ROWS])) if P4_ROWS else float("nan")
    P4_MAE = float(np.mean([abs(r["A1"] - r["true_err"]) for r in P4_ROWS])) if P4_ROWS else float("nan")
    P4_LOCREJ = float(np.mean([r["loc_p"] < 0.05 for r in P4_ROWS if r["loc_p"] == r["loc_p"]])) if P4_ROWS else float("nan")
    P4_COV = float(np.mean([r["A1_ci"][0] <= r["true_err"] <= r["A1_ci"][1] for r in P4_ROWS])) if P4_ROWS else float("nan")
    _zs_ok = [r for r in P4_ROWS if not (r["loc_p"] < 0.05)]
    print(f"P4 zero-shot reader: |A1 - true| <= .05 in {P4_FRAC:.3f} of {len(P4_ROWS)} cells (MAE {P4_MAE:.4f}, A1-CI covers truth {P4_COV:.3f}); "
          f"locality rejected in {P4_LOCREJ:.3f}; among non-rejected cells: {np.mean([abs(r['A1'] - r['true_err']) <= .05 for r in _zs_ok]) if _zs_ok else float('nan'):.3f}")
    P6_ROWS = [r for r in AUD_ROWS if r["audit_code"] == "design2" and r["method"] != "oracle" and _ok2(r)]
    P6_SP = spearmanr([r["A1"] for r in P6_ROWS], [r["true_err"] for r in P6_ROWS]).correlation if len(P6_ROWS) >= 5 else float("nan")
    P6_MAE = float(np.mean([abs(r["A1"] - r["true_err"]) for r in P6_ROWS])) if P6_ROWS else float("nan")
    print(f"P6 all readers, independent audit code design2: Spearman(A1, true error) = {P6_SP:.3f} over {len(P6_ROWS)} cells, MAE {P6_MAE:.4f}")
    for _cname in CODES:
        _cc = [r for r in P6_ROWS if r["code"] == _cname and r["split"] == "E3"]
        if _cc:
            print(f"   trained on {_cname:9s} E3: mean A1 {np.mean([r['A1'] for r in _cc]):.3f} vs true {np.mean([r['true_err'] for r in _cc]):.3f}")

    # P7: label-free model selection on E3 with the independent audit code
    P7_CAND = ["zeroshot", "kdr_soft_all_rounds", "kdr_hard_all_rounds", "nesy_conf", "ensemble", "bears_style", "knn", "lfcbm_style", "nesy_zsinit"]
    _sel, _zsv, _med, _best, _n7 = [], [], [], [], 0
    for _ti, _tr in enumerate(S7):
        for _cname, _res in _tr["codes"].items():
            by_m = _res["audit"]["E3"]["design2"]
            for _jj in range(3):
                cand = [(m, by_m[m][_jj]) for m in P7_CAND if m in by_m]
                scored = [(m, a) for m, a in cand if _ok2(a)]
                if len(scored) < 2:
                    continue
                m_sel = min(scored, key=lambda x: x[1]["A1"])[0]; _te = {m: a["true_err"] for m, a in cand}
                _sel.append(_te[m_sel]); _zsv.append(_te["zeroshot"]); _med.append(float(np.median(list(_te.values())))); _best.append(min(_te.values())); _n7 += 1
    P7 = {"n_cells": _n7, "selected_err": float(np.mean(_sel)) if _sel else float("nan"), "zeroshot_err": float(np.mean(_zsv)) if _zsv else float("nan"),
          "median_err": float(np.mean(_med)) if _med else float("nan"), "best_err": float(np.mean(_best)) if _best else float("nan")}
    P7_OK = bool(_sel) and P7["selected_err"] <= P7["zeroshot_err"] and P7["selected_err"] <= P7["median_err"] and P7["selected_err"] <= P7["best_err"] + 0.03
    print("P7 label-free model selection (E3, audit code design2): pooled true error", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in P7.items()})

    # measured: A0 width vs forced mass, locality rejection by reader
    A0_WIDTH = {ac: float(np.mean([r["A0_sample"][1] - r["A0_sample"][0] for r in AUD_ROWS if r["audit_code"] == ac and r["method"] == "zeroshot"]))
                for ac in AUDIT_SPECS}
    LOC_REJ = {m: float(np.mean([r["loc_p"] < 0.05 for r in AUD_ROWS if r["method"] == m and r["loc_p"] == r["loc_p"]] or [float("nan")]))
               for m in sorted({r["method"] for r in AUD_ROWS})}
    print("A0 bracket width (= unforced mass) by audit code:", {k: round(v, 3) for k, v in A0_WIDTH.items()})
    print("locality rejected (p < .05) by reader:", {k: round(v, 3) for k, v in LOC_REJ.items()})
    save_json(ckpt("s7c_audit.json"), {"G6_violations": G6_VIOL, "P4": {"frac": P4_FRAC, "mae": P4_MAE, "ci_cov": P4_COV, "loc_rej": P4_LOCREJ, "n": len(P4_ROWS)},
                                       "P6": {"spearman": P6_SP, "mae": P6_MAE, "n": len(P6_ROWS)}, "P7": P7, "A0_width": A0_WIDTH, "loc_rej": LOC_REJ})
    hf_push(RUN_NAME)
    return (A0_WIDTH, G6_VIOL, LOC_REJ, P4_COV, P4_FRAC, P4_LOCREJ, P4_MAE, P4_ROWS, P6_MAE, P6_ROWS, P6_SP, P7, P7_OK)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. MNIST with a foundation-model channel: sums (an identifying code) and sum parity (a pure-symmetry code)
    Real MNIST digits, encoded by the same frozen CLIP ViT-L/14; zero-shot digit readout from text prompts. Training sees
    only pairs of images and a label: **MNIST-Add** y = a + b; **MNIST-SumParity** y = (a + b) mod 2 (one bit per pair: the
    code forces no digit). Digit labels are used only for evaluation and for the oracle. Methods: zero-shot CLIP, NeSy
    (exact likelihood, random init), NeSy initialised at the zero-shot readout, KDR (cross-fitted decode-and-retrain), oracle.
    Ledger rule: KDR is issued only if no digit transposition preserves every label (exact check; MNIST-Add: none, so KDR; SumParity: many, so the zero-shot readout is kept). P3 is the MNIST-Add row.
    """)
    return



@app.cell
def _(CFG, CLIP_MEAN, CLIP_STD, DEV, FEATS, RUN_NAME, SMOKE, atomic_write, check_deadline, ckpt, clip_image_embed, clip_text_embed, ds_audit, ds_pairs, fit_softmax, fit_softmax_soft, gpu_free, have, hf_push, itertools, load_clip, load_json, np, os, save_json, torch):
    MN_TEMPLATES = ['a photo of the number "{d}".', "a handwritten digit {d}.", "the number {w} written by hand."]
    MN_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


    _SEG = {0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg", 5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg"}
    _SEG_BOX = {"a": (4, 6, 4, 20), "b": (4, 14, 18, 20), "c": (14, 24, 18, 20), "d": (22, 24, 6, 18), "e": (14, 24, 4, 6), "f": (4, 14, 4, 6), "g": (13, 15, 6, 18)}


    def synthetic_mnist(n, seed):
        """SMOKE ONLY: 7-segment digits with random shift, stroke dropout and pixel noise (28x28, MNIST format)."""
        r = np.random.default_rng(seed); X = np.zeros((n, 28, 28), np.float32); y = r.integers(0, 10, n)
        for i in range(n):
            dy, dx = r.integers(-2, 3, 2)
            for sgm in _SEG[int(y[i])]:
                if r.random() < 0.1:
                    continue
                y0, y1, x0, x1 = _SEG_BOX[sgm]; X[i, max(0, y0 + dy):y1 + dy, max(0, x0 + dx):x1 + dx] = 220
            X[i] += r.normal(0, 40, (28, 28))
        return np.clip(X, 0, 255).astype(np.uint8), y


    MNIST_MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/", "https://storage.googleapis.com/cvdf-datasets/mnist/"]
    MNIST_FILES = {"train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873", "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
                   "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3", "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c"}


    def _mnist_file(root, fn):
        """Download one MNIST IDX file (md5-checked, two mirrors, retries) and return its decompressed bytes."""
        import gzip, hashlib, urllib.request
        path = os.path.join(root, fn)
        ok = os.path.exists(path) and hashlib.md5(open(path, "rb").read()).hexdigest() == MNIST_FILES[fn]
        last = None
        for url in [m + fn for m in MNIST_MIRRORS] * 2:
            if ok:
                break
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if hashlib.md5(data).hexdigest() != MNIST_FILES[fn]:
                    raise IOError(f"incomplete or wrong file ({len(data)} bytes)")
                with open(path + ".part", "wb") as f:
                    f.write(data)
                os.replace(path + ".part", path); ok = True
            except Exception as e:
                last = e; print("  MNIST", url, "failed:", repr(e)[:160])
        if not ok:
            raise RuntimeError(f"Could not download MNIST {fn}; put it in {root}. Last error: {last}")
        with gzip.open(path, "rb") as f:
            return f.read()


    def get_mnist():
        if SMOKE:
            return synthetic_mnist(CFG["mnist_train"], 1), synthetic_mnist(CFG["mnist_test"], 2)
        root = os.path.join(CFG["work_dir"], "data", "mnist"); os.makedirs(root, exist_ok=True)
        def imgs(fn):
            return np.frombuffer(_mnist_file(root, fn), np.uint8, offset=16).reshape(-1, 28, 28).copy()
        def labs(fn):
            return np.frombuffer(_mnist_file(root, fn), np.uint8, offset=8).astype(np.int64)
        Xtr, ytr = imgs("train-images-idx3-ubyte.gz"), labs("train-labels-idx1-ubyte.gz")
        Xte, yte = imgs("t10k-images-idx3-ubyte.gz"), labs("t10k-labels-idx1-ubyte.gz")
        assert Xtr.shape == (60000, 28, 28) and Xte.shape == (10000, 28, 28) and len(ytr) == 60000 and len(yte) == 10000
        r = np.random.default_rng(CFG["seed"] + 3)
        a = r.choice(len(Xtr), CFG["mnist_train"], replace=False); b = r.choice(len(Xte), min(len(Xte), CFG["mnist_test"]), replace=False)
        return (Xtr[a], ytr[a]), (Xte[b], yte[b])


    def encode_mnist(imgs, tag):
        if have(f"mnist_{tag}.npy"):
            return np.load(ckpt(f"mnist_{tag}.npy"))
        if SMOKE:
            P = np.random.default_rng(5).normal(size=(28 * 28, FEATS.shape[1])).astype(np.float32) / 30
            f = (imgs.reshape(len(imgs), -1).astype(np.float32) / 255 - 0.1) @ P
        else:
            model, _ = load_clip(CFG["encoder"], text=False)
            mu, sd = torch.tensor(CLIP_MEAN, device=DEV).view(1, 3, 1, 1), torch.tensor(CLIP_STD, device=DEV).view(1, 3, 1, 1); res = []
            for b0 in range(0, len(imgs), 256):
                x = torch.from_numpy(imgs[b0:b0 + 256]).float().to(DEV)[:, None].repeat(1, 3, 1, 1) / 255
                x = torch.nn.functional.interpolate(x, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
                with torch.inference_mode():
                    res.append(clip_image_embed(model, (x - mu) / sd).cpu().numpy())
            f = np.concatenate(res); del model; gpu_free()
        f = (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float32)
        atomic_write(ckpt(f"mnist_{tag}.npy"), lambda t: np.save(t, f)); hf_push(RUN_NAME); return f


    def mnist_text(F_tr, y_tr):
        if have("mnist_txt.npy"):
            return np.load(ckpt("mnist_txt.npy"))
        if SMOKE:                                                    # SMOKE-ONLY planted stand-in for text, deliberately noisy
            r = np.random.default_rng(9); T = np.stack([F_tr[y_tr == k].mean(0) for k in range(10)])
            T = T - T.mean(0); T = T / np.linalg.norm(T, axis=1, keepdims=True) + r.normal(0, 0.12, T.shape)
        else:
            model, tok = load_clip(CFG["encoder"])
            T = 0
            with torch.inference_mode():
                for tm in MN_TEMPLATES:
                    T = T + torch.nn.functional.normalize(clip_text_embed(model, tok, [tm.format(d=k, w=MN_WORDS[k]) for k in range(10)]), dim=1).cpu().numpy()
            del model; gpu_free()
        T = (T / np.linalg.norm(T, axis=1, keepdims=True)).astype(np.float32); np.save(ckpt("mnist_txt.npy"), T); return T


    def pair_lab(a, b, task):
        return a + b if task == "add" else (a + b) % 2


    def pair_mask(task):
        """M[y, a, b] = True iff digits (a, b) are consistent with label y."""
        ny = 19 if task == "add" else 2; M = np.zeros((ny, 10, 10), bool)
        for a in range(10):
            for b in range(10):
                M[pair_lab(a, b, task), a, b] = True
        return M


    def decode_pairs(L1, L2, y, M):
        """MAP (a, b) inside the fibre of y from per-image log-posteriors (n,10)."""
        J = L1[:, :, None] + L2[:, None, :]; J = np.where(M[y], J, -np.inf)
        k = J.reshape(len(J), -1).argmax(1); return k // 10, k % 10


    def logsoftmax(L):
        L = L - L.max(1, keepdims=True); return L - np.log(np.exp(L).sum(1, keepdims=True))


    def kdr_mnist(X, pairs, y, Z0, task, rounds, X_te, seed=0, soft=True, folds=5):
        """Cross-fitted decode-and-retrain over digit pairs: the readout producing fold f's logits is fit on the other folds."""
        M = pair_mask(task); pf = np.random.default_rng(seed).permutation(len(pairs)) % folds
        img_fold = np.zeros(len(X), np.int64); img_fold[pairs[:, 0]] = pf; img_fold[pairs[:, 1]] = pf
        logit = Z0.copy(); hist, cons = [], []
        def consist(lg):
            p = lg.argmax(1); return float(np.mean(pair_lab(p[pairs[:, 0]], p[pairs[:, 1]], task) == y))
        cons.append(consist(logit))
        for r in range(rounds):
            lp = logsoftmax(logit); J = lp[pairs[:, 0]][:, :, None] + lp[pairs[:, 1]][:, None, :]; J = np.where(M[y], J, -np.inf)
            Q = np.zeros((len(X), 10))
            if soft:
                J = J - J.reshape(len(J), -1).max(1)[:, None, None]; Pj = np.exp(J); Pj /= Pj.sum((1, 2), keepdims=True)
                Q[pairs[:, 0]] = Pj.sum(2); Q[pairs[:, 1]] = Pj.sum(1)
            else:
                k = J.reshape(len(J), -1).argmax(1); Q[pairs[:, 0], k // 10] = 1; Q[pairs[:, 1], k % 10] = 1
            new = np.zeros_like(logit); te = 0
            for f in range(folds):
                ix = np.where(img_fold != f)[0]; ox = np.where(img_fold == f)[0]
                pr = fit_softmax_soft(X[torch.from_numpy(ix).to(DEV)], torch.from_numpy(Q[ix]).float().to(DEV))
                with torch.no_grad():
                    new[ox] = pr(X[torch.from_numpy(ox).to(DEV)]).cpu().numpy(); te = te + pr(X_te).cpu().numpy() / folds
                del pr
            logit = new; hist.append(te); cons.append(consist(logit))
        return hist, cons


    def nesy_mnist(X, pairs, y, task, epochs, W0=None, b0=None, seed=0):
        M = torch.from_numpy(pair_mask(task)).to(DEV); g = torch.Generator().manual_seed(seed)
        W = ((torch.randn(10, X.shape[1], generator=g) * 0.01).to(DEV) if W0 is None else W0.clone()).requires_grad_(True)
        b = (torch.zeros(10, device=DEV) if b0 is None else b0.clone()).requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=0.02); p1, p2 = torch.from_numpy(pairs[:, 0]).to(DEV), torch.from_numpy(pairs[:, 1]).to(DEV)
        yt = torch.from_numpy(y).to(DEV)
        for ep in range(epochs):
            lp = torch.log_softmax(X @ W.T + b, 1); J = lp[p1][:, :, None] + lp[p2][:, None, :]
            loss = -torch.logsumexp(J.masked_fill(~M[yt], -1e9).reshape(len(yt), -1), 1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return W.detach(), b.detach()


    def code_symmetries(task):
        """Exact: digit transpositions (a b) that preserve the label of every pair. Any one of them makes the code non-identifying."""
        out = []
        for a, b in itertools.combinations(range(10), 2):
            sg = list(range(10)); sg[a], sg[b] = b, a
            if all(pair_lab(sg[u], sg[v], task) == pair_lab(u, v, task) for u in range(10) for v in range(10)):
                out.append((a, b))
        return out


    def perm_acc(pred, y):
        """Digit accuracy after the best relabelling of predicted digits (Hungarian): measures grounding up to symmetry."""
        from scipy.optimize import linear_sum_assignment
        Cm = np.zeros((10, 10))
        for p, t in zip(pred, y):
            Cm[p, t] += 1
        r, c = linear_sum_assignment(-Cm); return float(Cm[r, c].sum() / len(y))


    if have("s8_mnist.json"):
        SM = load_json(ckpt("s8_mnist.json"))
    else:
        (Itr, ytr), (Ite, yte) = get_mnist()
        _Ftr, Fte = encode_mnist(Itr, "train"), encode_mnist(Ite, "test")
        TM = mnist_text(_Ftr, ytr)
        Xm, Xm_te = torch.from_numpy(_Ftr).to(DEV), torch.from_numpy(Fte).to(DEV)
        Tm = torch.from_numpy(TM).to(DEV); scale = 100.0
        with torch.no_grad():
            _Z0, Z0_te = (scale * Xm @ Tm.T).cpu().numpy(), (scale * Xm_te @ Tm.T).cpu().numpy()
        pairs = np.random.default_rng(CFG["seed"] + 4).permutation(len(ytr))[: 2 * (len(ytr) // 2)].reshape(-1, 2)
        SM = {"n_train_images": int(len(ytr)), "n_test_images": int(len(yte)), "zeroshot_acc": float((Z0_te.argmax(1) == yte).mean()), "tasks": {}}
        _orc = fit_softmax(Xm, torch.from_numpy(ytr).to(DEV), C=10)
        with torch.no_grad():
            SM["oracle_acc"] = float((_orc(Xm_te).argmax(1).cpu().numpy() == yte).mean())
        del _orc
        for task in ["add", "parity"]:
            check_deadline()
            y = pair_lab(ytr[pairs[:, 0]], ytr[pairs[:, 1]], task)
            _runs = {m: kdr_mnist(Xm, pairs, y, _Z0, task, CFG["kdr_rounds"], Xm_te, seed=CFG["seed"], soft=(m == "soft")) for m in ("soft", "hard")}
            best = ("zeroshot", 0, _runs["soft"][1][0])
            for _m, (_, _cons) in _runs.items():
                for _r in range(1, CFG["kdr_rounds"] + 1):
                    if _cons[_r] > best[2] + 1e-9:
                        best = (_m, _r, _cons[_r])
            hist = _runs["soft"][0]
            pk_sel = Z0_te.argmax(1) if best[0] == "zeroshot" else _runs[best[0]][0][best[1] - 1].argmax(1)
            Wr, br = nesy_mnist(Xm, pairs, y, task, CFG["nesy_epochs"] * 2, seed=1)
            _Wz, _bz = nesy_mnist(Xm, pairs, y, task, CFG["nesy_epochs"] * 2, W0=scale * Tm, b0=torch.zeros(10, device=DEV), seed=1)
            with torch.no_grad():
                pr_r = (Xm_te @ Wr.T + br).argmax(1).cpu().numpy(); pr_z = (Xm_te @ _Wz.T + _bz).argmax(1).cpu().numpy()
            sym = code_symmetries(task); SM.setdefault("symmetries", {})[task] = len(sym)
            pk = _runs["soft"][0][-1].argmax(1) if not sym else Z0_te.argmax(1)        # ledger rule: KDR only on an identifying code
            tp = np.random.default_rng(11).permutation(len(yte))[: 2 * (len(yte) // 2)].reshape(-1, 2)
            def sum_acc(p):
                return float(np.mean(pair_lab(p[tp[:, 0]], p[tp[:, 1]], task) == pair_lab(yte[tp[:, 0]], yte[tp[:, 1]], task)))
            _R = {}; Mtask = pair_mask(task); ytp = pair_lab(yte[tp[:, 0]], yte[tp[:, 1]], task)
            for _nm, p in [("zeroshot", Z0_te.argmax(1)), ("nesy_random", pr_r), ("nesy_zsinit", pr_z), ("kdr", pk),
                          ("kdr_soft_all_rounds", _runs["soft"][0][-1].argmax(1)), ("kdr_consistency_selected", pk_sel)]:
                _R[_nm] = {"digit_acc": float((p == yte).mean()), "digit_acc_up_to_relabel": perm_acc(p, yte), "task_acc": sum_acc(p)}
                # v6 A2 audit: digit accuracy from (reader outputs, pair label) only; truth used only to score it
                au = ds_audit(p[tp[:, 0]], p[tp[:, 1]], ytp, Mtask, n_boot=CFG["ds_boot"], seed=CFG["seed"]); au.pop("confusion")
                au["true_acc_pairs"] = float(np.mean(np.concatenate([p[tp[:, 0]] == yte[tp[:, 0]], p[tp[:, 1]] == yte[tp[:, 1]]])))
                _R[_nm]["audit"] = au
            # P8: label-free choice of the KDR round (soft and hard runs) by A2-estimated accuracy on the test pairs
            _est = []
            for m_ in ("soft", "hard"):
                for r_ in range(CFG["kdr_rounds"]):
                    p = _runs[m_][0][r_].argmax(1); Cq, pq = ds_pairs(p[tp[:, 0]], p[tp[:, 1]], ytp, Mtask, iters=150)
                    _est.append((float(pq @ np.diag(Cq)), float((p == yte).mean()), m_, r_ + 1))
            _b = max(_est, key=lambda x: x[0])
            SM.setdefault("audit_round_selection", {})[task] = {"chosen": [_b[2], _b[3]], "chosen_est_acc": _b[0], "chosen_true_acc": _b[1],
                                                               "best_true_acc": max(e[1] for e in _est), "all": _est}
            print(f"MNIST-{task} A2 audit (est / true digit acc): " + " ".join(f"{k} {v['audit']['est_acc']:.3f}/{v['audit']['true_acc_pairs']:.3f}" for k, v in _R.items()) +
                  f" | audit-chosen KDR round {_b[2]}@{_b[3]} true {_b[1]:.3f} (best round {max(e[1] for e in _est):.3f})", flush=True)
            _R["kdr_rounds_acc"] = [float((h.argmax(1) == yte).mean()) for h in hist]
            _R["kdr_hard_rounds_acc"] = [float((h.argmax(1) == yte).mean()) for h in _runs["hard"][0]]
            SM.setdefault("selection", {})[task] = {"mode": best[0], "round": best[1], "consistency": best[2], "consistency_by_round": {m: _runs[m][1] for m in _runs}}
            SM["tasks"][task] = _R
            print(f"MNIST-{task:6s} digit acc | " + " ".join(f"{k} {v['digit_acc']:.3f} (task {v['task_acc']:.3f})" for k, v in _R.items() if isinstance(v, dict)) +
                  f" | label-preserving digit transpositions {len(sym)} | consistency-selected {best[0]}@{best[1]} | soft by round {' '.join(f'{x:.3f}' for x in _R['kdr_rounds_acc'])} | hard {' '.join(f'{x:.3f}' for x in _R['kdr_hard_rounds_acc'])} | zero-shot {SM['zeroshot_acc']:.3f} oracle {SM['oracle_acc']:.3f}", flush=True)
            gpu_free()
        save_json(ckpt("s8_mnist.json"), SM); hf_push(RUN_NAME, force=True)
    return (SM,)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9b. External benchmark we did not build: BDD-OIA (real dashcam frames, 21 reason concepts, 4 actions)
    Data: the authors' official release (Xu et al., CVPR 2020; Google Drive id below), last frames + `*_actions.json` /
    `*_reasons.json`. Knowledge: **exactly the rsbench rules** (Bortolotti et al., NeurIPS 2024 D&B, `utils_problog.py`:
    forward/stop over concepts 0–8 with obstacle = car ∨ person ∨ rider ∨ other; left over 9–14; right over 15–20; worlds that
    rsbench marks invalid are excluded; the all-zero left/right worlds are ambiguous). Concept order = BDD-OIA reason order.
    Frames are squashed to 224×224 (left/right content must stay in view) and encoded by the same frozen CLIP ViT-L/14.
    Training uses **action labels only**; the 21 per-image reason labels are used only for evaluation (and for COCOCO-style's
    calibration split). Predictions LB1–LB3 in the pre-registration.
    """)
    return



@app.cell
def _(CFG, CLIP_MEAN, CLIP_STD, DEV, FEATS, Image, RUN_NAME, SMOKE, atomic_write, check_deadline, ckpt, clip_image_embed, clip_text_embed, ensure_pkg, gpu_free, have, hf_push, json, load_clip, make_block, np, os, torch):
    BDD_DRIVE_ID = "1NzF-UKaakHRNcyghtaWDmc-Vpem7lyQ6"
    BDD_CONCEPTS = ["the traffic light is green", "following the car ahead", "the road is clear", "the traffic light is red",
                    "there is a traffic sign", "a car is in the way", "a person is in the way", "a rider is in the way",
                    "another obstacle is in the way", "there is a lane on the left", "the left-turn traffic light is green",
                    "following a car turning left", "there is no lane on the left", "there is an obstacle on the left",
                    "there is a solid line on the left", "there is a lane on the right", "the right-turn traffic light is green",
                    "following a car turning right", "there is no lane on the right", "there is an obstacle on the right",
                    "there is a solid line on the right"]
    BDD_TEMPLATES = ["a dashcam photo where {c}.", "a driving scene: {c}."]


    def bdd_fs(v):                       # rsbench build_world_queries_matrix_FS, concepts 0..8 -> (forward, stop)
        g, f, cl, r, s_, car, per, rid, oth = v; obs = int(car or per or rid or oth)
        if g + f + cl > 0:
            if (g and r) or (cl and obs):
                return []                                    # invalid world (rsbench: 'pass')
            return [(0, 1)] if (r or s_ or obs) else [(1, 0)]
        return [(0, int(bool(r or s_ or obs)))]


    def bdd_left(v):                     # rsbench build_world_queries_matrix_L, concepts 9..14 -> (left,)
        ll, gl, fl, nll, lo, ls = v
        if ll + gl + fl + nll + lo + ls == 0:
            return [(0,), (1,)]                              # ambiguous (0.5 / 0.5)
        return [(1,)] if (ll or gl or fl) else [(0,)]


    def bdd_right(v):                    # rsbench build_world_queries_matrix_R, concepts 15..20 -> (right,)
        rl, gr, fr, nrl, ro, rs_ = v
        if rl + gr + fr + nrl + ro + rs_ == 0:
            return [(0,), (1,)]
        if rl or gr or fr:
            return [(0,)] if (ro or rs_ or nrl) else [(1,)]
        return [(0,)]


    BDD_SPECS = [make_block(list(range(0, 9)), bdd_fs), make_block(list(range(9, 15)), bdd_left), make_block(list(range(15, 21)), bdd_right)]


    def make_synthetic_bdd(root, n=900, seed=0):
        """SMOKE ONLY: BDD-OIA-format replica (same JSON schema as the official loader); concepts painted as patches."""
        rng = np.random.default_rng(seed); img_dir = os.path.join(root, "lastframe", "data"); info = os.path.join(root, "info")
        os.makedirs(img_dir, exist_ok=True); os.makedirs(info, exist_ok=True)
        loc = rng.integers(0, 50, size=(21, 2)); col = rng.integers(0, 255, size=(21, 3))
        splits = {"train": range(0, int(n * .6)), "val": range(int(n * .6), int(n * .7)), "test": range(int(n * .7), n)}
        for sp, rr in splits.items():
            imgs, anns, reasons = [], [], []
            for k, i in enumerate(rr):
                while True:
                    c = (rng.random(21) < 0.25).astype(int)
                    o_fs = bdd_fs(tuple(c[:9]))
                    if o_fs:
                        break
                act = list(o_fs[0]) + [int(rng.choice([x[0] for x in bdd_left(tuple(c[9:15]))])), int(rng.choice([x[0] for x in bdd_right(tuple(c[15:21]))]))]
                im = (rng.integers(0, 60, (64, 64, 3))).astype(np.uint8)
                for j in np.where(c == 1)[0]:
                    y0, x0 = loc[j]; im[y0:y0 + 9, x0:x0 + 9] = col[j]
                fn = f"{sp}_{i:05d}.jpg"; Image.fromarray(im).save(os.path.join(img_dir, fn))
                imgs.append({"id": k, "file_name": fn}); anns.append({"id": k, "category": act}); reasons.append({"file_name": fn, "reason": c.tolist()})
            with open(os.path.join(info, f"{sp}_25k_images_actions.json"), "w") as f:
                json.dump({"images": imgs, "annotations": anns}, f)
            with open(os.path.join(info, f"{sp}_25k_images_reasons.json"), "w") as f:
                json.dump(reasons, f)
        return root


    def get_bdd(root):
        if SMOKE:
            return root if os.path.exists(os.path.join(root, "info")) else make_synthetic_bdd(root)
        if any("reason" in f for _, _, fs in os.walk(root) for f in fs):
            return root
        os.makedirs(root, exist_ok=True)
        zp = os.environ.get("BDDOIA_ZIP", os.path.join(root, "bddoia.zip"))
        if not os.path.exists(zp):
            try:
                ensure_pkg("gdown")
                import gdown
                gdown.download(id=BDD_DRIVE_ID, output=zp, quiet=False)
            except Exception as e:
                raise FileNotFoundError(f"BDD-OIA download from Google Drive failed: {repr(e)[:200]}") from None
            if not os.path.exists(zp):
                raise FileNotFoundError("BDD-OIA download from Google Drive returned no file (quota or permission)")
        import zipfile
        if not zipfile.is_zipfile(zp):
            raise FileNotFoundError(f"{zp} is not a complete zip file (interrupted download?); delete it and download again")
        with zipfile.ZipFile(zp) as z:
            z.extractall(root)
        return root


    def parse_bdd(root):
        jpgs = {}
        for d, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    jpgs.setdefault(f, os.path.join(d, f))
        out = {}
        for sp in ["train", "val", "test"]:
            fa = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.startswith(sp) and "action" in f and f.endswith(".json")]
            fr = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.startswith(sp) and "reason" in f and f.endswith(".json")]
            if not fa or not fr:
                raise FileNotFoundError(f"BDD-OIA {sp}: action/reason json not found under {root}")
            with open(sorted(fa)[0]) as f:
                A = json.load(f)
            with open(sorted(fr)[0]) as f:
                R = json.load(f)
            annl = A["annotations"]                                    # official loader: annotations[img['id']]['category']
            def _cat(i):
                a = annl[i] if isinstance(annl, list) else annl.get(i, annl.get(str(i)))
                return a["category"] if isinstance(a, dict) else a
            rmap = {r["file_name"]: r["reason"] for r in R}
            paths, acts, reas = [], [], []
            for im in A["images"]:
                cat = _cat(im["id"])
                if not (len(cat) == 4 or cat[4] == 0):                  # official loader's filter
                    continue
                fn = os.path.basename(im["file_name"])
                if fn in jpgs and (im["file_name"] in rmap or fn in rmap):
                    r = rmap.get(im["file_name"], rmap.get(fn))
                    paths.append(jpgs[fn]); acts.append(cat[:4]); reas.append(r[:21])
            out[sp] = (paths, np.array(acts, np.int64), np.array(reas, np.int64))
            print(f"BDD-OIA {sp}: {len(paths)} frames")
        return out


    def _load_squash(path, size=224):
        return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0


    def encode_bdd(paths, tag):
        out = ckpt(f"bdd_{tag}.npy")
        if have(f"bdd_{tag}.npy"):
            return np.load(out)
        if SMOKE:
            rng = np.random.default_rng(1); P = rng.normal(size=(32 * 32 * 3, FEATS.shape[1])).astype(np.float32) / 50
            f = np.stack([(_load_squash(p, 32) - 0.3).reshape(-1) @ P for p in paths]).astype(np.float32)
            f /= np.linalg.norm(f, axis=1, keepdims=True); np.save(out, f); return f
        model, _ = load_clip(CFG["encoder"], text=False)
        mu, sd = torch.tensor(CLIP_MEAN, device=DEV).view(1, 3, 1, 1), torch.tensor(CLIP_STD, device=DEV).view(1, 3, 1, 1)
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(8); chunk_dir = ckpt(f"img_chunks_bdd_{tag}"); os.makedirs(chunk_dir, exist_ok=True)
        for c0 in range(0, len(paths), 1024):
            cf = os.path.join(chunk_dir, f"c{c0:06d}.npy")
            if os.path.exists(cf):
                continue
            check_deadline(); res = []
            for b0 in range(c0, min(c0 + 1024, len(paths)), CFG["img_batch"]):
                arr = np.stack(list(pool.map(_load_squash, paths[b0:min(b0 + CFG["img_batch"], c0 + 1024, len(paths))])))
                x = (torch.from_numpy(arr).permute(0, 3, 1, 2).to(DEV) - mu) / sd
                with torch.inference_mode():
                    res.append(torch.nn.functional.normalize(clip_image_embed(model, x), dim=1).cpu().numpy())
            atomic_write(cf, lambda t: np.save(t, np.concatenate(res))); hf_push(RUN_NAME)
            print(f"  [bdd {tag}] {min(c0 + 1024, len(paths))}/{len(paths)}", flush=True)
        f = np.concatenate([np.load(os.path.join(chunk_dir, x)) for x in sorted(os.listdir(chunk_dir)) if x.endswith(".npy")])
        atomic_write(out, lambda t: np.save(t, f)); del model; gpu_free(); return f


    def bdd_text_dirs():
        if have("bdd_txt.npy"):
            T = np.load(ckpt("bdd_txt.npy"))
        elif SMOKE:
            T = None
        else:
            model, tok = load_clip(CFG["encoder"])
            with torch.inference_mode():
                T = 0
                for tm in BDD_TEMPLATES:
                    T = T + torch.nn.functional.normalize(clip_text_embed(model, tok, [tm.format(c=c) for c in BDD_CONCEPTS]), dim=1)
            T = torch.nn.functional.normalize(T, dim=1).cpu().numpy(); np.save(ckpt("bdd_txt.npy"), T); del model; gpu_free()
        return T
    return (BDD_DRIVE_ID, BDD_SPECS, bdd_fs, bdd_left, bdd_right, bdd_text_dirs, encode_bdd, get_bdd, parse_bdd)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. External benchmark: BDD-OIA under the rsbench knowledge (measured; the theory says what to expect)
    Exact fact F3 (computed below from the rules, no data): no action label forces any of the 21 concepts TRUE. The labels can
    therefore only *remove* false positives and enforce "at least one reason"; everything else must come from the channel.
    Methods: zero-shot CLIP, the label-only baselines, NeSy initialised at zero-shot, KDR. Metrics on test frames with the
    per-frame reason labels (evaluation only): concept macro-F1, mean accuracy, and action accuracy implied by the concepts.
    """)
    return



@app.cell
def _(BDD_DRIVE_ID, BDD_SPECS, CFG, DEV, RUN_NAME, audit_reader, bdd_fs, bdd_left, bdd_right, bdd_text_dirs, ckpt, concept_baselines, cons_mask, encode_bdd, forced_from_cons, get_bdd, have, hf_push, kdr_select, label_ids, load_json, nesy_train, np, os, parse_bdd, save_json, text_dirs_centered, torch, zs_logits):
    if have("s9k_bdd.json"):
        SB = load_json(ckpt("s9k_bdd.json"))
    else:
        SB = {"status": "not run"}
        try:
            pos_force, neg_force = np.zeros(21, bool), np.zeros(21, bool)
            for _s_ in BDD_SPECS:
                for o in range(_s_["WL"].shape[1]):
                    Vw = _s_["V"][_s_["WL"][:, o]]
                    for _jj, _j in enumerate(_s_["ix"]):
                        pos_force[_j] |= bool(len(Vw) and Vw[:, _jj].min() == 1); neg_force[_j] |= bool(len(Vw) and Vw[:, _jj].max() == 0)
            broot = get_bdd(os.path.join(CFG["work_dir"], "data", "bddoia"))
            BD = parse_bdd(broot)
            Fb = {sp: encode_bdd(BD[sp][0], sp) for sp in BD}
            Tb = bdd_text_dirs()
            if Tb is None:                                                # SMOKE-ONLY planted stand-in text directions
                _tr = Fb["train"]; Cr = BD["train"][2]
                Tb = np.stack([_tr[Cr[:, j] == 1].mean(0) - _tr[Cr[:, j] == 0].mean(0) if 0 < Cr[:, j].sum() < len(Cr) else np.random.randn(_tr.shape[1]) for j in range(21)])
                Tb = Tb / np.linalg.norm(Tb, axis=1, keepdims=True) + np.random.default_rng(3).normal(0, 0.1, Tb.shape)
            TbC = text_dirs_centered(Tb.astype(np.float32))
            rsb = np.random.default_rng(CFG["seed"] + 21)
            tr_sub = np.sort(rsb.choice(len(BD["train"][0]), size=min(len(BD["train"][0]), CFG["bdd_train_cap"]), replace=False))
            te_sub = np.sort(rsb.choice(len(BD["test"][0]), size=min(len(BD["test"][0]), CFG["bdd_test_cap"]), replace=False))
            Xb = torch.nn.functional.normalize(torch.from_numpy(Fb["train"][tr_sub]).float().to(DEV), dim=1)
            Xbt = torch.nn.functional.normalize(torch.from_numpy(Fb["test"][te_sub]).float().to(DEV), dim=1)
            acts_tr = BD["train"][1][tr_sub]; Cte = BD["test"][2][te_sub]; Ate = BD["test"][1][te_sub]
            LB = [label_ids(BDD_SPECS[0], [tuple(a[:2]) for a in acts_tr]), label_ids(BDD_SPECS[1], [(a[2],) for a in acts_tr]),
                  label_ids(BDD_SPECS[2], [(a[3],) for a in acts_tr])]
            _cons = [cons_mask(s_, l) for s_, l in zip(BDD_SPECS, LB)]
            _Ftr = np.concatenate([forced_from_cons(c, len(s_["ix"])) for c, s_ in zip(_cons, BDD_SPECS)], 1)
            _Z0, (_W0, _b0) = zs_logits(Xb, TbC, Xb); Z0t = zs_logits(Xbt, TbC, Xb)[0]
            _chosen, _runs, _sel = kdr_select(Xb, BDD_SPECS, LB, _Z0, [Xbt], [Z0t], CFG["kdr_rounds"], CFG["kdr_folds"], CFG["seed"])
            _hist_ev = _runs["soft"][0]
            _ident = pos_force & neg_force                                     # exact: concepts the rsbench code identifies
            _meth = {"zeroshot": Z0t, "kdr": np.where(_ident[None, :], _runs["soft"][0][-1][0], Z0t), "kdr_soft_all_rounds": _runs["soft"][0][-1][0],
                    "kdr_hard_all_rounds": _runs["hard"][0][-1][0], "kdr_consistency_selected": _chosen[0]}
            _base = concept_baselines(Xb, BDD_SPECS, _cons, 21, Xbt, _Ftr, T_dirs=TbC, cal=None, n_ens=CFG["n_ens"], knn_k=CFG["knn_k"])
            for _bn, (_sc, _pr) in _base.items():
                if _bn != "zeroshot_clip":
                    _meth[_bn] = np.where(_pr, _sc + 1e-3, -_sc - 1e-3)
            _Wz, _bz = nesy_train(Xb, BDD_SPECS, _cons, 21, seed=7, epochs=CFG["nesy_epochs"], W0=_W0, b0=_b0)
            with torch.no_grad():
                _meth["nesy_zsinit"] = (Xbt @ _Wz.T + _bz).cpu().numpy()
            from sklearn.metrics import f1_score
            def acts_from(c):
                out = []
                for v in c.astype(int):
                    fs = bdd_fs(tuple(v[:9])); l = bdd_left(tuple(v[9:15])); r = bdd_right(tuple(v[15:21]))
                    out.append(list(fs[0] if fs else (0, 1)) + [l[0][0], r[0][0]])
                return np.array(out)
            _R = {}
            for _mn, _lg in _meth.items():
                _pr = (_lg > 0).astype(int)
                _R[_mn] = {"concept_macroF1": float(f1_score(Cte, _pr, average="macro", zero_division=0)),
                         "concept_acc": float((_pr == Cte).mean()), "positive_rate": float(_pr.mean()),
                         "action_acc": float((acts_from(_pr) == Ate).mean())}
            # ---- v6 audit on test frames: action labels only; reason labels score the audit ----
            LBt = [label_ids(BDD_SPECS[0], [tuple(a[:2]) for a in Ate]), label_ids(BDD_SPECS[1], [(a[2],) for a in Ate]),
                   label_ids(BDD_SPECS[2], [(a[3],) for a in Ate])]
            cons_t = [cons_mask(s_, l) for s_, l in zip(BDD_SPECS, LBt)]
            sat_t = np.stack([c_.any(1) for c_ in cons_t], 1).all(1)
            Ft = np.concatenate([forced_from_cons(c_, len(s_["ix"])) for c_, s_ in zip(cons_t, BDD_SPECS)], 1)
            strata_t = np.concatenate([np.repeat(l[:, None], len(s_["ix"]), 1) for l, s_ in zip(LBt, BDD_SPECS)], 1)
            BAUD = {mn: audit_reader((lg[sat_t] > 0).astype(int), Ft[sat_t], strata_t[sat_t], CFG["audit_alpha"], truth=Cte[sat_t]) for mn, lg in _meth.items()}
            fva = [float((Ft[sat_t, j] == Cte[sat_t, j])[Ft[sat_t, j] >= 0].mean()) if (Ft[sat_t, j] >= 0).any() else float("nan") for j in range(21)]
            def _contain(a, key):
                iv = a.get(key)
                return None if iv is None else bool(iv[0] - 1e-12 <= a["true_err"] <= iv[1] + 1e-12)
            AUD_SUM = {mn: {"A0_contains": float(np.mean([_contain(a, "A0_sample") for a in al])),
                            "A1_contains": float(np.mean([x for x in [_contain(a, "A1_interval") for a in al] if x is not None] or [float("nan")])),
                            "A0_width": float(np.mean([a["A0_sample"][1] - a["A0_sample"][0] for a in al]))} for mn, al in BAUD.items()}
            AUD_SUM["_knowledge"] = {"test_frames_contradicting_rules": float(1 - sat_t.mean()), "forced_value_accuracy_mean": float(np.nanmean(fva)),
                                     "types": [a["type"] for a in BAUD["zeroshot"]]}
            print("BDD-OIA audit:", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in AUD_SUM.items() if k in ("zeroshot", "kdr_soft_all_rounds", "nesy_conf", "_knowledge")})
            SB = {"status": "ok", "n_train": int(len(tr_sub)), "n_test": int(len(te_sub)),
                  "positive_forcing_possible": pos_force.tolist(), "negative_forcing_possible": neg_force.tolist(),
                  "forced_frac": (_Ftr >= 0).mean(0).tolist(), "methods": _R, "audit": BAUD, "audit_summary": AUD_SUM, "forced_value_accuracy": fva,
                  "selection": _sel, "kdr_rounds_F1": [float(f1_score(Cte, (h[0] > 0).astype(int), average="macro", zero_division=0)) for h in _hist_ev]}
            print(f"F3 (exact): action labels can force TRUE {int(pos_force.sum())}/21 concepts, FALSE {int(neg_force.sum())}/21")
            for _mn, _v in _R.items():
                print(f"BDD-OIA {_mn:13s} concept macro-F1 {_v['concept_macroF1']:.3f} acc {_v['concept_acc']:.3f} positives {_v['positive_rate']:.3f} action acc {_v['action_acc']:.3f}")
            save_json(ckpt("s9k_bdd.json"), SB); hf_push(RUN_NAME, force=True)
        except Exception as _e:
            if not isinstance(_e, FileNotFoundError):            # a missing dataset is reported in one line below
                import traceback; traceback.print_exc()
            SB = {"status": f"failed: {repr(_e)[:300]}"}
            print(f"BDD-OIA section skipped: {repr(_e)[:300]}\nThe rest of the notebook is unaffected. To include it, download "
                  f"https://drive.google.com/open?id={BDD_DRIVE_ID}, save it as "
                  f"{os.path.abspath(os.path.join(CFG['work_dir'], 'data', 'bddoia', 'bddoia.zip'))} and re-run this cell.")
    return (SB,)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Unseen-class world elimination (L7, L8): perception refutes knowledge-generated worlds
    Worlds = partial signatures (y, σ_y on committed set D_y) for classes of **unseen families**; σ, D_y and the knowledge
    error bound ε_y = Σ_{j∈D_y} min(π_yj, 1−π_yj) come from the unseen classes' *train-split* images only, and the test uses
    their *test-split* images (no leakage). Per-concept probes are trained on E1-train (per-image reliable labels) and
    calibrated on E1-test; Mondrian conformal p-values per concept value. A world is rejected iff
    min_{j∈D_y} p_j(x, σ_yj) < α/|D_y|. **Theorem (union bound, CCE):** P(true world rejected) ≤ α + ε_y for every class,
    seen or unseen. CCE (score of concept j given its value does not depend on the class) is audited on held-out families.
    """)
    return



@app.cell
def _(CFG, DEV, E1_TE, E1_TR, E2, E3, FAMILY, GROUP_OF_FAM, IS_TRAIN, K_CON, LABELS, N_CLS, RELIABLE, RUN_NAME, Xall, Y_INST, check_deadline, ckpt, fit_logreg, have, hf_push, load_json, np, risk_at_cov, safe_auc, save_json, torch):
    from scipy.stats import ks_2samp

    def world_ncf(s, v):
        return -s if v == 1 else s


    def conformal_tables(SC, cal, K):
        """Sorted calibration nonconformities per (concept, hypothesised value), reliable labels only (Mondrian)."""
        T = {}
        for j in range(K):
            for v in (0, 1):
                m = RELIABLE[cal, j] & (Y_INST[cal, j] == v)
                T[(j, v)] = np.sort(world_ncf(SC[cal[m], j], v))
        return T


    def world_pval(T, j, v, s):
        """Split-conformal p-value of 'concept j has value v' for test scores s: (1 + #{cal r_i >= r}) / (n + 1)."""
        arr = T[(j, v)]; n = len(arr)
        if n == 0:
            return np.ones_like(s)
        return (1 + n - np.searchsorted(arr, world_ncf(s, v), side="left")) / (n + 1)


    def build_worlds(CU, allowed, labels, is_train, Yi, Rel, m_max, gamma, eps_max=1.0):
        """Partial worlds of classes CU from their TRAIN-split images only: committed set D_y, signature, eps_y bound."""
        W = {}
        for c in CU:
            tr = np.where((labels == c) & is_train)[0]
            n_rel = Rel[tr].sum(0); pos = (Yi[tr] * Rel[tr]).sum(0)
            pi = np.where(n_rel > 0, pos / np.maximum(n_rel, 1), 0.5)
            minority = np.minimum(pos, n_rel - pos)
            amb = (minority + 0.5) / (n_rel + 1.0)          # Jeffreys-smoothed declared deviation rate (never 0 from finite data)
            ok = np.where((n_rel >= 3) & (amb <= gamma) & allowed)[0]
            Dy = ok[np.argsort(amb[ok], kind="stable")][:m_max]
            Dy = Dy[np.cumsum(amb[Dy]) <= eps_max]           # commit only as much as the declared error budget allows
            W[int(c)] = {"D": Dy, "sigma": (pi[Dy] > 0.5).astype(int), "eps": float(min(1.0, amb[Dy].sum()))}
        return W


    def evaluate_worlds(W, te, labels, SC, PROB, T, alpha):
        cls = list(W.keys()); surv = np.zeros((len(te), len(cls)), bool)
        for ci, c in enumerate(cls):
            Dy, sg = W[c]["D"], W[c]["sigma"]
            if len(Dy) == 0:
                surv[:, ci] = True; continue
            P = np.stack([world_pval(T, j, int(v), SC[te, j]) for j, v in zip(Dy, sg)], 1)
            surv[:, ci] = P.min(1) >= alpha / len(Dy)
        truth_col = np.array([cls.index(int(labels[i])) for i in te])
        alive = surv[np.arange(len(te)), truth_col]
        eps_bar = float(np.mean([W[int(labels[i])]["eps"] for i in te]))
        single = surv.sum(1) == 1
        ll = np.zeros((len(te), len(cls)))
        for ci, c in enumerate(cls):
            Dy, sg = W[c]["D"], W[c]["sigma"]
            if len(Dy):
                pp = np.clip(PROB[te][:, Dy], 1e-4, 1 - 1e-4)
                ll[:, ci] = (np.log(pp) * sg + np.log(1 - pp) * (1 - sg)).sum(1)
        srt = np.sort(ll, 1); marg = srt[:, -1] - srt[:, -2]; dap_ok = ll.argmax(1) == truth_col
        return {"coverage": float(alive.mean()), "bound": 1 - alpha - eps_bar, "eps_bar": eps_bar, "mean_survivors": float(surv.sum(1).mean()),
                "n_worlds": len(cls), "empty_rate": float((surv.sum(1) == 0).mean()), "singleton_cov": float(single.mean()),
                "singleton_precision": float((alive & single).sum() / max(single.sum(), 1)),
                "dap_precision_at_cov": (1 - risk_at_cov(marg, dap_ok, single.mean())) if single.any() else float("nan"),
                "dap_acc": float(dap_ok.mean()), "mean_D": float(np.mean([len(W[c]["D"]) for c in cls]))}


    def evaluate_worlds_classcal(CU, labels, SC, T, alpha, gamma, m_max, eps_max, rng):
        """G4 (guaranteed): per unseen class, OUR random three-way split of its images into KB / calibration / test.
        World score s_c(x) = |D_c| * min_j p_j(x, sigma_cj) (any fixed function of x; built only from the KB third and seen-class
        tables). Class-conditional conformal p-value from the calibration third; class c survives iff P_c(x) > alpha.
        Theorem: KB, score and tables are independent of the cal/test thirds, which are exchangeable by our randomisation,
        hence P(true class rejected) <= alpha for every class (Vovk 2005, class-conditional / Mondrian conformal)."""
        kb_mask = np.zeros(len(labels), bool); calI, teI = {}, []
        for c in CU:
            ii = rng.permutation(np.where(labels == c)[0]); a, b = len(ii) // 3, 2 * len(ii) // 3
            kb_mask[ii[:a]] = True; calI[int(c)] = ii[a:b]; teI.append(ii[b:])
        te = np.concatenate(teI)
        W = build_worlds(CU, np.ones(SC.shape[1], bool), labels, kb_mask, Y_INST, RELIABLE, m_max, gamma, eps_max)
        def score(c, idx):
            Dy, sg = W[c]["D"], W[c]["sigma"]
            if len(Dy) == 0:
                return np.zeros(len(idx))
            P = np.stack([world_pval(T, j, int(v), SC[idx, j]) for j, v in zip(Dy, sg)], 1)
            return -len(Dy) * P.min(1)                                   # larger = more nonconforming
        cls = [int(c) for c in CU]; surv = np.zeros((len(te), len(cls)), bool)
        for ci, c in enumerate(cls):
            sc_cal = np.sort(score(c, calI[c])); sc_te = score(c, te); n = len(sc_cal)
            pv = (1 + n - np.searchsorted(sc_cal, sc_te, side="left")) / (n + 1)      # ties counted as nonconforming (conservative)
            surv[:, ci] = pv > alpha
        truth_col = np.array([cls.index(int(labels[i])) for i in te]); alive = surv[np.arange(len(te)), truth_col]
        n_cal = np.array([len(calI[int(labels[i])]) for i in te])
        exact = float(np.mean(np.floor(alpha * (n_cal + 1)) / (n_cal + 1)))          # exact finite-sample miss bound (<= alpha)
        return {"coverage": float(alive.mean()), "bound": 1 - exact, "n_test": int(len(te)), "mean_survivors": float(surv.sum(1).mean()),
                "n_worlds": len(cls), "singleton_cov": float((surv.sum(1) == 1).mean()),
                "singleton_precision": float((alive & (surv.sum(1) == 1)).sum() / max((surv.sum(1) == 1).sum(), 1))}


    if have("s9_world_v4.json"):
        S9 = load_json(ckpt("s9_world_v4.json"))
    else:
        YI_tr = torch.from_numpy(Y_INST[E1_TR]).to(DEV); MI_tr = torch.from_numpy(RELIABLE[E1_TR]).to(DEV)
        probe = fit_logreg(Xall[E1_TR], YI_tr, MI_tr)
        with torch.no_grad():
            SC = probe(Xall).cpu().numpy()
        PROB = 1 / (1 + np.exp(-SC)); cal = E1_TE; TAB = conformal_tables(SC, cal, K_CON)
        auc_cal = np.array([safe_auc(Y_INST[cal, j][RELIABLE[cal, j]], SC[cal, j][RELIABLE[cal, j]]) for j in range(K_CON)])
        nonE1_fams = sorted({f for f, g in GROUP_OF_FAM.items() if g != 1})
        S9 = {"alpha": CFG["alpha"], "splits": [], "auc_cal": auc_cal.tolist()}
        rs = np.random.default_rng(CFG["seed"] + 9)
        for _sp in range(CFG["n_world_splits"]):
            check_deadline()
            U = set(rs.choice(nonE1_fams, size=max(1, len(nonE1_fams) // 2), replace=False))
            CU = np.array([c for c in range(N_CLS) if FAMILY[c] in U]); CUs = set(CU.tolist())
            audit = np.array([i for i in np.concatenate([E2, E3]) if FAMILY[LABELS[i]] not in U])
            _te = np.array([i for i in np.where(~IS_TRAIN)[0] if LABELS[i] in CUs])
            _gap = np.zeros(K_CON)
            for _j in range(K_CON):
                _g = 0.0
                for _v in (0, 1):
                    a1 = SC[cal, _j][RELIABLE[cal, _j] & (Y_INST[cal, _j] == _v)]; a2 = SC[audit, _j][RELIABLE[audit, _j] & (Y_INST[audit, _j] == _v)]
                    if len(a1) > 5 and len(a2) > 5:
                        _g = max(_g, ks_2samp(a1, a2).statistic)
                _gap[_j] = _g
            half = K_CON // 2
            _sel = {"all": np.ones(K_CON, bool)}
            for _nm, order in [("low_cce_gap", np.argsort(_gap)), ("high_auc", np.argsort(-np.nan_to_num(auc_cal, nan=0))), ("random_half", rs.permutation(K_CON))]:
                _m = np.zeros(K_CON, bool); _m[order[:half]] = True; _sel[_nm] = _m
            _res = {"unseen_families": sorted(U), "n_test": int(len(_te)), "n_classes": int(len(CU)), "cce_gap_mean": float(_gap.mean())}
            for _nm, _al in _sel.items():
                _res[_nm] = evaluate_worlds(build_worlds(CU, _al, LABELS, IS_TRAIN, Y_INST, RELIABLE, CFG["world_m"], CFG["world_gamma"], CFG["world_eps_max"]),
                                          _te, LABELS, SC, PROB, TAB, CFG["alpha"])
            _res["class_cal"] = evaluate_worlds_classcal(CU, LABELS, SC, TAB, CFG["alpha"], CFG["world_gamma"], CFG["world_m"], CFG["world_eps_max"],
                                                        np.random.default_rng(CFG["seed"] + 90 + _sp))
            print(f"  G4 class-calibrated (guaranteed): coverage {_res['class_cal']['coverage']:.3f} >= {_res['class_cal']['bound']:.3f} "
                  f"| survivors {_res['class_cal']['mean_survivors']:.1f}/{_res['class_cal']['n_worlds']} | singleton cov {_res['class_cal']['singleton_cov']:.3f} "
                  f"prec {_res['class_cal']['singleton_precision']:.3f}", flush=True)
            S9["splits"].append(_res)
            print(f"split {_sp}: {len(CU)} unseen classes, {len(_te)} test imgs | coverage {_res['all']['coverage']:.3f} (bound {_res['all']['bound']:.3f}, "
                  f"eps {_res['all']['eps_bar']:.3f}) survivors {_res['all']['mean_survivors']:.1f}/{_res['all']['n_worlds']} | singleton prec "
                  f"{_res['all']['singleton_precision']:.3f} vs DAP {_res['all']['dap_precision_at_cov']:.3f} @cov {_res['all']['singleton_cov']:.3f} | "
                  f"low-CCE-gap cov {_res['low_cce_gap']['coverage']:.3f} high-AUC {_res['high_auc']['coverage']:.3f} random-half {_res['random_half']['coverage']:.3f}", flush=True)
        save_json(ckpt("s9_world_v4.json"), S9); hf_push(RUN_NAME, force=True)
    return (S9,)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. G3 — risk-controlled concept claims on unseen families (guaranteed by construction)
    Units are E3 images, split calibration / test at random by us. Every (image, concept) cell of the design-1 code can be
    claimed; per image ONE of its claims (pre-drawn random priority, label-free) is scored, so each unit's loss is an exact
    Bernoulli. Operating points: top-q fraction by |logit| per cell. Learn-then-Test with Bonferroni over the fixed grid
    (Angelopoulos et al. 2021): P(selected risk > δ) ≤ α for ANY method; the test-half check fails w.p. ≤ α + .01.
    Guaranteed for every method; what differs (measured) is how much each can claim at risk δ.
    """)
    return



@app.cell
def _(CFG, CNAMES, E3, RELIABLE, RUN_NAME, S7, Y_INST, ckpt, hf_push, np, rankdata, save_json):
    from scipy.stats import binom


    def ltt_select(levels, delta, alpha):
        """Learn-then-Test with Bonferroni over the (pre-fixed) operating points: reject H_k: risk_k > delta iff the exact
        binomial p-value P(Bin(n_k, delta) <= e_k) <= alpha / K. Any rejected point has risk <= delta with FWER <= alpha, so the
        largest-coverage rejected point is safe (no ordering or monotonicity assumption, no early stopping). Returns index or -1."""
        K = len(levels); ok = [k for k, (n, e) in enumerate(levels) if n > 0 and binom.cdf(e, n, delta) <= alpha / K]
        return max(ok, key=lambda k: levels[k][0]) if ok else -1


    def risk_control(units, levels_fn, n_levels, delta, alpha, check):
        """units: dict cal/test -> image ids; levels_fn(ids, k) -> (n_images_with_a_claim, errors_of_the_picked_claims,
        n_claims, n_eligible_pairs). Risk = P(wrong | random image with a claim, one of its claims picked uniformly at random)."""
        cal = [levels_fn(units["cal"], k) for k in range(n_levels)]
        k = ltt_select([(c[0], c[1]) for c in cal], delta, alpha)
        if k < 0:
            return {"level": -1, "cov_test": 0.0, "err_test": float("nan"), "n_test_sel": 0, "pass": True}
        n_t, e_t, nc, ne = levels_fn(units["test"], k)
        p_check = binom.sf(e_t - 1, n_t, delta) if n_t else 1.0                 # P(>= e_t errors | risk = delta)
        return {"level": k, "cov_test": nc / max(ne, 1), "img_cov_test": n_t / max(len(units["test"]), 1),
                "err_test": e_t / n_t if n_t else float("nan"), "n_test_sel": int(n_t), "p_check": float(p_check), "pass": bool(p_check >= check)}


    def rank01(x):
        return (rankdata(x, method="average") - 1) / max(len(x) - 1, 1)


    Q_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


    def run_G3_methods(methods, truth, rel, delta, seed):
        """methods: name -> logits (n_img, n_cells); truth, rel: (n_img, n_cells)."""
        rng = np.random.default_rng(seed); n, m = truth.shape
        prio = rng.random((n, m)); perm = rng.permutation(n); units = {"cal": perm[: n // 2], "test": perm[n // 2:]}
        out = {}
        for mn, L in methods.items():
            R = np.stack([rank01(np.abs(L[:, c])) for c in range(m)], 1); P = L > 0
            def fn(ids, k, R=R, P=P):
                S = (R[ids] >= 1 - Q_GRID[k]) & rel[ids]; has = S.any(1)
                pick = np.where(S, prio[ids], -1.0).argmax(1); ar = np.arange(len(ids))
                wrong = P[ids][ar, pick] != truth[ids][ar, pick]
                return int(has.sum()), int((wrong & has).sum()), int(S.sum()), int(rel[ids].sum())
            out[mn] = risk_control(units, fn, len(Q_GRID), delta, CFG["risk_alpha"], CFG["risk_check"])
        return out


    E3n = len(E3); G3_METHOD_NAMES = sorted(S7[0]["codes"]["design1"]["methods"].keys())
    G3_IN = {mn: np.concatenate([np.array(tr["codes"]["design1"]["methods"][mn]["E3_logits"]) for tr in S7], 1) for mn in G3_METHOD_NAMES}
    _tr_cols = [j for tr in S7 for j in [CNAMES.index(c) for c in tr["triple"]]]
    G3_TRUTH = Y_INST[E3][:, _tr_cols].astype(bool); G3_REL = RELIABLE[E3][:, _tr_cols]
    G3_ALL = {dl: run_G3_methods({k: v for k, v in G3_IN.items() if k != "oracle"}, G3_TRUTH, G3_REL, dl, CFG["seed"] + 33) for dl in CFG["risk_deltas"]}
    for _dl, _g in G3_ALL.items():
        print(f"G3 delta={_dl}: " + " | ".join(f"{m} cov {r['cov_test']:.3f} err {r['err_test']:.3f}{'' if r['pass'] else ' CHECK-FAILED'}" for m, r in _g.items()))
    save_json(ckpt("s10_G3.json"), {str(k): v for k, v in G3_ALL.items()}); hf_push(RUN_NAME)
    return (G3_ALL, G3_METHOD_NAMES)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Verdicts — guarantees (PASS/FAIL), exact facts, pre-registered predictions (PASS/FAIL), measurements
    """)
    return



@app.cell
def _(A0_WIDTH, CFG, CODES, CODE_SPECS, G3_ALL, G3_METHOD_NAMES, G5, G5_OK, G6_VIOL, HF_STATE, HF_URI, HW, LABELS, LOC_REJ, P2_MAE, P2_SP, P2_SP3, P4_COV, P4_FRAC, P4_LOCREJ, P4_MAE, P4_ROWS, P6_MAE, P6_ROWS, P6_SP, P7, P7_OK, RELIABLE, RUN_NAME, S7, S9, SB, SEP, SEP_ROWS, SE_CHECK, SIG, SM, SMOKE, T0, Y_INST, ckpt, hf_push, load_json, np, os, rss_gb, save_json, time, vram_report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = {}
    def put(kind, name, val, ok=True):
        V[name] = (kind, val, bool(ok))

    def mean_acc(tr, code, mn, split="E3"):
        return float(np.nanmean([c["acc"] for c in tr["codes"][code]["methods"][mn][split]["concepts"]]))

    def pooled(code, mn, split="E3", key="acc"):
        return float(np.nanmean([mean_acc(tr, code, mn, split) if key == "acc" else tr["codes"][code]["methods"][mn][split]["task_acc"] for tr in S7]))

    g1 = sum(tr["codes"][c]["G1_violations"] for tr in S7 for c in CODES)
    put("GUARANTEE", "G1 decoded concept vectors satisfy the knowledge for their label (violations)", g1, g1 == 0)
    put("GUARANTEE", "G2 state evolution = fresh-sample simulation of cross-fitted KDR (max gap <= .025)", round(SE_CHECK["max_gap"], 4), SE_CHECK["max_gap"] <= 0.025)
    gk = G3_ALL[CFG["risk_delta"]]["kdr"]
    put("GUARANTEE", f"G3 KDR claims on unseen families: risk <= {CFG['risk_delta']} not refuted on the held-out half "
        f"({'EMPTY: nothing claimed at this delta' if gk['level'] < 0 else 'non-empty'})", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in gk.items()}, gk["pass"])
    g4 = [s_["class_cal"] for s_ in S9["splits"]]; g4ok = [x["coverage"] >= x["bound"] - 0.02 for x in g4]
    put("GUARANTEE", "G4 unseen-class world elimination (class-calibrated): coverage >= exact bound - .02 on >= 4/5 splits",
        {"coverage": [round(x["coverage"], 3) for x in g4], "bound": [round(x["bound"], 3) for x in g4], "survivors": [round(x["mean_survivors"], 1) for x in g4]},
        sum(g4ok) >= max(1, int(0.8 * len(g4ok))))
    _dev = ((Y_INST != SIG[LABELS]) & RELIABLE).any(1)
    put("FACT", "F1 images whose reliable per-image attributes deviate from their class code", round(float(_dev.mean()), 4))
    put("FACT", "F2 which concepts each code can force (T/F), exact", {k: ["".join(["T" if x["force_true"] else "-", "F" if x["force_false"] else "-"]) for x in fx] for k, (_, fx) in CODE_SPECS.items()})
    if SB.get("status") == "ok":
        put("FACT", "F3 BDD-OIA rsbench knowledge: concepts an action label can force TRUE / FALSE", {"true": int(np.sum(SB["positive_forcing_possible"])), "false": int(np.sum(SB["negative_forcing_possible"]))})
    LABEL_ONLY = [m for m in ["nesy_conf", "ensemble", "bears_style", "knn"] if m in G3_METHOD_NAMES]
    p1 = {c: {m: round(pooled(c, m), 4) for m in ["kdr", "zeroshot"] + LABEL_ONLY + ["nesy_zsinit", "oracle"]} for c in CODES}
    TWO_SIDED = [c for c, (_, fx) in CODE_SPECS.items() if all(x["force_true"] and x["force_false"] for x in fx)]
    p1a = {c: (round(pooled(c, "kdr", "E1te"), 4), round(pooled(c, "oracle", "E1te"), 4)) for c in TWO_SIDED}
    p1b = all(p1[c]["kdr"] >= p1[c]["zeroshot"] - 0.01 for c in CODES)
    put("PREDICTION", "P1 CUB-Logic: (a) identifying codes: KDR within .05 of the instance-supervised oracle (seen families); (b) ledger-guided KDR >= zero-shot - .01 on every code (unseen families)",
        {"a_kdr_vs_oracle_E1te": p1a, "b_no_harm": p1b, "E3": p1}, all(k >= o - 0.05 for k, o in p1a.values()) and p1b)
    put("PREDICTION", "P2 state evolution ranks measured KDR errors (Spearman >= .5, seen families)",
        {"spearman_E1te": round(float(P2_SP), 3), "spearman_E3": round(float(P2_SP3), 3), "mae": round(P2_MAE, 4), "cells": len(SEP_ROWS)}, P2_SP >= 0.5)
    ma = SM["tasks"]["add"]
    put("PREDICTION", "P3 MNIST-Add (identifying code): KDR digit acc >= oracle - .05 and >= zero-shot + .10",
        {k: round(v["digit_acc"], 4) for k, v in ma.items() if isinstance(v, dict)} | {"zeroshot_overall": round(SM["zeroshot_acc"], 4), "oracle": round(SM["oracle_acc"], 4)},
        ma["kdr"]["digit_acc"] >= SM["oracle_acc"] - 0.05 and ma["kdr"]["digit_acc"] >= SM["zeroshot_acc"] + 0.10)
    _gap = {c: {m: round(pooled(c, m, key="task") - pooled(c, m), 4) for m in ["kdr", "nesy_conf"] if m in G3_METHOD_NAMES} for c in CODES}
    put("MEASURED", "M6 shortcut gap (task acc - concept acc, E3) per code", _gap)
    put("MEASURED", "M7 non-identifying codes (where the ledger keeps zero-shot): what unrestricted KDR and label-only methods do",
        {"cub_parity": {m: round(pooled("parity", m), 3) for m in ["zeroshot", "kdr_soft_all_rounds", "kdr_hard_all_rounds", "kdr_consistency_selected"] + LABEL_ONLY},
         "mnist_sumparity": {k: round(v["digit_acc"], 3) for k, v in SM["tasks"]["parity"].items() if isinstance(v, dict)}})
    put("MEASURED", "M1 coverage at guaranteed risk (delta -> method -> coverage, error), CUB-Logic design1, unseen families",
        {str(dl): {m: (round(r["cov_test"], 3), round(r["err_test"], 3) if r["err_test"] == r["err_test"] else None) for m, r in g.items()} for dl, g in G3_ALL.items()})
    put("MEASURED", "M2 (identifying codes: label-only NeSy is expected near the oracle, KDR is not claimed to beat it) CUB-Logic all methods, concept acc on E3 / E1te, per code", {c: {m: (round(pooled(c, m), 3), round(pooled(c, m, "E1te"), 3)) for m in G3_METHOD_NAMES} for c in CODES})
    put("MEASURED", "M3 MNIST digit acc / task acc per task (+ number of label-preserving digit transpositions)", {t: {"symmetries": SM["symmetries"][t], **{k: (round(v["digit_acc"], 3), round(v["task_acc"], 3)) for k, v in R.items() if isinstance(v, dict)}} for t, R in SM["tasks"].items()})
    put("MEASURED", "M4 MNIST digit acc up to relabelling (grounding modulo symmetry)", {t: {k: round(v["digit_acc_up_to_relabel"], 3) for k, v in R.items() if isinstance(v, dict)} for t, R in SM["tasks"].items()})
    if SB.get("status") == "ok":
        put("MEASURED", "M5 BDD-OIA: concept macro-F1 / action acc", {k: (round(v["concept_macroF1"], 3), round(v["action_acc"], 3)) for k, v in SB["methods"].items()})
    # ---- v6: the instrument ----
    put("GUARANTEE", "G5 planted: A0 CI always covers (even with cross-talk); A1 CI covers under locality; A2 exact on pairs",
        {"cases": [(c["code"], c["crosstalk"], round(c["A0_cov"], 3), round(c["A1_cov"], 3), round(c["locality_rejected"], 3)) for c in G5["cases"]],
         "A2_true_vs_est": [(round(x["true_acc"], 4), round(x["est_acc"], 4)) for x in G5["A2"]]}, G5_OK)
    put("GUARANTEE", "G6 real CUB-Logic: the A0 bracket contains every reader's true error (violations)", G6_VIOL, G6_VIOL == 0)
    put("PREDICTION", "P4 zero-shot reader: |A1 - true error| <= .05 in >= 70% of two-sided cells (CUB-Logic, E1te + E3)",
        {"frac": round(P4_FRAC, 3), "mae": round(P4_MAE, 4), "A1_ci_covers": round(P4_COV, 3), "locality_rejected": round(P4_LOCREJ, 3), "cells": len(P4_ROWS)}, P4_FRAC >= 0.70)
    _ma = {k: (round(v["audit"]["est_acc"], 4), round(v["audit"]["true_acc_pairs"], 4)) for k, v in SM["tasks"]["add"].items() if isinstance(v, dict) and "audit" in v}
    put("PREDICTION", "P5 MNIST-Add: label-free A2 digit accuracy within .02 of the truth for every reader (est, true)", _ma,
        bool(_ma) and all(abs(e - t) <= 0.02 for e, t in _ma.values()))
    put("PREDICTION", "P6 all readers, independent audit code design2: Spearman(A1, true error) >= .70",
        {"spearman": round(float(P6_SP), 3), "mae": round(P6_MAE, 4), "cells": len(P6_ROWS)}, P6_SP >= 0.70)
    put("PREDICTION", "P7 label-free model selection on unseen families (pooled true error: selected <= zero-shot, <= median, <= best + .03)",
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in P7.items()}, P7_OK)
    _rs = SM["audit_round_selection"]["add"]
    put("PREDICTION", "P8 MNIST-Add: KDR round chosen by the A2 audit is within .01 of the best round", {k: v for k, v in _rs.items() if k != "all"},
        _rs["chosen_true_acc"] >= _rs["best_true_acc"] - 0.01)
    _mp = {k: (round(v["audit"]["est_acc"], 4), round(v["audit"]["true_acc_pairs"], 4)) for k, v in SM["tasks"]["parity"].items() if isinstance(v, dict) and "audit" in v}
    put("MEASURED", "M8 MNIST-SumParity A2 (NOT identified: label-preserving digit transpositions exist) est vs true", _mp)
    put("MEASURED", "M9 A0 width (= unforced mass) by audit code; locality-test rejection rate by reader",
        {"A0_width": {k: round(v, 3) for k, v in A0_WIDTH.items()}, "loc_rej": {k: round(v, 3) for k, v in LOC_REJ.items()}})
    if SB.get("status") == "ok" and "audit_summary" in SB:
        put("MEASURED", "M10 BDD-OIA audit (one-sided code): A0 / A1 containment of the true error, A0 width, knowledge errors", SB["audit_summary"])
    for _k, (_kind, _val, _ok) in V.items():
        _tag = ("PASS" if _ok else "FAIL") if _kind in ("GUARANTEE", "PREDICTION") else _kind
        print(f"{_tag:8s} [{_kind[:4]}] {_k}\n         {_val}")

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.2))
    for _c, mk in zip(CODES, "os^"):
        ax[0].plot(range(CFG["kdr_rounds"] + 1), [np.mean([1 - np.nanmean(tr["codes"][_c]["kdr_rounds"][str(r)]["E3"]) for tr in S7]) for r in range(CFG["kdr_rounds"] + 1)], mk + "-", label=_c)
    ax[0].set_xlabel("KDR round"); ax[0].set_ylabel("concept error, unseen families"); ax[0].set_title("Decoding corrects the channel"); ax[0].legend(fontsize=8)
    ax[1].scatter([c["pred_final"] for c in SEP_ROWS], [c["meas_E1te"] for c in SEP_ROWS], c=[list(CODES).index(c["code"]) for c in SEP_ROWS], cmap="viridis")
    ax[1].plot([0, .5], [0, .5], "k--", lw=.8); ax[1].set_xlabel("state evolution prediction"); ax[1].set_ylabel("measured KDR error (E1te)"); ax[1].set_title(f"P2: Spearman {P2_SP:.2f}")
    for _t, mk in zip(["add", "parity"], "os"):
        ax[2].plot(range(1, CFG["kdr_rounds"] + 1), SM["tasks"][_t]["kdr_rounds_acc"], mk + "-", label=f"KDR MNIST-{_t}")
        ax[2].axhline(SM["tasks"][_t]["nesy_random"]["digit_acc"], ls=":", color="gray")
    ax[2].axhline(SM["zeroshot_acc"], ls="--", color="k", label="zero-shot"); ax[2].set_xlabel("round"); ax[2].set_ylabel("digit accuracy"); ax[2].legend(fontsize=8)
    ax[3].scatter([r["true_err"] for r in P6_ROWS], [r["A1"] for r in P6_ROWS], s=8, alpha=.5)
    ax[3].plot([0, .6], [0, .6], "k--", lw=.8); ax[3].set_xlabel("true concept error (per-image labels)"); ax[3].set_ylabel("label-free audit A1")
    ax[3].set_title(f"P6: audit vs truth, Spearman {P6_SP:.2f}")
    plt.tight_layout(); fig.savefig(ckpt("summary.png"), dpi=130); plt.close(fig)
    SUMMARY = dict(smoke=SMOKE, run=RUN_NAME, hardware=HW, triples=[tr["triple"] for tr in S7], cub=S7, se_check=SE_CHECK, se_pred=SEP, mnist=SM, bdd=SB,
                   world=S9, audit_check=G5, audit=load_json(ckpt("s7c_audit.json")), G3={str(k): v for k, v in G3_ALL.items()}, verdicts={k: [v[0], v[1], v[2]] for k, v in V.items()},
                   minutes=(time.time() - T0) / 60, vram=vram_report(), ram_gb=rss_gb(), hf=HF_STATE)
    save_json(ckpt("SUMMARY.json"), SUMMARY)
    ok_push = hf_push(RUN_NAME, force=True)
    print(f"\nResults: {os.path.abspath(CFG['work_dir'])} (SUMMARY.json, summary.png) | HF bucket push ok={ok_push}: {HF_URI}/{RUN_NAME}")
    print("wall time %.1f min | RAM %.2f GB | VRAM %s" % ((time.time() - T0) / 60, rss_gb(), vram_report()))
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. v7 — knowledge as a *comparator*: what the real run taught us, and three new results

    **What the ViT-L/14 run showed.** All guarantees held; the audit *ranks* readers almost perfectly (P6 Spearman 0.87, P8 exact,
    MNIST-Add A2 within 0.002), but absolute CUB levels are slightly off (P4 0.687 vs 0.70) and A1-based selection missed "near best"
    by 0.004 (P7). The locality test rejects in 64–73% of cells for **every** reader, the oracle included: CLIP's error on one
    attribute depends on other attributes (cross-talk), and that distortion is largely **shared** by all readers. This section turns
    both facts into theorems. It reuses the saved E3 logits, so it adds only seconds of compute.

    **Theorem S (assumption-free label-free selection; proved).** err_r = P(F, r wrong) + P(U)·e_r^U, and the first term is observed
    exactly (A0). Choosing r̂ = argmin P(F, r wrong) gives, for every reader r, err_r̂ − err_r ≤ P(U)(e_r̂^U − e_r^U) ≤ **P(U)**, on the
    sample and in the population. Regret is bounded by the unforced mass, a property of the rules alone: more knowledge ⇒ provably
    better selection, with no channel assumption.

    **Theorem D (reader differences under shared cross-talk; proved).** If E[1(r wrong) − 1(s wrong) | c] depends on c only through
    c_j ("difference-locality", much weaker than locality), then err_r − err_s = π d₁ + (1−π) d₀ with d_v observed exactly on F_v and
    π ∈ [P(F₁), 1 − P(F₀)] assumption-free. The interval is valid; when d₀, d₁ share a sign, the **ranking is identified**.

    **Theorem A1\* (identification iff the code forces the error's dependency set; proved).** If the joint law of (ĉ_j, ĉ_k) given c
    depends on c only through s = (c_j, c_k), every s is forced jointly by some label, and Q[t, s] = P(ĉ_S = t | F_s) has full rank, then
    m = Q π_S has a unique solution and err_j = Σ_s π_s P(ĉ_j ≠ s_j | F_s). Locality (A1) is the case S = {j}; MNIST-Add is exact because
    the sum forces the digit pair. The partner k is found from data: on labels forcing both c_j and c_k, test whether the error on j
    changes with c_k (falls back to A1 when no partner is detected). design12 forces all four (c_j, c_k) configurations for every pair.

    **Status.** G7 (S regret ≤ P(U)) is an identity, so a FAIL is a bug. G8 is a planted check of D and A1\*. The real-CUB numbers below
    are **exploratory**: they are computed on the same run that motivated the theorems, so they are not pre-registered tests.
    Pre-registration 5 (bottom) states what a fresh run must show.
    """)
    return



@app.cell
def _(SMOKE, ckpt, have, itertools, load_json, np, save_json):
    # ---- 12a. v7 core functions + G8 planted check (pairwise cross-talk: A1 locality FALSE by construction) ----



    def code_fibres(fn):
        """Exact, data-free: label -> list of 3-bit configurations in its fibre."""
        fib = {}
        for v in itertools.product([0, 1], repeat=3):
            fib.setdefault(tuple(int(x) for x in fn(v)), []).append(v)
        return fib


    def labels_of(fn, C):
        return [tuple(int(x) for x in fn(tuple(int(a) for a in c))) for c in C]


    def forced_from_labels(fn, labs):
        """(n, 3) value each label forces per concept, -1 = unforced (A0's forced set)."""
        fib = code_fibres(fn); F = np.full((len(labs), 3), -1, np.int64)
        for i, l in enumerate(labs):
            V = np.array(fib[l])
            for j in range(3):
                if V[:, j].min() == V[:, j].max():
                    F[i, j] = V[0, j]
        return F


    def s_terms(chat_j, F_j):
        """Theorem S ingredients: forced-wrong rate (observed exactly) and unforced mass P(U)."""
        f = F_j >= 0
        return float(np.mean(f & (chat_j != F_j))), float(np.mean(~f))


    def diff_interval(chat_r, chat_s, F_j):
        """Theorem D interval for err_r - err_s and the identified sign (0 = not identified)."""
        d = {}
        for v in (0, 1):
            m = F_j == v
            if m.sum() == 0:
                return None
            d[v] = float(np.mean((chat_r[m] != v).astype(float) - (chat_s[m] != v)))
        ends = [p * d[1] + (1 - p) * d[0] for p in (float(np.mean(F_j == 1)), float(1 - np.mean(F_j == 0)))]
        return {"d0": d[0], "d1": d[1], "lo": min(ends), "hi": max(ends), "sign": int(np.sign(d[0])) if np.sign(d[0]) == np.sign(d[1]) else 0}


    def proj_simplex(v):
        u = np.sort(v)[::-1]; css = np.cumsum(u); r = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
        return np.maximum(v - (css[r] - 1) / (r + 1.0), 0)


    def simplex_lsq(Q, m, iters=3000):
        """min ||Q p - m||^2 over the probability simplex (projected gradient; a 4 x 4 problem)."""
        p = np.full(Q.shape[1], 1.0 / Q.shape[1]); L = np.linalg.norm(Q, 2) ** 2 + 1e-12
        for _ in range(iters):
            p = proj_simplex(p - (Q.T @ (Q @ p - m)) / L)
        return p


    def a1_point(chat_j, F_j):
        """v6's A1 (locality, two-sided code): eps_v on F_v, pi from q; nan if not two-sided or degenerate."""
        if not ((F_j == 0).any() and (F_j == 1).any()):
            return float("nan")
        e1, e0, q = np.mean(chat_j[F_j == 1] != 1), np.mean(chat_j[F_j == 0] != 0), np.mean(chat_j)
        if abs(1 - e0 - e1) < 1e-6:
            return float("nan")
        p = float(np.clip((q - e0) / (1 - e0 - e1), 0, 1)); return float(p * e1 + (1 - p) * e0)


    def a1star_pair(chat, labs, fn, j, k):
        """Theorem A1* for S = (j, k); None when some configuration is never forced jointly or Q is singular."""
        fib = code_fibres(fn); cfg = [(a, b) for a in (0, 1) for b in (0, 1)]; Q = np.zeros((4, 4))
        lab_ix = {l: i for i, l in enumerate(fib)}; L = np.array([lab_ix[l] for l in labs])
        for si, s in enumerate(cfg):
            m = np.isin(L, [lab_ix[l] for l, V in fib.items() if all((v[j], v[k]) == s for v in V)])
            if m.sum() < 5:
                return None
            for ti, t in enumerate(cfg):
                Q[ti, si] = np.mean((chat[m, j] == t[0]) & (chat[m, k] == t[1]))
        if np.linalg.svd(Q, compute_uv=False)[-1] < 1e-3:
            return None
        mvec = np.array([np.mean((chat[:, j] == t[0]) & (chat[:, k] == t[1])) for t in cfg])
        pi = simplex_lsq(Q, mvec)
        wrong = np.array([sum(Q[ti, si] for ti, t in enumerate(cfg) if t[0] != s[0]) for si, s in enumerate(cfg)])
        return float(pi @ wrong)


    def crosstalk_partner(chat, F, j, z_min=2.0):
        """Data-driven dependency set: on images whose label forces both c_j and c_k, does the error on j change with c_k?"""
        best = (None, 0.0)
        for k in range(3):
            if k == j:
                continue
            zs = []
            for v in (0, 1):
                a = (F[:, j] == v) & (F[:, k] == 0); b = (F[:, j] == v) & (F[:, k] == 1)
                if a.sum() >= 10 and b.sum() >= 10:
                    pa, pb = np.mean(chat[a, j] != v), np.mean(chat[b, j] != v)
                    zs.append(abs(pa - pb) / (np.sqrt(pa * (1 - pa) / a.sum() + pb * (1 - pb) / b.sum()) + 1e-9))
            z = float(np.sqrt(np.mean(np.square(zs)))) if zs else 0.0
            if z > best[1]:
                best = (k, z)
        return best if best[1] >= z_min else (None, best[1])


    def a1star_auto(chat, labs, fn, j):
        """A1* with the detected partner; falls back to A1 (dependency set {j}) when none is found or it is not identified."""
        F = forced_from_labels(fn, labs); base = a1_point(chat[:, j], F[:, j])
        k, z = crosstalk_partner(chat, F, j)
        est = None if k is None else a1star_pair(chat, labs, fn, j, k)
        return {"err": base if est is None else est, "A1": base, "partner": None if est is None else k, "z": z}


    def planted_v7(reps, seed=0):
        """G8: S, D and A1* on planted readers whose error on concept 0 depends on a partner concept (locality false)."""
        _D12 = lambda v: (v[0] & v[1], v[1] & (1 - v[2]), int(v[0] + v[1] + v[2] >= 2), v[0] & v[2], v[2] & (1 - v[0]), v[1] | v[2])
        rng = np.random.default_rng(seed); out = {"S_viol": 0, "S_n": 0, "D_cov": [], "D_sign": [], "A1": [], "A1s": []}
        for rep in range(reps):
            n = 4000; C = (rng.random((n, 3)) < rng.uniform(.2, .8, 3)).astype(int); xt = rng.uniform(.05, .25)
            partner = [{0: 1, 1: 0}, {0: 2}, {}][rep % 3]
            R = []
            for _r in range(4):
                Rr = C.copy()
                for j in range(3):
                    p = np.array([rng.uniform(.02, .3), rng.uniform(.02, .3)])[C[:, j]] + (xt * C[:, partner[j]] if j in partner else 0)
                    f = rng.random(n) < p; Rr[f, j] = 1 - Rr[f, j]
                R.append(Rr)
            labs = labels_of(_D12, C); F = forced_from_labels(_D12, labs)
            for j in range(3):
                tr = [float(np.mean(r[:, j] != C[:, j])) for r in R]; fw = [s_terms(r[:, j], F[:, j]) for r in R]
                sel = int(np.argmin([a for a, _ in fw])); out["S_n"] += 1; out["S_viol"] += int(tr[sel] - min(tr) > fw[sel][1] + 1e-12)
                for a in range(4):
                    for b in range(a + 1, 4):
                        di = diff_interval(R[a][:, j], R[b][:, j], F[:, j]); td = tr[a] - tr[b]
                        if di is not None:
                            out["D_cov"].append(di["lo"] - .02 <= td <= di["hi"] + .02)
                            if di["sign"] != 0 and abs(td) > .02:
                                out["D_sign"].append(np.sign(td) == di["sign"])
            r0 = R[0]; t0 = float(np.mean(r0[:, 0] != C[:, 0])); est = a1star_auto(r0, labs, _D12, 0)
            out["A1"].append(abs(est["A1"] - t0)); out["A1s"].append(abs(est["err"] - t0))
        return {"S_violations": out["S_viol"], "S_cells": out["S_n"], "D_coverage": float(np.mean(out["D_cov"])),
                "D_sign_correct": float(np.mean(out["D_sign"])), "A1_mae": float(np.nanmean(out["A1"])), "A1star_mae": float(np.nanmean(out["A1s"]))}


    if have("s12a_planted.json"):
        G8 = load_json(ckpt("s12a_planted.json"))
    else:
        G8 = planted_v7(30 if SMOKE else 150); save_json(ckpt("s12a_planted.json"), G8)
    G8_OK = G8["S_violations"] == 0 and G8["D_coverage"] >= 0.95 and G8["D_sign_correct"] >= 0.98 and G8["A1star_mae"] < G8["A1_mae"]
    print("G8 planted (cross-talk on 2/3 of reps):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in G8.items()}, "->", "PASS" if G8_OK else "FAIL")
    return (G8, G8_OK, a1_point, a1star_auto, diff_interval, forced_from_labels, labels_of, s_terms)



@app.cell
def _(AUDIT_SPECS, CNAMES, E3, RELIABLE, RUN_NAME, S7, Y_INST, a1_point, a1star_auto, ckpt, diff_interval, forced_from_labels, have, hf_push, itertools, labels_of, load_json, np, s_terms, save_json):
    # ---- 12b. v7 on real CUB-Logic, unseen families (E3): S selection, D rankings, A1* levels. Saved logits only. ----
    from scipy.stats import chi2 as v7_chi2

    V7_CAND = ["zeroshot", "kdr_soft_all_rounds", "kdr_hard_all_rounds", "nesy_conf", "ensemble", "bears_style", "knn", "lfcbm_style", "nesy_zsinit"]


    def homog_p(x, groups):
        """Between-label homogeneity of the mean of x (chi-square, df = #labels - 1); labels with < 10 images dropped."""
        ok = [g for g in np.unique(groups) if (groups == g).sum() >= 10]
        if len(ok) < 2:
            return float("nan")
        x = x.astype(float); keep = np.isin(groups, ok); mu, var = x[keep].mean(), x[keep].var() + 1e-12
        stat = sum((groups == g).sum() * (x[groups == g].mean() - mu) ** 2 for g in ok) / var
        return float(v7_chi2.sf(stat, len(ok) - 1))


    def v7_real():
        rows, sel_rows, d_rows = [], [], []
        for ti, tr in enumerate(S7):
            trip = [CNAMES.index(nm) for nm in tr["triple"]]
            r3 = RELIABLE[E3][:, trip].all(1); Ct = Y_INST[E3][r3][:, trip].astype(int)
            for cname, res in tr["codes"].items():
                chats = {mn: (np.asarray(res["methods"][mn]["E3_logits"])[r3] > 0).astype(int) for mn in res["methods"]}
                for acn, (_aspec, afn) in AUDIT_SPECS.items():
                    labs = labels_of(afn, Ct); F = forced_from_labels(afn, labs); lab_id = np.array([hash(l) for l in labs])
                    cand = [m for m in V7_CAND if m in chats]
                    for j in range(3):
                        true = {m: float(np.mean(chats[m][:, j] != Ct[:, j])) for m in chats}
                        st = {m: s_terms(chats[m][:, j], F[:, j]) for m in cand}
                        m_sel = min(cand, key=lambda m: st[m][0])
                        sel_rows.append({"triple": ti, "code": cname, "audit_code": acn, "concept": j, "selected": m_sel, "sel_err": true[m_sel],
                                         "best_err": min(true[m] for m in cand), "zeroshot_err": true["zeroshot"], "median_err": float(np.median([true[m] for m in cand])),
                                         "PU": st[m_sel][1], "regret": true[m_sel] - min(true[m] for m in cand)})
                        for mn, ch in chats.items():
                            row = {"triple": ti, "code": cname, "audit_code": acn, "concept": j, "method": mn, "true_err": true[mn], "A1": a1_point(ch[:, j], F[:, j])}
                            if acn == "design12":
                                a = a1star_auto(ch, labs, afn, j); row.update({"A1star": a["err"], "partner": a["partner"], "z": a["z"]})
                            fv = F[:, j] >= 0
                            row["level_p"] = homog_p((ch[fv, j] != F[fv, j]), np.char.add(lab_id[fv].astype(str), F[fv, j].astype(str)))
                            rows.append(row)
                        if acn == "design2":                                       # independent audit code for the D ranking test
                            for a_, b_ in itertools.combinations(cand, 2):
                                di = diff_interval(chats[a_][:, j], chats[b_][:, j], F[:, j])
                                if di is None:
                                    continue
                                fv = F[:, j] >= 0; dd = (chats[a_][fv, j] != F[fv, j]).astype(float) - (chats[b_][fv, j] != F[fv, j])
                                d_rows.append({"td": true[a_] - true[b_], **di,
                                               "diff_p": homog_p(dd, np.char.add(lab_id[fv].astype(str), F[fv, j].astype(str)))})
        return rows, sel_rows, d_rows


    if have("s12b_v7.json"):
        V7R = load_json(ckpt("s12b_v7.json"))
    else:
        _rows, _sel, _d = v7_real()
        V7R = {"rows": _rows, "sel": _sel, "d": _d}; save_json(ckpt("s12b_v7.json"), V7R); hf_push(RUN_NAME, force=True)

    # G7: S regret <= P(U) on every cell (an identity on the sample)
    G7_VIOL = sum(1 for r in V7R["sel"] if r["regret"] > r["PU"] + 1e-12)
    # S selection pooled exactly like P7 (E3, audit code design2), and with the strongest rules (design12)
    S_SEL = {}
    for _acn in ("design2", "design12", "design1"):
        _s = [r for r in V7R["sel"] if r["audit_code"] == _acn]
        S_SEL[_acn] = {k: float(np.mean([r[k] for r in _s])) for k in ("sel_err", "best_err", "zeroshot_err", "median_err", "PU", "regret")} | {"cells": len(_s)}
    # D: rankings from differences (independent audit code design2)
    _d = V7R["d"]; _ds = [r for r in _d if r["sign"] != 0 and abs(r["td"]) > 0.02]
    D_SUM = {"pairs": len(_d), "sign_identified": float(np.mean([r["sign"] != 0 for r in _d])) if _d else float("nan"),
             "sign_correct": float(np.mean([np.sign(r["td"]) == r["sign"] for r in _ds])) if _ds else float("nan"),
             "interval_covers": float(np.mean([r["lo"] - .02 <= r["td"] <= r["hi"] + .02 for r in _d])) if _d else float("nan"),
             "level_test_rejects": float(np.nanmean([r["level_p"] < .05 for r in V7R["rows"] if r["audit_code"] == "design2" and r["level_p"] == r["level_p"]])),
             "diff_test_rejects": float(np.nanmean([r["diff_p"] < .05 for r in _d if r["diff_p"] == r["diff_p"]]))}
    # A1* vs A1 on design12 (all pairs jointly forced): zero-shot (P4 analogue) and readers NOT trained on design12
    def _cmp(rs):
        rs = [r for r in rs if r.get("A1star") is not None and r["A1"] == r["A1"]]
        if not rs:
            return {"cells": 0}
        e1 = np.array([abs(r["A1"] - r["true_err"]) for r in rs]); e2 = np.array([abs(r["A1star"] - r["true_err"]) for r in rs])
        return {"cells": len(rs), "A1_within05": float((e1 <= .05).mean()), "A1star_within05": float((e2 <= .05).mean()),
                "A1_mae": float(e1.mean()), "A1star_mae": float(e2.mean()), "partner_found": float(np.mean([r["partner"] is not None for r in rs]))}
    A1S = {"zeroshot": _cmp([r for r in V7R["rows"] if r["audit_code"] == "design12" and r["method"] == "zeroshot" and r["code"] == "design1"]),
           "readers_not_trained_on_design12": _cmp([r for r in V7R["rows"] if r["audit_code"] == "design12" and r["code"] != "design12" and r["method"] != "oracle"])}
    print(f"G7 S-selection regret <= P(U): {G7_VIOL} violations over {len(V7R['sel'])} cells")
    for _acn, _v in S_SEL.items():
        print(f"S-selection (E3, audit code {_acn:8s}): selected {_v['sel_err']:.4f} | best {_v['best_err']:.4f} | median {_v['median_err']:.4f} | zero-shot {_v['zeroshot_err']:.4f} | mean regret {_v['regret']:.4f} <= mean P(U) {_v['PU']:.3f}")
    print("D rankings (design2):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in D_SUM.items()})
    print("A1* vs A1 (design12):", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in A1S.items()})
    return (A1S, D_SUM, G7_VIOL, S_SEL)



@app.cell
def _(A1S, D_SUM, G7_VIOL, G8, G8_OK, RUN_NAME, S_SEL, ckpt, hf_push, save_json):
    # ---- 12c. v7 verdicts and Pre-registration 5 (for a FRESH run: new triples / seeds / encoder) ----
    V7V = {
        "G7": ("GUAR", G7_VIOL == 0, "S-selection regret <= unforced mass P(U) on every E3 cell (identity; violations)", G7_VIOL),
        "G8": ("GUAR", G8_OK, "planted cross-talk: S holds, D interval covers & signs right, A1* beats A1", G8),
        "X1": ("EXPL", S_SEL["design2"]["sel_err"] <= S_SEL["design2"]["best_err"] + 0.03, "S-selection with design2 within .03 of best (the rule v6's P7 missed)", S_SEL["design2"]),
        "X2": ("EXPL", S_SEL["design12"]["regret"] <= S_SEL["design2"]["regret"], "stronger rules (smaller P(U)) give smaller selection regret", {k: S_SEL[k]["regret"] for k in S_SEL}),
        "X3": ("EXPL", D_SUM["diff_test_rejects"] < D_SUM["level_test_rejects"], "difference-locality is rejected less often than level-locality (cross-talk is shared)", D_SUM),
        "X4": ("EXPL", A1S["zeroshot"].get("A1star_mae", 1) < A1S["zeroshot"].get("A1_mae", 0), "A1* reduces the zero-shot audit error vs A1 on design12", A1S),
    }
    for _k, (_kind, _ok, desc, _val) in V7V.items():
        _tag = ("PASS" if _ok else "FAIL") if _kind == "GUAR" else ("YES " if _ok else "NO  ")
        print(f"{_tag:8s} [{_kind}] {_k} {desc}\n         {_val}")
    print("\nPre-registration 5 (to be scored on a fresh run, never on this one):\n"
          "  P9  S-selection regret <= P(U) on every cell (guarantee).\n"
          "  P10 S-selection (design12 audit) within .02 of the best reader, pooled.\n"
          "  P11 difference-locality rejection rate < half the level-locality rate.\n"
          "  P12 A1* puts >= 75% of zero-shot design12 cells within .05 of the truth.\n"
          "  P13 part-cropped encoding (CUB keypoints) halves the locality rejection rate.")
    save_json(ckpt("s12c_v7_verdicts.json"), {k: [v[0], bool(v[1]), v[2], v[3]] for k, v in V7V.items()}); hf_push(RUN_NAME, force=True)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. v8 — knowledge turns *disagreement* into a certificate: the sharp, minimax label-free guarantee

    **What v7 showed on ViT-L/14.** Theorem S held with 0 violations and selected a reader within 0.006 of the best on design12. But
    its guarantee, regret ≤ P(U) = 0.17, is 27× larger than the regret actually realised. The bound charges every unforced image as if
    the two readers disagreed on it. They mostly agree. This section replaces the unforced *mass* with the unforced *disagreement*,
    and proves that nothing label-free can do better.

    **Setting.** Each image i has a task label ℓ_i, and the knowledge gives its fibre V_i = {c : program(c) = ℓ_i}. The truth c_i lies
    in V_i, and that is all we assume. ĉ_r is reader r's prediction. F is the forced set (V_i fixes c_j) and U is the unforced set.

    **Theorem C (sharp pairwise certificate; proved).** For every pair of readers r, s and every concept j,

      err_r − err_s ≤ Δ_F(r, s) + D_U(r, s),  with Δ_F = P(F, r wrong) − P(F, s wrong) and D_U = P(U, ĉ_r,j ≠ ĉ_s,j),

    and the bound is **attained**: some truth consistent with every label makes it an equality. The same holds with the sign flipped
    for the lower bound. *Proof.* On F the truth is the forced value, so the F terms are observed exactly. On U, where the readers agree,
    both are right or both are wrong, which contributes 0. Where they disagree, exactly one is wrong, which contributes ±1. Setting
    c_j := ĉ_s,j on those images is allowed, because U means both values of c_j occur in V_i. That choice attains +1 on every one of
    them. ∎

    **Corollary C1 (certified regret, and the minimax selector).** CR(r) = max_s [Δ_F(r, s) + D_U(r, s)]₊ is the *exact* worst-case
    regret of choosing r over all truths consistent with the labels. So r† = argmin_r CR(r) is **minimax-optimal among all
    label-free selectors**: any rule, whether confidence, agreement or A1, picks some r, and is guaranteed no better than CR(r) ≥ CR(r†).
    Theorem S is the special case D_U ≤ P(U): CR(r̂_S) ≤ max_s D_U(r̂_S, s) ≤ P(U).

    **Corollary C2 (assumption-free certified rankings).** If Δ_F(r, s) + D_U(r, s) < 0, then r is *certainly* better than s. This
    needs no locality and no difference-locality, unlike Theorem D.

    **Corollary C3 (knowledge coupling).** For the triple error, the sharp bound is E_i max_{v ∈ V_i} [Ham(ĉ_r, v) − Ham(ĉ_s, v)] / 3.
    It is never larger than the per-concept sum, and is strictly smaller when the fibre ties the concepts together. Here logic
    does work that per-concept reasoning cannot.

    **Finite population.** All statements above are deterministic on the evaluated sample: the truth is fixed, so there is no
    probability anywhere. For the population, each per-image term lies in [−1, 1], so Hoeffding plus a union bound over the R²
    pairs adds √(2 ln(R²/δ)/n).

    **Guarantees and checks.** G9: the true regret is ≤ CR on every cell, for per-concept and triple error (a FAIL is a bug). G10:
    every certified ordering is correct. G11 (sharpness): for each reader, the constructed worst-case truth reproduces CR exactly.
    The baselines are the standard label-free selectors: v6's A1, NeSy label consistency (how often a reader reproduces the task
    label), agreement with the majority vote, and mean confidence. The comparisons are **exploratory** on this run.
    Pre-registration 6 (bottom) is for a fresh run.
    """)
    return


@app.cell
def _(AUDIT_SPECS, CNAMES, E3, RELIABLE, RUN_NAME, S7, V7_CAND, Y_INST, a1_point, ckpt, code_fibres, forced_from_labels, have, hf_push, labels_of, load_json, np, save_json):
    # ---- 13b. v8: sharp label-free certificates (Theorem C), minimax selection, and label-free baselines. Saved logits only. ----
    def cert_tables(A, F):
        """A (R,n,3) binary predictions, F (n,3) forced values (-1 = unforced).
        Returns Wm (R,3) forced-wrong rates, UP (R,R,3) sharp upper bound of err_r - err_s, DU (R,R,3) disagreement on U."""
        n = A.shape[1]; f = F >= 0; M = (~f).astype(float)
        Wm = ((A != F[None]) & f[None]).mean(1)
        a = np.einsum("rnj,nj->rj", A, M)
        DU = (a[:, None] + a[None] - 2 * np.einsum("rnj,snj,nj->rsj", A, A, M)) / n
        return Wm, Wm[:, None] - Wm[None] + DU, DU


    def cert_joint(A, labs, fn):
        """Sharp upper bound of (triple error of r) - (triple error of s): mean over images of max over the label's fibre."""
        fib = code_fibres(fn); R, n = A.shape[:2]; UJ = np.zeros((R, R)); keys = np.array([hash(l) for l in labs])
        for l, V in fib.items():
            idx = np.nonzero(keys == hash(l))[0]
            if len(idx):
                H = (A[:, idx, None, :] != np.array(V)[None, None]).sum(-1)          # (R, n_l, k)
                UJ += (H[:, None] - H[None]).max(-1).sum(-1)
        return UJ / (3 * n)


    def worst_truth(A, F, r, s, j):
        """The truth on concept j that realises the certificate for r against s (forced value on F, s's value on U)."""
        return np.where(F[:, j] >= 0, F[:, j], A[s, :, j])


    def v8_real():
        sel, cert, pairs = [], [], []
        for ti, tr in enumerate(S7):
            trip = [CNAMES.index(nm) for nm in tr["triple"]]
            r3 = RELIABLE[E3][:, trip].all(1); Ct = Y_INST[E3][r3][:, trip].astype(int)
            for cname, res in tr["codes"].items():
                cand = [m for m in V7_CAND if m in res["methods"]]
                Z = np.stack([np.asarray(res["methods"][m]["E3_logits"])[r3] for m in cand]); A = (Z > 0).astype(int)
                err = (A != Ct[None]).mean(1)                                        # (R,3) true per-concept errors
                terr = (A != Ct[None]).mean((1, 2))                                  # (R,) true triple errors
                conf = (1 / (1 + np.exp(-np.abs(Z)))).mean(1)                        # mean confidence per concept
                mv = (A.mean(0) > .5).astype(int); agree = (A == mv[None]).mean(1)
                for acn, (_aspec, afn) in AUDIT_SPECS.items():
                    labs = labels_of(afn, Ct); F = forced_from_labels(afn, labs)
                    Wm, UP, DU = cert_tables(A, F)
                    CR = np.maximum(UP, 0).max(1)                                    # (R,3) sharp certified regret of each reader
                    consist = np.array([np.mean([l == m for l, m in zip(labels_of(afn, A[r]), labs)]) for r in range(len(cand))])
                    for j in range(3):
                        a1 = np.array([a1_point(A[r, :, j], F[:, j]) for r in range(len(cand))]); a1 = np.where(np.isnan(a1), np.inf, a1)
                        picks = {"S": int(np.argmin(Wm[:, j])), "S_dagger": int(np.lexsort((Wm[:, j], CR[:, j]))[0]),
                                 "A1_v6": int(np.argmin(a1)), "confidence": int(np.argmax(conf[:, j])),
                                 "majority_agreement": int(np.argmax(agree[:, j])), "label_consistency": int(np.argmax(consist))}
                        best = float(err[:, j].min()); PU = float(np.mean(F[:, j] < 0))
                        sel.append({"triple": ti, "code": cname, "audit_code": acn, "concept": j, "PU": PU, "best_err": best,
                                    "zeroshot_err": float(err[cand.index("zeroshot"), j]),
                                    **{f"regret_{k}": float(err[v, j] - best) for k, v in picks.items()},
                                    "cert_S": float(CR[picks["S"], j]), "cert_S_dagger": float(CR[picks["S_dagger"], j])})
                        reg = err[:, j] - best
                        viol = int(np.sum(reg > CR[:, j] + 1e-12))
                        sharp = 0.0                                                    # realise each reader's certificate exactly
                        for r in range(len(cand)):
                            s = int(np.argmax(UP[r, :, j])); c_star = worst_truth(A, F, r, s, j)
                            e_star = (A[:, :, j] != c_star[None]).mean(1)
                            sharp = max(sharp, abs((e_star[r] - e_star.min()) - CR[r, j]))
                        off = ~np.eye(len(cand), dtype=bool); certified = (UP[:, :, j] < -1e-9) & off
                        td = err[:, j][:, None] - err[:, j][None]
                        cert.append({"audit_code": acn, "viol": viol, "sharp_gap": float(sharp), "PU": PU,
                                     "mean_cert": float(CR[:, j].mean()), "mean_true_regret": float(reg.mean()),
                                     "pairs": int(off.sum()), "certified": int(certified.sum()), "certified_wrong": int((certified & (td >= 0)).sum())})
                    UJ = cert_joint(A, labs, afn); CRJ = np.maximum(UJ, 0).max(1)
                    CRsum = np.maximum(UP.mean(2), 0).max(1)                          # per-concept bound summed (ignores the fibre coupling)
                    pj = int(np.lexsort((Wm.mean(1), CRJ))[0]); tbest = float(terr.min())
                    pairs.append({"audit_code": acn, "viol": int(np.sum(terr - tbest > CRJ + 1e-12)), "joint_cert": float(CRJ.mean()),
                                  "sum_cert": float(CRsum.mean()), "pick_regret": float(terr[pj] - tbest), "pick_cert": float(CRJ[pj]),
                                  "S_regret": float(terr[int(np.argmin(Wm.mean(1)))] - tbest)})
        return sel, cert, pairs


    if have("s13b_v8.json"):
        V8R = load_json(ckpt("s13b_v8.json"))
    else:
        _sel, _cert, _pairs = v8_real()
        V8R = {"sel": _sel, "cert": _cert, "joint": _pairs}; save_json(ckpt("s13b_v8.json"), V8R); hf_push(RUN_NAME, force=True)

    V8_METHODS = ["S_dagger", "S", "A1_v6", "label_consistency", "majority_agreement", "confidence"]
    G9_VIOL = sum(c["viol"] for c in V8R["cert"]) + sum(c["viol"] for c in V8R["joint"])
    G10_WRONG = sum(c["certified_wrong"] for c in V8R["cert"])
    G11_GAP = max(c["sharp_gap"] for c in V8R["cert"])
    V8_SEL = {}
    for _acn in ("design12", "design2", "design1"):
        _s = [r for r in V8R["sel"] if r["audit_code"] == _acn]; _c = [r for r in V8R["cert"] if r["audit_code"] == _acn]
        _j = [r for r in V8R["joint"] if r["audit_code"] == _acn]
        V8_SEL[_acn] = {"cells": len(_s), "PU": float(np.mean([r["PU"] for r in _s])), "best_err": float(np.mean([r["best_err"] for r in _s])),
                        "regret": {m: float(np.mean([r[f"regret_{m}"] for r in _s])) for m in V8_METHODS},
                        "cert_S_dagger": float(np.mean([r["cert_S_dagger"] for r in _s])), "cert_S": float(np.mean([r["cert_S"] for r in _s])),
                        "certified_pairs": float(sum(r["certified"] for r in _c) / max(1, sum(r["pairs"] for r in _c))),
                        "joint_cert": float(np.mean([r["joint_cert"] for r in _j])), "sum_cert": float(np.mean([r["sum_cert"] for r in _j])),
                        "joint_pick_regret": float(np.mean([r["pick_regret"] for r in _j])), "joint_pick_cert": float(np.mean([r["pick_cert"] for r in _j]))}
    print(f"G9  true regret <= sharp certificate: {G9_VIOL} violations | G10 certified orderings wrong: {G10_WRONG} | G11 max |worst-case regret - certificate| = {G11_GAP:.2e}")
    for _acn, _v in V8_SEL.items():
        print(f"\naudit code {_acn:8s} ({_v['cells']} cells): best reader error {_v['best_err']:.4f} | P(U) {_v['PU']:.3f} | "
              f"certificate of S-dagger pick {_v['cert_S_dagger']:.3f} (S pick {_v['cert_S']:.3f}) | pairs certified {_v['certified_pairs']:.3f}")
        print("   mean selection regret:", " | ".join(f"{m} {_v['regret'][m]:.4f}" for m in V8_METHODS))
        print(f"   triple level: joint certificate {_v['joint_cert']:.3f} vs per-concept sum {_v['sum_cert']:.3f} | joint pick regret {_v['joint_pick_regret']:.4f} <= cert {_v['joint_pick_cert']:.3f}")
    return (G10_WRONG, G11_GAP, G9_VIOL, V8_SEL,)


@app.cell
def _(G10_WRONG, G11_GAP, G9_VIOL, RUN_NAME, V8_SEL, ckpt, hf_push, save_json):
    # ---- 13c. v8 verdicts and Pre-registration 6 (for a FRESH run: new triples / seeds / encoder) ----
    _d12, _d2, _d1 = V8_SEL["design12"], V8_SEL["design2"], V8_SEL["design1"]
    _base = ["A1_v6", "label_consistency", "majority_agreement", "confidence"]
    V8V = {
        "G9": ("GUAR", G9_VIOL == 0, "true regret <= sharp certificate CR on every cell, per concept and per triple (violations)", G9_VIOL),
        "G10": ("GUAR", G10_WRONG == 0, "every assumption-free certified ordering is correct (wrong orderings)", G10_WRONG),
        "G11": ("GUAR", G11_GAP < 1e-9, "sharpness: the constructed worst-case truth attains CR exactly (max gap)", G11_GAP),
        "X5": ("EXPL", _d12["cert_S_dagger"] <= 0.5 * _d12["PU"], "minimax certificate is at most half of v7's bound P(U) on design12",
               {"cert": _d12["cert_S_dagger"], "PU": _d12["PU"]}),
        "X6": ("EXPL", all(_d["regret"]["S"] < _d["regret"][m] for _d in (_d12, _d2) for m in _base),
               "S beats every label-free baseline in realised regret (design12 and design2)", {k: V8_SEL[k]["regret"] for k in V8_SEL}),
        "X7": ("EXPL", _d12["certified_pairs"] >= 0.25, "at least 25% of reader pairs certified assumption-free on design12",
               {k: V8_SEL[k]["certified_pairs"] for k in V8_SEL}),
        "X8": ("EXPL", _d1["joint_cert"] < _d1["sum_cert"], "knowledge coupling: joint triple certificate tighter than per-concept sum (design1)",
               {k: (V8_SEL[k]["joint_cert"], V8_SEL[k]["sum_cert"]) for k in V8_SEL}),
    }
    for _k, (_kind, _ok, _desc, _val) in V8V.items():
        _tag = ("PASS" if _ok else "FAIL") if _kind == "GUAR" else ("YES " if _ok else "NO  ")
        print(f"{_tag:8s} [{_kind}] {_k} {_desc}\n         {_val}")
    print("\nPre-registration 6 (to be scored on a fresh run, never on this one):\n"
          "  P14 G9-G11 hold on every cell (guarantees).\n"
          "  P15 minimax certificate CR(S-dagger) <= 0.35 x P(U), pooled, on design12.\n"
          "  P16 S has lower realised regret than A1, label consistency, majority agreement and confidence on design12.\n"
          "  P17 >= 25% of reader pairs certified assumption-free on design12.\n"
          "  P18 joint triple certificate < per-concept sum on design1.")
    save_json(ckpt("s13c_v8_verdicts.json"), {k: [v[0], bool(v[1]), v[2], v[3]] for k, v in V8V.items()}); hf_push(RUN_NAME, force=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. v9 — partial identification of perception from symbolic knowledge (the main theory) and experiment E-A at scale

    **Connecting v6–v8.** Every result that held on real CLIP was *assumption-free*: A0 (0 violations), S (0) and C (0, attained to
    1e-16). Every result that failed used an assumption about CLIP's errors: A1 locality (refuted in 64–73% of cells) and A1\*
    (worse than A1). The lesson is the paradigm: **knowledge does not measure a model, it bounds it, and the sharp bounds are
    enough to decide.** A0, S, C, D, A1, A1\* and the exactness of MNIST-Add are one object, the *identified set*, seen at
    different rungs of an assumption ladder. Full proofs are in `theory_v9.md`.

    **T1 (fibre principle).** Under (K), "truth ∈ fibre of its label", the identified set of any additive evaluation functional is
    the Minkowski sum of per-image fibre images. Its extremes are per-image min and max. Its convex hull is sharp up to O(d/n)
    (Shapley–Folkman). A0, C and the joint certificate are corollaries.

    **T2 (minimax).** CR(r) is the exact worst-case regret of deploying r, so argmin CR is optimal among *all* label-free
    selectors. No method using the same information can certify less regret. Certified rankings and a certified top set, which
    always contains the best reader, come for free.

    **T3 (logic beats tables).** The joint-over-per-concept gain Γ is zero for every box fibre (class-attribute tables), and positive
    exactly when rules tie concepts together. *Measured:* Γ > 0 on design1 (0.240 → 0.206 on ViT-L/14), and Γ = 0 on design2 and
    design12.

    **T4 (imperfect knowledge, sharp).** If the knowledge is wrong on at most k images, the sharp bound adds the top-k of per-image
    gains. For differences this is ≤ 2k/n, and it charges *only* forced disagreements.

    **T5 (certify the knowledge with a few labels, fast rate).** A Clopper–Pearson bound on the violation rate from m labels, then
    a Binomial quantile, gives k⁺. All certificates then hold simultaneously with probability ≥ 1 − δ. For exact knowledge the slack
    is ≈ ln(2/δ)/m (rate 1/m), against Ω(√(D/m)) for any label-only certificate.

    **T6 (knowledge-stratified evaluation, KSE).** Images where all readers agree never need labels, and forced disagreements are
    settled by the knowledge. Label only *unforced disagreements*. The labels needed for precision η drop from D/η² to
    (|C|/n)²/η².

    **T7 (referee).** Projecting any label-free estimate (ATC, confidence, A1, Dawid–Skene) onto the robust identified set never
    increases its error when the truth is in the set. Landing outside the set refutes the estimator's assumption.

    **Experiment E-A (natural, noisy knowledge at scale).**
    - **Audit set:** every CUB *test* image (5,794 images, 200 species), all 86 concepts, scored where the concept label is reliable.
    - **Knowledge:** a class-attribute "field guide" computed from *train* images only. With threshold τ, a species forces an
      attribute if at least 1−τ of its train images have it (or at most τ do).
    - **Readers:** an encoder zoo (up to 10 CLIP-family models), each with three heads: source-calibrated zero-shot, linear probe,
      and 16-shot probe. All heads are trained on E1-family train images only, so most audit species are a distribution shift.
    - **Baselines:** source-validation accuracy, ATC (Garg et al. 2022), mean confidence, majority agreement, and selection from
      m = 100 random concept labels. KSE is compared with random labelling and active testing (labelling all disagreements).
    """)
    return


@app.cell
def _(CFG, CLIP_MEAN, CLIP_STD, CNAMES, CUB, DEV, FEATS, Image, N_IMG, PATHS, RUN_NAME, SMOKE, TXT, atomic_write, check_deadline, ckpt, clip_image_embed, clip_text_embed, gpu_free, hf_push, load_clip, np, os, re, rss_gb, torch, trim_ram, vram_report):
    # ---- 14b. v9 encoder zoo: image + concept-text embeddings for every CLIP-family model (cached, resumable, skips failures) ----
    V9_ZOO = [m for m in os.environ.get("V9_ZOO", "").split(",") if m] or [
        "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14",
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "facebook/metaclip-b32-400m", "facebook/metaclip-l14-fullcc2.5b",
        "facebook/metaclip-h14-fullcc2.5b"]
    V9_TEMPLATES = ["a photo of a bird whose {part} is {val}.", "a close-up photo of a bird; its {part} is {val}.", "a bird with {part}: {val}."]


    def v9_prompt(name, tmpl):
        part, val = name.split("::") if "::" in name else (name, "")
        return tmpl.format(part=part.replace("has_", "").replace("_", " ").strip(),
                           val=val.replace("_", " ").replace("(", "").replace(")", "").strip())


    def v9_load_img(rel, size=224):
        im = Image.open(os.path.join(CUB, "images", rel)).convert("RGB")
        w, h = im.size; s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        w, h = im.size; l, t = (w - size) // 2, (h - size) // 2
        return np.asarray(im.crop((l, t, l + size, t + size)), dtype=np.float32) / 255.0


    def v9_encode(name):
        """(N, d) normalised image embeddings and (K, d) prompt-ensemble concept directions for one model, cached per model."""
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[-60:]
        fi, ft = ckpt(f"zoo_{slug}_img.npy"), ckpt(f"zoo_{slug}_txt.npy")
        if os.path.exists(fi) and os.path.exists(ft):
            return np.load(fi), np.load(ft)
        if name == CFG["encoder"] and not SMOKE:
            img, txt = FEATS, TXT                                              # the main run already encoded this model
        elif SMOKE:
            r = np.random.default_rng(sum(map(ord, name))); Q = np.linalg.qr(r.normal(size=(FEATS.shape[1],) * 2))[0]
            img = FEATS @ Q + r.normal(size=FEATS.shape) * r.uniform(.02, .08); txt = TXT @ Q + r.normal(size=TXT.shape) * r.uniform(.05, .2)
            img /= np.linalg.norm(img, axis=1, keepdims=True); txt /= np.linalg.norm(txt, axis=1, keepdims=True)
        else:
            model, tok = load_clip(name)
            mu = torch.tensor(CLIP_MEAN, device=DEV).view(1, 3, 1, 1); sd = torch.tensor(CLIP_STD, device=DEV).view(1, 3, 1, 1)
            cdir = ckpt(f"zoo_{slug}_chunks"); os.makedirs(cdir, exist_ok=True)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(8) as pool:
                for c0 in range(0, N_IMG, 2048):
                    cf = os.path.join(cdir, f"c{c0:06d}.npy")
                    if os.path.exists(cf):
                        continue
                    check_deadline(); res = []
                    for b0 in range(c0, min(c0 + 2048, N_IMG), CFG["img_batch"]):
                        arr = np.stack(list(pool.map(v9_load_img, PATHS[b0:min(b0 + CFG["img_batch"], c0 + 2048, N_IMG)])))
                        with torch.inference_mode():
                            f = clip_image_embed(model, (torch.from_numpy(arr).permute(0, 3, 1, 2).to(DEV) - mu) / sd)
                        res.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
                    atomic_write(cf, lambda t_, a=np.concatenate(res): np.save(t_, a))
                    print(f"  [{slug}] {min(c0 + 2048, N_IMG)}/{N_IMG} | VRAM {vram_report()} | RAM {rss_gb():.2f} GB", flush=True)
            img = np.concatenate([np.load(os.path.join(cdir, f)) for f in sorted(os.listdir(cdir))])
            with torch.inference_mode():
                txt = sum(torch.nn.functional.normalize(clip_text_embed(model, tok, [v9_prompt(n, tm) for n in CNAMES]), dim=1)
                          for tm in V9_TEMPLATES)
            txt = torch.nn.functional.normalize(txt, dim=1).cpu().numpy()
            del model; gpu_free()
        atomic_write(fi, lambda t_: np.save(t_, img.astype(np.float32))); atomic_write(ft, lambda t_: np.save(t_, txt.astype(np.float32)))
        hf_push(RUN_NAME)
        return img.astype(np.float32), txt.astype(np.float32)


    ZOO_FEATS = {}
    for _name in (V9_ZOO[:3] if SMOKE else V9_ZOO):
        try:
            ZOO_FEATS[_name] = v9_encode(_name)
            print(f"zoo {_name}: image {ZOO_FEATS[_name][0].shape}, text {ZOO_FEATS[_name][1].shape}")
        except Exception as _e:
            print(f"zoo {_name}: skipped ({type(_e).__name__}: {str(_e)[:160]})"); gpu_free()
    print(f"encoder zoo: {len(ZOO_FEATS)} models | RAM {trim_ram():.2f} GB")
    return (ZOO_FEATS,)


@app.cell
def _(CFG, E1_TR, IS_TRAIN, LABELS, N_CLS, RELIABLE, RUN_NAME, Y_INST, ZOO_FEATS, ckpt, have, hf_push, load_json, np, save_json, torch):
    # ---- 14c. v9 experiment E-A: full CUB, natural (noisy) class knowledge, encoder zoo x 3 heads. Theorems 2, 4, 5, 6, 7. ----
    from scipy.stats import hypergeom as v9_hyper

    V9 = dict(taus=(0.0, 0.1, 0.2), tau_main=0.1, delta=0.05, m_cal=100, m_grid=(10, 25, 50, 100, 200), reps=20, few=16, min_cls_imgs=5)


    def v9_fit(X, Y, M, one_d=False, l2=1e-3, steps=300, lr=0.05, seed=0):
        """Batched logistic regression for all K concepts at once. X (n,d) shared features, or (n,K) per-concept scores if one_d.
        Y, M (n,K): labels and a 0/1 mask of which samples train which concept. Returns a function features -> probabilities (n,K)."""
        torch.manual_seed(seed); X = torch.as_tensor(X, dtype=torch.float32); Y = torch.as_tensor(Y, dtype=torch.float32)
        M = torch.as_tensor(M, dtype=torch.float32); K = Y.shape[1]
        W = torch.zeros(1 if one_d else X.shape[1], K, requires_grad=True); b = torch.zeros(K, requires_grad=True)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        opt = torch.optim.Adam([W, b], lr=lr)
        lin = (lambda Z: ((Z - mu) / sd) * W[0] + b) if one_d else (lambda Z: ((Z - mu) / sd) @ W + b)
        for _ in range(steps):
            opt.zero_grad()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(lin(X), Y, reduction="none") * M).sum(0) / M.sum(0).clamp(min=1)
            (loss.sum() + l2 * (W ** 2).sum()).backward(); opt.step()
        W, b = W.detach(), b.detach()
        return lambda Z: torch.sigmoid(lin(torch.as_tensor(Z, dtype=torch.float32))).numpy()


    def v9_readers():
        """Every encoder x {source-calibrated zero-shot, linear probe, 16-shot probe}, trained on E1-train concept labels only."""
        rng = np.random.default_rng(CFG["seed"] + 91); tr = rng.permutation(E1_TR); val, fit = tr[: len(tr) // 5], tr[len(tr) // 5:]
        Yf, Mf = Y_INST[fit].astype(float), RELIABLE[fit].astype(float)
        Mfew = np.zeros_like(Mf)
        for j in range(Yf.shape[1]):
            for v in (0, 1):
                ok = np.nonzero((Mf[:, j] > 0) & (Yf[:, j] == v))[0]
                Mfew[rng.choice(ok, min(V9["few"], len(ok)), replace=False), j] = 1
        names, probs = [], []
        for mname, (img, txt) in ZOO_FEATS.items():
            short = mname.rstrip("/").split("/")[-1]
            S = img @ txt.T
            heads = {"zs": (v9_fit(S[fit], Yf, Mf, one_d=True), S), "lin": (v9_fit(img[fit], Yf, Mf), img),
                     "few": (v9_fit(img[fit], Yf, Mfew), img)}
            for h, (f, Z) in heads.items():
                names.append(f"{short}:{h}"); probs.append(f(Z).astype(np.float32))
        return names, np.stack(probs), val


    def v9_knowledge(tau):
        """Class-attribute table from TRAIN images only (a field guide); forced 1 if >= 1-tau of a class has it, 0 if <= tau."""
        K = Y_INST.shape[1]; F = np.full((N_CLS, K), -1, np.int64)
        for c in range(N_CLS):
            sel = (LABELS == c) & IS_TRAIN
            for j in range(K):
                m = sel & RELIABLE[:, j]
                if m.sum() >= V9["min_cls_imgs"]:
                    p = Y_INST[m, j].mean()
                    F[c, j] = 1 if p >= 1 - tau else (0 if p <= tau else -1)
        return F


    def v9_kplus(viol, m, N, delta):
        """Theorems 4-5, sharpened: only forced images where readers disagree (B, size N) can move a certificate. From m labels
        drawn without replacement from B with `viol` violations, the exact hypergeometric upper bound on violations in B."""
        if m >= N:
            return viol
        lo, hi = viol, N                                                           # largest K with P(X <= viol | K) > delta
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if v9_hyper.cdf(viol, N, mid, m) > delta else (lo, mid - 1)
        return lo


    def v9_tables(A, f, k):
        """Theorems C + 4 on one concept. A (R,n) binary, f (n,) forced value or -1. Returns forced-wrong rates, robust UP (R,R)."""
        n = A.shape[1]; fm = f >= 0; um = ~fm
        W = ((A != f[None]) & fm[None]).mean(1)
        DU = ((A[:, None, :] != A[None, :, :]) & um[None, None]).sum(-1) / n
        agree_f = (A == f[None]) & fm[None]
        cnt = (agree_f[:, None, :] & (A[:, None, :] != A[None, :, :])).sum(-1)      # forced, disagree, r agrees with knowledge
        UP = W[:, None] - W[None] + DU + 2 * np.minimum(k, cnt) / n
        return W, UP


    def v9_project(A, f, k, e_hat, iters=400):
        """Theorem 7: Euclidean projection of an error-vector estimate onto the k-robust identified set (FISTA over image groups)."""
        R, n = A.shape; code = (A.T.astype(np.int64) * (1 << np.arange(R, dtype=np.int64))).sum(1) * 3 + (f + 1)
        _, first, cnt = np.unique(code, return_index=True, return_counts=True)
        ag, fg = A[:, first].astype(float), f[first]; cnt = cnt.astype(float)
        base = A.sum(1) / n; Mx = (1 - 2 * ag) / n; L = np.linalg.norm(Mx, 2) ** 2 + 1e-12
        unf, f1, f0 = fg < 0, fg == 1, fg == 0

        def proj(w):
            w = np.clip(w, 0, cnt); y = np.where(f1, cnt - w, np.where(f0, w, 0.0))
            if y.sum() > k:
                lo, hi = 0.0, y.max()
                for _ in range(60):
                    th = (lo + hi) / 2; lo, hi = (th, hi) if np.clip(y - th, 0, cnt).sum() > k else (lo, th)
                y = np.where(unf, 0.0, np.clip(y - hi, 0, cnt))
            return np.where(unf, w, np.where(f1, cnt - y, y))

        w = proj(np.where(f1, cnt, np.where(f0, 0.0, cnt / 2))); z, t = w.copy(), 1.0
        for _ in range(iters):
            w_new = proj(z - Mx.T @ (base + Mx @ z - e_hat) / L)
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2; z = w_new + (t - 1) / t_new * (w_new - w); w, t = w_new, t_new
        return base + Mx @ w


    def v9_atc(p_val, y_val, p_aud):
        """ATC (Garg et al. 2022): threshold on source confidence matching source accuracy; predicted target error."""
        cv, ca = np.maximum(p_val, 1 - p_val), np.maximum(p_aud, 1 - p_aud)
        acc = np.mean((p_val > .5) == y_val); t = np.quantile(cv, 1 - acc) if 0 < acc < 1 else 0.5
        return float(1 - np.mean(ca > t))


    def v9_run():
        names, P, val = v9_readers(); R = len(names); A_all = (P > .5).astype(np.int8)
        AUD = np.nonzero(~IS_TRAIN)[0]; rng = np.random.default_rng(CFG["seed"] + 92); rows, curves, projs = [], [], []
        for tau in V9["taus"]:
            FK = v9_knowledge(tau)
            for j in range(Y_INST.shape[1]):
                idx = AUD[RELIABLE[AUD, j]]; y = Y_INST[idx, j].astype(int); n = len(idx)
                if n < 200:
                    continue
                A = A_all[:, idx, j]; Pj = P[:, idx, j]; f = FK[LABELS[idx], j]
                err = (A != y[None]).mean(1); best = err.min()
                dis = ~np.all(A == A[:1], axis=0); Bset = np.nonzero(dis & (f >= 0))[0]; Cset = np.nonzero(dis & (f < 0))[0]
                cal = rng.choice(Bset, min(V9["m_cal"], len(Bset)), replace=False) if len(Bset) else Bset
                k = v9_kplus(int(np.sum(f[cal] != y[cal])), len(cal), len(Bset), V9["delta"]); eps = k / max(1, len(Bset))
                true_viol = int(np.sum(f[Bset] != y[Bset])); rnd = rng.choice(n, V9["m_cal"], replace=False)
                W, UPk = v9_tables(A, f, k); _, UP0 = v9_tables(A, f, 0)
                CRk, CR0 = np.maximum(UPk, 0).max(1), np.maximum(UP0, 0).max(1)
                vm = RELIABLE[val, j]; yv = Y_INST[val[vm], j].astype(int)
                src_acc = np.array([np.mean((P[r, val[vm], j] > .5) == yv) for r in range(R)])
                atc = np.array([v9_atc(P[r, val[vm], j], yv, Pj[r]) for r in range(R)])
                conf = np.maximum(Pj, 1 - Pj).mean(1); mv = (A.mean(0) > .5); agree = (A == mv[None]).mean(1)
                picks = {"S_dagger": int(np.lexsort((W, CRk))[0]), "S": int(np.argmin(W)), "source_acc": int(np.argmax(src_acc)),
                         "ATC": int(np.argmin(atc)), "confidence": int(np.argmax(conf)), "majority_agreement": int(np.argmax(agree)),
                         f"labels_{V9['m_cal']}": int(np.argmin((A[:, rnd] != y[rnd][None]).mean(1)))}
                off = ~np.eye(R, dtype=bool); cert = (UPk < -1e-9) & off; td = err[:, None] - err[None]
                topset = ~np.any(cert, axis=0)                                         # readers no one is certified better than
                picks["source_acc_in_top_set"] = int(np.argmax(np.where(topset, src_acc, -1)))   # knowledge vetoes, source decides
                rows.append({"tau": tau, "concept": j, "n": n, "PU": float(np.mean(f < 0)), "B_frac": len(Bset) / n,
                             "true_eps": float(np.mean(f[f >= 0] != y[f >= 0])) if (f >= 0).any() else 0.0, "true_eps_B": true_viol / max(1, len(Bset)),
                             "eps_plus": eps, "k_over_n": k / n, "true_viol_over_n": true_viol / n, "best_err": float(best),
                             **{f"regret_{m}": float(err[r] - best) for m, r in picks.items()},
                             "cert_S_dagger": float(CRk[picks["S_dagger"]]), "valid_robust": bool(np.all(err - best <= CRk + 1e-12)),
                             "valid_naive": bool(np.all(err - best <= CR0 + 1e-12)), "certified_pairs": float(cert.sum() / off.sum()),
                             "certified_wrong": int((cert & (td >= 0)).sum()), "top_set_size": int(topset.sum()),
                             "top_set_has_best": bool(topset[int(np.argmin(err))])})
                if tau != V9["tau_main"]:
                    continue
                Dset = np.nonzero(dis)[0]
                for m in V9["m_grid"]:
                    for rep in range(V9["reps"]):
                        rr = rng.choice(n, m, replace=False)
                        dd = rng.choice(Dset, min(m, len(Dset)), replace=False) if len(Dset) else Dset
                        cc = rng.choice(Cset, min(m, len(Cset)), replace=False) if len(Cset) else Cset
                        e_kse = (A[:, Bset] != f[Bset][None]).sum(1) + (len(Cset) / max(1, len(cc))) * (A[:, cc] != y[cc][None]).sum(1)
                        est = {"random": (A[:, rr] != y[rr][None]).mean(1), "active_testing": (A[:, dd] != y[dd][None]).mean(1),
                               "KSE": e_kse}
                        curves.append({"concept": j, "m": m, "rep": rep, **{kk: float(err[int(np.argmin(v))] - best) for kk, v in est.items()}})
                for est_name, e_hat in {"ATC": atc, "confidence": 1 - conf}.items():
                    e_proj = v9_project(A, np.where(dis, f, -1), k, e_hat)       # agreeing images: truth free (shifts all readers alike)
                    projs.append({"concept": j, "estimator": est_name, "mae_raw": float(np.abs(e_hat - err).mean()),
                                  "mae_proj": float(np.abs(e_proj - err).mean()), "l2_raw": float(np.linalg.norm(e_hat - err)),
                                  "l2_proj": float(np.linalg.norm(e_proj - err)), "truth_in_set": true_viol <= k,
                                  "refuted": bool(np.abs(e_proj - e_hat).max() > 0.005)})
        return {"readers": names, "rows": rows, "curves": curves, "proj": projs}


    if have("s14c_v9.json"):
        V9R = load_json(ckpt("s14c_v9.json"))
    else:
        V9R = v9_run(); save_json(ckpt("s14c_v9.json"), V9R); hf_push(RUN_NAME, force=True)

    V9_SELECTORS = ["S_dagger", "S", "source_acc_in_top_set", f"labels_{V9['m_cal']}", "source_acc", "ATC", "confidence", "majority_agreement"]
    V9_SUM = {}
    for _tau in V9["taus"]:
        _r = [x for x in V9R["rows"] if x["tau"] == _tau]
        V9_SUM[_tau] = {"concepts": len(_r), "PU": float(np.mean([x["PU"] for x in _r])), "true_eps": float(np.mean([x["true_eps"] for x in _r])),
                        "eps_plus": float(np.mean([x["eps_plus"] for x in _r])), "B_frac": float(np.mean([x["B_frac"] for x in _r])),
                        "true_eps_B": float(np.mean([x["true_eps_B"] for x in _r])), "valid_robust": float(np.mean([x["valid_robust"] for x in _r])),
                        "valid_naive": float(np.mean([x["valid_naive"] for x in _r])), "cert_S_dagger": float(np.mean([x["cert_S_dagger"] for x in _r])),
                        "certified_pairs": float(np.mean([x["certified_pairs"] for x in _r])), "certified_wrong": int(sum(x["certified_wrong"] for x in _r)),
                        "top_set_size": float(np.mean([x["top_set_size"] for x in _r])), "top_set_has_best": float(np.mean([x["top_set_has_best"] for x in _r])),
                        "best_err": float(np.mean([x["best_err"] for x in _r])), "regret": {s: float(np.mean([x[f"regret_{s}"] for x in _r])) for s in V9_SELECTORS}}
    V9_CURVE = {m: {k: float(np.mean([c[k] for c in V9R["curves"] if c["m"] == m])) for k in ("random", "active_testing", "KSE")} for m in V9["m_grid"]}
    V9_PROJ = {e: {k: float(np.mean([p[k] for p in V9R["proj"] if p["estimator"] == e])) for k in ("mae_raw", "mae_proj", "refuted")}
               | {"never_worse_when_truth_in_set": bool(all(p["l2_proj"] <= p["l2_raw"] + 1e-6 for p in V9R["proj"] if p["estimator"] == e and p["truth_in_set"]))}
               for e in ("ATC", "confidence")}
    print(f"E-A: {len(V9R['readers'])} readers ({len(ZOO_FEATS)} encoders x 3 heads) | audit = all CUB test images, per concept reliable labels")
    for _tau, _v in V9_SUM.items():
        print(f"\ntau {_tau}: {_v['concepts']} concepts | P(U) {_v['PU']:.3f} | true knowledge error {_v['true_eps']:.3f} | "
              f"forced disagreements B {_v['B_frac']:.3f} (knowledge error there {_v['true_eps_B']:.3f}, certified {_v['eps_plus']:.3f}) | "
              f"certificate valid: robust {_v['valid_robust']:.3f} vs naive {_v['valid_naive']:.3f}")
        print(f"   minimax certificate {_v['cert_S_dagger']:.3f} | pairs certified {_v['certified_pairs']:.3f} (wrong {_v['certified_wrong']}) | "
              f"certified top set {_v['top_set_size']:.1f} readers, contains best {_v['top_set_has_best']:.3f} | best error {_v['best_err']:.4f}")
        print("   mean selection regret:", " | ".join(f"{s} {_v['regret'][s]:.4f}" for s in V9_SELECTORS))
    print(f"\nlabel efficiency at tau {V9['tau_main']} (mean regret of the reader picked with m concept labels):")
    for _m, _v in V9_CURVE.items():
        print(f"   m={_m:4d}: random {_v['random']:.4f} | active testing {_v['active_testing']:.4f} | KSE (knowledge-stratified) {_v['KSE']:.4f}")
    print("projection repair (Theorem 7):", V9_PROJ)
    return (V9, V9_CURVE, V9_PROJ, V9_SUM,)


@app.cell
def _(RUN_NAME, V9, V9_CURVE, V9_PROJ, V9_SUM, ckpt, hf_push, np, save_json):
    # ---- 14d. v9 verdicts and Pre-registration 7 (for a FRESH run: new seed, the full 10-encoder zoo) ----
    _vm = V9_SUM[V9["tau_main"]]
    _viol_ok = all(V9_SUM[t]["valid_robust"] >= 1 - V9["delta"] - 2 * np.sqrt(V9["delta"] * (1 - V9["delta"]) / max(1, V9_SUM[t]["concepts"])) for t in V9_SUM)
    _base9 = ["source_acc", "ATC", "confidence", "majority_agreement"]
    V9V = {
        "G12": ("GUAR", _viol_ok, "robust certificate (T4+T5) valid on >= 1-delta of concepts at every tau (up to 2 binomial s.d.)",
                {t: V9_SUM[t]["valid_robust"] for t in V9_SUM}),
        "G13": ("GUAR", all(V9_SUM[t]["certified_wrong"] == 0 for t in V9_SUM), "no certified ordering is wrong (k-robust)",
                {t: V9_SUM[t]["certified_wrong"] for t in V9_SUM}),
        "G14": ("GUAR", all(V9_PROJ[e]["never_worse_when_truth_in_set"] for e in V9_PROJ), "projection never increases L2 error when truth is in the set (T7)",
                {e: V9_PROJ[e]["never_worse_when_truth_in_set"] for e in V9_PROJ}),
        "X9": ("EXPL", any(V9_SUM[t]["valid_naive"] < V9_SUM[t]["valid_robust"] for t in V9_SUM), "natural knowledge is noisy: the naive (k=0) certificate breaks, the robust one does not",
               {t: (V9_SUM[t]["valid_naive"], V9_SUM[t]["valid_robust"]) for t in V9_SUM}),
        "X10": ("EXPL", min(_vm["regret"]["S"], _vm["regret"]["S_dagger"]) < min(_vm["regret"][b] for b in _base9),
                "knowledge selection (S or S-dagger) beats every label-free baseline at tau_main", _vm["regret"]),
        "X11": ("EXPL", V9_CURVE[50]["KSE"] <= V9_CURVE[100]["random"] and V9_CURVE[100]["KSE"] <= V9_CURVE[200]["random"],
                "KSE halves the labels: m labels beat random labelling with 2m (m = 50, 100)",
                V9_CURVE),
        "X12": ("EXPL", all(V9_PROJ[e]["mae_proj"] < V9_PROJ[e]["mae_raw"] for e in V9_PROJ), "projection repairs ATC and confidence error estimates (lower MAE)", V9_PROJ),
    }
    for _k, (_kind, _ok, _desc, _val) in V9V.items():
        _tag = ("PASS" if _ok else "FAIL") if _kind == "GUAR" else ("YES " if _ok else "NO  ")
        print(f"{_tag:8s} [{_kind}] {_k} {_desc}\n         {_val}")
    print("\nPre-registration 7 (to be scored on a fresh run with the full zoo, never on this one):\n"
          "  P19 G12-G14 hold (guarantees).\n"
          "  P20 S or S-dagger has lower mean regret than source accuracy, ATC, confidence and majority agreement at tau = 0.1.\n"
          "  P21 KSE with m labels has regret <= random labelling with 2m, for m = 50 and 100.\n"
          "  P22 projection cuts the MAE of confidence-based error estimates by >= 20%.\n"
          "  P23 the certified top set contains the best reader on >= 95% of concepts.")
    save_json(ckpt("s14d_v9_verdicts.json"), {k: [v[0], bool(v[1]), v[2], v[3]] for k, v in V9V.items()}); hf_push(RUN_NAME, force=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. v10 — exact logical rules on real digits: instance-level versus population-level identification

    **Question.** CUB tables are weak knowledge (box fibres, Theorem 3). Do *exact logical rules* fix that? Experiment E-B runs eight
    programs over the same real MNIST test digits, read by the encoder zoo (each encoder gives zero-shot, 10-shot and 100-shot
    digit readers):
    - sum, product, max, |a − b|, carry and parity on pairs;
    - a **box control**, (a mod 2, b mod 2), whose fibre is a product set;
    - sum of three digits.

    Tuples are disjoint, so (K) stays separable and every bound is sharp.

    **What the theory predicts.**
    - **T1–T3 (exact knowledge, proved).** The joint certificate is valid for every reader and every program (G15). It is never
      above the per-position bound, and is exactly equal to it on the box control (G16).
    - **Instance versus population (new, non-obvious).** A sum rarely *forces* a digit: only sums of 0 and 18 do. So the
      assumption-free per-reader bracket is nearly vacuous, even though MNIST-Add is the textbook *identifiable* NeSy task. v6's
      exact MNIST-Add audit (A2) came from an **assumption**, class-conditional noise (CCN), not from the rule. *Shortcut-freeness
      is a population property; certification is an instance property.* The cell measures the gap between rung 0 (the bracket) and
      rung 3 (A2, code-constrained Dawid–Skene) on the same data.
    - **T8 (over-identification test, proved).** Under CCN every identifying program (sum, product, max, |a − b|) has the same
      population solution: the reader's confusion matrix. So the spread of A2 estimates *across programs* is a label-free test of
      CCN, the analogue of the Sargan–Hansen J test. A non-identified program (parity) should fall outside the range the
      identifying programs agree on.
    - **KSE with propagation.** Labelling one digit of a pair can force its partner through the rule, so a label can be worth two.

    *Proof of T8.* Under CCN and identification, each program's A2 estimator is consistent for the same π·diag(C). The difference
    of two consistent estimators converges to 0, so a persistent spread refutes CCN or identification. ∎
    """)
    return


@app.cell
def _(CFG, CLIP_MEAN, CLIP_STD, DEV, RUN_NAME, SMOKE, ZOO_FEATS, atomic_write, ckpt, clip_image_embed, clip_text_embed, fit_softmax, gpu_free, hf_push, load_clip, np, os, re, torch, urllib):
    # ---- 15b. v10 experiment E-B data: real MNIST through the encoder zoo; digit readers = encoder x {zero-shot, 10-shot, 100-shot} ----
    import gzip

    V10 = dict(n_test=int(os.environ.get("V10_N_TEST", 10000)), n_train=2000, seed=CFG["seed"] + 101)
    V10_MNIST = {"train-images-idx3-ubyte.gz": 16, "train-labels-idx1-ubyte.gz": 8, "t10k-images-idx3-ubyte.gz": 16, "t10k-labels-idx1-ubyte.gz": 8}
    V10_TEMPLATES = ['a photo of the number "{d}".', "a handwritten digit {d}.", "the number {w} written by hand."]
    V10_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


    def v10_mnist():
        """Real MNIST (IDX files in <work>/data/mnist, downloaded from the S3 / GCS mirrors if missing)."""
        root = os.path.join(CFG["work_dir"], "data", "mnist"); os.makedirs(root, exist_ok=True); out = {}
        for fn, off in V10_MNIST.items():
            path = os.path.join(root, fn)
            if not os.path.exists(path):
                for url in ["https://ossci-datasets.s3.amazonaws.com/mnist/" + fn, "https://storage.googleapis.com/cvdf-datasets/mnist/" + fn]:
                    try:
                        urllib.request.urlretrieve(url, path + ".part"); os.replace(path + ".part", path); break
                    except Exception as e:
                        print("  MNIST", url, "failed:", repr(e)[:120])
            with gzip.open(path, "rb") as f:
                out[fn] = np.frombuffer(f.read(), np.uint8, offset=off)
        Xtr, ytr = out["train-images-idx3-ubyte.gz"].reshape(-1, 28, 28), out["train-labels-idx1-ubyte.gz"].astype(np.int64)
        Xte, yte = out["t10k-images-idx3-ubyte.gz"].reshape(-1, 28, 28), out["t10k-labels-idx1-ubyte.gz"].astype(np.int64)
        r = np.random.default_rng(V10["seed"]); a = r.choice(len(Xtr), V10["n_train"], replace=False); b = r.permutation(len(Xte))[: V10["n_test"]]
        return Xtr[a], ytr[a], Xte[b], yte[b]


    def v10_encode(name, imgs, tag):
        """Normalised image embeddings of 28x28 digits (upsampled to 224) and 10 prompt-ensemble digit directions; cached."""
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[-60:]
        fi, ft = ckpt(f"zoo_mnist_{tag}_{slug}.npy"), ckpt(f"zoo_mnist_txt_{slug}.npy")
        if os.path.exists(fi) and os.path.exists(ft):
            return np.load(fi), np.load(ft)
        model, tok = load_clip(name)
        mu = torch.tensor(CLIP_MEAN, device=DEV).view(1, 3, 1, 1); sd = torch.tensor(CLIP_STD, device=DEV).view(1, 3, 1, 1); res = []
        for b0 in range(0, len(imgs), CFG["img_batch"]):
            x = torch.from_numpy(imgs[b0:b0 + CFG["img_batch"]]).float().to(DEV)[:, None].repeat(1, 3, 1, 1) / 255
            x = torch.nn.functional.interpolate(x, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
            with torch.inference_mode():
                res.append(torch.nn.functional.normalize(clip_image_embed(model, (x - mu) / sd), dim=1).cpu().numpy())
        with torch.inference_mode():
            T = sum(torch.nn.functional.normalize(clip_text_embed(model, tok, [tm.format(d=k, w=V10_WORDS[k]) for k in range(10)]), dim=1)
                    for tm in V10_TEMPLATES)
        T = torch.nn.functional.normalize(T, dim=1).cpu().numpy(); del model; gpu_free()
        img = np.concatenate(res).astype(np.float32)
        atomic_write(fi, lambda t_: np.save(t_, img)); atomic_write(ft, lambda t_: np.save(t_, T.astype(np.float32))); hf_push(RUN_NAME)
        return img, T.astype(np.float32)


    def v10_readers():
        """(R, n_test, 10) digit probabilities, reader names, test labels, source-validation probabilities and labels."""
        if SMOKE:
            r = np.random.default_rng(V10["seed"]); ytr, yte = r.integers(0, 10, V10["n_train"]), r.integers(0, 10, 2000)
            mu_c = r.normal(size=(10, 64)); enc = {f"smoke{i}": (s, r.normal(size=(10, 64))) for i, s in enumerate((0.9, 1.3, 1.8))}
            feats = {k: (mu_c[ytr] + r.normal(size=(len(ytr), 64)) * s, mu_c[yte] + r.normal(size=(len(yte), 64)) * s,
                         mu_c + T * 0.6) for k, (s, T) in enc.items()}
        else:
            Itr, ytr, Ite, yte = v10_mnist(); feats = {}
            for name in ZOO_FEATS:
                try:
                    Ftr, T = v10_encode(name, Itr, "train"); Fte, _ = v10_encode(name, Ite, "test"); feats[name] = (Ftr, Fte, T)
                    print(f"mnist zoo {name}: train {Ftr.shape}, test {Fte.shape}")
                except Exception as e:
                    print(f"mnist zoo {name}: skipped ({type(e).__name__}: {str(e)[:120]})"); gpu_free()
        r = np.random.default_rng(V10["seed"] + 1); perm = r.permutation(len(ytr)); fit, val = perm[: len(perm) // 2], perm[len(perm) // 2:]
        shots = {s: np.concatenate([fit[ytr[fit] == c][:s] for c in range(10)]) for s in (10, 100)}
        names, P, Pval = [], [], []
        for name, (Ftr, Fte, T) in feats.items():
            short = name.rstrip("/").split("/")[-1]
            Xtr_t, Xte_t = torch.as_tensor(Ftr, dtype=torch.float32, device=DEV), torch.as_tensor(Fte, dtype=torch.float32, device=DEV)
            heads = {"zs": (lambda Z, T=T: torch.softmax(100 * Z @ torch.as_tensor(T, dtype=torch.float32, device=DEV).T, 1))}
            for s, idx in shots.items():
                lin = fit_softmax(Xtr_t[idx], torch.as_tensor(ytr[idx], device=DEV), l2=1e-3, C=10)
                heads[f"{s}shot"] = (lambda Z, lin=lin: torch.softmax(lin(Z), 1))
            with torch.no_grad():
                for h, f in heads.items():
                    names.append(f"{short}:{h}"); P.append(f(Xte_t).cpu().numpy()); Pval.append(f(Xtr_t[val]).cpu().numpy())
        return names, np.stack(P).astype(np.float32), yte, np.stack(Pval).astype(np.float32), ytr[val]


    V10_NAMES, V10_P, V10_Y, V10_PVAL, V10_YVAL = v10_readers()
    print(f"E-B readers: {len(V10_NAMES)} | test digits {V10_P.shape[1]} | digit accuracy per reader:",
          {n: round(float((p.argmax(1) == V10_Y).mean()), 3) for n, p in zip(V10_NAMES, V10_P)})
    return (V10, V10_NAMES, V10_P, V10_PVAL, V10_Y, V10_YVAL,)


@app.cell
def _(RUN_NAME, V10, V10_NAMES, V10_P, V10_PVAL, V10_Y, V10_YVAL, ckpt, ds_pairs, have, hf_push, itertools, load_json, np, save_json, spearmanr):
    # ---- 15c. v10 experiment E-B: exact logical rules on real digits (MNIST-Add family). Theorems 1-3 and 6 with exact knowledge. ----
    V10_PROGRAMS = {                                    # name -> (arity, program). box_parity is the control: a product (box) fibre
        "product": (2, lambda v: v[0] * v[1]), "sum": (2, lambda v: v[0] + v[1]), "max": (2, lambda v: max(v)),
        "absdiff": (2, lambda v: abs(v[0] - v[1])), "carry": (2, lambda v: int(v[0] + v[1] >= 10)), "parity": (2, lambda v: (v[0] + v[1]) % 2),
        "box_parity": (2, lambda v: (v[0] % 2, v[1] % 2)), "sum3": (3, lambda v: v[0] + v[1] + v[2])}


    def v10_fibres(fn, k):
        fib = {}
        for v in itertools.product(range(10), repeat=k):
            fib.setdefault(fn(v), []).append(v)
        return {l: np.array(V, np.int64) for l, V in fib.items()}


    def v10_cert(C, labs, fib, k):
        """Sharp joint (T1) and per-position (marginal) upper bounds of err_r - err_s, and each reader's sharp error bracket
        [LO, HI], all as means over the n*k digit slots. C (R, n, k) predicted digits. Returns UPJ, UPM (R, R), LO, HI (R,)."""
        R, n = C.shape[:2]; UPJ = np.zeros((R, R)); UPM = np.zeros((R, R)); LO = np.zeros(R); HI = np.zeros(R)
        for l in np.unique(labs):
            idx = np.nonzero(labs == l)[0]; V = fib[l]
            miss = (C[:, idx, None, :] != V[None, None]).astype(np.int16)            # (R, n_l, m, k)
            H = miss.sum(-1)                                                         # (R, n_l, m)
            LO += H.min(-1).sum(-1); HI += H.max(-1).sum(-1)
            for r in range(R):
                UPJ[r] += (H[r][None] - H).max(-1).sum(-1)
                UPM[r] += (miss[r][None] - miss).max(2).sum((1, 2))
        return UPJ / (n * k), UPM / (n * k), LO / (n * k), HI / (n * k)


    def v10_run():
        A = V10_P.argmax(2); R = A.shape[0]; rng = np.random.default_rng(V10["seed"] + 2); out = {"programs": {}, "kse": []}
        src_acc = (V10_PVAL.argmax(2) == V10_YVAL[None]).mean(1); conf = V10_P.max(2).mean(1)
        mv = np.array([np.bincount(A[:, i], minlength=10).argmax() for i in range(A.shape[1])]); agree = (A == mv[None]).mean(1)
        for pname, (k, fn) in V10_PROGRAMS.items():
            n = A.shape[1] // k; slots = rng.permutation(A.shape[1])[: n * k].reshape(n, k)      # disjoint tuples: (K) stays separable
            Y = V10_Y[slots]; C = A[:, slots]; labs = np.array([hash(fn(tuple(int(x) for x in y))) for y in Y])
            fib = {hash(l): V for l, V in v10_fibres(fn, k).items()}
            forced = np.stack([[len(np.unique(fib[l][:, p])) == 1 for p in range(k)] for l in labs])     # (n, k)
            fval = np.stack([[fib[l][0, p] for p in range(k)] for l in labs])
            err = (C != Y[None]).mean((1, 2)); best = err.min()
            UPJ, UPM, LO, HI = v10_cert(C, labs, fib, k); CRJ, CRM = np.maximum(UPJ, 0).max(1), np.maximum(UPM, 0).max(1)
            W = ((C != fval[None]) & forced[None]).mean((1, 2))
            consist = np.array([np.mean([hash(fn(tuple(int(x) for x in c))) == l for c, l in zip(C[r], labs)]) for r in range(R)])
            picks = {"S_dagger": int(np.lexsort((W, CRJ))[0]), "S": int(np.argmin(W)), "label_consistency": int(np.argmax(consist)),
                     "source_acc": int(np.argmax(src_acc)), "confidence": int(np.argmax(conf)), "majority_agreement": int(np.argmax(agree))}
            off = ~np.eye(R, dtype=bool); cert = (UPJ < -1e-9) & off; td = err[:, None] - err[None]; top = ~np.any(cert, axis=0)
            out["programs"][pname] = {"arity": k, "tuples": n, "PU": float(1 - forced.mean()), "best_err": float(best),
                                      "log2_fibre": float(np.mean([np.log2(len(fib[l])) for l in labs])),
                                      "cert_joint": float(CRJ[picks["S_dagger"]]), "cert_marginal": float(CRM[picks["S_dagger"]]),
                                      "gain_mean": float((CRM - CRJ).mean()), "gain_min": float((UPM - UPJ).min()),
                                      "valid": bool(np.all(err - best <= CRJ + 1e-12)), "certified_pairs": float(cert.sum() / off.sum()),
                                      "certified_wrong": int((cert & (td >= 0)).sum()), "top_set": int(top.sum()), "top_has_best": bool(top[int(np.argmin(err))]),
                                      "regret": {m: float(err[r] - best) for m, r in picks.items()}}
            if k == 2:                                                                  # assumption ladder: code-constrained Dawid-Skene (A2)
                lid = {l: i for i, l in enumerate(fib)}; M = np.zeros((len(fib), 10, 10), bool)
                for l, V in fib.items():
                    M[lid[l], V[:, 0], V[:, 1]] = True
                li = np.array([lid[l] for l in labs]); a2 = np.zeros(R)
                for r in range(R):
                    Cm, pi = ds_pairs(C[r, :, 0], C[r, :, 1], li, M); a2[r] = 1 - float(pi @ np.diag(Cm))
                out["programs"][pname]["ladder"] = {"bracket_width": float((HI - LO).mean()), "truth_in_bracket": bool(np.all((LO - 1e-12 <= err) & (err <= HI + 1e-12))),
                                                   "A2_mae": float(np.abs(a2 - err).mean()), "A2_refuted": float(np.mean((a2 < LO - 1e-9) | (a2 > HI + 1e-9))),
                                                   "A2_clipped_mae": float(np.abs(np.clip(a2, LO, HI) - err).mean()),
                                                   "A2_err": a2.tolist(), "true_err": err.tolist()}
            if pname != "sum":
                continue
            # KSE with exact knowledge, plus propagation: labelling one digit of a pair forces its partner through the rule
            dis = ~np.all(C == C[:1], axis=0)                                           # (n, k) slots where readers disagree
            Cs = [tuple(x) for x in np.argwhere(dis & ~forced)]; Bs = dis & forced
            for m in (10, 25, 50, 100, 200):
                for rep in range(20):
                    rr = rng.choice(n * k, m, replace=False); e_rand = (C.reshape(R, -1)[:, rr] != Y.reshape(-1)[rr][None]).mean(1)
                    pick = [Cs[i] for i in rng.choice(len(Cs), min(m, len(Cs)), replace=False)] if Cs else []
                    known = np.where(Bs, fval, -1)
                    for (t, p) in pick:                                                 # label, then propagate within the tuple
                        known[t, p] = Y[t, p]
                        rest = fib[labs[t]][fib[labs[t]][:, p] == Y[t, p]]
                        for q in range(k):
                            if q != p and dis[t, q] and len(np.unique(rest[:, q])) == 1:
                                known[t, q] = rest[0, q]
                    lab = (known >= 0) & dis & ~forced; nC = int((dis & ~forced).sum())
                    scale = nC / max(1, int(lab.sum()))
                    e_kse = ((C != fval[None]) & Bs[None]).sum((1, 2)) + scale * ((C != known[None]) & lab[None]).sum((1, 2))
                    out["kse"].append({"m": m, "random": float(err[int(np.argmin(e_rand))] - best),
                                       "KSE_prop": float(err[int(np.argmin(e_kse))] - best), "labelled_slots": int(lab.sum())})
        return out


    if have("s15c_v10.json"):
        V10R = load_json(ckpt("s15c_v10.json"))
    else:
        V10R = v10_run(); save_json(ckpt("s15c_v10.json"), V10R); hf_push(RUN_NAME, force=True)

    V10_KSE = {m: {k: float(np.mean([x[k] for x in V10R["kse"] if x["m"] == m])) for k in ("random", "KSE_prop", "labelled_slots")}
               for m in (10, 25, 50, 100, 200)}
    print(f"E-B: {len(V10_NAMES)} digit readers on {V10_P.shape[1]} real MNIST test digits, disjoint tuples per program")
    print(f"{'program':11s} {'log2|V|':>7s} {'P(U)':>6s} {'joint cert':>10s} {'marginal':>9s} {'gain':>7s} {'pairs cert':>10s} {'top set':>7s} | regret: S_dagger / S / consistency / source / confidence / majority")
    for _p, _v in sorted(V10R["programs"].items(), key=lambda kv: kv[1]["log2_fibre"]):
        _g = _v["regret"]
        print(f"{_p:11s} {_v['log2_fibre']:7.2f} {_v['PU']:6.3f} {_v['cert_joint']:10.4f} {_v['cert_marginal']:9.4f} {_v['gain_mean']:7.4f} {_v['certified_pairs']:10.3f} {_v['top_set']:7d} | "
              f"{_g['S_dagger']:.4f} / {_g['S']:.4f} / {_g['label_consistency']:.4f} / {_g['source_acc']:.4f} / {_g['confidence']:.4f} / {_g['majority_agreement']:.4f}")
    print("assumption ladder (pairs): sharp bracket [rung 0] vs code-constrained Dawid-Skene A2 [rung 3] point estimates:")
    for _p, _v in V10R["programs"].items():
        if "ladder" in _v:
            _l = _v["ladder"]
            print(f"   {_p:11s} bracket width {_l['bracket_width']:.3f} (truth inside: {_l['truth_in_bracket']}) | A2 MAE {_l['A2_mae']:.4f}, "
                  f"refuted by the bracket {_l['A2_refuted']:.2f}, after clipping to it {_l['A2_clipped_mae']:.4f}")
    # T8 (over-identification): under class-conditional noise every identifying program estimates the SAME error of a reader,
    # so the spread of A2 across programs is a label-free test of the assumption (cf. Sargan-Hansen J test).
    _idp = [p for p in ("sum", "product", "max", "absdiff") if "ladder" in V10R["programs"].get(p, {})]
    _A2 = np.array([V10R["programs"][p]["ladder"]["A2_err"] for p in _idp]); _TE = np.array(V10R["programs"]["sum"]["ladder"]["true_err"])
    _spread = _A2.max(0) - _A2.min(0); _a2err = np.abs(_A2.mean(0) - _TE)
    _par = np.array(V10R["programs"]["parity"]["ladder"]["A2_err"])
    V10_OVERID = {"programs": _idp, "mean_spread": float(_spread.mean()), "spearman_spread_vs_A2_error": float(spearmanr(_spread, _a2err).correlation),
                  "parity_outside_range": float(np.mean((_par < _A2.min(0) - 0.01) | (_par > _A2.max(0) + 0.01)))}
    print("over-identification test (label-free):", V10_OVERID)
    print("KSE with rule propagation on MNIST-Add (mean regret; labelled slots include propagated ones):")
    for _m, _v in V10_KSE.items():
        print(f"   m={_m:4d}: random {_v['random']:.4f} | KSE+propagation {_v['KSE_prop']:.4f} (slots known {_v['labelled_slots']:.0f})")
    return (V10R, V10_KSE, V10_OVERID,)


@app.cell
def _(RUN_NAME, V10R, V10_KSE, V10_OVERID, ckpt, hf_push, save_json, spearmanr):
    # ---- 15d. v10 verdicts and Pre-registration 8 (for a FRESH run: new seed / tuples) ----
    _pr = V10R["programs"]; _coupled = [p for p in ("sum", "product", "max", "absdiff", "sum3") if p in _pr]
    _order = sorted(_pr, key=lambda p: _pr[p]["log2_fibre"])
    _rho10 = spearmanr([_pr[p]["log2_fibre"] for p in _order], [_pr[p]["cert_joint"] for p in _order]).correlation
    _s = _pr["sum"]["regret"]
    V10V = {
        "G15": ("GUAR", all(v["valid"] for v in _pr.values()), "exact knowledge: true regret <= joint certificate for every reader, every program",
                {p: v["valid"] for p, v in _pr.items()}),
        "G16": ("GUAR", all(v["gain_min"] >= -1e-12 for v in _pr.values()) and _pr["box_parity"]["gain_mean"] <= 1e-12,
                "T3: joint bound never above the per-position bound, and exactly equal on the box-fibre control",
                {p: (round(v["gain_min"], 6), round(v["gain_mean"], 6)) for p, v in _pr.items()}),
        "G18": ("GUAR", all(v["ladder"]["truth_in_bracket"] and v["ladder"]["A2_clipped_mae"] <= v["ladder"]["A2_mae"] + 1e-12 for v in _pr.values() if "ladder" in v),
                "rung 0 always contains the truth, and clipping A2 to it never increases its error (T7)",
                {p: v["ladder"] for p, v in _pr.items() if "ladder" in v}),
        "G17": ("GUAR", all(v["certified_wrong"] == 0 for v in _pr.values()), "no certified ordering is wrong",
                {p: v["certified_wrong"] for p, v in _pr.items()}),
        "X13": ("EXPL", all(_pr[p]["gain_mean"] > 0 for p in _coupled), "logic beats tables: rules that tie digits together tighten the certificate",
                {p: _pr[p]["gain_mean"] for p in _coupled}),
        "X14": ("EXPL", _rho10 > 0.7, "knowledge budget law with exact logic: certificate rises with fibre size (log2|V|) across programs (Spearman)",
                {"spearman": _rho10, "log2_fibre": {p: _pr[p]["log2_fibre"] for p in _order}}),
        "X15": ("EXPL", all(_pr[p]["regret"]["S_dagger"] <= 0.01 for p in ("sum", "product", "sum3")),
                "minimax selection lands within .01 of the best reader on sum, product and sum3", {p: _pr[p]["regret"] for p in ("sum", "product", "sum3")}),
        "X17": ("EXPL", V10_OVERID["parity_outside_range"] >= 0.5, "over-identification flags the non-identified program: parity A2 falls outside the identifying programs' range",
                V10_OVERID),
        "X16": ("EXPL", all(v["KSE_prop"] < v["random"] for v in V10_KSE.values()) and V10_KSE[10]["KSE_prop"] <= 0.5 * V10_KSE[10]["random"],
                "KSE with rule propagation beats random labelling at every budget, and halves its regret at m = 10", V10_KSE),
    }
    for _k, (_kind, _ok, _desc, _val) in V10V.items():
        _tag = ("PASS" if _ok else "FAIL") if _kind == "GUAR" else ("YES " if _ok else "NO  ")
        print(f"{_tag:8s} [{_kind}] {_k} {_desc}\n         {_val}")
    print("\nPre-registration 8 (to be scored on a fresh run, never on this one):\n"
          "  P24 G15-G17 hold (guarantees).\n"
          "  P25 the joint certificate is strictly tighter than the per-position one for sum, product, max, absdiff and sum3.\n"
          "  P26 strong rules certify tightly and weak rules do not: joint certificate <= .25 for product and sum, >= .25 for parity and carry.\n"
          "  P27 S-dagger lands within .01 of the best reader on sum, product and sum3.\n"
          "  P28 A2 MAE <= .01 on sum, product, max and absdiff, and the over-identification test flags parity on >= 50% of readers.\n"
          "  P29 KSE with rule propagation beats random labelling at every budget, and halves its regret at m = 10.")
    save_json(ckpt("s15d_v10_verdicts.json"), {k: [v[0], bool(v[1]), v[2], v[3]] for k, v in V10V.items()}); hf_push(RUN_NAME, force=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16. v11 — what the numbers say: three levels of identification, reasoning shortcuts, and select-by-assumption / certify-without

    **What E-B taught us.** Three facts from the 30-reader MNIST run, read together:
    1. Per-reader brackets are 0.76–0.93 wide, yet A2 estimates the same errors to within 0.001. *The data identify the population
       error; the knowledge does not identify the instance.*
    2. carry pins down far less than sum, yet its A2 error is only 0.02. parity's is 0.13, six times larger. The number of
       worlds per label (fibre size) does not explain this.
    3. S† (minimax) regret is 0.23 on sum, while label consistency gets 0. These readers make random errors. A random error breaks
       the rule, so it is visible. A *reasoning shortcut* is an error the rule cannot see, and that is where consistency must fail.

    **Theorem 9 (shortcut indistinguishability; proved, exact).** Let r(x), s(x) ∈ V(x) (both readers satisfy the rule on frame x),
    and let D(x) be their Hamming distance. Then max_{v∈V(x)} [H(r,v) − H(s,v)] = D(x) and min = −D(x).
    Hence, with separable knowledge, the identified set of err_r − err_s restricted to such frames is exactly the symmetric
    interval [−D̄, D̄].
    *Proof.* By the triangle inequality, H(r,v) − H(s,v) ≤ H(r,s), with equality at v = s(x) ∈ V(x). The minimum is symmetric, at
    v = r(x). Theorem 1 then sums over frames. ∎
    Corollaries:
    - A rule-consistent reader can **never** be certified worse than any reader, by any label-free method. Its frames contribute
      ≥ 0 to every UP(·, it).
    - Among rule-consistent readers, label consistency ties at 1. The label-free minimax regret equals the Hamming radius
      min_r max_s D̄(r,s).
    - The *only* information that separates shortcuts is a label on a frame where they disagree. That is stratum C of KSE
      (Theorem 6), and nothing else.

    **Theorem 10 (identification of the error functional; Rothenberg-type local criterion).** Under class-conditional noise (CCN),
    the observable law is P_θ(h₁, h₂, ℓ) = Gᵀ M_ℓ G, where G = diag(π)C and M_ℓ is the fibre mask. Let J be its Jacobian in the
    tangent space of θ = (C, π), and g the gradient of err = 1 − Σ_a π_a C_aa. Then err is locally identified at θ iff g ⊥ null(J).
    The number ρ = ‖P_null(J) g‖ / ‖g‖ is computable **from the program and the reader's outputs alone**, with no labels:
    - ρ = 0 ⇒ point-identified;
    - ρ > 0 ⇒ an exact direction exists along which the observable law is flat while the error moves.

    The criterion is non-obvious. carry leaves **28** directions of C unidentified, yet ρ = 0, so its *error* is identified.
    parity leaves 72, with ρ ≈ 0.94. We computed this offline before running anything: σ_min(J) orders the A2 errors with
    Spearman −0.93.

    **Theorem 11 (spread is a certified lower bound; proved, assumption-free).** For any estimates ê_p of the same quantity e,
    max_p |ê_p − e| ≥ (max_p ê_p − min_p ê_p) / 2. So the over-identification spread of Theorem 8 is a label-free *proof* that some
    program's estimate is off by at least half of it. That is a refutation with no assumption at all.

    **Proposition 12 (select by assumption, certify without; proved).** The certificate CR(r) of Theorem 2 holds for *every* r,
    whatever chose it. So choose r̂ by the assumption-based estimate, after projecting it onto the identified set (Theorem 7), and
    report CR(r̂). The guarantee is unchanged. The typical regret is that of the assumption whenever the over-identification test
    passes. When the test fails, fall back to S†.

    **This section's cells.**
    - **16b (MNIST, no new encoding):** ρ, nullity and σ_min per program and per reader at the reader's own A2 estimate; the
      Theorem 11 check; pooled multi-program estimates; the test-then-select ladder.
    - **16c–16e (BDD-OIA, real dashcam frames, exact rsbench rules):** encoder zoo × {zero-shot, 50-label probe, action-only NeSy
      from random init, action-only NeSy from zero-shot}. Action-only NeSy is the textbook source of reasoning shortcuts. The cells
      test Theorem 9 exactly, and test whether label consistency picks a shortcut. They also check whether multi-reader,
      rule-constrained Dawid–Skene (rung 3), with a cross-fit over-identification test, gives a ladder selector that beats every
      label-free baseline, and whether KSE (labels on stratum C only) beats random labelling.
    """)
    return


@app.cell
def _(CFG, RUN_NAME, V10_P, V10_PROGRAMS, V10_Y, ckpt, ds_pairs, have, hf_push, load_json, np, save_json, spearmanr, v10_fibres):
    # ---- 16b. v11 identification geometry on MNIST (Theorems 10, 11, Prop. 12). Re-uses V10_P; fresh disjoint pairs, no encoding ----
    V11M = dict(seed=CFG["seed"] + 303, null_tol=1e-8, tau_spread=0.02)
    V11_PAIR_PROGS = [p for p, (k, _) in V10_PROGRAMS.items() if k == 2]


    def v11_jac(Cm, pi, M):
        """Jacobian of P(h1, h2, l) = G^T M_l G (G = diag(pi) C) in the tangent space of (C rows, pi), and the gradient of
        err = 1 - sum_a pi_a C_aa. Returns J (n_obs, n_dir), g (n_dir,)."""
        K = len(pi); B = np.linalg.qr(np.vstack([np.eye(K - 1), -np.ones(K - 1)]))[0]          # orthonormal sum-zero basis
        dC, dpi = [], []
        for c in range(K):
            for t in range(K - 1):
                d = np.zeros((K, K)); d[c] = B[:, t]; dC.append(d); dpi.append(np.zeros(K))
        for t in range(K - 1):
            dC.append(np.zeros((K, K))); dpi.append(B[:, t])
        dC, dpi = np.array(dC), np.array(dpi); G = pi[:, None] * Cm
        dG = dpi[:, :, None] * Cm[None] + pi[None, :, None] * dC
        Mf = M.astype(float)
        J = np.einsum("dah,lab,bk->dlhk", dG, Mf, G, optimize=True) + np.einsum("ah,lab,dbk->dlhk", G, Mf, dG, optimize=True)
        g = -(dpi * np.diag(Cm)[None]).sum(1) - (pi[None] * np.einsum("daa->da", dC)).sum(1)
        return J.reshape(len(dC), -1).T, g


    def v11_ident(Cm, pi, M):
        J, g = v11_jac(Cm, pi, M); U, s, Vt = np.linalg.svd(J, full_matrices=True)
        r = int((s > V11M["null_tol"] * s[0]).sum()); N = Vt[r:]
        return {"nullity": int(len(N)), "rho": float(np.linalg.norm(N @ g) / (np.linalg.norm(g) + 1e-300)), "sigma_min": float(s[-1]) if r == len(s) else 0.0}


    def v11_mnist():
        A = V10_P.argmax(2); R = A.shape[0]; rng = np.random.default_rng(V11M["seed"]); out = {}
        for pname in V11_PAIR_PROGS:
            fn = V10_PROGRAMS[pname][1]; n = A.shape[1] // 2; slots = rng.permutation(A.shape[1])[: 2 * n].reshape(n, 2)
            Y = V10_Y[slots]; C = A[:, slots]; fib = v10_fibres(fn, 2); lid = {l: i for i, l in enumerate(fib)}
            M = np.zeros((len(fib), 10, 10), bool)
            for l, V in fib.items():
                M[lid[l], V[:, 0], V[:, 1]] = True
            li = np.array([lid[fn(tuple(int(x) for x in y))] for y in Y]); err = (C != Y[None]).mean((1, 2))
            a2, idn = np.zeros(R), []
            for r in range(R):
                Cm, pi = ds_pairs(C[r, :, 0], C[r, :, 1], li, M); a2[r] = 1 - float(pi @ np.diag(Cm)); idn.append(v11_ident(Cm, pi, M))
            out[pname] = {"A2": a2.tolist(), "err": err.tolist(), "A2_mae": float(np.abs(a2 - err).mean()),
                          "rho": [d["rho"] for d in idn], "nullity": [d["nullity"] for d in idn], "sigma_min": [d["sigma_min"] for d in idn]}
        return out


    if have("s16b_v11_mnist.json"):
        V11MR = load_json(ckpt("s16b_v11_mnist.json"))
    else:
        V11MR = v11_mnist(); save_json(ckpt("s16b_v11_mnist.json"), V11MR); hf_push(RUN_NAME, force=True)

    print("Theorem 10 (label-free identification of each reader's error under CCN, at the reader's own A2 estimate):")
    print(f"{'program':11s} {'nullity':>8s} {'rho (median)':>12s} {'sigma_min':>10s} {'A2 MAE':>8s}")
    for _p, _v in V11MR.items():
        print(f"{_p:11s} {np.median(_v['nullity']):8.0f} {np.median(_v['rho']):12.4f} {np.median(_v['sigma_min']):10.2e} {_v['A2_mae']:8.4f}")
    _ident_p = [p for p, v in V11MR.items() if np.median(v["rho"]) < 1e-3]
    _nonid_p = [p for p in V11MR if p not in _ident_p]
    _deficit = [np.median(V11MR[p]["rho"]) - np.median(V11MR[p]["sigma_min"]) for p in V11MR]; _mae = [V11MR[p]["A2_mae"] for p in V11MR]
    V11_RHO = {"identified": _ident_p, "not_identified": _nonid_p,
               "max_mae_identified": float(max([V11MR[p]["A2_mae"] for p in _ident_p] or [np.nan])),
               "min_mae_not_identified": float(min([V11MR[p]["A2_mae"] for p in _nonid_p] or [np.nan])),
               "spearman_deficit_vs_mae": float(spearmanr(_deficit, _mae).correlation)}          # deficit = rho - sigma_min
    print("rho separates programs whose error is identified from those whose is not:", V11_RHO)

    # Theorem 11 (assumption-free): over identifying programs, max_p |A2_p - err| >= spread / 2 for every reader
    _E = np.array(V11MR[V11_PAIR_PROGS[0]]["err"]); _idp = [p for p in ("sum", "product", "max", "absdiff") if p in V11MR]
    _A = np.array([V11MR[p]["A2"] for p in _idp]); _spread = _A.max(0) - _A.min(0); _worst = np.abs(_A - _E[None]).max(0)
    _Sg = np.array([V11MR[p]["sigma_min"] for p in _idp]) ** 2                # label-free weights: identification strength
    _pooled_eq = _A.mean(0); _pooled = (_Sg * _A).sum(0) / (_Sg.sum(0) + 1e-300)    # over-identified (GMM-style) pooled estimates
    V11_T11 = {"holds_every_reader": bool(np.all(_worst >= _spread / 2 - 1e-12)), "mean_spread": float(_spread.mean()),
               "pooled_mae": float(np.abs(_pooled - _E).mean()), "pooled_equal_weights_mae": float(np.abs(_pooled_eq - _E).mean()), "single_program_mae": {p: V11MR[p]["A2_mae"] for p in _idp},
               "spearman_spread_vs_pooled_error": float(spearmanr(_spread, np.abs(_pooled - _E)).correlation)}
    print("Theorem 11 + pooling:", V11_T11)

    # Proposition 12: select by the (tested) assumption, certify without it
    _pass = float(np.median(_spread)) <= V11M["tau_spread"]
    _pick = int(np.argmin(_pooled)) if _pass else None
    V11_LADDER_M = {"test_passes": _pass, "median_spread": float(np.median(_spread)),
                    "regret_ladder": float(_E[_pick] - _E.min()) if _pass else None,
                    "regret_per_program_argmin": {p: float(_E[int(np.argmin(V11MR[p]["A2"]))] - _E.min()) for p in V11MR}}
    print("test-then-select on MNIST:", V11_LADDER_M)
    return (V11_LADDER_M, V11_RHO, V11_T11,)


@app.cell
def _(BDD_SPECS, CFG, CLIP_MEAN, CLIP_STD, DEV, Image, RUN_NAME, SMOKE, ZOO_FEATS, atomic_write, check_deadline, ckpt, clip_image_embed, clip_text_embed, cons_mask, encode_bdd, get_bdd, gpu_free, hf_push, label_ids, load_clip, nesy_train, np, os, parse_bdd, re, text_dirs_centered, torch, zs_logits):
    # ---- 16c. v11 experiment E-C data: real BDD-OIA dashcam frames through the encoder zoo. Readers per encoder:
    #      zero-shot, 50-label concept probe, action-only NeSy (random init) and action-only NeSy (zero-shot init) ----
    V11 = dict(n_train=int(os.environ.get("V11_N_TRAIN", 3000)), n_test=int(os.environ.get("V11_N_TEST", 3000)), n_probe=50, n_val=100,
               seed=CFG["seed"] + 202)
    V11_CONCEPTS = ["the traffic light is green", "following the car ahead", "the road is clear", "the traffic light is red",
                    "there is a traffic sign", "a car is in the way", "a person is in the way", "a rider is in the way",
                    "another obstacle is in the way", "there is a lane on the left", "the left-turn traffic light is green",
                    "following a car turning left", "there is no lane on the left", "there is an obstacle on the left",
                    "there is a solid line on the left", "there is a lane on the right", "the right-turn traffic light is green",
                    "following a car turning right", "there is no lane on the right", "there is an obstacle on the right",
                    "there is a solid line on the right"]
    V11_TEMPLATES = ["a dashcam photo where {c}.", "a driving scene: {c}.", "a photo taken from a car: {c}."]


    def v11_frames():
        """Frame subsets (indices into the official split order of parse_bdd), paths, action labels, reason labels."""
        BD = parse_bdd(get_bdd(os.path.join(CFG["work_dir"], "data", "bddoia"))); r = np.random.default_rng(V11["seed"]); out = {}
        for sp in ("train", "test"):
            P, A, C = BD[sp]; idx = np.sort(r.choice(len(P), min(V11[f"n_{sp}"], len(P)), replace=False))
            out[sp] = (idx, [P[i] for i in idx], A[idx], C[idx])
        return out


    try:
        V11_FR = v11_frames()
    except FileNotFoundError as _e:                     # Drive refused: put the zip at <work_dir>/data/bddoia/bddoia.zip (or set BDDOIA_ZIP)
        V11_FR = None
        print("BDD-OIA not available, E-C is skipped (MNIST cells still run).", str(_e)[:200],
              f"\nFix: download the zip in your browser and upload it to {os.path.join(CFG['work_dir'], 'data', 'bddoia', 'bddoia.zip')}, then re-run this cell.")


    def v11_pixels(sp):
        """uint8 (n, 224, 224, 3) frames, squashed like section 9b; cached under data/ (never uploaded)."""
        f = os.path.join(CFG["work_dir"], "data", f"bdd_u8_{sp}_{len(V11_FR[sp][0])}_{V11['seed']}.npy")
        if not os.path.exists(f):
            from concurrent.futures import ThreadPoolExecutor
            load = lambda p: np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.uint8)
            with ThreadPoolExecutor(8) as pool:
                arr = np.stack(list(pool.map(load, V11_FR[sp][1])))
            atomic_write(f, lambda t_: np.save(t_, arr)); del arr
        return np.load(f, mmap_mode="r")


    def v11_encode(name):
        """{'train', 'test'} normalised image embeddings and (21, d) prompt-ensemble concept directions for one model; cached."""
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[-60:]
        fi = {sp: ckpt(f"zoo_bdd_{sp}_{len(V11_FR[sp][0])}_{V11['seed']}_{slug}.npy") for sp in ("train", "test")}; ft = ckpt(f"zoo_bdd_txt_{slug}.npy")
        if all(os.path.exists(f) for f in fi.values()) and os.path.exists(ft):
            return {sp: np.load(f) for sp, f in fi.items()}, np.load(ft)
        if SMOKE:                                       # synthetic plumbing only: rotated copies of the section-9b features
            r = np.random.default_rng(sum(map(ord, name))); base = {sp: encode_bdd(parse_bdd(get_bdd(os.path.join(CFG["work_dir"], "data", "bddoia")))[sp][0], sp)[V11_FR[sp][0]] for sp in ("train", "test")}
            d = base["train"].shape[1]; Q = np.linalg.qr(r.normal(size=(d, d)))[0]
            img = {sp: (b @ Q + r.normal(size=b.shape) * r.uniform(.02, .08)).astype(np.float32) for sp, b in base.items()}
            Ctr = V11_FR["train"][3]
            txt = np.stack([img["train"][Ctr[:, j] == 1].mean(0) - img["train"][Ctr[:, j] == 0].mean(0) if 0 < Ctr[:, j].sum() < len(Ctr) else r.normal(size=d) for j in range(21)])
            txt = txt / np.linalg.norm(txt, axis=1, keepdims=True) + r.normal(0, 0.1, txt.shape)
        else:
            model, tok = load_clip(name)
            mu = torch.tensor(CLIP_MEAN, device=DEV).view(1, 3, 1, 1); sd = torch.tensor(CLIP_STD, device=DEV).view(1, 3, 1, 1); img = {}
            for sp in ("train", "test"):
                X = v11_pixels(sp); res = []
                for b0 in range(0, len(X), CFG["img_batch"]):
                    check_deadline()
                    x = torch.from_numpy(np.ascontiguousarray(X[b0:b0 + CFG["img_batch"]])).to(DEV).permute(0, 3, 1, 2).float() / 255
                    with torch.inference_mode():
                        res.append(torch.nn.functional.normalize(clip_image_embed(model, (x - mu) / sd), dim=1).cpu().numpy())
                img[sp] = np.concatenate(res).astype(np.float32)
            with torch.inference_mode():
                txt = sum(torch.nn.functional.normalize(clip_text_embed(model, tok, [tm.format(c=c) for c in V11_CONCEPTS]), dim=1) for tm in V11_TEMPLATES)
            txt = torch.nn.functional.normalize(txt, dim=1).cpu().numpy(); del model; gpu_free()
        img = {sp: v / np.linalg.norm(v, axis=1, keepdims=True) for sp, v in img.items()}
        for sp, f in fi.items():
            atomic_write(f, lambda t_, a=img[sp]: np.save(t_, a.astype(np.float32)))
        atomic_write(ft, lambda t_: np.save(t_, txt.astype(np.float32))); hf_push(RUN_NAME)
        return img, txt.astype(np.float32)


    def v11_bce(X, Y, l2=1e-3, iters=300):
        """Multi-label logistic probe (21 independent concepts) fit with L-BFGS; returns logits function."""
        lin = torch.nn.Linear(X.shape[1], Y.shape[1]).to(X.device)
        opt = torch.optim.LBFGS(lin.parameters(), lr=0.5, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(X), Y) + l2 * (lin.weight ** 2).sum(); loss.backward(); return loss
        opt.step(closure)
        return lin


    def v11_readers():
        """Logits (R, n_test, 21) on test frames, reader names, and logits on V11['n_val'] labelled train frames (source accuracy)."""
        if V11_FR is None:
            return [], np.zeros((0, 0, 21), np.float32), np.zeros((0, 0, 21), np.float32), np.zeros((0, 21), np.int64)
        acts = V11_FR["train"][2]; Ctr = V11_FR["train"][3]
        LB = [label_ids(BDD_SPECS[0], [tuple(a[:2]) for a in acts]), label_ids(BDD_SPECS[1], [(a[2],) for a in acts]),
              label_ids(BDD_SPECS[2], [(a[3],) for a in acts])]
        cons_tr = [cons_mask(s_, l) for s_, l in zip(BDD_SPECS, LB)]
        r = np.random.default_rng(V11["seed"] + 1); perm = r.permutation(len(acts)); probe, val = perm[:V11["n_probe"]], perm[V11["n_probe"]:V11["n_probe"] + V11["n_val"]]
        names, Z, Zval = [], [], []
        for name in (list(ZOO_FEATS)[:3] if SMOKE else list(ZOO_FEATS)):
            try:
                img, T = v11_encode(name)
            except Exception as e:
                print(f"bdd zoo {name}: skipped ({type(e).__name__}: {str(e)[:120]})"); gpu_free(); continue
            short = name.rstrip("/").split("/")[-1]
            Xtr = torch.as_tensor(img["train"], device=DEV); Xte = torch.as_tensor(img["test"], device=DEV)
            Tc = text_dirs_centered(T); zte, (W0, b0) = zs_logits(Xte, Tc, Xtr); ztr = zs_logits(Xtr, Tc, Xtr)[0]
            heads = {"zs": (zte, ztr)}
            lin = v11_bce(Xtr[probe], torch.as_tensor(Ctr[probe], dtype=torch.float32, device=DEV))
            with torch.no_grad():
                heads["probe50"] = (lin(Xte).cpu().numpy(), lin(Xtr).cpu().numpy())
            for hn, init in (("nesy_rand", (None, None)), ("nesy_zs", (W0, b0))):
                W, b = nesy_train(Xtr, BDD_SPECS, cons_tr, 21, seed=V11["seed"], epochs=CFG["nesy_epochs"], W0=init[0], b0=init[1])
                with torch.no_grad():
                    heads[hn] = ((Xte @ W.T + b).cpu().numpy(), (Xtr @ W.T + b).cpu().numpy())
            for hn, (zt, ztr_) in heads.items():
                names.append(f"{short}:{hn}"); Z.append(zt); Zval.append(ztr_[val])
            print(f"bdd zoo {short}: train {img['train'].shape}, test {img['test'].shape}", flush=True)
        return names, np.stack(Z).astype(np.float32), np.stack(Zval).astype(np.float32), Ctr[val]


    V11_NAMES, V11_Z, V11_ZVAL, V11_CVAL = v11_readers()
    if V11_FR is not None:
        print(f"E-C readers: {len(V11_NAMES)} | test frames {V11_Z.shape[1]} | concept accuracy per reader:",
              {n: round(float(((z > 0) == V11_FR['test'][3]).mean()), 3) for n, z in zip(V11_NAMES, V11_Z)})
    return (V11, V11_CVAL, V11_FR, V11_NAMES, V11_Z, V11_ZVAL, v11_bce,)


@app.cell
def _(BDD_SPECS, DEV, RUN_NAME, V11, V11_CVAL, V11_FR, V11_NAMES, V11_Z, V11_ZVAL, ckpt, cons_mask, forced_from_cons, have, hf_push, label_ids, load_json, np, save_json, torch):
    # ---- 16d. v11 experiment E-C: certificates, shortcut indistinguishability (Theorem 9), rule-constrained multi-reader
    #      Dawid-Skene with a cross-fit over-identification test, select-by-assumption / certify-without (Prop. 12), and KSE ----
    V11X = dict(tau_spread=0.02, ds_iters=60, kse_m=(10, 25, 50, 100), kse_reps=20)


    def v11_dist(A, V):
        """Hamming distance of each reader's block prediction to every world: A (R, n, kb) in {0,1}, V (W, kb) -> (R, n, W)."""
        A = torch.as_tensor(A, dtype=torch.float32, device=DEV); V = torch.as_tensor(V, dtype=torch.float32, device=DEV)
        return A @ (1 - V).T + (1 - A) @ V.T


    def v11_cert(A, cons, w):
        """Sharp joint (T1) and per-concept (box) upper bounds of err_r - err_s, per-reader brackets [LO, HI] and the
        Theorem 9 check, all weighted by w (n,) and normalised to error per concept slot. A (R, n, 21) binary."""
        R, n, K = A.shape; wt = torch.as_tensor(w, dtype=torch.float32, device=DEV); den = float(w.sum()) * K
        UPJ = torch.zeros(R, R, device=DEV); UPM = torch.zeros(R, R, device=DEV); LO = torch.zeros(R, device=DEV); HI = torch.zeros(R, device=DEV)
        PF = torch.zeros(R, R, n, device=DEV); inside = torch.ones(R, n, dtype=torch.bool, device=DEV)
        At = torch.as_tensor(A, dtype=torch.float32, device=DEV)
        for s_, c in zip(BDD_SPECS, cons):
            ix, V = s_["ix"], s_["V"]; H = v11_dist(A[:, :, ix], V); C = torch.as_tensor(c, device=DEV)
            Hm = H.masked_fill(~C[None], float("inf")); LO += (Hm.min(-1).values * wt).sum(1)
            HI += (H.masked_fill(~C[None], -float("inf")).max(-1).values * wt).sum(1)
            inside &= (Hm.min(-1).values == 0)
            Vt = torch.as_tensor(V, dtype=torch.float32, device=DEV)
            can1 = (C.float() @ Vt) > 0; can0 = (C.float() @ (1 - Vt)) > 0              # (n, kb): value allowed by the fibre
            for r in range(R):
                per = (H[r][None] - H).masked_fill(~C[None], -float("inf")).max(-1).values      # (R, n): max over the fibre
                PF[r] += per; UPJ[r] += (per * wt).sum(1)
                a_r = At[r][:, ix]; a_s = At[:, :, ix]; diff = a_r[None] != a_s                 # (R, n, kb)
                ok_s = torch.where(a_s > 0.5, can1[None], can0[None])                          # can the truth equal s's value?
                UPM[r] += ((diff.float() * (2 * ok_s.float() - 1)).sum(-1) * wt).sum(1)
        ham = (At[:, None] != At[None]).float().sum(-1)                                          # (R, R, n)
        both = inside[:, None] & inside[None]
        t9 = float((PF - ham).abs().masked_fill(~both, 0).max()) if bool(both.any()) else 0.0
        return (UPJ / den).cpu().numpy(), (UPM / den).cpu().numpy(), (LO / den).cpu().numpy(), (HI / den).cpu().numpy(), inside.cpu().numpy(), t9, (ham / K).cpu().numpy()


    def v11_ds(A, cons, readers, iters):
        """Rule-constrained multi-reader Dawid-Skene: latent world per block inside its fibre, per-reader per-concept 2x2
        confusion, conditional independence across readers. Returns concept posteriors M (n, 21)."""
        As = A[readers].astype(np.float64); M = As.mean(0); pis = [np.full(len(s_["V"]), 1.0 / len(s_["V"])) for s_ in BDD_SPECS]
        for _ in range(iters):
            q1 = ((As * M[None]).sum(1) + 1) / (M.sum(0)[None] + 2); q0 = ((As * (1 - M)[None]).sum(1) + 1) / ((1 - M).sum(0)[None] + 2)
            a = (As * np.log(q1)[:, None] + (1 - As) * np.log(1 - q1)[:, None]).sum(0)
            b = (As * np.log(q0)[:, None] + (1 - As) * np.log(1 - q0)[:, None]).sum(0)
            for bi, (s_, c) in enumerate(zip(BDD_SPECS, cons)):
                ix, V = s_["ix"], s_["V"]
                L = np.log(pis[bi] + 1e-12)[None] + a[:, ix] @ V.T + b[:, ix] @ (1 - V).T
                L = np.where(c, L, -np.inf); L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
                M[:, ix] = P @ V; pis[bi] = P.mean(0)
        return M


    def v11_run():
        A = (V11_Z > 0).astype(np.int8); R, n, K = A.shape; Y = V11_FR["test"][3]; acts = V11_FR["test"][2]
        LB = [label_ids(BDD_SPECS[0], [tuple(a[:2]) for a in acts]), label_ids(BDD_SPECS[1], [(a[2],) for a in acts]),
              label_ids(BDD_SPECS[2], [(a[3],) for a in acts])]
        cons_all = [cons_mask(s_, l) for s_, l in zip(BDD_SPECS, LB)]
        sat = np.stack([c.any(1) for c in cons_all], 1).all(1)                         # label-free: the action label is possible
        A, Y, cons = A[:, sat], Y[sat], [c[sat] for c in cons_all]; n = int(sat.sum())
        kin = np.ones(n, bool)                                                          # evaluation only: is the truth in the fibre (K)?
        for s_, c in zip(BDD_SPECS, cons):
            wi = (Y[:, s_["ix"]] * (2 ** np.arange(len(s_["ix"]))[::-1])).sum(1)
            vi = (s_["V"] * (2 ** np.arange(len(s_["ix"]))[::-1])).sum(1); pos = np.argsort(vi)
            kin &= c[np.arange(n), pos[wi]]
        F = np.concatenate([forced_from_cons(c, len(s_["ix"])) for c, s_ in zip(cons, BDD_SPECS)], 1)
        err = (A != Y[None]).mean((1, 2)); best = err.min(); err_k = (A[:, kin] != Y[kin][None]).mean((1, 2))
        UPJ, UPM, LO, HI, inside, t9, ham = v11_cert(A, cons, np.ones(n))
        UPJk, _, LOk, HIk, _, _, _ = v11_cert(A[:, kin], [c[kin] for c in cons], np.ones(int(kin.sum())))
        CRJ = np.maximum(UPJ, 0).max(1); CRJk = np.maximum(UPJk, 0).max(1); CRM = np.maximum(UPM, 0).max(1)
        off = ~np.eye(R, dtype=bool); certk = (UPJk < -1e-9) & off; tdk = err_k[:, None] - err_k[None]
        W = ((A != F[None]) & (F[None] >= 0)).mean((1, 2))
        cons_rate = inside.mean(1)
        P = 1 / (1 + np.exp(-V11_Z[:, sat])); conf = np.maximum(P, 1 - P).mean((1, 2))
        agree = (A == (A.mean(0) > 0.5)[None]).mean((1, 2)); src = ((V11_ZVAL > 0) == V11_CVAL[None]).mean((1, 2))
        # rung 3: rule-constrained multi-reader Dawid-Skene; cross-fit over encoder halves for the over-identification test
        enc = np.array([nm.split(":")[0] for nm in V11_NAMES]); ue = sorted(set(enc)); half = np.isin(enc, ue[0::2])
        Mfull = v11_ds(A, cons, np.arange(R), V11X["ds_iters"])
        Mh = {h: v11_ds(A, cons, np.where(half == h)[0], V11X["ds_iters"]) if (half == h).any() else Mfull for h in (True, False)}
        est = lambda M: (A * (1 - M)[None] + (1 - A) * M[None]).mean((1, 2))
        e_full = est(Mfull); e_cross = np.where(half, est(Mh[False]), est(Mh[True]))
        spread = np.abs(e_full - e_cross); e_clip = np.clip(e_cross, LO, HI)
        test_pass = bool(np.median(spread) <= V11X["tau_spread"])
        picks = {"S_dagger": int(np.lexsort((W, CRJ))[0]), "S": int(np.argmin(W)), "label_consistency": int(np.argmax(cons_rate)),
                 "confidence": int(np.argmax(conf)), "majority_agreement": int(np.argmax(agree)), "source_acc_100_labels": int(np.argmax(src)),
                 "DS_clipped": int(np.argmin(e_clip)), "ladder": int(np.argmin(e_clip)) if test_pass else int(np.lexsort((W, CRJ))[0])}
        heads = np.array([nm.split(":")[1] for nm in V11_NAMES])
        by_head = {h: {"err": float(err[heads == h].mean()), "rule_consistency": float(cons_rate[heads == h].mean()),
                       "best_err": float(err[heads == h].min())} for h in sorted(set(heads))}
        lc = np.flatnonzero(cons_rate >= 0.9)
        radius = float(min(ham[r][:, inside[r]][lc].mean(1).max() if inside[r].any() else 0 for r in lc)) if len(lc) > 1 else None
        out = {"n_frames": int(len(sat)), "n_sat": n, "K_violation_rate": float(1 - kin.mean()), "forced_frac": float((F >= 0).mean()),
               "err": err.tolist(), "best_err": float(best), "names": V11_NAMES,
               "valid_all_frames": bool(np.all(err - best <= CRJ + 1e-12)), "valid_K_frames": bool(np.all(err_k - err_k.min() <= CRJk + 1e-12)),
               "certified_wrong_K": int((certk & (tdk >= 0)).sum()), "certified_pairs_K": float(certk.sum() / off.sum()),
               "gain_mean": float((CRM - CRJ).mean()), "gain_min": float((UPM - UPJ).min()), "theorem9_max_dev": t9,
               "cert_selected": {m: float(CRJ[r]) for m, r in picks.items()}, "regret": {m: float(err[r] - best) for m, r in picks.items()},
               "picked_head": {m: str(heads[r]) for m, r in picks.items()}, "by_head": by_head, "consistent_readers": int(len(lc)), "hamming_radius": radius,
               "rule_consistency": cons_rate.tolist(), "DS": {"mae_full": float(np.abs(e_full - err).mean()), "mae_cross": float(np.abs(e_cross - err).mean()),
               "mae_clipped": float(np.abs(e_clip - err).mean()), "median_spread": float(np.median(spread)), "test_passes": test_pass,
               "bracket_contains_truth_K": bool(np.all((LOk - 1e-12 <= err_k) & (err_k <= HIk + 1e-12)))}, "kse": []}
        # KSE at frame level: forced concepts are known exactly; label only frames where readers disagree on an unforced concept
        rng = np.random.default_rng(V11["seed"] + 7); unf = F < 0
        Cf = np.flatnonzero(((A != A[:1]) & unf[None]).any((0, 2))); forced_err = ((A != F[None]) & ~unf[None]).sum((1, 2))
        for m in V11X["kse_m"]:
            for rep in range(V11X["kse_reps"]):
                rr = rng.choice(n, min(m, n), replace=False); e_rand = (A[:, rr] != Y[rr][None]).mean((1, 2))
                cc = rng.choice(Cf, min(m, len(Cf)), replace=False) if len(Cf) else np.array([], int)
                e_kse = forced_err + (len(Cf) / max(1, len(cc))) * ((A[:, cc] != Y[cc][None]) & unf[cc][None]).sum((1, 2))
                out["kse"].append({"m": m, "random": float(err[int(np.argmin(e_rand))] - best), "KSE": float(err[int(np.argmin(e_kse))] - best)})
        return out


    if V11_FR is None:
        V11R, V11_KSE = None, {}
        print("E-C skipped: BDD-OIA frames are not available.")
    elif have("s16d_v11_bdd.json"):
        V11R = load_json(ckpt("s16d_v11_bdd.json"))
    else:
        V11R = v11_run(); save_json(ckpt("s16d_v11_bdd.json"), V11R); hf_push(RUN_NAME, force=True)

    if V11R is not None:
        V11_KSE = {m: {k: float(np.mean([x[k] for x in V11R["kse"] if x["m"] == m])) for k in ("random", "KSE")} for m in V11X["kse_m"]}
        print(f"E-C: {len(V11R['names'])} readers on {V11R['n_sat']} real BDD-OIA test frames whose action label the rules allow "
              f"(of {V11R['n_frames']}); knowledge violated on {V11R['K_violation_rate']:.3f} of them; concepts forced: {V11R['forced_frac']:.3f}")
        print("by head (mean concept error / mean rule consistency / best error):")
        for _h, _v in V11R["by_head"].items():
            print(f"   {_h:10s} {_v['err']:.4f} / {_v['rule_consistency']:.3f} / {_v['best_err']:.4f}")
        print(f"certificate valid (all frames / frames where the rules hold): {V11R['valid_all_frames']} / {V11R['valid_K_frames']} | "
              f"wrong certified orderings: {V11R['certified_wrong_K']} | pairs certified: {V11R['certified_pairs_K']:.3f} | joint-vs-box gain {V11R['gain_mean']:.4f}")
        print(f"Theorem 9 exact identity, max deviation over all reader pairs and frames: {V11R['theorem9_max_dev']} | "
              f"readers consistent on >= 90% of frames: {V11R['consistent_readers']}, their Hamming radius (= label-free minimax regret among them): {V11R['hamming_radius']}")
        print("regret by selector (and the head it picked):")
        for _m, _r in V11R["regret"].items():
            print(f"   {_m:22s} {_r:.4f}  [{V11R['picked_head'][_m]}]  certificate of the pick {V11R['cert_selected'][_m]:.4f}")
        print("rung 3 (rule-constrained multi-reader Dawid-Skene):", V11R["DS"])
        print("KSE on BDD-OIA (mean regret):", {m: {k: round(v, 4) for k, v in d.items()} for m, d in V11_KSE.items()})
    return (V11R, V11_KSE,)


@app.cell
def _(RUN_NAME, V11R, V11_KSE, V11_LADDER_M, V11_RHO, V11_T11, ckpt, hf_push, save_json):
    # ---- 16e. v11 verdicts and Pre-registration 9 (for a FRESH run: new seed, full zoo) ----
    _bdd = V11R is not None
    _bh = V11R["by_head"] if _bdd else {}; _rg = V11R["regret"] if _bdd else {}; _lf = ["S_dagger", "S", "label_consistency", "confidence", "majority_agreement"]
    V11V = {
        "G22": ("GUAR", V11_T11["holds_every_reader"], "Theorem 11 (MNIST): the over-identification spread is a certified lower bound on the worst program's error",
                V11_T11),
        "X23": ("EXPL", set(V11_RHO["identified"]) >= {"sum", "product", "max", "absdiff"} and "parity" in V11_RHO["not_identified"]
                and V11_RHO["max_mae_identified"] < V11_RHO["min_mae_not_identified"],
                "Theorem 10 (MNIST): rho, computed without labels, separates programs whose error is identified (small A2 error) from those whose is not", V11_RHO),
        "X24": ("EXPL", V11_T11["pooled_mae"] <= min(V11_T11["single_program_mae"].values()), "pooling identifying programs, weighted by identification strength sigma_min^2 (label-free), beats the best single program", V11_T11),
        "X25": ("EXPL", V11_LADDER_M["test_passes"] and V11_LADDER_M["regret_ladder"] <= 0.01, "MNIST: test passes and test-then-select lands within .01 of the best reader", V11_LADDER_M),
    }
    if _bdd:
        V11V.update({
        "G19": ("GUAR", V11R["theorem9_max_dev"] <= 1e-6, "Theorem 9: for every pair of readers that both satisfy the rules on a frame, the sharp bound equals their Hamming distance",
                V11R["theorem9_max_dev"]),
        "G20": ("GUAR", V11R["valid_K_frames"] and V11R["certified_wrong_K"] == 0, "where the knowledge holds, the certificate is valid and no certified ordering is wrong",
                {"valid": V11R["valid_K_frames"], "wrong": V11R["certified_wrong_K"]}),
        "G21": ("GUAR", V11R["gain_min"] >= -1e-9 and V11R["DS"]["bracket_contains_truth_K"], "joint bound never above the per-concept bound; bracket contains every reader's error",
                {"gain_min": V11R["gain_min"], "bracket_ok": V11R["DS"]["bracket_contains_truth_K"]}),
        "X18": ("EXPL", min(_bh.get("nesy_rand", {}).get("rule_consistency", 0), _bh.get("nesy_zs", {}).get("rule_consistency", 0)) > _bh["probe50"]["rule_consistency"]
                and min(_bh.get("nesy_rand", {}).get("err", 0), _bh.get("nesy_zs", {}).get("err", 0)) > _bh["probe50"]["err"],
                "shortcut signature on real dashcam frames: action-only NeSy readers obey the rules more but read the concepts worse than a 50-label probe", _bh),
        "X19": ("EXPL", V11R["picked_head"]["label_consistency"].startswith("nesy") and _rg["label_consistency"] > _rg["ladder"],
                "label consistency (the NeSy default) picks a shortcut-prone reader, and does worse than test-then-select", {"picked": V11R["picked_head"], "regret": _rg}),
        "X20": ("EXPL", _rg["ladder"] <= min(_rg[m] for m in _lf), "test-then-select beats every label-free baseline on BDD-OIA", _rg),
        "X21": ("EXPL", V11R["DS"]["mae_clipped"] <= 0.02, "rule-constrained multi-reader Dawid-Skene, cross-fit and clipped, estimates every reader's concept error within .02", V11R["DS"]),
        "X22": ("EXPL", all(v["KSE"] <= v["random"] for v in V11_KSE.values()), "KSE beats random frame labelling at every budget on BDD-OIA", V11_KSE),
        })
    for _k, (_kind, _ok, _desc, _val) in V11V.items():
        _tag = ("PASS" if _ok else "FAIL") if _kind == "GUAR" else ("YES " if _ok else "NO  ")
        print(f"{_tag:8s} [{_kind}] {_k} {_desc}\n         {_val}")
    print("\nPre-registration 9 (to be scored on a fresh run, never on this one):\n"
          "  P30 G19-G22 hold (guarantees).\n"
          "  P31 action-only NeSy readers have higher rule consistency and higher concept error than the 50-label probe (shortcut signature).\n"
          "  P32 label consistency picks an action-only NeSy reader; test-then-select has lower regret than label consistency.\n"
          "  P33 test-then-select has regret <= every label-free baseline on BDD-OIA and <= .01 on MNIST.\n"
          "  P34 the label-free identification deficit (rho - sigma_min) ranks the pair programs by A2 error (Spearman >= .8), and rho > .5 on parity.\n"
          "  P35 KSE beats random frame labelling at every budget on BDD-OIA.")
    save_json(ckpt("s16e_v11_verdicts.json"), {k: [v[0], bool(v[1]), v[2], v[3]] for k, v in V11V.items()}); hf_push(RUN_NAME, force=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17. v12 — real traffic signs with exact traffic-law knowledge (GTSRB; downloaded from Hugging Face, no Google Drive)

    **Why.** BDD-OIA is reachable only through a Google Drive file with a download quota. GTSRB is the standard real-world benchmark
    for traffic signs: 12,630 official test photos taken from a moving car, read from the Hugging Face copy tanganke/gtsrb, already cropped to the sign, labels in the official order.

    **Knowledge.** Each of the 43 sign classes has an exact visual design, encoded as 12 concepts: shape (circle, triangle,
    inverted triangle, octagon, diamond), red, blue background, yellow centre, grey "end" stripe, number, arrow or curve, and
    pictogram. The task label is the **legal action** the sign imposes. There are 8 actions: limit, end of restriction, prohibit,
    warn, priority, yield, stop and mandatory direction. An action allows several designs. For example, "warn" allows a plain red
    triangle, one with an arrow and one with a pictogram. So the fibre is a small set of joint concept vectors. It couples concepts
    (Theorem 3), and part of it is unforced.

    **What is tested.** The same audit as 16d, written once for any list of rule blocks:
    - the sharp certificate and its validity;
    - Theorem 9;
    - the shortcut signature of action-only NeSy readers;
    - rule-constrained Dawid–Skene with the cross-fit over-identification test;
    - test-then-select (Proposition 12);
    - KSE.

    Here the concept truth is the class design, so the knowledge (K) holds exactly.
    """)
    return


@app.cell
def _(CFG, CLIP_MEAN, CLIP_STD, DEV, Image, RUN_NAME, SMOKE, atomic_write, check_deadline, ckpt, clip_image_embed, clip_text_embed, ensure_pkg, gpu_free, hf_push, load_clip, np, os, re, torch):
    # ---- 17b. v12 experiment E-D data: real German traffic-sign photos (GTSRB official test set, 12,630 sign crops, from the
    #      Hugging Face dataset tanganke/gtsrb, whose label order equals the official ClassId 0-42),
    #      a 12-concept visual table for the 43 sign classes, and the traffic-law grouping of signs into 8 actions (the task label) ----
    V12 = dict(n_train=int(os.environ.get("V12_N_TRAIN", 3000)), n_test=int(os.environ.get("V12_N_TEST", 3000)), n_probe=50, n_val=100,
               seed=CFG["seed"] + 404)
    V12_HF = ("tanganke/gtsrb", "data/test-00000-of-00001.parquet")
    V12_CONCEPTS = ["the sign is circular", "the sign is an upward triangle", "the sign is an inverted triangle", "the sign is an octagon",
                    "the sign is a diamond", "the sign has red on it", "the sign has a blue background", "the sign has a yellow centre",
                    "the sign has a diagonal grey stripe", "the sign shows a number", "the sign shows an arrow or curve",
                    "the sign shows a vehicle, person or animal"]
    V12_TEMPLATES = ["a photo of a traffic sign: {c}.", "a close-up of a road sign; {c}.", "a German road sign where {c}."]
    #        circ tri inv oct dia red blue yel stripe num arrow pict      (exact visual facts of each official sign design)
    _t = {"speed": (1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0), "warn": (0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0), "warn_arrow": (0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
          "warn_pict": (0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1), "prohib": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0), "prohib_pict": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1),
          "mand": (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0), "end": (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0), "end_num": (1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
          "end_pict": (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1), "yield": (0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0), "stop": (0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
          "priority": (0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0)}
    V12_CLASS = [("speed", "limit")] * 6 + [("end_num", "end")] + [("speed", "limit")] * 2 + [("prohib_pict", "prohibit")] * 2 + \
                [("warn", "warn"), ("priority", "priority"), ("yield", "yield"), ("stop", "stop"), ("prohib", "prohibit"), ("prohib_pict", "prohibit"),
                 ("prohib", "prohibit"), ("warn", "warn"), ("warn_arrow", "warn"), ("warn_arrow", "warn"), ("warn_arrow", "warn"), ("warn", "warn"),
                 ("warn_pict", "warn"), ("warn", "warn"), ("warn_pict", "warn"), ("warn", "warn"), ("warn_pict", "warn"), ("warn_pict", "warn"),
                 ("warn_pict", "warn"), ("warn", "warn"), ("warn_pict", "warn"), ("end", "end")] + [("mand", "mandatory")] * 8 + [("end_pict", "end")] * 2
    V12_CVEC = np.array([_t[k] for k, _ in V12_CLASS], np.int64)                 # (43, 12) concept truth of every class
    V12_ACTIONS = sorted({a for _, a in V12_CLASS}); V12_ACT = np.array([V12_ACTIONS.index(a) for _, a in V12_CLASS])
    _Vu = np.unique(V12_CVEC, axis=0)                                            # the worlds the traffic law allows
    _WL = np.zeros((len(_Vu), len(V12_ACTIONS)), bool)
    for _c in range(43):
        _WL[np.flatnonzero((_Vu == V12_CVEC[_c]).all(1))[0], V12_ACT[_c]] = True
    V12_SPECS = [{"ix": list(range(12)), "V": _Vu, "WL": _WL, "labels": [(a,) for a in range(len(V12_ACTIONS))], "lid": {(a,): a for a in range(len(V12_ACTIONS))}}]


    def v12_replica(path, n=1600, seed=0):
        """SMOKE / offline test only: a parquet with the same schema as tanganke/gtsrb (image bytes + label), drawn sign shapes."""
        from PIL import ImageDraw
        import io, pyarrow as pa, pyarrow.parquet as pq
        r = np.random.default_rng(seed); imgs, labs = [], []
        col = {"red": (200, 20, 20), "blue": (20, 60, 200), "yel": (240, 200, 20), "white": (240, 240, 240), "grey": (90, 90, 90)}
        for i in range(n):
            c = int(r.integers(43)); v = V12_CVEC[c]; s = int(r.integers(40, 90)); im = Image.new("RGB", (s, s), tuple(int(x) for x in r.integers(0, 255, 3)))
            g = ImageDraw.Draw(im); m = s // 8; box = [m, m, s - m, s - m]; fill = col["blue"] if v[6] else col["yel"] if v[7] else col["white"]
            out = col["red"] if v[5] else col["grey"]
            if v[0]:
                g.ellipse(box, fill=fill, outline=out, width=max(2, s // 10))
            elif v[1]:
                g.polygon([(s // 2, m), (s - m, s - m), (m, s - m)], fill=fill, outline=out)
            elif v[2]:
                g.polygon([(m, m), (s - m, m), (s // 2, s - m)], fill=fill, outline=out)
            elif v[3]:
                g.regular_polygon((s // 2, s // 2, s // 2 - m), 8, fill=col["red"])
            else:
                g.polygon([(s // 2, m), (s - m, s // 2), (s // 2, s - m), (m, s // 2)], fill=fill, outline=col["white"])
            if v[8]:
                g.line([(m, s - m), (s - m, m)], fill=col["grey"], width=max(2, s // 12))
            if v[9]:
                g.text((s // 3, s // 3), str(int(r.integers(2, 13)) * 10), fill=(0, 0, 0))
            if v[10]:
                g.line([(s // 3, s // 2), (2 * s // 3, s // 2)], fill=(0, 0, 0), width=3)
            if v[11]:
                g.rectangle([s // 3, s // 3, s // 2, s // 2], fill=(0, 0, 0))
            buf = io.BytesIO(); im.save(buf, format="PNG"); imgs.append({"bytes": buf.getvalue(), "path": f"{i:05d}.png"}); labs.append(c)
        pq.write_table(pa.table({"image": imgs, "label": labs}), path)


    def v12_gtsrb():
        """PNG/JPEG bytes and official class ids (0-42) of the GTSRB test set, read from the Hugging Face parquet (cached)."""
        root = os.path.join(CFG["work_dir"], "data", "gtsrb"); os.makedirs(root, exist_ok=True)
        ensure_pkg("pyarrow")
        import pyarrow.parquet as pq
        f = os.environ.get("GTSRB_PARQUET", os.path.join(root, V12_HF[1]))
        if SMOKE and not os.path.exists(f):
            os.makedirs(os.path.dirname(f), exist_ok=True); v12_replica(f)
        if not os.path.exists(f):
            from huggingface_hub import hf_hub_download
            print("  downloading", V12_HF, flush=True)
            f = hf_hub_download(repo_id=V12_HF[0], filename=V12_HF[1], repo_type="dataset", local_dir=root)
        t = pq.read_table(f, columns=["image", "label"]).to_pydict()
        return [d["bytes"] for d in t["image"]], np.array(t["label"], np.int64)


    V12_IMGS, V12_CLS = v12_gtsrb()
    _r12 = np.random.default_rng(V12["seed"]); _perm12 = _r12.permutation(len(V12_IMGS))
    _ntr12 = min(V12["n_train"], len(V12_IMGS) // 2); _nte12 = min(V12["n_test"], len(V12_IMGS) - _ntr12)
    V12_IDX = {"train": np.sort(_perm12[:_ntr12]), "test": np.sort(_perm12[_ntr12:_ntr12 + _nte12])}


    def v12_pixels(sp):
        """uint8 (n, 224, 224, 3): the sign crop (already cut to the sign by the dataset), resized; cached under data/ (never uploaded)."""
        f = os.path.join(CFG["work_dir"], "data", f"gtsrb_u8_{sp}_{len(V12_IDX[sp])}_{V12['seed']}.npy")
        if not os.path.exists(f):
            import io
            load = lambda i: np.asarray(Image.open(io.BytesIO(V12_IMGS[i])).convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.uint8)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(8) as pool:
                arr = np.stack(list(pool.map(load, V12_IDX[sp])))
            atomic_write(f, lambda t_: np.save(t_, arr)); del arr
        return np.load(f, mmap_mode="r")


    def v12_encode(name):
        """{'train', 'test'} normalised image embeddings and (12, d) prompt-ensemble concept directions for one model; cached."""
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[-60:]
        fi = {sp: ckpt(f"zoo_gtsrb_{sp}_{len(V12_IDX[sp])}_{V12['seed']}_{slug}.npy") for sp in ("train", "test")}; ft = ckpt(f"zoo_gtsrb_txt_{slug}.npy")
        if all(os.path.exists(f) for f in fi.values()) and os.path.exists(ft):
            return {sp: np.load(f) for sp, f in fi.items()}, np.load(ft)
        if SMOKE:                                       # synthetic plumbing only: concept-structured random features
            r = np.random.default_rng(sum(map(ord, name))); d = 64; P = r.normal(size=(12, d))
            img = {sp: V12_CVEC[V12_CLS[ix]] @ P + r.normal(size=(len(ix), d)) * r.uniform(1, 2) for sp, ix in V12_IDX.items()}
            txt = P + r.normal(size=P.shape) * 0.5
        else:
            model, tok = load_clip(name)
            mu = torch.tensor(CLIP_MEAN, device=DEV).view(1, 3, 1, 1); sd = torch.tensor(CLIP_STD, device=DEV).view(1, 3, 1, 1); img = {}
            for sp in ("train", "test"):
                X = v12_pixels(sp); res = []
                for b0 in range(0, len(X), CFG["img_batch"]):
                    check_deadline()
                    x = torch.from_numpy(np.ascontiguousarray(X[b0:b0 + CFG["img_batch"]])).to(DEV).permute(0, 3, 1, 2).float() / 255
                    with torch.inference_mode():
                        res.append(torch.nn.functional.normalize(clip_image_embed(model, (x - mu) / sd), dim=1).cpu().numpy())
                img[sp] = np.concatenate(res)
            with torch.inference_mode():
                txt = sum(torch.nn.functional.normalize(clip_text_embed(model, tok, [tm.format(c=c) for c in V12_CONCEPTS]), dim=1) for tm in V12_TEMPLATES)
            txt = torch.nn.functional.normalize(txt, dim=1).cpu().numpy(); del model; gpu_free()
        img = {sp: (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32) for sp, v in img.items()}
        txt = (txt / np.linalg.norm(txt, axis=1, keepdims=True)).astype(np.float32)
        for sp, f in fi.items():
            atomic_write(f, lambda t_, a=img[sp]: np.save(t_, a))
        atomic_write(ft, lambda t_: np.save(t_, txt)); hf_push(RUN_NAME)
        return img, txt


    print(f"GTSRB: {len(V12_IMGS)} real test photos (Hugging Face {V12_HF[0]}), {len(set(V12_CLS.tolist()))} classes | worlds the traffic law allows: {len(_Vu)} | "
          f"actions: {V12_ACTIONS} | per action, number of allowed concept vectors: {_WL.sum(0).tolist()}")
    return (V12, V12_ACT, V12_CLS, V12_CVEC, V12_HF, V12_IDX, V12_SPECS, v12_encode, v12_replica,)


@app.cell
def _(CFG, DEV, RUN_NAME, SMOKE, V12, V12_ACT, V12_CLS, V12_CVEC, V12_IDX, V12_SPECS, ZOO_FEATS, ckpt, gpu_free, have, hf_push, load_json, nesy_train, np, save_json, text_dirs_centered, torch, v11_bce, v12_encode, zs_logits):
    # ---- 17c. v12 experiment E-D: readers on traffic signs (encoder zoo x {zero-shot, 50-label probe, action-only NeSy x2}),
    #      certificates, Theorem 9, rule-constrained Dawid-Skene with cross-fit test, test-then-select and KSE. Same analysis as
    #      16d, written once for any list of program blocks so it can be reused on new knowledge ----
    V12X = dict(tau_spread=0.02, ds_iters=60, kse_m=(10, 25, 50, 100), kse_reps=20)


    def v12_forced(specs, cons):
        """(n, K) forced concept values (-1 = not forced) from per-block consistency masks over the listed worlds."""
        K = sum(len(s_["ix"]) for s_ in specs); n = cons[0].shape[0]; F = np.full((n, K), -1, np.int64)
        for s_, c in zip(specs, cons):
            V = s_["V"]
            for jj, j in enumerate(s_["ix"]):
                h1 = (c & (V[:, jj] == 1)[None]).any(1); h0 = (c & (V[:, jj] == 0)[None]).any(1)
                F[h1 & ~h0, j] = 1; F[h0 & ~h1, j] = 0
        return F


    def v12_cert(A, specs, cons):
        """Sharp joint and per-concept bounds of err_r - err_s, brackets [LO, HI], rule-consistency and the Theorem 9 check."""
        R, n, K = A.shape; den = n * K; At = torch.as_tensor(A, dtype=torch.float32, device=DEV)
        UPJ = torch.zeros(R, R, device=DEV); UPM = torch.zeros(R, R, device=DEV); LO = torch.zeros(R, device=DEV); HI = torch.zeros(R, device=DEV)
        PF = torch.zeros(R, R, n, device=DEV); inside = torch.ones(R, n, dtype=torch.bool, device=DEV)
        for s_, c in zip(specs, cons):
            ix = s_["ix"]; Vt = torch.as_tensor(s_["V"], dtype=torch.float32, device=DEV); C = torch.as_tensor(c, device=DEV)
            a = At[:, :, ix]; H = a @ (1 - Vt).T + (1 - a) @ Vt.T
            Hm = H.masked_fill(~C[None], float("inf")); LO += Hm.min(-1).values.sum(1); inside &= (Hm.min(-1).values == 0)
            HI += H.masked_fill(~C[None], -float("inf")).max(-1).values.sum(1)
            can1 = (C.float() @ Vt) > 0; can0 = (C.float() @ (1 - Vt)) > 0
            for r in range(R):
                per = (H[r][None] - H).masked_fill(~C[None], -float("inf")).max(-1).values; PF[r] += per; UPJ[r] += per.sum(1)
                ok_s = torch.where(a > 0.5, can1[None], can0[None])
                UPM[r] += ((a[r][None] != a).float() * (2 * ok_s.float() - 1)).sum((1, 2))
        ham = (At[:, None] != At[None]).float().sum(-1); both = inside[:, None] & inside[None]
        t9 = float((PF - ham).abs().masked_fill(~both, 0).max()) if bool(both.any()) else 0.0
        return (UPJ / den).cpu().numpy(), (UPM / den).cpu().numpy(), (LO / den).cpu().numpy(), (HI / den).cpu().numpy(), inside.cpu().numpy(), t9


    def v12_ds(A, specs, cons, readers, iters):
        """Rule-constrained multi-reader Dawid-Skene over the listed worlds of each block; returns concept posteriors (n, K)."""
        As = A[readers].astype(np.float64); M = As.mean(0); pis = [np.full(len(s_["V"]), 1.0 / len(s_["V"])) for s_ in specs]
        for _ in range(iters):
            q1 = ((As * M[None]).sum(1) + 1) / (M.sum(0)[None] + 2); q0 = ((As * (1 - M)[None]).sum(1) + 1) / ((1 - M).sum(0)[None] + 2)
            a = (As * np.log(q1)[:, None] + (1 - As) * np.log(1 - q1)[:, None]).sum(0); b = (As * np.log(q0)[:, None] + (1 - As) * np.log(1 - q0)[:, None]).sum(0)
            for bi, (s_, c) in enumerate(zip(specs, cons)):
                ix, V = s_["ix"], s_["V"]; L = np.log(pis[bi] + 1e-12)[None] + a[:, ix] @ V.T + b[:, ix] @ (1 - V).T
                L = np.where(c, L, -np.inf); L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True); M[:, ix] = P @ V; pis[bi] = P.mean(0)
        return M


    def v12_audit(Z, Zval, Cval, Y, specs, cons, names, seed):
        """Full label-free audit of R concept readers (logits Z (R, n, K)) against exact knowledge; Y (n, K) used only to score."""
        A = (Z > 0).astype(np.int8); R, n, K = A.shape; F = v12_forced(specs, cons)
        err = (A != Y[None]).mean((1, 2)); best = err.min()
        UPJ, UPM, LO, HI, inside, t9 = v12_cert(A, specs, cons)
        CRJ = np.maximum(UPJ, 0).max(1); CRM = np.maximum(UPM, 0).max(1); off = ~np.eye(R, dtype=bool); cert = (UPJ < -1e-9) & off
        W = ((A != F[None]) & (F[None] >= 0)).mean((1, 2)); cons_rate = inside.mean(1)
        P = 1 / (1 + np.exp(-Z)); conf = np.maximum(P, 1 - P).mean((1, 2)); agree = (A == (A.mean(0) > 0.5)[None]).mean((1, 2))
        src = ((Zval > 0) == Cval[None]).mean((1, 2))
        enc = np.array([nm.split(":")[0] for nm in names]); ue = sorted(set(enc)); half = np.isin(enc, ue[0::2])
        Mfull = v12_ds(A, specs, cons, np.arange(R), V12X["ds_iters"])
        Mh = {h: v12_ds(A, specs, cons, np.where(half == h)[0], V12X["ds_iters"]) if (half == h).any() else Mfull for h in (True, False)}
        est = lambda M: (A * (1 - M)[None] + (1 - A) * M[None]).mean((1, 2))
        e_full = est(Mfull); e_cross = np.where(half, est(Mh[False]), est(Mh[True])); spread = np.abs(e_full - e_cross); e_clip = np.clip(e_cross, LO, HI)
        test_pass = bool(np.median(spread) <= V12X["tau_spread"]); sdag = int(np.lexsort((W, CRJ))[0])
        picks = {"S_dagger": sdag, "S": int(np.argmin(W)), "label_consistency": int(np.argmax(cons_rate)), "confidence": int(np.argmax(conf)),
                 "majority_agreement": int(np.argmax(agree)), "source_acc_100_labels": int(np.argmax(src)), "DS_clipped": int(np.argmin(e_clip)),
                 "ladder": int(np.argmin(e_clip)) if test_pass else sdag}
        heads = np.array([nm.split(":")[1] for nm in names])
        out = {"n": n, "forced_frac": float((F >= 0).mean()), "names": names, "err": err.tolist(), "best_err": float(best),
               "valid": bool(np.all(err - best <= CRJ + 1e-12)), "certified_wrong": int((cert & ((err[:, None] - err[None]) >= 0)).sum()),
               "certified_pairs": float(cert.sum() / off.sum()), "top_set": int((~np.any(cert, axis=0)).sum()),
               "gain_mean": float((CRM - CRJ).mean()), "gain_min": float((UPM - UPJ).min()), "theorem9_max_dev": t9,
               "regret": {m: float(err[r] - best) for m, r in picks.items()}, "picked_head": {m: str(heads[r]) for m, r in picks.items()},
               "cert_selected": {m: float(CRJ[r]) for m, r in picks.items()},
               "by_head": {str(h): {"err": float(err[heads == h].mean()), "rule_consistency": float(cons_rate[heads == h].mean())} for h in sorted(set(heads))},
               "DS": {"mae_full": float(np.abs(e_full - err).mean()), "mae_cross": float(np.abs(e_cross - err).mean()), "mae_clipped": float(np.abs(e_clip - err).mean()),
                      "median_spread": float(np.median(spread)), "test_passes": test_pass, "bracket_contains_truth": bool(np.all((LO - 1e-12 <= err) & (err <= HI + 1e-12)))},
               "kse": []}
        rng = np.random.default_rng(seed); unf = F < 0
        Cf = np.flatnonzero(((A != A[:1]) & unf[None]).any((0, 2))); forced_err = ((A != F[None]) & ~unf[None]).sum((1, 2))
        for m in V12X["kse_m"]:
            for _ in range(V12X["kse_reps"]):
                rr = rng.choice(n, min(m, n), replace=False); e_rand = (A[:, rr] != Y[rr][None]).mean((1, 2))
                cc = rng.choice(Cf, min(m, len(Cf)), replace=False) if len(Cf) else np.array([], int)
                e_kse = forced_err + (len(Cf) / max(1, len(cc))) * ((A[:, cc] != Y[cc][None]) & unf[cc][None]).sum((1, 2))
                out["kse"].append({"m": m, "random": float(err[int(np.argmin(e_rand))] - best), "KSE": float(err[int(np.argmin(e_kse))] - best)})
        return out


    def v12_readers():
        ytr = V12_CLS[V12_IDX["train"]]; Ctr = V12_CVEC[ytr]; cons_tr = [V12_SPECS[0]["WL"][:, V12_ACT[ytr]].T]
        r = np.random.default_rng(V12["seed"] + 1); perm = r.permutation(len(ytr)); probe, val = perm[:V12["n_probe"]], perm[V12["n_probe"]:V12["n_probe"] + V12["n_val"]]
        names, Z, Zval = [], [], []
        for name in (list(ZOO_FEATS)[:3] if SMOKE else list(ZOO_FEATS)):
            try:
                img, T = v12_encode(name)
            except Exception as e:
                print(f"gtsrb zoo {name}: skipped ({type(e).__name__}: {str(e)[:120]})"); gpu_free(); continue
            short = name.rstrip("/").split("/")[-1]; Xtr = torch.as_tensor(img["train"], device=DEV); Xte = torch.as_tensor(img["test"], device=DEV)
            Tc = text_dirs_centered(T); zte, (W0, b0) = zs_logits(Xte, Tc, Xtr); heads = {"zs": (zte, zs_logits(Xtr, Tc, Xtr)[0])}
            lin = v11_bce(Xtr[probe], torch.as_tensor(Ctr[probe], dtype=torch.float32, device=DEV))
            with torch.no_grad():
                heads["probe50"] = (lin(Xte).cpu().numpy(), lin(Xtr).cpu().numpy())
            for hn, init in (("nesy_rand", (None, None)), ("nesy_zs", (W0, b0))):
                W, b = nesy_train(Xtr, V12_SPECS, cons_tr, 12, seed=V12["seed"], epochs=CFG["nesy_epochs"], W0=init[0], b0=init[1])
                with torch.no_grad():
                    heads[hn] = ((Xte @ W.T + b).cpu().numpy(), (Xtr @ W.T + b).cpu().numpy())
            for hn, (zt, ztr_) in heads.items():
                names.append(f"{short}:{hn}"); Z.append(zt); Zval.append(ztr_[val])
            print(f"gtsrb zoo {short}: train {img['train'].shape}, test {img['test'].shape}", flush=True)
        return names, np.stack(Z).astype(np.float32), np.stack(Zval).astype(np.float32), Ctr[val]


    if have("s17c_v12_gtsrb.json"):
        V12R = load_json(ckpt("s17c_v12_gtsrb.json"))
    else:
        _n12, _Z12, _Zv12, _Cv12 = v12_readers(); _yte = V12_CLS[V12_IDX["test"]]
        V12R = v12_audit(_Z12, _Zv12, _Cv12, V12_CVEC[_yte], V12_SPECS, [V12_SPECS[0]["WL"][:, V12_ACT[_yte]].T], _n12, V12["seed"] + 7)
        save_json(ckpt("s17c_v12_gtsrb.json"), V12R); hf_push(RUN_NAME, force=True)

    V12_KSE = {m: {k: float(np.mean([x[k] for x in V12R["kse"] if x["m"] == m])) for k in ("random", "KSE")} for m in V12X["kse_m"]}
    print(f"E-D: {len(V12R['names'])} readers on {V12R['n']} real traffic-sign photos; concepts forced by the traffic law: {V12R['forced_frac']:.3f}")
    print("by head (mean concept error / mean rule consistency):", {h: (round(v["err"], 4), round(v["rule_consistency"], 3)) for h, v in V12R["by_head"].items()})
    print(f"certificate valid: {V12R['valid']} | wrong certified orderings: {V12R['certified_wrong']} | pairs certified: {V12R['certified_pairs']:.3f} | "
          f"certified top set: {V12R['top_set']} of {len(V12R['names'])} | joint-vs-box gain {V12R['gain_mean']:.4f} | Theorem 9 max deviation {V12R['theorem9_max_dev']}")
    print("regret by selector [head picked] (certificate of the pick):")
    for _m, _r in V12R["regret"].items():
        print(f"   {_m:22s} {_r:.4f}  [{V12R['picked_head'][_m]}]  ({V12R['cert_selected'][_m]:.4f})")
    print("rung 3 (rule-constrained multi-reader Dawid-Skene):", V12R["DS"])
    print("KSE on traffic signs (mean regret):", {m: {k: round(v, 4) for k, v in d.items()} for m, d in V12_KSE.items()})
    return (V12R, V12_KSE, v12_cert, v12_forced,)


@app.cell
def _(RUN_NAME, V12R, V12_KSE, ckpt, hf_push, save_json):
    # ---- 17d. v12 verdicts and Pre-registration 10 (for a FRESH run: new seed, full zoo) ----
    _b12 = V12R["by_head"]; _g12 = V12R["regret"]; _lf12 = ["S_dagger", "S", "label_consistency", "confidence", "majority_agreement"]
    V12V = {
        "G23": ("GUAR", V12R["valid"] and V12R["certified_wrong"] == 0, "traffic law as exact knowledge: certificate valid for every reader, no certified ordering wrong",
                {"valid": V12R["valid"], "wrong": V12R["certified_wrong"]}),
        "G24": ("GUAR", V12R["theorem9_max_dev"] <= 1e-6 and V12R["gain_min"] >= -1e-9 and V12R["DS"]["bracket_contains_truth"],
                "Theorem 9 identity exact; joint bound never above the per-concept bound; every reader's error inside its bracket",
                {"t9": V12R["theorem9_max_dev"], "gain_min": V12R["gain_min"], "bracket": V12R["DS"]["bracket_contains_truth"]}),
        "X26": ("EXPL", min(_b12["nesy_rand"]["rule_consistency"], _b12["nesy_zs"]["rule_consistency"]) > _b12["probe50"]["rule_consistency"]
                and min(_b12["nesy_rand"]["err"], _b12["nesy_zs"]["err"]) > _b12["probe50"]["err"],
                "shortcut signature on real signs: action-only NeSy readers obey the law more but read the concepts worse than a 50-label probe", _b12),
        "X27": ("EXPL", _g12["ladder"] <= min(_g12[m] for m in _lf12), "test-then-select beats every label-free baseline on traffic signs", _g12),
        "X28": ("EXPL", V12R["certified_pairs"] > 0, "exact traffic law certifies some reader orderings with no labels at all", V12R["certified_pairs"]),
        "X29": ("EXPL", all(v["KSE"] <= v["random"] for v in V12_KSE.values()), "KSE beats random labelling at every budget on traffic signs", V12_KSE),
    }
    for _k, (_kind, _ok, _desc, _val) in V12V.items():
        _tag = ("PASS" if _ok else "FAIL") if _kind == "GUAR" else ("YES " if _ok else "NO  ")
        print(f"{_tag:8s} [{_kind}] {_k} {_desc}\n         {_val}")
    print("\nPre-registration 10 (to be scored on a fresh run, never on this one):\n"
          "  P36 G23-G24 hold.\n"
          "  P37 action-only NeSy readers show the shortcut signature (higher rule consistency, higher concept error than the 50-label probe).\n"
          "  P38 test-then-select has regret <= every label-free baseline.\n"
          "  P39 the law certifies > 0 reader orderings label-free, and KSE beats random labelling at every budget.")
    save_json(ckpt("s17d_v12_verdicts.json"), {k: [v[0], bool(v[1]), v[2], v[3]] for k, v in V12V.items()}); hf_push(RUN_NAME, force=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 18. v13: certified, label-free drift alarms (GTSRB under 7 real corruptions)

    **Why.** A deployed perception model meets fog, blur, sensor noise and compression. Today's tools give no certified answer to
    "did my model just get worse?" when no labels are available. ATC, DoC and average confidence all give *estimates*, and all of
    them can be silently wrong under shift.

    **Theorem 13 (certified drift).** The traffic law alone, with no labels at all, says the true concept vector is one of
    the 13 legal sign designs. Every frame's error is therefore at least its distance to the nearest legal design. Call this the
    certified floor, LO_free. On the clean source, task labels give a certified ceiling, HI.
    - **Alarm.** If LO_free(corrupted) > HI(clean), the error certainly increased, with zero target labels. A false alarm is impossible.
    - **Population form.** With Hoeffding margins, the same comparison certifies the population error, with probability ≥ 1 − 2δ.
    - **Rank reversal.** If the clean labels certify r better than s and the unlabelled corrupted photos certify s better than r, the corruption provably reversed the order.
    - **Projection.** Any estimate below the floor is provably impossible. Projecting ATC, DoC or confidence onto [LO_free, HI_free] never increases their error (Theorem 7).

    **Data.** The Hugging Face dataset tanganke/gtsrb ships the GTSRB test photos under 7 real corruptions: contrast, Gaussian
    noise, impulse noise, JPEG compression, motion blur, pixelation and spatter. Each reader from section 17 is trained once on
    clean photos and then read on clean photos and on 2,000 photos of each corruption.

    **What is tested.**
    - G25 brackets: LO_free ≤ LO_task ≤ error ≤ HI_task.
    - G26: no false alarms.
    - G27: every certified ordering and rank reversal is correct.
    - G28: projection never hurts.
    - The exploratory checks cover:
      - recall of the certified alarm;
      - how often the standard estimators are provably impossible;
      - ranking the corruptions by harm without labels;
      - false alarms of the ATC alarm;
      - label-free reader selection under shift.
    """)
    return


@app.cell
def _(CFG, CLIP_MEAN, CLIP_STD, DEV, Image, RUN_NAME, SMOKE, V12, V12_ACT, V12_CLS, V12_CVEC, V12_HF, V12_IDX, V12_SPECS, ZOO_FEATS, atomic_write, check_deadline, ckpt, clip_image_embed, gpu_free, hf_push, load_clip, nesy_train, np, os, re, text_dirs_centered, torch, v11_bce, v12_encode, v12_replica, zs_logits):
    # ---- 18b. v13 experiment E-E data: the same GTSRB test photos under 7 real corruptions (Hugging Face tanganke/gtsrb splits).
    #      Every reader is trained once on clean photos (as in 17c) and then read on clean and on each corrupted domain ----
    V13 = dict(n_shift=int(os.environ.get("V13_N_SHIFT", 2000)), seed=CFG["seed"] + 505)
    V13_SPLITS = ["contrast", "gaussian_noise", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "spatter"]


    def v13_split(split):
        """(image bytes, class ids) of a random V13['n_shift'] subset of one corrupted split (same parquet schema as the test split)."""
        root = os.path.join(CFG["work_dir"], "data", "gtsrb"); fn = f"data/{split}-00000-of-00001.parquet"
        f = os.environ.get("GTSRB_PARQUET_" + split.upper(), os.path.join(root, fn))
        if SMOKE and not os.path.exists(f):
            os.makedirs(os.path.dirname(f), exist_ok=True); v12_replica(f, n=800, seed=sum(map(ord, split)))
        if not os.path.exists(f):
            from huggingface_hub import hf_hub_download
            print("  downloading", V12_HF[0], fn, flush=True)
            f = hf_hub_download(repo_id=V12_HF[0], filename=fn, repo_type="dataset", local_dir=root)
        import pyarrow.parquet as pq
        t = pq.read_table(f, columns=["image", "label"]).to_pydict(); n = len(t["label"])
        idx = np.sort(np.random.default_rng(V13["seed"] + sum(map(ord, split))).choice(n, min(V13["n_shift"], n), replace=False))
        return [t["image"][i]["bytes"] for i in idx], np.array(t["label"], np.int64)[idx]


    def v13_encode(name, split, imgs):
        """Normalised image embeddings of one corrupted split for one model; cached per (split, model)."""
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[-60:]; f = ckpt(f"zoo_gtsrbc_{split}_{len(imgs)}_{V13['seed']}_{slug}.npy")
        if os.path.exists(f):
            return np.load(f)
        if SMOKE:                                       # synthetic plumbing only: the 17b smoke features with extra noise
            r = np.random.default_rng(sum(map(ord, name + split))); P = np.random.default_rng(sum(map(ord, name))).normal(size=(12, 64))
            x = V12_CVEC[V13_DATA[split][1]] @ P + r.normal(size=(len(imgs), 64)) * r.uniform(1.5, 3)
        else:
            import io
            model, _ = load_clip(name, text=False)
            mu = torch.tensor(CLIP_MEAN, device=DEV).view(1, 3, 1, 1); sd = torch.tensor(CLIP_STD, device=DEV).view(1, 3, 1, 1); res = []
            for b0 in range(0, len(imgs), CFG["img_batch"]):
                check_deadline()
                arr = np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.uint8) for b in imgs[b0:b0 + CFG["img_batch"]]])
                xb = torch.from_numpy(arr).to(DEV).permute(0, 3, 1, 2).float() / 255
                with torch.inference_mode():
                    res.append(torch.nn.functional.normalize(clip_image_embed(model, (xb - mu) / sd), dim=1).cpu().numpy())
            x = np.concatenate(res); del model; gpu_free()
        x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
        atomic_write(f, lambda t_: np.save(t_, x)); hf_push(RUN_NAME)
        return x


    V13_DATA = {}
    for _sp in V13_SPLITS:
        try:
            V13_DATA[_sp] = v13_split(_sp)
        except Exception as _e:
            print(f"corruption {_sp}: skipped ({type(_e).__name__}: {str(_e)[:160]})")


    def v13_readers():
        """For every encoder: heads trained on clean photos exactly as in 17c, read on clean test photos and on each corruption.
        Returns names and logits {domain: (R, n, 12)}, plus clean validation logits/labels (100 labelled clean photos, for ATC)."""
        ytr = V12_CLS[V12_IDX["train"]]; Ctr = V12_CVEC[ytr]; cons_tr = [V12_SPECS[0]["WL"][:, V12_ACT[ytr]].T]
        r = np.random.default_rng(V12["seed"] + 1); perm = r.permutation(len(ytr)); probe, val = perm[:V12["n_probe"]], perm[V12["n_probe"]:V12["n_probe"] + V12["n_val"]]
        names, Z, Zval = [], {d: [] for d in ["clean"] + list(V13_DATA)}, []
        for name in (list(ZOO_FEATS)[:3] if SMOKE else list(ZOO_FEATS)):
            try:
                img, T = v12_encode(name); X = {"clean": img["test"]}
                for sp, (imgs, _) in V13_DATA.items():
                    X[sp] = v13_encode(name, sp, imgs)
            except Exception as e:
                print(f"drift zoo {name}: skipped ({type(e).__name__}: {str(e)[:120]})"); gpu_free(); continue
            short = name.rstrip("/").split("/")[-1]; Xtr = torch.as_tensor(img["train"], device=DEV); Tc = text_dirs_centered(T)
            heads = {"zs": lambda Q: zs_logits(Q, Tc, Xtr)[0]}
            lin = v11_bce(Xtr[probe], torch.as_tensor(Ctr[probe], dtype=torch.float32, device=DEV))
            heads["probe50"] = lambda Q, lin=lin: lin(Q).detach().cpu().numpy()
            W0, b0 = zs_logits(Xtr, Tc, Xtr)[1]
            for hn, init in (("nesy_rand", (None, None)), ("nesy_zs", (W0, b0))):
                W, b = nesy_train(Xtr, V12_SPECS, cons_tr, 12, seed=V12["seed"], epochs=CFG["nesy_epochs"], W0=init[0], b0=init[1])
                heads[hn] = lambda Q, W=W, b=b: (Q @ W.T + b).cpu().numpy()
            with torch.no_grad():
                for hn, f in heads.items():
                    names.append(f"{short}:{hn}"); Zval.append(f(Xtr[val]))
                    for d, x in X.items():
                        Z[d].append(f(torch.as_tensor(x, device=DEV)))
            print(f"drift zoo {short}: domains {list(X)}", flush=True)
        return names, {d: np.stack(z).astype(np.float32) for d, z in Z.items() if z}, np.stack(Zval).astype(np.float32), Ctr[val]


    V13_NAMES, V13_Z, V13_ZVAL, V13_CVAL = v13_readers()
    V13_Y = {"clean": V12_CLS[V12_IDX["test"]], **{sp: v[1] for sp, v in V13_DATA.items()}}
    print(f"E-E: {len(V13_NAMES)} readers | photos per domain: {dict((d, z.shape[1]) for d, z in V13_Z.items())} | mean concept error per domain:",
          {d: round(float(((z > 0) != V12_CVEC[V13_Y[d]][None]).mean()), 4) for d, z in V13_Z.items()})
    return (V13_CVAL, V13_NAMES, V13_SPLITS, V13_Y, V13_Z, V13_ZVAL,)


@app.cell
def _(DEV, RUN_NAME, V12_ACT, V12_CVEC, V12_SPECS, V13_CVAL, V13_NAMES, V13_SPLITS, V13_Y, V13_Z, V13_ZVAL, ckpt, have, hf_push, load_json, np, save_json, spearmanr, torch, v12_cert, v12_forced):
    # ---- 18c. v13 experiment E-E: certified label-free drift alarms (Theorem 13), certified rank reversal, and the projection of
    #      ATC / DoC / average-confidence error estimates onto the certified bracket (Theorem 7), on 7 real corruptions ----
    V13X = dict(delta=0.05, degrade=0.02, atc_alarm=0.02)


    def v13_frame_bounds(A, cons):
        """Per-frame lower / upper bounds (R, n) of the concept error: Hamming distance to the nearest / farthest allowed world."""
        V = torch.as_tensor(V12_SPECS[0]["V"], dtype=torch.float32, device=DEV); C = torch.as_tensor(cons, device=DEV)
        a = torch.as_tensor(A, dtype=torch.float32, device=DEV); H = (a @ (1 - V).T + (1 - a) @ V.T) / A.shape[2]
        lo = H.masked_fill(~C[None], float("inf")).min(-1).values; hi = H.masked_fill(~C[None], -float("inf")).max(-1).values
        return lo.cpu().numpy(), hi.cpu().numpy()


    def v13_baselines(Zt, Zval, Cval):
        """Standard label-free error estimators per reader, calibrated on 100 labelled clean photos: ATC (Garg et al. 2022),
        DoC (Guillory et al. 2021) and average confidence."""
        out = {"ATC": [], "DoC": [], "AC": []}
        for r in range(Zt.shape[0]):
            pv = 1 / (1 + np.exp(-Zval[r].astype(np.float64))); pt = 1 / (1 + np.exp(-Zt[r].astype(np.float64)))
            sv = np.maximum(pv, 1 - pv).ravel(); st = np.maximum(pt, 1 - pt).ravel(); ev = float(((Zval[r] > 0) != Cval).mean())
            t = np.quantile(sv, ev); out["ATC"].append(float((st < t).mean())); out["DoC"].append(ev + float(sv.mean() - st.mean())); out["AC"].append(float(1 - st.mean()))
        return {k: np.array(v) for k, v in out.items()}


    def v13_run():
        doms = [d for d in ["clean"] + V13_SPLITS if d in V13_Z]; R = len(V13_NAMES); nW = len(V12_SPECS[0]["V"]); hz = lambda n: np.sqrt(np.log(2 / V13X["delta"]) / (2 * n))
        D = {}
        for d in doms:
            A = (V13_Z[d] > 0).astype(np.int8); y = V13_Y[d]; Y = V12_CVEC[y]; n = A.shape[1]
            task = V12_SPECS[0]["WL"][:, V12_ACT[y]].T; free = np.ones((n, nW), bool)       # task-label knowledge / no labels at all
            err = (A != Y[None]).mean((1, 2)); lo_t, hi_t = v13_frame_bounds(A, task); lo_f, hi_f = v13_frame_bounds(A, free)
            UPJ_f = v12_cert(A, V12_SPECS, [free])[0]; UPJ_t = v12_cert(A, V12_SPECS, [task])[0]
            F = v12_forced(V12_SPECS, [free]); W = ((A != F[None]) & (F[None] >= 0)).mean((1, 2))
            base = v13_baselines(V13_Z[d], V13_ZVAL, V13_CVAL)
            D[d] = dict(n=n, err=err, lo_t=lo_t.mean(1), hi_t=hi_t.mean(1), lo_f=lo_f.mean(1), hi_f=hi_f.mean(1), UPJ_f=UPJ_f, UPJ_t=UPJ_t, W=W, base=base,
                        lo_f_lcb=lo_f.mean(1) - hz(n), lo_t_lcb=lo_t.mean(1) - hz(n), hi_t_ucb=hi_t.mean(1) + hz(n), rule_ok=(lo_f == 0).mean(1))
        c = D["clean"]; out = {"domains": doms, "names": V13_NAMES, "R": R, "n": {d: D[d]["n"] for d in doms}}
        # Theorem 13 brackets: LO_free <= LO_task <= err <= HI_task for every reader and domain
        out["bracket_ok"] = bool(all(np.all((D[d]["lo_f"] - 1e-9 <= D[d]["lo_t"]) & (D[d]["lo_t"] - 1e-9 <= D[d]["err"]) & (D[d]["err"] <= D[d]["hi_t"] + 1e-9)) for d in doms))
        out["mean_err"] = {d: float(D[d]["err"].mean()) for d in doms}; out["mean_lo_free"] = {d: float(D[d]["lo_f"].mean()) for d in doms}
        out["rule_violation_rate"] = {d: float(1 - D[d]["rule_ok"].mean()) for d in doms}
        out["certified_share_of_error"] = {d: float(D[d]["lo_f"].sum() / max(D[d]["err"].sum(), 1e-12)) for d in doms}
        # certified drift alarms: shifted LO (no target labels / target task labels) above clean HI (clean task labels)
        al, tot = {}, {k: [0, 0, 0, 0] for k in ("free", "task", "free_pop", "ATC")}               # alarms, false alarms, true degradations caught, degradations
        for d in doms[1:]:
            dl = D[d]["err"] - c["err"]; deg = dl > V13X["degrade"]
            rule = {"free": D[d]["lo_f"] > c["hi_t"] + 1e-12, "task": D[d]["lo_t"] > c["hi_t"] + 1e-12, "free_pop": D[d]["lo_f_lcb"] > c["hi_t_ucb"],
                    "ATC": D[d]["base"]["ATC"] - c["base"]["ATC"] > V13X["atc_alarm"]}
            al[d] = {}
            for k, a in rule.items():
                al[d][k] = {"alarms": int(a.sum()), "false": int((a & (dl <= 0)).sum()), "caught": int((a & deg).sum()), "degraded": int(deg.sum())}
                for i, key in enumerate(("alarms", "false", "caught", "degraded")):
                    tot[k][i] += al[d][k][key]
        out["alarms"] = al; out["alarm_totals"] = {k: {"alarms": v[0], "false": v[1], "recall": v[2] / max(v[3], 1), "degraded": v[3]} for k, v in tot.items()}
        # certified rank reversal: clean task labels certify r < s, the unlabelled shifted domain certifies s < r
        rev, rev_wrong, cert_wrong = {}, 0, 0
        off = ~np.eye(R, dtype=bool)
        for d in doms[1:]:
            cc = (c["UPJ_t"] < -1e-9) & off; cs = (D[d]["UPJ_f"] < -1e-9) & off; rv = cc & cs.T
            rev[d] = int(rv.sum()); e0, e1 = c["err"], D[d]["err"]
            rev_wrong += int((rv & ~((e0[:, None] < e0[None]) & (e1[None] < e1[:, None]))).sum())
            cert_wrong += int((cs & ((e1[:, None] - e1[None]) >= 0)).sum())
        out["rank_reversals"] = rev; out["rank_reversal_wrong"] = rev_wrong; out["certified_wrong_shifted_free"] = cert_wrong
        out["certified_pairs_shifted_free"] = {d: float(((D[d]["UPJ_f"] < -1e-9) & off).sum() / off.sum()) for d in doms}
        # Theorem 7 on the estimators: projection onto [LO_free, HI_free] (no target labels) and onto [LO_task, HI_task]
        est = {}
        for k in ("ATC", "DoC", "AC"):
            e, pf, pt, imp, worse = [], [], [], 0, 0
            for d in doms:
                x = D[d]["base"][k]; t = D[d]["err"]; xf = np.clip(x, D[d]["lo_f"], D[d]["hi_f"]); xt = np.clip(x, D[d]["lo_t"], D[d]["hi_t"])
                e.append(np.abs(x - t)); pf.append(np.abs(xf - t)); pt.append(np.abs(xt - t)); imp += int((x < D[d]["lo_f"] - 1e-12).sum())
                worse += int((np.abs(xf - t) > np.abs(x - t) + 1e-12).sum() + (np.abs(xt - t) > np.abs(x - t) + 1e-12).sum())
            est[k] = {"mae": float(np.concatenate(e).mean()), "mae_proj_free": float(np.concatenate(pf).mean()), "mae_proj_task": float(np.concatenate(pt).mean()),
                      "provably_impossible": imp, "of": R * len(doms), "projection_made_worse": worse}
        out["estimators"] = est
        # domain severity ranking without labels: mean certified floor vs mean true error, across the 8 domains
        out["severity_spearman"] = float(spearmanr([D[d]["lo_f"].mean() for d in doms], [D[d]["err"].mean() for d in doms])[0]) if len(doms) > 2 else float("nan")
        out["severity_spearman_ATC"] = float(spearmanr([D[d]["base"]["ATC"].mean() for d in doms], [D[d]["err"].mean() for d in doms])[0]) if len(doms) > 2 else float("nan")
        # reader selection on each shifted domain with no target labels
        rg = {m: [] for m in ("S_dagger_free", "min_floor", "ATC", "ATC_projected", "DoC", "DoC_projected", "confidence", "clean_best")}
        for d in doms[1:]:
            x = D[d]; e = x["err"]; b = e.min(); CRJ = np.maximum(x["UPJ_f"], 0).max(1)
            pk = {"S_dagger_free": int(np.lexsort((x["W"], CRJ))[0]), "min_floor": int(np.argmin(x["lo_f"])), "ATC": int(np.argmin(x["base"]["ATC"])),
                  "ATC_projected": int(np.argmin(np.clip(x["base"]["ATC"], x["lo_f"], x["hi_f"]))),
                  "DoC": int(np.argmin(x["base"]["DoC"])), "DoC_projected": int(np.argmin(np.clip(x["base"]["DoC"], x["lo_f"], x["hi_f"]))), "confidence": int(np.argmin(x["base"]["AC"])), "clean_best": int(np.argmin(c["err"]))}
            for m, r in pk.items():
                rg[m].append(float(e[r] - b))
        out["selection_regret"] = {m: float(np.mean(v)) if v else float("nan") for m, v in rg.items()}
        out["per_reader"] = {d: {"err": D[d]["err"].tolist(), "lo_free": D[d]["lo_f"].tolist(), "lo_task": D[d]["lo_t"].tolist(), "hi_task": D[d]["hi_t"].tolist(),
                                 **{k: v.tolist() for k, v in D[d]["base"].items()}} for d in doms}
        return out


    if have("s18c_v13_drift.json"):
        V13R = load_json(ckpt("s18c_v13_drift.json"))
    else:
        V13R = v13_run(); save_json(ckpt("s18c_v13_drift.json"), V13R); hf_push(RUN_NAME, force=True)

    print(f"E-E: {V13R['R']} readers x {len(V13R['domains'])} domains {V13R['n']}")
    print("mean concept error per domain:", {d: round(v, 4) for d, v in V13R["mean_err"].items()})
    print("mean certified floor, no labels at all:", {d: round(v, 4) for d, v in V13R["mean_lo_free"].items()})
    print("share of the error certified without labels:", {d: round(v, 3) for d, v in V13R["certified_share_of_error"].items()})
    print("brackets LO_free <= LO_task <= err <= HI_task hold for every reader and domain:", V13R["bracket_ok"])
    print("drift alarms, all shifted domains (false = alarm while the error did not increase; recall over increases > .02):")
    for _k, _v in V13R["alarm_totals"].items():
        print(f"   {_k:9s} alarms {_v['alarms']:4d}  false {_v['false']:3d}  recall {_v['recall']:.3f}  (of {_v['degraded']} degradations)")
    print("certified rank reversals per domain:", V13R["rank_reversals"], "| wrong:", V13R["rank_reversal_wrong"],
          "| wrong certified orderings on unlabelled shifted data:", V13R["certified_wrong_shifted_free"])
    print("error estimators (MAE; after projection onto the no-label bracket; onto the task-label bracket; estimates below the certified floor):")
    for _k, _v in V13R["estimators"].items():
        print(f"   {_k:4s} {_v['mae']:.4f} -> {_v['mae_proj_free']:.4f} -> {_v['mae_proj_task']:.4f}   impossible {_v['provably_impossible']}/{_v['of']}   made worse {_v['projection_made_worse']}")
    print(f"domain severity ranking without labels (Spearman with true error): certified floor {V13R['severity_spearman']:.3f} | ATC {V13R['severity_spearman_ATC']:.3f}")
    print("mean regret of reader selection on shifted domains, no target labels:", {m: round(v, 4) for m, v in V13R["selection_regret"].items()})
    return (V13R, v13_frame_bounds,)


@app.cell
def _(RUN_NAME, V13R, ckpt, hf_push, save_json):
    # ---- 18d. v13 verdicts and Pre-registration 11 (for a FRESH run: new seed, full zoo) ----
    _at = V13R["alarm_totals"]; _es = V13R["estimators"]; _sr = V13R["selection_regret"]
    V13V = {
        "G25": ("GUAR", V13R["bracket_ok"], "Theorem 13: on every domain, LO_free <= LO_task <= error <= HI_task for every reader (no target labels needed for LO_free)",
                V13R["bracket_ok"]),
        "G26": ("GUAR", _at["free"]["false"] == 0 and _at["task"]["false"] == 0, "certified drift alarms never fire when the error did not increase",
                {k: _at[k] for k in ("free", "task")}),
        "G27": ("GUAR", V13R["rank_reversal_wrong"] == 0 and V13R["certified_wrong_shifted_free"] == 0,
                "every certified ordering on unlabelled corrupted photos, and every certified rank reversal, is correct",
                {"reversal_wrong": V13R["rank_reversal_wrong"], "ordering_wrong": V13R["certified_wrong_shifted_free"]}),
        "G28": ("GUAR", all(v["projection_made_worse"] == 0 for v in _es.values()), "Theorem 7: projecting ATC / DoC / confidence onto the certified bracket never increases any error",
                {k: v["projection_made_worse"] for k, v in _es.items()}),
        "X30": ("EXPL", _at["free"]["recall"] > 0, "certified alarms with no target labels catch real degradations", _at["free"]),
        "X31": ("EXPL", sum(v["provably_impossible"] for v in _es.values()) > 0 and all(v["mae_proj_free"] <= v["mae"] for v in _es.values()),
                "standard estimators output provably impossible errors (below the certified floor); projection fixes them for free", _es),
        "X32": ("EXPL", V13R["severity_spearman"] >= 0.8, "the certified floor ranks corruptions by harm without labels (Spearman >= .8)",
                {"floor": V13R["severity_spearman"], "ATC": V13R["severity_spearman_ATC"]}),
        "X33": ("EXPL", _at["ATC"]["false"] > 0, "the conventional ATC alarm raises false alarms; the certified alarm cannot", _at["ATC"]),
        "X34": ("EXPL", _sr["S_dagger_free"] <= min(_sr["ATC"], _sr["DoC"], _sr["confidence"]), "label-free certificate selection on shifted data beats ATC, DoC and confidence", _sr),
        "X35": ("EXPL", sum(V13R["rank_reversals"].values()) > 0, "corruption certifiably reverses some clean reader rankings, with no target labels", V13R["rank_reversals"]),
    }
    for _k, (_kind, _ok, _desc, _val) in V13V.items():
        _tag = ("PASS" if _ok else "FAIL") if _kind == "GUAR" else ("YES " if _ok else "NO  ")
        print(f"{_tag:8s} [{_kind}] {_k} {_desc}\n         {_val}")
    print("\nPre-registration 11 (to be scored on a fresh run, never on this one):\n"
          "  P40 G25-G28 hold.\n"
          "  P41 certified alarms with no target labels have recall > 0 and zero false alarms; ATC alarms have > 0 false alarms.\n"
          "  P42 projection onto the no-label bracket lowers the MAE of ATC, DoC and confidence.\n"
          "  P43 the certified floor ranks the 7 corruptions plus clean by mean error with Spearman >= .8.")
    save_json(ckpt("s18d_v13_verdicts.json"), {k: [v[0], bool(v[1]), v[2], v[3]] for k, v in V13V.items()}); hf_push(RUN_NAME, force=True)
    return


@app.cell
def _(RUN_NAME, V12_ACT, V12_SPECS, V13R, V13_NAMES, V13_Y, V13_Z, ckpt, have, hf_push, load_json, np, save_json, v12_cert, v13_frame_bounds):
    # ---- 18e. Proposition 12 under shift: pick a reader with any estimator (ATC / DoC / confidence), then certify the pick with the law.
    #      The certificate CR(r) = max_s max(0, UP(r, s)) bounds the regret of WHATEVER was picked, with no target labels.
    #      "Law veto": restrict the estimator to the certified top set (readers no other reader certifiably beats) ----
    def v13_prop12():
        doms = [d for d in V13R["domains"] if d != "clean"]; R = len(V13_NAMES); nW = len(V12_SPECS[0]["V"]); out = {"per_domain": {}}
        tot = {}
        for d in doms:
            A = (V13_Z[d] > 0).astype(np.int8); y = V13_Y[d]; n = A.shape[1]; pr = V13R["per_reader"][d]; err = np.array(pr["err"]); best = err.min()
            free = np.ones((n, nW), bool); task = V12_SPECS[0]["WL"][:, V12_ACT[y]].T
            lo_f, hi_f = (v.mean(1) for v in v13_frame_bounds(A, free))
            CR = {}
            for kn, cons in (("free", free), ("task", task)):
                U = v12_cert(A, V12_SPECS, [cons])[0]; np.fill_diagonal(U, 0.0)
                CR[kn] = (np.maximum(U, 0).max(1), ~np.any(U < -1e-9, axis=0))                 # (certificate per reader, certified top set)
            row = {}
            for est in ("ATC", "DoC", "AC"):
                e = np.clip(np.array(pr[est]), lo_f, hi_f); r = int(np.argmin(e))
                for kn, (cr, top) in CR.items():
                    rv = int(np.flatnonzero(top)[np.argmin(e[top])])
                    row[f"{est}|{kn}"] = {"regret": float(err[r] - best), "certificate": float(cr[r]), "covered": bool(err[r] - best <= cr[r] + 1e-6),
                                          "veto_regret": float(err[rv] - best), "veto_certificate": float(cr[rv]), "veto_covered": bool(err[rv] - best <= cr[rv] + 1e-6),
                                          "top_set": int(top.sum())}
                    for k, v in row[f"{est}|{kn}"].items():
                        tot.setdefault(f"{est}|{kn}", {}).setdefault(k, []).append(v)
            out["per_domain"][d] = row
        out["summary"] = {m: {k: (bool(all(v)) if isinstance(v[0], bool) else float(np.mean(v))) for k, v in d.items()} for m, d in tot.items()}
        out["all_covered"] = bool(all(v["covered"] and v["veto_covered"] for v in out["summary"].values()))
        return out


    if have("s18e_v13_prop12.json"):
        V13P = load_json(ckpt("s18e_v13_prop12.json"))
    else:
        V13P = v13_prop12(); save_json(ckpt("s18e_v13_prop12.json"), V13P); hf_push(RUN_NAME, force=True)
    print("Proposition 12 under shift (mean over the corrupted domains): pick by estimator, certify by the law with no target labels (free)\n"
          "or with the target task labels (task). certificate = guaranteed upper bound on the regret of the pick")
    for _m, _v in V13P["summary"].items():
        print(f"   {_m:9s} regret {_v['regret']:.4f}  certificate {_v['certificate']:.4f}  | with law veto: regret {_v['veto_regret']:.4f}  "
              f"certificate {_v['veto_certificate']:.4f}  | certified top set {_v['top_set']:.1f} of {len(V13_NAMES)}")
    _g29 = V13P["all_covered"]; _sd = V13R["selection_regret"]["S_dagger_free"]
    _x36 = min(V13P["summary"]["DoC|task"]["regret"], V13P["summary"]["DoC|task"]["veto_regret"]) <= _sd
    print(f"{'PASS' if _g29 else 'FAIL':8s} [GUAR] G29 Proposition 12: every pick's regret is inside its certificate, for every estimator, knowledge level and domain")
    print(f"{'YES ' if _x36 else 'NO  ':8s} [EXPL] X36 estimator pick + law certificate has regret <= certificate-only selection (S_dagger_free {_sd:.4f})")
    save_json(ckpt("s18e_v13_verdicts.json"), {"G29": ["GUAR", _g29], "X36": ["EXPL", bool(_x36)]}); hf_push(RUN_NAME, force=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 19. v14: reasoning shortcuts on MNIST-EvenOdd (the rsbench protocol), caught and repaired without concept labels

    **Protocol.** This follows rsbench and Marconato et al. (2023). The task label is the sum of two real MNIST digits. Training pairs
    come only from 0+6, 2+8, 4+6, 4+8, 1+5, 3+7, 1+9 and 3+9 (and their mirrors). On this support **36** digit maps give exactly
    the same sums: the identity and 35 reasoning shortcuts. NeSy heads trained on sums alone can learn any of them.

    **rsbench versus ours.**
    - rsbench diagnoses shortcuts **with concept labels**: concept accuracy, collapse, and the learned map.
    - We use **only sum labels**:
      - knowledge brackets on each reader's digit error;
      - the sharp certificate;
      - Theorem 9, which shows that inside the training support *nothing* label-free can separate a shortcut from the truth;
      - on a test set covering all digit pairs, a certified floor that flags every shortcut reader with no wrong flag.

    **Theorem 14 (new).** The number of shortcuts of an additive task is exact and cheap to compute: one root value fixes each
    connected component of the pair graph. The **minimal repair** is the fewest extra pair types whose sum labels leave only the
    identity. For EvenOdd this is **one** pair type, for example 0+1. We retrain the same sum-only heads with 10% extra sum labels on
    that pair and measure the shortcuts disappearing.

    **What is tested.**
    - G30: brackets hold.
    - G31: no wrong certified ordering or flag.
    - G32: the Theorem 9 identity is exact.
    - Exploratory checks:
      - shortcuts are learned;
      - they are invisible in distribution;
      - they are caught on all pairs;
      - shortcut-detection AUROC against the standard checks;
      - the Theorem 14 repair works.
    """)
    return


@app.cell
def _(CFG, DEV, SMOKE, ZOO_FEATS, fit_softmax, gpu_free, itertools, np, os, torch, v10_encode, v10_mnist):
    # ---- 19b. v14 experiment E-F data: the reasoning-shortcut benchmark protocol of rsbench / Marconato et al. (2023), MNIST-EvenOdd.
    #      Two real MNIST digits, task label = their sum. Training pairs come ONLY from 8 digit pairs (and their mirrors):
    #      0+6, 2+8, 4+6, 4+8, 1+5, 3+7, 1+9, 3+9, so the sum alone cannot tell the digits apart (reasoning shortcuts exist).
    #      Readers (shared digit head on frozen CLIP features): zero-shot, a 100-concept-label probe, and NeSy heads trained on sums
    #      only (3 random inits + zero-shot init). Test sets: in-distribution pairs (same 8) and all 100 digit pairs ----
    V14 = dict(n_pairs_train=int(os.environ.get("V14_N_TRAIN", 4000)), n_pairs_test=int(os.environ.get("V14_N_TEST", 3000)), nesy_seeds=(0, 1, 2),
               epochs=int(os.environ.get("V14_EPOCHS", 500)), repair_share=0.10, seed=CFG["seed"] + 606)
    V14_PAIRS = [(0, 6), (2, 8), (4, 6), (4, 8), (1, 5), (3, 7), (1, 9), (3, 9)]
    V14_ALLOWED = sorted(set(V14_PAIRS) | {(b, a) for a, b in V14_PAIRS})



    def v14_count_rs(allowed):
        """Number of deterministic digit maps alpha with alpha(a) + alpha(b) = a + b on every allowed pair (rsbench's shortcut count).
        Exact: on each connected component of the pair graph, alpha(root) fixes every other value (alpha(b) = a + b - alpha(a))."""
        adj = {d: [] for d in range(10)}
        for a, b in allowed:
            adj[a].append(b); adj[b].append(a)
        seen, total = set(), 1
        for d0 in range(10):
            if d0 in seen:
                continue
            if not adj[d0]:
                seen.add(d0); total *= 10; continue
            comp, stack = [], [d0]
            while stack:
                u = stack.pop()
                if u not in seen:
                    seen.add(u); comp.append(u); stack.extend(adj[u])
            good = 0
            for v0 in range(10):
                al, q, ok = {d0: v0}, [d0], True
                while q and ok:
                    u = q.pop()
                    for w in adj[u]:
                        vw = u + w - al[u]
                        if not 0 <= vw <= 9 or (w in al and al[w] != vw):
                            ok = False; break
                        if w not in al:
                            al[w] = vw; q.append(w)
                good += ok
            total *= good
        return total


    def v14_min_repair(allowed, kmax=3):
        """Theorem 14 search: the smallest set of extra digit pairs whose SUM labels make the shortcut count 1 (exhaustive up to kmax)."""
        cand = [(a, b) for a in range(10) for b in range(a, 10) if (a, b) not in allowed]
        for k in range(1, kmax + 1):
            sols = [c for c in itertools.combinations(cand, k) if v14_count_rs(list(allowed) + list(c) + [(b, a) for a, b in c]) == 1]
            if sols:
                return k, sols
        return None, []

    def v14_digits():
        """Train / test digit images and labels (real MNIST via 15b; SMOKE: none, features are synthesised in v14_feats)."""
        if SMOKE:
            r = np.random.default_rng(V14["seed"]); return None, r.integers(0, 10, 2000), None, r.integers(0, 10, 3000)
        return v10_mnist()


    V14_ITR, V14_YTR, V14_ITE, V14_YTE = v14_digits()


    def v14_make_pairs(y, n, allowed, seed):
        """n index pairs (i, j) into y whose digit pair is drawn uniformly from `allowed` (None = all 100 pairs)."""
        r = np.random.default_rng(seed); by = {d: np.flatnonzero(y == d) for d in range(10)}
        al = allowed if allowed is not None else [(a, b) for a in range(10) for b in range(10)]
        al = [p for p in al if len(by[p[0]]) and len(by[p[1]])]; k = r.integers(0, len(al), n)
        return np.array([[r.choice(by[al[t][0]]), r.choice(by[al[t][1]])] for t in k], np.int64)


    V14_IDX = {"train": v14_make_pairs(V14_YTR, V14["n_pairs_train"], V14_ALLOWED, V14["seed"]),
               "test_id": v14_make_pairs(V14_YTE, V14["n_pairs_test"], V14_ALLOWED, V14["seed"] + 1),
               "test_all": v14_make_pairs(V14_YTE, V14["n_pairs_test"], None, V14["seed"] + 2)}
    V14_K, V14_REPAIRS = v14_min_repair(V14_ALLOWED)
    V14_FIX = sorted(set(V14_ALLOWED) | {p for c in V14_REPAIRS[:1] for a, b in c for p in ((a, b), (b, a))})
    _nfix = int(round(V14["n_pairs_train"] * V14["repair_share"]))         # the repair adds only 10% extra sum-labelled pairs
    V14_IDX["train_fix"] = np.concatenate([V14_IDX["train"], v14_make_pairs(V14_YTR, _nfix, [p for p in V14_FIX if p not in V14_ALLOWED], V14["seed"] + 5)])
    print(f"Theorem 14: {len(V14_REPAIRS)} minimal repairs of size {V14_K}; first: {V14_REPAIRS[:1]} | shortcut count {v14_count_rs(V14_ALLOWED)} -> {v14_count_rs(V14_FIX)}")


    def v14_feats():
        """{encoder: (train features, test features, text directions)}; real CLIP via the 15b cache, or synthetic in SMOKE."""
        if SMOKE:
            r = np.random.default_rng(V14["seed"] + 3); mu = r.normal(size=(10, 64)); out = {}
            for i, s in enumerate((0.8, 1.2, 1.6)):
                T = mu + r.normal(size=mu.shape) * 0.6
                out[f"smoke{i}"] = (mu[V14_YTR] + r.normal(size=(len(V14_YTR), 64)) * s, mu[V14_YTE] + r.normal(size=(len(V14_YTE), 64)) * s, T)
            return {k: tuple((v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32) for v in t) for k, t in out.items()}
        out = {}
        for name in ZOO_FEATS:
            try:
                Ftr, T = v10_encode(name, V14_ITR, "train"); Fte, _ = v10_encode(name, V14_ITE, "test"); out[name] = (Ftr, Fte, T)
            except Exception as e:
                print(f"evenodd zoo {name}: skipped ({type(e).__name__}: {str(e)[:120]})"); gpu_free()
        return out


    def v14_nesy(Xa, Xb, s, W0, seed, b0=None):
        """Digit head trained from SUMS ONLY: maximise log sum_{a+b=s} p(a|x1) p(b|x2) (exact semantic / DeepProbLog loss).
        Inputs are centred, x10-scaled features (see v14_readers)."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = ((torch.randn(10, Xa.shape[1], generator=g) * 0.01).to(DEV) if W0 is None else W0.clone()).requires_grad_(True)
        b = (torch.zeros(10, device=DEV) if b0 is None else b0.clone()).requires_grad_(True); opt = torch.optim.Adam([W, b], lr=0.02)
        ar = torch.arange(10, device=DEV); M = ((ar[:, None] + ar[None]) [None] == s[:, None, None]).float()
        for _ in range(V14["epochs"]):
            la = torch.log_softmax(Xa @ W.T + b, 1); lb = torch.log_softmax(Xb @ W.T + b, 1)
            J = la[:, :, None] + lb[:, None, :]; loss = -torch.logsumexp(J.masked_fill(M == 0, -1e9), (1, 2)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return W.detach(), b.detach()


    def v14_readers():
        """Digit predictions (R, n_pairs, 2) on both test sets, names, and training task accuracy of every reader."""
        feats = v14_feats(); itr = V14_IDX["train"]; s_tr = torch.as_tensor(V14_YTR[itr].sum(1), device=DEV)
        r = np.random.default_rng(V14["seed"] + 4); probe = np.concatenate([r.permutation(np.flatnonzero(V14_YTR == d))[:10] for d in range(10)])
        names, pred = [], {"test_id": [], "test_all": [], "train": []}
        for name, (Ftr, Fte, T) in feats.items():
            short = name.rstrip("/").split("/")[-1]; Xtr = torch.as_tensor(Ftr, device=DEV); Xte = torch.as_tensor(Fte, device=DEV); Tt = torch.as_tensor(T, device=DEV)
            mu = Xtr.mean(0); C = lambda Q: (Q - mu) * 10                       # centred, scaled features for the sum-trained heads
            heads = {"zs": lambda Q: Q @ (100 * Tt).T}
            lin = fit_softmax(Xtr[probe], torch.as_tensor(V14_YTR[probe], device=DEV), l2=1e-3, C=10)
            heads["probe100"] = lambda Q, lin=lin: lin(Q)
            Xc = C(Xtr)
            for sd in V14["nesy_seeds"]:
                W, b = v14_nesy(Xc[itr[:, 0]], Xc[itr[:, 1]], s_tr, None, V14["seed"] + sd)
                heads[f"nesy_s{sd}"] = lambda Q, W=W, b=b: C(Q) @ W.T + b
            W, b = v14_nesy(Xc[itr[:, 0]], Xc[itr[:, 1]], s_tr, 10 * Tt, V14["seed"], b0=100 * mu @ Tt.T)   # starts exactly at the zero-shot head
            heads["nesy_zs"] = lambda Q, W=W, b=b: C(Q) @ W.T + b
            ifx = V14_IDX["train_fix"]; s_fx = torch.as_tensor(V14_YTR[ifx].sum(1), device=DEV)
            for sd in V14["nesy_seeds"]:                                         # same sum-only training, on the Theorem 14 repaired support
                W, b = v14_nesy(Xc[ifx[:, 0]], Xc[ifx[:, 1]], s_fx, None, V14["seed"] + sd)
                heads[f"repair_s{sd}"] = lambda Q, W=W, b=b: C(Q) @ W.T + b
            with torch.no_grad():
                for hn, f in heads.items():
                    names.append(f"{short}:{hn}"); dtr = f(Xtr).argmax(1).cpu().numpy(); dte = f(Xte).argmax(1).cpu().numpy()
                    pred["train"].append(dtr[itr]); pred["test_id"].append(dte[V14_IDX["test_id"]]); pred["test_all"].append(dte[V14_IDX["test_all"]])
            print(f"evenodd zoo {short}: {len(heads)} readers", flush=True)
        return names, {k: np.stack(v) for k, v in pred.items()}


    V14_NAMES, V14_PRED = v14_readers()
    V14_TRUE = {"train": V14_YTR[V14_IDX["train"]], "test_id": V14_YTE[V14_IDX["test_id"]], "test_all": V14_YTE[V14_IDX["test_all"]]}
    print(f"E-F (MNIST-EvenOdd): {len(V14_NAMES)} readers | train pairs {len(V14_IDX['train'])} from {len(V14_ALLOWED)} ordered digit pairs | "
          f"test pairs: in-distribution {len(V14_IDX['test_id'])}, all-pairs {len(V14_IDX['test_all'])}")
    return (V14, V14_ALLOWED, V14_FIX, V14_K, V14_NAMES, V14_PRED, V14_REPAIRS, V14_TRUE, v14_count_rs,)


@app.cell
def _(DEV, RUN_NAME, V14, V14_ALLOWED, V14_FIX, V14_K, V14_NAMES, V14_PRED, V14_REPAIRS, V14_TRUE, ckpt, have, hf_push, load_json, np, rankdata, save_json, spearmanr, torch, v14_count_rs):
    # ---- 19c. v14 experiment E-F: what rsbench measures WITH concept labels (concept accuracy, collapse, learned shortcut map) versus
    #      what the knowledge certifies WITHOUT them (brackets, sharp certificate, Theorem 9), on in-distribution and all-pairs test sets ----
    def v14_bounds(P, s):
        """P (R, n, 2) predicted digits, s (n,) sums. Per-frame digit-error bounds over all digit pairs with that sum, the sharp
        pairwise certificate UP(r, t) >= err_r - err_t, and the Theorem 9 deviation (max |UP_frame - Hamming| over jointly consistent frames)."""
        w = np.array([(a, b) for a in range(10) for b in range(10)]); ok = torch.as_tensor(w.sum(1)[None] == s[:, None], device=DEV)
        Pt = torch.as_tensor(P, device=DEV); wt = torch.as_tensor(w, device=DEV)
        mm = ((Pt[:, :, None, 0] != wt[None, None, :, 0]).double() + (Pt[:, :, None, 1] != wt[None, None, :, 1]).double()) / 2      # (R, n, 100)
        lo = mm.masked_fill(~ok[None], float("inf")).min(-1).values; hi = mm.masked_fill(~ok[None], -float("inf")).max(-1).values
        R = P.shape[0]; UP = torch.zeros(R, R, device=DEV, dtype=torch.float64); t9 = 0.0; inside = lo == 0
        for r in range(R):
            per = (mm[r][None] - mm).masked_fill(~ok[None], -float("inf")).max(-1).values; UP[r] = per.mean(1)
            ham = (Pt[r][None] != Pt).double().sum(-1) / 2; both = inside[r][None] & inside
            if bool(both.any()):
                t9 = max(t9, float((per - ham).abs().masked_fill(~both, 0).max()))
        return lo.cpu().numpy(), hi.cpu().numpy(), UP.cpu().numpy(), t9


    def v14_auroc(score, lab):
        """AUROC of `score` for the binary label `lab` (nan when only one class is present)."""
        lab = np.asarray(lab, bool)
        if lab.all() or not lab.any():
            return float("nan")
        rk = rankdata(score); return float((rk[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum()))


    def v14_run():
        R = len(V14_NAMES); off = ~np.eye(R, dtype=bool); heads = np.array([n.split(":")[1] for n in V14_NAMES]); out = {"names": V14_NAMES}
        out["rs_count"] = {"evenodd_support": v14_count_rs(V14_ALLOWED), "repaired_support": v14_count_rs(V14_FIX),
                           "all_pairs": v14_count_rs([(a, b) for a in range(10) for b in range(10)])}
        out["repair"] = {"k": V14_K, "n_minimal": len(V14_REPAIRS), "used": [list(map(list, c)) for c in V14_REPAIRS[:1]], "extra_share": V14["repair_share"]}
        # rsbench-style diagnostics (need concept labels): concept accuracy, collapse, the learned digit map
        Pa, Ya = V14_PRED["test_all"], V14_TRUE["test_all"]
        alpha = np.array([[np.bincount(Pa[r][Ya == d], minlength=10).argmax() if (Ya == d).any() else d for d in range(10)] for r in range(R)])
        task_id = (V14_PRED["test_id"].sum(2) == V14_TRUE["test_id"].sum(1)[None]).mean(1); task_all = (Pa.sum(2) == Ya.sum(1)[None]).mean(1)
        err = {k: (V14_PRED[k] != V14_TRUE[k][None]).mean((1, 2)) for k in ("test_id", "test_all")}
        collapse = 1 - np.array([len(np.unique(Pa[r])) for r in range(R)]) / 10
        has_rs = (task_id >= 0.7) & (err["test_all"] >= 0.2)                  # rsbench's notion: solves the task, gets the concepts wrong
        det_rs = np.array([np.any(al != np.arange(10)) and all(al[p] + al[q] == p + q for p, q in V14_ALLOWED) for al in alpha])
        out["labelled"] = {"concept_acc_all": (1 - err["test_all"]).tolist(), "collapse": collapse.tolist(), "has_shortcut": has_rs.tolist(), "deterministic_shortcut_map": det_rs.tolist(),
                           "alpha": alpha.tolist(), "task_acc_id": task_id.tolist(), "task_acc_all": task_all.tolist()}
        out["n_shortcut_readers"] = int(has_rs.sum()); out["n_deterministic_shortcut_maps"] = int(det_rs.sum()); out["n_readers"] = R
        out["by_head"] = {str(h): {"concept_err_all": float(err["test_all"][heads == h].mean()), "task_acc_id": float(task_id[heads == h].mean()),
                                   "shortcut_share": float(has_rs[heads == h].mean())} for h in sorted(set(heads))}
        # knowledge certificates (task labels only, zero concept labels)
        cert = {}
        for k in ("test_id", "test_all"):
            lo, hi, UP, t9 = v14_bounds(V14_PRED[k], V14_TRUE[k].sum(1)); e = err[k]; LO, HI = lo.mean(1), hi.mean(1); c = (UP < -1e-9) & off
            CR = np.maximum(np.where(off, UP, 0), 0).max(1)
            cert[k] = {"bracket_ok": bool(np.all((LO - 1e-9 <= e) & (e <= HI + 1e-9))), "certified_pairs": float(c.sum() / off.sum()),
                       "certified_wrong": int((c & ((e[:, None] - e[None]) >= 0)).sum()), "theorem9_max_dev": t9,
                       "shortcut_vs_clean_certified": int((c & (~has_rs)[:, None] & has_rs[None]).sum()),
                       "mean_width": float((HI - LO).mean()), "LO": LO.tolist(), "HI": HI.tolist(), "CR": CR.tolist(),
                       "spearman_mid_vs_err": float(spearmanr((LO + HI) / 2, e)[0]) if np.ptp(e) > 0 else float("nan"),
                       "mae_mid": float(np.abs((LO + HI) / 2 - e).mean())}
            cert[k]["auroc_floor"] = v14_auroc(LO, has_rs)
            best = e.min(); sdag = int(np.lexsort((LO, CR))[0])                    # S_dagger, ties broken by the certified floor
            cert[k]["regret"] = {"S_dagger": float(e[sdag] - best), "task_accuracy": float(e[int(np.argmax(task_id if k == 'test_id' else task_all))] - best),
                                 "min_floor": float(e[int(np.argmin(LO))] - best)}
            cert[k]["cert_of_pick"] = {"S_dagger": float(CR[sdag])}
        out["cert"] = cert
        out["auroc_detect_shortcut"] = {"task_acc_id (standard NeSy check)": v14_auroc(-task_id, has_rs), "floor_id (ours, in-distribution)": cert["test_id"]["auroc_floor"],
                                        "task_acc_all (no concept labels)": v14_auroc(-task_all, has_rs), "floor_all (ours, certified)": cert["test_all"]["auroc_floor"]}
        # certified detection: readers whose certified floor on all pairs is >= .2 digit error, with zero concept labels
        fl = np.array(cert["test_all"]["LO"]); flag = fl >= 0.2
        out["certified_flag"] = {"flagged": int(flag.sum()), "flagged_with_shortcut": int((flag & has_rs).sum()),
                                 "flagged_wrongly": int((flag & (err["test_all"] < 0.2)).sum()), "shortcut_readers": int(has_rs.sum())}
        return out


    if have("s19c_v14_evenodd.json"):
        V14R = load_json(ckpt("s19c_v14_evenodd.json"))
    else:
        V14R = v14_run(); save_json(ckpt("s19c_v14_evenodd.json"), V14R); hf_push(RUN_NAME, force=True)

    print(f"E-F: {V14R['n_readers']} readers | deterministic reasoning shortcuts (rsbench count): EvenOdd support {V14R['rs_count']['evenodd_support']}, "
          f"all pairs {V14R['rs_count']['all_pairs']} | readers that learned a shortcut (concept labels needed to see it): {V14R['n_shortcut_readers']}")
    print("by head (digit error on all pairs / task accuracy in distribution / share with a shortcut):",
          {h: (round(v["concept_err_all"], 3), round(v["task_acc_id"], 3), round(v["shortcut_share"], 2)) for h, v in V14R["by_head"].items()})
    for _k, _c in V14R["cert"].items():
        print(f"{_k:8s} bracket ok {_c['bracket_ok']} | certified pairs {_c['certified_pairs']:.3f} (wrong {_c['certified_wrong']}) | clean reader certified better than a "
              f"shortcut reader: {_c['shortcut_vs_clean_certified']} | Theorem 9 dev {_c['theorem9_max_dev']:.2e} | width {_c['mean_width']:.3f} | "
              f"Spearman(mid, err) {_c['spearman_mid_vs_err']:.3f} | regret {_c['regret']}")
    print("AUROC for detecting shortcut readers:", {k: round(v, 3) for k, v in V14R["auroc_detect_shortcut"].items()})
    print("certified flag (floor >= .2 digit error on all pairs, zero concept labels):", V14R["certified_flag"])
    return (V14R,)


@app.cell
def _(RUN_NAME, V14R, ckpt, hf_push, np, save_json):
    # ---- 19d. v14 verdicts and Pre-registration 12 (for a FRESH run: new seed, full zoo) ----
    _ci, _ca = V14R["cert"]["test_id"], V14R["cert"]["test_all"]; _au = V14R["auroc_detect_shortcut"]; _cf = V14R["certified_flag"]
    V14V = {
        "G30": ("GUAR", _ci["bracket_ok"] and _ca["bracket_ok"], "every reader's digit error lies in its knowledge bracket, in distribution and on all pairs",
                {"id": _ci["bracket_ok"], "all": _ca["bracket_ok"]}),
        "G31": ("GUAR", _ci["certified_wrong"] == 0 and _ca["certified_wrong"] == 0 and _cf["flagged_wrongly"] == 0,
                "no certified ordering is wrong, and no reader is flagged by its certified floor unless its digit error really is >= .2",
                {"wrong_id": _ci["certified_wrong"], "wrong_all": _ca["certified_wrong"], "flag_wrong": _cf["flagged_wrongly"]}),
        "G32": ("GUAR", _ci["theorem9_max_dev"] <= 1e-6 and _ca["theorem9_max_dev"] <= 1e-6, "Theorem 9 identity exact on MNIST-EvenOdd",
                {"id": _ci["theorem9_max_dev"], "all": _ca["theorem9_max_dev"]}),
        "X37": ("EXPL", V14R["n_shortcut_readers"] > 0 and V14R["rs_count"]["evenodd_support"] > 1, "sum-only NeSy training on EvenOdd learns real reasoning shortcuts",
                {"rs_count": V14R["rs_count"], "shortcut_readers": V14R["n_shortcut_readers"]}),
        "X38": ("EXPL", _ci["shortcut_vs_clean_certified"] == 0, "Theorem 9 in practice: in distribution, no label-free certificate separates shortcut readers from clean ones",
                _ci["shortcut_vs_clean_certified"]),
        "X39": ("EXPL", _cf["shortcut_readers"] > 0 and _cf["flagged_with_shortcut"] == _cf["shortcut_readers"] and _cf["flagged_wrongly"] == 0,
                "with task labels on all pairs, the certified floor flags every shortcut reader and no clean one, with zero concept labels",
                {"pairwise_certified": _ca["shortcut_vs_clean_certified"], **_cf}),
        "X40": ("EXPL", (_au["floor_all (ours, certified)"] or 0) >= 0.9, "the certified floor detects shortcut readers (AUROC >= .9), matching rsbench's concept-labelled diagnosis", _au),
        "X42": ("EXPL", V14R["rs_count"]["repaired_support"] == 1 and
                np.mean([V14R["by_head"][f"repair_s{_s}"]["concept_err_all"] for _s in (0, 1, 2)]) < np.mean([V14R["by_head"][f"nesy_s{_s}"]["concept_err_all"] for _s in (0, 1, 2)]) and
                np.mean([V14R["by_head"][f"repair_s{_s}"]["shortcut_share"] for _s in (0, 1, 2)]) <= np.mean([V14R["by_head"][f"nesy_s{_s}"]["shortcut_share"] for _s in (0, 1, 2)]),
                "Theorem 14: a few extra sum-labelled pair types (10% more sum labels) make the shortcut count 1 and remove learned shortcuts",
                {"repair": V14R["repair"], "rs_count": V14R["rs_count"], **{h: v for h, v in V14R["by_head"].items() if h.startswith(("nesy_s", "repair"))}}),
        "X41": ("EXPL", _ca["regret"]["min_floor"] <= _ca["regret"]["task_accuracy"], "picking the reader with the lowest certified floor (no concept labels) is as good as picking by task accuracy", _ca["regret"]),
    }
    for _k, (_kind, _ok, _desc, _val) in V14V.items():
        _tag = ("PASS" if _ok else "FAIL") if _kind == "GUAR" else ("YES " if _ok else "NO  ")
        print(f"{_tag:8s} [{_kind}] {_k} {_desc}\n         {_val}")
    print("\nPre-registration 12 (to be scored on a fresh run, never on this one):\n"
          "  P44 G30-G32 hold.\n"
          "  P45 sum-only NeSy heads learn shortcuts on EvenOdd; in distribution none is certified worse than a clean reader (Theorem 9).\n"
          "  P46 on all pairs the certified floor detects shortcut readers with AUROC >= .9 and no wrong flag, using zero concept labels.\n"
          "  P47 the Theorem 14 repair (minimal extra pair types, 10% extra sum labels) lowers the mean digit error and the shortcut share of sum-only NeSy heads.")
    save_json(ckpt("s19d_v14_verdicts.json"), {k: [v[0], bool(v[1]), v[2], v[3]] for k, v in V14V.items()}); hf_push(RUN_NAME, force=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 20. Confirmatory scoring of Pre-registrations 9–12

    Every exploratory "X" check above was chosen after looking at one run, so it is a hypothesis. The pre-registrations turn those
    hypotheses into claims that must hold on a **fresh run** that never saw them. That run uses a new seed, which gives new
    train/test splits, new probe and validation photos and new NeSy initialisations, and a new run name, so nothing resumes from old
    checkpoints.

    How to run it: in the config cell set `seed=1` and the run name `"audit_v6_confirm_s1"`, or set the environment variables
    `FAG_SEED=1 FAG_RUN=audit_v6_confirm_s1`. Then run the whole notebook. The cell below reads the verdict files of sections 16–19
    and prints CONFIRMED, NOT CONFIRMED or NOT SCORABLE for each pre-registration. BDD-OIA claims are not scorable without its frames.
    """)
    return


@app.cell
def _(CFG, RUN_NAME, ckpt, hf_push, load_json, os, save_json):
    # ---- 20b. Confirmatory scoring of Pre-registrations 9-12. Only meaningful on a FRESH run (new seed, new run name), never on the
    #      run that produced the hypotheses. Reads the verdict files written by 16e, 17d, 18d, 18e and 19d ----
    V20_PREREG = {
        "P30": ("G19-G22 hold", ["G19", "G20", "G21", "G22"]), "P31": ("BDD shortcut signature", ["X18"]),
        "P32": ("label consistency picks a NeSy reader; ladder beats it", ["X19"]), "P33": ("ladder <= every label-free baseline", ["X20", "X25"]),
        "P34": ("rho / sigma_min deficit ranks programs by A2 error", ["X23"]), "P35": ("KSE beats random on BDD-OIA", ["X22"]),
        "P36": ("G23-G24 hold", ["G23", "G24"]), "P37": ("GTSRB shortcut signature", ["X26"]), "P38": ("ladder <= every label-free baseline (GTSRB)", ["X27"]),
        "P39": ("law certifies orderings; KSE beats random", ["X28", "X29"]),
        "P40": ("G25-G28 hold", ["G25", "G26", "G27", "G28"]), "P41": ("no-label alarm recall > 0, zero false; ATC has false alarms", ["X30", "X33"]),
        "P42": ("projection lowers ATC / DoC / AC error", ["X31"]), "P43": ("certified floor ranks corruptions (Spearman >= .8)", ["X32"]),
        "P44": ("G30-G32 hold", ["G30", "G31", "G32"]), "P45": ("shortcuts learned and invisible in distribution", ["X37", "X38"]),
        "P46": ("certified floor flags every shortcut reader, AUROC >= .9", ["X39", "X40"]), "P47": ("Theorem 14 repair lowers error and shortcut share", ["X42"]),
        "P48": ("Proposition 12 under shift: every pick inside its certificate", ["G29"]),
    }
    V20_FILES = ["s16e_v11_verdicts.json", "s17d_v12_verdicts.json", "s18d_v13_verdicts.json", "s18e_v13_verdicts.json", "s19d_v14_verdicts.json"]
    V20_V = {}
    for _f in V20_FILES:
        if os.path.exists(ckpt(_f)):
            V20_V.update({_k: bool(_v[1]) for _k, _v in load_json(ckpt(_f)).items()})
    V20_SCORE = {}
    for _p, (_desc, _keys) in V20_PREREG.items():
        _miss = [_k for _k in _keys if _k not in V20_V]
        V20_SCORE[_p] = "NOT SCORABLE" if _miss else ("CONFIRMED" if all(V20_V[_k] for _k in _keys) else "NOT CONFIRMED")
        print(f"{V20_SCORE[_p]:14s} {_p} {_desc}  [{', '.join(_keys)}]" + (f"  missing: {_miss}" if _miss else ""))
    _sc = [v for v in V20_SCORE.values() if v != "NOT SCORABLE"]
    print(f"\nrun {RUN_NAME}, seed {CFG['seed']}: {sum(v == 'CONFIRMED' for v in _sc)} of {len(_sc)} scorable pre-registrations confirmed "
          f"({len(V20_SCORE) - len(_sc)} not scorable, e.g. BDD-OIA absent)")
    save_json(ckpt("s20b_confirmatory.json"), {"run": RUN_NAME, "seed": CFG["seed"], "score": V20_SCORE}); hf_push(RUN_NAME, force=True)
    return


@app.cell
def _(SMOKE, gpu_free, load_json, mo, os, rss_gb, subprocess, sys, time, trim_ram):
    # ---- 20c. ONE-CLICK CONFIRMATORY RUN. Runs a copy of THIS notebook as a separate process with a NEW seed (1) and a NEW run name,
    #      so nothing resumes from the run that produced the hypotheses, then prints the pre-registration scores (cell 20b of the copy).
    #      Nothing to edit. Re-running this cell after a disconnect resumes the copy from its own checkpoints (Hugging Face sync included).
    #      Full log: <work dir of the copy>/confirm_run.log ----
    V20C = dict(seed=1, run="audit_v6_confirm_s1")
    V20C_RESULT = None
    if os.environ.get("FAG_CONFIRM_CHILD") == "1":
        print("20c: this is the confirmatory copy itself; nothing to launch.")
    else:
        import glob, re as _re

        def _v20c_source():
            """Path of this notebook's .py file: marimo's runtime context first, then a search for the file that contains this cell."""
            try:
                from marimo._runtime.context import get_context
                _p = get_context().filename
                if _p and os.path.exists(_p):
                    return os.path.abspath(_p)
            except Exception:
                pass
            _dirs = []
            try:
                _dirs.append(str(mo.notebook_dir()))
            except Exception:
                pass
            _dirs += [os.getcwd(), os.path.expanduser("~")]
            for _d in _dirs:
                for _p in sorted(glob.glob(os.path.join(_d, "**", "*.py"), recursive=True))[:2000]:
                    try:
                        with open(_p, encoding="utf-8") as _fh:
                            _t = _fh.read()
                        if "20c. ONE-CLICK CONFIRMATORY RUN" in _t and "app = marimo.App" in _t and "_confirm_copy" not in _p:
                            return os.path.abspath(_p)
                    except Exception:
                        continue
            return None

        _src = _v20c_source()
        if _src is None:
            raise RuntimeError("20c: could not find this notebook's .py file. Save the notebook (Ctrl+S) and run this cell again.")
        with open(_src, encoding="utf-8") as _fh:
            _code = _fh.read()
        # make the seed configurable in the copy: the 'seed=' line directly above 'work_dir=' in the config cell
        _code, _k = _re.subn(r"(\n[ \t]*)seed=\d+,(?=[^\n]*\n[ \t]*work_dir=)", r'\1seed=int(os.environ.get("FAG_SEED", 0)),', _code, count=1)
        if _k == 0 and not _re.search(r'seed=int\(os\.environ\.get\("FAG_SEED", 0\)\),[^\n]*\n[ \t]*work_dir=', _code):
            raise RuntimeError("20c: could not find the 'seed=' line of the config cell.")
        _work = os.path.abspath("fag_work_" + V20C["run"]); os.makedirs(_work, exist_ok=True)
        _done = os.path.join(_work, "s20b_confirmatory.json")
        _copy = os.path.join(_work, "_confirm_copy_notebook.py")
        with open(_copy, "w", encoding="utf-8") as _fh:
            _fh.write(_code)
        if os.path.exists(_done):
            print(f"20c: the confirmatory run already finished ({_done}); showing its result. Delete that file to run it again.")
        else:
            _env = dict(os.environ, FAG_CONFIRM_CHILD="1", FAG_SEED=str(V20C["seed"]), FAG_RUN=V20C["run"], FAG_WORK=_work, SMOKE="1" if SMOKE else "0")
            gpu_free(); trim_ram()
            print(f"20c: launching the confirmatory copy | seed {V20C['seed']} | run {V20C['run']} | work dir {_work} | this kernel uses {rss_gb():.1f} GB RAM", flush=True)
            _log = os.path.join(_work, "confirm_run.log"); _keep = ("PASS", "FAIL", "YES ", "NO  ", "CONFIRMED", "NOT ", "run ", "E-", "Traceback", "Error", "error:")
            _t0 = time.time(); _tail = []
            with open(_log, "a", encoding="utf-8") as _lf:
                _proc = subprocess.Popen([sys.executable, _copy], cwd=os.getcwd(), env=_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for _line in _proc.stdout:
                    _lf.write(_line); _tail = (_tail + [_line.rstrip()])[-60:]
                    if _line.startswith(_keep):
                        print(f"[{(time.time() - _t0) / 60:6.1f} min] {_line.rstrip()[:300]}", flush=True)
                _rc = _proc.wait()
            if _rc != 0:
                print("\n".join(_tail))
                raise RuntimeError(f"20c: the confirmatory copy stopped with exit code {_rc}. The last 60 lines are above; full log: {_log}. "
                                   "Run this cell again to resume from its checkpoints.")
        V20C_RESULT = load_json(_done) if os.path.exists(_done) else None
        if V20C_RESULT is not None and int(V20C_RESULT["seed"]) != V20C["seed"]:
            raise RuntimeError(f"20c: the copy ran with seed {V20C_RESULT['seed']}, not {V20C['seed']}; delete {_work} and run this cell again.")
        if V20C_RESULT is None:
            print(f"20c: the copy finished but wrote no score file; see {_log}")
        else:
            _s = V20C_RESULT["score"]; _sc = [v for v in _s.values() if v != "NOT SCORABLE"]
            print(f"\n===== CONFIRMATORY RESULT: run {V20C_RESULT['run']}, seed {V20C_RESULT['seed']} =====")
            for _p, _v in _s.items():
                print(f"   {_v:14s} {_p}")
            print(f"   {sum(v == 'CONFIRMED' for v in _sc)} of {len(_sc)} scorable pre-registrations confirmed; {len(_s) - len(_sc)} not scorable "
                  )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 21. Head-to-head with COCOCO on MNIST-EvenOdd

    COCOCO (Bortolotti et al., 2026) builds conformal sets for the concepts and the label, then revises them with the knowledge. It needs concept labels for calibration and its coverage needs exchangeability. Our certificate needs task labels only. This section runs both on the same 90 readers and asks two questions. Does COCOCO keep its coverage when the test pairs leave the training support? Which method detects reasoning-shortcut readers better? It also reports rsbench's own concept metrics (macro concept F1 and concept collapse).
    """)
    return


@app.cell
def _(DEV, RUN_NAME, V14, V14_IDX, V14_YTE, V14_YTR, ckpt, fit_softmax, gpu_free, have, hf_push, load_json, save_json, v14_auroc, v14_feats, v14_nesy):
    # ---- 21b. v15 head-to-head on MNIST-EvenOdd: COCOCO (Bortolotti et al. 2026, conformal concept + label sets revised by the knowledge,
    #      calibrated WITH concept labels) versus the knowledge certificate of this paper (task labels only, ZERO concept labels).
    #      Same 90 readers as 19b (retrained here with the same seeds so their probabilities are available), plus rsbench's own
    #      concept metrics (macro concept F1 and concept collapse). COCOCO is re-implemented from the paper (Eq. 2, 5, 10; score -log p).
    #      Questions: (1) does COCOCO keep its coverage when the test pairs move from the training support to all pairs?
    #      (2) which detects reasoning-shortcut readers better: COCOCO set size (1200 concept labels) or our certified floor (none)? ----
    import numpy as np_21b
    import torch as torch_21b
    from scipy.stats import spearmanr as spearmanr_21b
    V15_21b = dict(alpha=0.1, beta=0.1, cal_share=0.2, n_splits=20, seed=V14["seed"] + 707)   # COCOCO is averaged over 20 random calibration splits, as its paper averages over seeds


    def v15_readers_21b():
        """Digit probabilities (R, n_test_digits, 10) on the MNIST test digits, for exactly the reader set of 19b."""
        feats = v14_feats(); itr = V14_IDX["train"]; s_tr = torch_21b.as_tensor(V14_YTR[itr].sum(1), device=DEV)
        r = np_21b.random.default_rng(V14["seed"] + 4); probe = np_21b.concatenate([r.permutation(np_21b.flatnonzero(V14_YTR == d))[:10] for d in range(10)])
        names, prob = [], []
        for name, (Ftr, Fte, T) in feats.items():
            short = name.rstrip("/").split("/")[-1]; Xtr = torch_21b.as_tensor(Ftr, device=DEV); Xte = torch_21b.as_tensor(Fte, device=DEV); Tt = torch_21b.as_tensor(T, device=DEV)
            mu = Xtr.mean(0); C = lambda Q, mu=mu: (Q - mu) * 10
            heads = {"zs": lambda Q, Tt=Tt: Q @ (100 * Tt).T}
            lin = fit_softmax(Xtr[probe], torch_21b.as_tensor(V14_YTR[probe], device=DEV), l2=1e-3, C=10)
            heads["probe100"] = lambda Q, lin=lin: lin(Q)
            Xc = C(Xtr)
            for sd in V14["nesy_seeds"]:
                W, b = v14_nesy(Xc[itr[:, 0]], Xc[itr[:, 1]], s_tr, None, V14["seed"] + sd)
                heads[f"nesy_s{sd}"] = lambda Q, W=W, b=b, C=C: C(Q) @ W.T + b
            W, b = v14_nesy(Xc[itr[:, 0]], Xc[itr[:, 1]], s_tr, 10 * Tt, V14["seed"], b0=100 * mu @ Tt.T)
            heads["nesy_zs"] = lambda Q, W=W, b=b, C=C: C(Q) @ W.T + b
            ifx = V14_IDX["train_fix"]; s_fx = torch_21b.as_tensor(V14_YTR[ifx].sum(1), device=DEV)
            for sd in V14["nesy_seeds"]:
                W, b = v14_nesy(Xc[ifx[:, 0]], Xc[ifx[:, 1]], s_fx, None, V14["seed"] + sd)
                heads[f"repair_s{sd}"] = lambda Q, W=W, b=b, C=C: C(Q) @ W.T + b
            with torch_21b.no_grad():
                for hn, f in heads.items():
                    names.append(f"{short}:{hn}"); prob.append(torch_21b.softmax(f(Xte).double(), 1).cpu().numpy())
            del Xtr, Xte, Xc; gpu_free()
            print(f"v15 readers {short}: {len(heads)}", flush=True)
        return names, np_21b.stack(prob)


    def v15_cp_quantile_21b(s, level):
        """Split-conformal quantile: the ceil((n+1) level)-th smallest score (inf when that exceeds n)."""
        s = np_21b.sort(np_21b.asarray(s, np_21b.float64)); k = int(np_21b.ceil((len(s) + 1) * level))
        return float(s[k - 1]) if k <= len(s) else float("inf")


    def v15_cococo_21b(p1, p2, c, q=None):
        """COCOCO on digit pairs. p1, p2 (n, 10) digit probabilities, c (n, 2) true digits. Calibrates when q is None
        (returns the quantiles), otherwise returns per-pair coverage and sizes of the plain and the revised sets."""
        grid = np_21b.add.outer(np_21b.arange(10), np_21b.arange(10)); ar = np_21b.arange(len(c))
        py = np_21b.zeros((len(c), 19))
        for a in range(10):
            for b in range(10):
                py[:, a + b] += p1[:, a] * p2[:, b]
        ls_y = -np_21b.log(np_21b.clip(py, 1e-300, None)); ls_1 = -np_21b.log(np_21b.clip(p1, 1e-300, None)); ls_2 = -np_21b.log(np_21b.clip(p2, 1e-300, None))
        y = c.sum(1)
        if q is None:
            a, b = V15_21b["alpha"], V15_21b["beta"] / 2                                     # Bonferroni beta_j = beta / k, k = 2 concepts
            return {"y": v15_cp_quantile_21b(ls_y[ar, y], 1 - a), "c1": v15_cp_quantile_21b(ls_1[ar, c[:, 0]], 1 - b), "c2": v15_cp_quantile_21b(ls_2[ar, c[:, 1]], 1 - b)}
        U = ls_y <= q["y"]; G1 = ls_1 <= q["c1"]; G2 = ls_2 <= q["c2"]
        Gx = G1[:, :, None] & G2[:, None, :]                                          # product concept set (Eq. 5)
        Grev = Gx & U[:, grid]                                                        # revision (Eq. 10): keep pairs whose sum is in the label set
        Urev = U & np_21b.stack([(Gx & (grid == s)[None]).any((1, 2)) for s in range(19)], 1)
        hit = Gx[ar, c[:, 0], c[:, 1]]; hit_rev = Grev[ar, c[:, 0], c[:, 1]]
        return {"cov_concept": float(hit.mean()), "cov_concept_rev": float(hit_rev.mean()), "cov_label": float(U[ar, y].mean()),
                "cov_label_rev": float(Urev[ar, y].mean()), "size_concept": float(Gx.sum((1, 2)).mean()), "size_concept_rev": float(Grev.sum((1, 2)).mean()),
                "size_label_rev": float(Urev.sum(1).mean()), "empty_rev": float((Grev.sum((1, 2)) == 0).mean())}


    def v15_floor_21b(P, s):
        """Our certified floor per reader: mean over pairs of min_{a+b=s} digit mismatch (task labels only)."""
        w = np_21b.array([(a, b) for a in range(10) for b in range(10)]); ok = w.sum(1)[None] == s[:, None]               # (n, 100)
        mm = ((P[:, :, None, 0] != w[None, None, :, 0]).astype(np_21b.float32) + (P[:, :, None, 1] != w[None, None, :, 1])) / 2  # (R, n, 100)
        lo = np_21b.where(ok[None], mm, np_21b.inf).min(-1)
        return lo.mean(1), lo


    def v15_f1_21b(pred, true):
        """rsbench concept metrics: macro F1 over the 10 digits and concept collapse (1 - share of digit values ever predicted)."""
        f1 = []
        for d in range(10):
            tp = np_21b.sum((pred == d) & (true == d)); fp = np_21b.sum((pred == d) & (true != d)); fn = np_21b.sum((pred != d) & (true == d))
            f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        return float(np_21b.mean(f1)), 1 - len(np_21b.unique(pred)) / 10


    def v15_run_21b():
        names, prob = v15_readers_21b(); R = len(names); out = {"names": names, "alpha": V15_21b["alpha"], "beta": V15_21b["beta"]}
        r = np_21b.random.default_rng(V15_21b["seed"]); idd = V14_IDX["test_id"]; ncal = int(round(V15_21b["cal_share"] * len(idd))); alls = V14_IDX["test_all"]
        splits = [r.permutation(len(idd)) for _ in range(V15_21b["n_splits"])]
        cal, ev = idd[splits[0][:ncal]], idd[splits[0][ncal:]]                    # split 0 also serves our certificate's in-distribution evaluation
        out["n_cal_pairs"] = ncal; out["n_splits"] = V15_21b["n_splits"]; out["concept_labels_used"] = {"cococo": 2 * ncal, "ours": 0}
        c_ev, c_all = V14_YTE[ev], V14_YTE[alls]

        def avg(ds):
            return {k: float(np_21b.mean([d[k] for d in ds])) for k in ds[0]}
        res = {"id": [], "all": []}; floor = {"id": [], "all": []}; err = {"id": [], "all": []}; task = {"id": [], "all": []}; f1, cls = [], []
        for k in range(R):
            rid, rall = [], []
            for sp in splits:
                ca, eva = idd[sp[:ncal]], idd[sp[ncal:]]
                q = v15_cococo_21b(prob[k][ca[:, 0]], prob[k][ca[:, 1]], V14_YTE[ca])
                rid.append(v15_cococo_21b(prob[k][eva[:, 0]], prob[k][eva[:, 1]], V14_YTE[eva], q)); rall.append(v15_cococo_21b(prob[k][alls[:, 0]], prob[k][alls[:, 1]], c_all, q))
            res["id"].append(avg(rid)); res["all"].append(avg(rall))
            for tag, ix, cc in (("id", ev, c_ev), ("all", alls, c_all)):
                P = prob[k].argmax(1)[ix]; err[tag].append(float((P != cc).mean())); task[tag].append(float((P.sum(1) == cc.sum(1)).mean()))
            Pa = prob[k].argmax(1)[alls]; a_, b_ = v15_f1_21b(Pa.ravel(), c_all.ravel()); f1.append(a_); cls.append(b_)
        for tag, ix, cc in (("id", ev, c_ev), ("all", alls, c_all)):
            P = np_21b.stack([prob[k].argmax(1)[ix] for k in range(R)]); fl, _ = v15_floor_21b(P, cc.sum(1)); floor[tag] = fl.tolist()
        E = {t: np_21b.array(err[t]) for t in err}; TK = {t: np_21b.array(task[t]) for t in task}; FL = {t: np_21b.array(floor[t]) for t in floor}
        has_rs = (TK["id"] >= 0.7) & (E["all"] >= 0.2)
        nominal = 1 - V15_21b["alpha"] - V15_21b["beta"]
        out["readers"] = {"err": {t: v.tolist() for t, v in E.items()}, "task": {t: v.tolist() for t, v in TK.items()}, "floor": floor, "has_shortcut": has_rs.tolist(),
                          "rsbench_f1_all": f1, "rsbench_collapse_all": cls, "cococo": res}
        out["n_shortcut_readers"] = int(has_rs.sum())
        cov = {t: np_21b.array([x["cov_concept_rev"] for x in res[t]]) for t in res}; covl = {t: np_21b.array([x["cov_label_rev"] for x in res[t]]) for t in res}
        size = {t: np_21b.array([x["size_concept_rev"] for x in res[t]]) for t in res}
        out["coverage"] = {t: {"concept_rev_mean": float(cov[t].mean()), "concept_rev_min": float(cov[t].min()), "label_rev_mean": float(covl[t].mean()),
                               "label_rev_min": float(covl[t].min()), "readers_below_nominal": int((cov[t] < nominal - 0.02).sum()),
                               "shortcut_readers_below_nominal": int(((cov[t] < nominal - 0.02) & has_rs).sum()),
                               "size_concept_rev_mean": float(size[t].mean())} for t in res}
        out["coverage"]["nominal"] = nominal
        out["ours_valid"] = {t: bool(np_21b.all(FL[t] <= E[t] + 1e-9)) for t in FL}
        out["auroc_detect_shortcut"] = {
            "cococo_concept_set_size_id (1200 concept labels)": v14_auroc(size["id"], has_rs),
            "cococo_label_set_size_all (concept labels for calibration)": v14_auroc(np_21b.array([x["size_label_rev"] for x in res["all"]]), has_rs),
            "task_acc_all (no concept labels)": v14_auroc(-TK["all"], has_rs),
            "ours_certified_floor_all (0 concept labels)": v14_auroc(FL["all"], has_rs),
            "rsbench_1_minus_f1_all (all concept labels)": v14_auroc(1 - np_21b.array(f1), has_rs)}
        flag = FL["all"] >= 0.2
        out["ours_flag"] = {"flagged": int(flag.sum()), "flagged_with_shortcut": int((flag & has_rs).sum()), "flagged_wrongly": int((flag & (E["all"] < 0.2)).sum()),
                            "shortcut_readers": int(has_rs.sum())}
        out["spearman_floor_vs_rsbench_f1"] = float(spearmanr_21b(FL["all"], -np_21b.array(f1))[0]) if np_21b.ptp(f1) > 0 else float("nan")
        return out


    if have("s21b_v15_cococo.json"):
        V15R = load_json(ckpt("s21b_v15_cococo.json"))
    else:
        V15R = v15_run_21b(); save_json(ckpt("s21b_v15_cococo.json"), V15R); hf_push(RUN_NAME, force=True)

    print(f"E-G (COCOCO head-to-head, MNIST-EvenOdd): {len(V15R['names'])} readers, {V15R['n_shortcut_readers']} with a reasoning shortcut | COCOCO averaged over {V15R['n_splits']} calibration splits | "
          f"concept labels used: {V15R['concept_labels_used']} | target coverage 1-alpha-beta = {V15R['coverage']['nominal']:.2f}")
    for t_21b, lab_21b in (("id", "in distribution (calibration support)"), ("all", "all digit pairs (shift)")):
        c_21b = V15R["coverage"][t_21b]
        print(f"  COCOCO {lab_21b:38s} concept coverage mean {c_21b['concept_rev_mean']:.3f} min {c_21b['concept_rev_min']:.3f} | readers below target "
              f"{c_21b['readers_below_nominal']} (shortcut readers {c_21b['shortcut_readers_below_nominal']}) | label coverage min {c_21b['label_rev_min']:.3f} | "
              f"concept set size {c_21b['size_concept_rev_mean']:.2f} | our bracket valid: {V15R['ours_valid'][t_21b]}")
    print("AUROC for detecting shortcut readers:", {k: round(v, 3) for k, v in V15R["auroc_detect_shortcut"].items()})
    print("our certified flag (floor >= .2, zero concept labels):", V15R["ours_flag"], "| Spearman(our floor, rsbench 1-F1):", round(V15R["spearman_floor_vs_rsbench_f1"], 3))
    return (V15R,)


@app.cell
def _(RUN_NAME, V15R, ckpt, hf_push, save_json):
    # ---- 21c. v15 verdicts: COCOCO vs the knowledge certificate ----
    cv_21c = V15R["coverage"]; au_21c = V15R["auroc_detect_shortcut"]; of_21c = V15R["ours_flag"]
    ours_21c = au_21c["ours_certified_floor_all (0 concept labels)"]; coc_21c = au_21c["cococo_concept_set_size_id (1200 concept labels)"]
    V15V_21c = {
        "G33": ("GUAR", V15R["ours_valid"]["id"] and V15R["ours_valid"]["all"] and of_21c["flagged_wrongly"] == 0,
                "our certified floor never exceeds a reader's true digit error, in distribution or under shift, and never flags a clean reader",
                {"valid": V15R["ours_valid"], **of_21c}),
        "X43": ("EXPL", cv_21c["id"]["concept_rev_min"] >= cv_21c["nominal"] - 0.02, "COCOCO works as advertised in distribution: every reader's revised concept set reaches 1-alpha-beta coverage (mean over calibration splits)",
                {k: cv_21c["id"][k] for k in ("concept_rev_mean", "concept_rev_min", "readers_below_nominal")}),
        "X44": ("EXPL", cv_21c["all"]["readers_below_nominal"] > 0, "under the shift that exposes shortcuts (all digit pairs), COCOCO's coverage breaks for some readers; our bracket stays valid by construction",
                {k: cv_21c["all"][k] for k in ("concept_rev_mean", "concept_rev_min", "readers_below_nominal", "shortcut_readers_below_nominal")}),
        "X45": ("EXPL", (ours_21c if ours_21c == ours_21c else 0) >= (coc_21c if coc_21c == coc_21c else 0) - 0.05,
                "with zero concept labels, our certified floor detects shortcut readers at least as well as COCOCO with 1200 concept labels (AUROC within .05)", au_21c),
        "X46": ("EXPL", V15R["spearman_floor_vs_rsbench_f1"] >= 0.7, "our label-free floor ranks readers like rsbench's concept-labelled F1 (Spearman >= .7)",
                V15R["spearman_floor_vs_rsbench_f1"]),
    }
    for k_21c, (kind_21c, ok_21c, desc_21c, val_21c) in V15V_21c.items():
        tag_21c = ("PASS" if ok_21c else "FAIL") if kind_21c == "GUAR" else ("YES " if ok_21c else "NO  ")
        print(f"{tag_21c:8s} [{kind_21c}] {k_21c} {desc_21c}\n         {val_21c}")
    save_json(ckpt("s21c_v15_verdicts.json"), {k: [v[0], bool(v[1]), v[2], v[3]] for k, v in V15V_21c.items()}); hf_push(RUN_NAME, force=True)
    return


@app.cell
def _(CFG, RUN_NAME, ckpt, load_json):
    # ---- 22. Paper numbers: collects every verdict (guarantees G, findings X) and the headline numbers of every result file of this run
    #      into one file, paper_numbers.txt in the work dir, and prints the verdicts. Attach that file in the chat so the paper uses this run's numbers ----
    import glob as glob_22, json as json_22, os as os_22


    def v22_flat_22(obj, path="", depth=0, out=None):
        """(path, value) for every number / bool / short string inside obj, down to depth 3; long lists are skipped."""
        out = [] if out is None else out
        if isinstance(obj, dict) and depth < 4:
            for k, v in obj.items():
                v22_flat_22(v, f"{path}.{k}" if path else str(k), depth + 1, out)
        elif isinstance(obj, (list, tuple)) and len(obj) <= 8 and depth < 4 and all(not isinstance(x, (dict, list)) for x in obj):
            out.append((path, list(obj)))
        elif isinstance(obj, bool) or isinstance(obj, (int, float)) or (isinstance(obj, str) and len(obj) <= 40):
            out.append((path, round(obj, 4) if isinstance(obj, float) else obj))
        return out


    lines_22 = [f"PAPER NUMBERS | run {RUN_NAME} | seed {CFG['seed']}"]
    for f_22 in sorted(glob_22.glob(ckpt("s*verdict*.json"))):
        lines_22.append(f"## {os_22.path.basename(f_22)}")
        try:
            for k_22, v_22 in load_json(f_22).items():
                v_22 = list(v_22) if isinstance(v_22, (list, tuple)) else [None, v_22]
                ok_22 = bool(v_22[1]) if len(v_22) > 1 else None
                tag_22 = ("PASS" if ok_22 else "FAIL") if v_22[0] == "GUAR" else ("YES" if ok_22 else "NO")
                desc_22 = str(v_22[2])[:90] if len(v_22) > 2 else ""
                val_22 = json_22.dumps(v_22[3], default=str)[:700] if len(v_22) > 3 else ""
                lines_22.append(f"{k_22} {tag_22} {desc_22} :: {val_22}")
        except Exception as e_22:
            lines_22.append(f"(could not read: {type(e_22).__name__})")
    nverd_22 = len(lines_22)
    for f_22 in sorted(glob_22.glob(ckpt("s*.json"))):
        if "verdict" in f_22 or "confirmatory" in f_22:
            continue
        try:
            fl_22 = v22_flat_22(load_json(f_22))
        except Exception as e_22:
            lines_22.append(f"## {os_22.path.basename(f_22)} (could not read: {type(e_22).__name__})"); continue
        if fl_22:
            lines_22.append(f"## {os_22.path.basename(f_22)} ({len(fl_22)} values{', first 120 shown' if len(fl_22) > 120 else ''})")
            lines_22 += [f"{p_22} = {x_22}" for p_22, x_22 in fl_22[:120]]
    PAPER_NUMBERS_22 = "\n".join(lines_22)
    with open(ckpt("paper_numbers.txt"), "w", encoding="utf-8") as fh_22:
        fh_22.write(PAPER_NUMBERS_22)
    print("\n".join(lines_22[:nverd_22]))
    print(f"\nThe full file ({len(lines_22)} lines, verdicts plus every result number) is saved at {ckpt('paper_numbers.txt')}. "
          "Download it from the file panel and attach it in the chat.")
    return (PAPER_NUMBERS_22,)


@app.cell
def _(PAPER_NUMBERS_22, V15R, ckpt, load_json):
    # ---- 23. Paper figures and runtime table. Draws every data figure of the paper from this run's saved results (no recomputation),
    #      times the certificate computations on CPU, and zips everything to <work dir>/paper_figures.zip.
    #      Download that zip from the file panel and attach it in the chat. Figures whose results are missing are skipped, not fatal ----
    import os as os_23, json as json_23, time as time_23, zipfile as zipfile_23
    import numpy as np_23
    import matplotlib.pyplot as plt_23

    FIG_DIR_23 = ckpt("paper_figures"); os_23.makedirs(FIG_DIR_23, exist_ok=True)
    plt_23.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    C_23 = {"ours": "#1f6fb2", "cococo": "#d1495b", "base": "#8c8c8c", "short": "#d1495b", "clean": "#1f6fb2", "repair": "#2a9d55"}
    made_23, skipped_23 = [], []


    def load_23(name):
        """Saved result of this run, or None."""
        return load_json(ckpt(name)) if os_23.path.exists(ckpt(name)) else None


    def save_23(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(os_23.path.join(FIG_DIR_23, f"{name}.{ext}"), bbox_inches="tight")
        plt_23.close(fig); made_23.append(name)


    def fig_drift_23():
        """Fig. drift: mean true error, certified floors and ceiling per corruption (GTSRB), and every reader-domain pair below the diagonal."""
        d = load_23("s18c_v13_drift.json")
        if d is None:
            return "s18c_v13_drift.json missing"
        dom = d["domains"]; pr = d["per_reader"]
        m = {k: [float(np_23.mean(pr[x][k])) for x in dom] for k in ("err", "lo_free", "lo_task", "hi_task")}
        fig, ax = plt_23.subplots(1, 2, figsize=(7.2, 2.6))
        xs = np_23.arange(len(dom))
        ax[0].fill_between(xs, m["lo_task"], m["hi_task"], color=C_23["ours"], alpha=0.15, label="certified bracket (logged decisions)")
        ax[0].plot(xs, m["err"], "k.-", label="true error"); ax[0].plot(xs, m["lo_free"], "--", color=C_23["ours"], label="certified floor (nothing logged)")
        short_23 = {"gaussian_noise": "gauss. noise", "impulse_noise": "impulse", "jpeg_compression": "jpeg", "motion_blur": "motion blur"}
        ax[0].set_xticks(xs); ax[0].set_xticklabels([short_23.get(x, x) for x in dom], fontsize=7, rotation=40, ha="right"); ax[0].set_ylabel("mean concept error"); ax[0].legend(fontsize=6, frameon=False)
        e = np_23.concatenate([pr[x]["err"] for x in dom]); lo = np_23.concatenate([pr[x]["lo_task"] for x in dom]); hi = np_23.concatenate([pr[x]["hi_task"] for x in dom])
        ax[1].scatter(e, lo, s=5, color=C_23["ours"], label="floor"); ax[1].scatter(e, hi, s=5, color=C_23["base"], label="ceiling")
        t = [0, max(float(hi.max()), float(e.max()))]; ax[1].plot(t, t, "k-", lw=0.6)
        ax[1].set_xlabel("true concept error"); ax[1].set_ylabel("certified bound"); ax[1].legend(fontsize=6, frameon=False)
        ax[1].set_title(f"{len(e)} model-domain pairs, all inside", fontsize=8)
        save_23(fig, "fig_drift")


    def fig_projection_23():
        """Fig. projection: error of ATC / DoC / average confidence before and after projection onto the certified bracket."""
        v = load_23("s18d_v13_verdicts.json")
        if v is None or "X31" not in v:
            return "s18d_v13_verdicts.json missing"
        x = v["X31"][3]; est = ["ATC", "DoC", "AC"]; lab = ["ATC", "DoC", "avg. confidence"]
        fig, ax = plt_23.subplots(figsize=(3.6, 2.4)); w = 0.26; xs = np_23.arange(3)
        for j, (k, nm, col) in enumerate((("mae", "raw", C_23["base"]), ("mae_proj_free", "projected, nothing logged", "#7fb3e0"), ("mae_proj_task", "projected, logged decisions", C_23["ours"]))):
            ax.bar(xs + (j - 1) * w, [x[e][k] for e in est], w, color=col, label=nm)
        for i, e in enumerate(est):
            ax.text(i, max(x[e]["mae"], 1e-9) * 1.03, f"{x[e]['provably_impossible']}/{x[e]['of']}\nimpossible", ha="center", fontsize=6)
        ax.set_xticks(xs); ax.set_xticklabels(lab); ax.set_ylabel("mean absolute error of the estimate"); ax.legend(fontsize=6, frameon=False)
        save_23(fig, "fig_projection")


    def fig_shortcut_23():
        """Fig. shortcuts: certified floor (no concept labels) vs true digit error on all pairs; Theorem 14 repair per seed."""
        c = load_23("s19c_v14_evenodd.json")
        if c is None:
            return "s19c_v14_evenodd.json missing"
        lo = np_23.array(c["cert"]["test_all"]["LO"]); err = 1 - np_23.array(c["labelled"]["concept_acc_all"]); rs = np_23.array(c["labelled"]["has_shortcut"], bool)
        fig, ax = plt_23.subplots(1, 2, figsize=(7.2, 2.6))
        ax[0].scatter(err[~rs], lo[~rs], s=10, color=C_23["clean"], label="no shortcut"); ax[0].scatter(err[rs], lo[rs], s=10, color=C_23["short"], marker="^", label="reasoning shortcut")
        t = [0, float(max(err.max(), lo.max()))]; ax[0].plot(t, t, "k-", lw=0.6); ax[0].axhline(0.2, color="k", ls=":", lw=0.6)
        ax[0].set_xlabel("true digit error (needs concept labels)"); ax[0].set_ylabel("certified floor (no concept labels)"); ax[0].legend(fontsize=6, frameon=False)
        ax[0].set_title(f"AUROC {c['auroc_detect_shortcut']['floor_all (ours, certified)']:.3f}; flag above dotted line", fontsize=8)
        bh = c["by_head"]; seeds = [0, 1, 2]; xs = np_23.arange(3); w = 0.38
        ax[1].bar(xs - w / 2, [bh[f"nesy_s{s}"]["concept_err_all"] for s in seeds], w, color=C_23["short"], label="sum-only training")
        ax[1].bar(xs + w / 2, [bh[f"repair_s{s}"]["concept_err_all"] for s in seeds], w, color=C_23["repair"], label="+ one extra pair type (Thm. 14)")
        ax[1].set_xticks(xs); ax[1].set_xticklabels([f"seed {s}" for s in seeds]); ax[1].set_ylabel("mean digit error, all pairs"); ax[1].legend(fontsize=6, frameon=False)
        save_23(fig, "fig_shortcut")


    def fig_cococo_23():
        """Fig. COCOCO: per-model concept coverage in distribution vs all pairs; detection AUROC with the labels each method uses."""
        d = V15R                                                                  # results of cell 21b (this also makes 23 run after 21b)
        cid = np_23.array([x["cov_concept_rev"] for x in d["readers"]["cococo"]["id"]]); call = np_23.array([x["cov_concept_rev"] for x in d["readers"]["cococo"]["all"]])
        rs = np_23.array(d["readers"]["has_shortcut"], bool); nom = d["coverage"]["nominal"]
        fig, ax = plt_23.subplots(1, 2, figsize=(7.6, 2.6), gridspec_kw={"wspace": 0.9})
        ax[0].scatter(cid[~rs], call[~rs], s=10, color=C_23["clean"], label="no shortcut"); ax[0].scatter(cid[rs], call[rs], s=10, color=C_23["short"], marker="^", label="reasoning shortcut")
        ax[0].axhline(nom, color="k", ls=":", lw=0.6); ax[0].axvline(nom, color="k", ls=":", lw=0.6)
        ax[0].set_xlabel("COCOCO coverage, in distribution"); ax[0].set_ylabel("COCOCO coverage, all digit pairs"); ax[0].legend(fontsize=6, frameon=False, loc="center right")
        ax[0].set_title(f"target {nom:.1f}; our bracket valid for all {len(cid)}", fontsize=8)
        au = d["auroc_detect_shortcut"]
        items = [("ours: certified floor\n0 concept labels", au["ours_certified_floor_all (0 concept labels)"], C_23["ours"]),
                 ("COCOCO set size\n1,200 concept labels", au["cococo_concept_set_size_id (1200 concept labels)"], C_23["cococo"]),
                 ("rsbench concept F1\nall concept labels", au["rsbench_1_minus_f1_all (all concept labels)"], C_23["base"])]
        ax[1].barh(range(3), [i[1] for i in items], color=[i[2] for i in items]); ax[1].set_yticks(range(3)); ax[1].set_yticklabels([i[0] for i in items], fontsize=7)
        ax[1].set_xlim(0.5, 1.0); ax[1].set_xlabel("AUROC for detecting shortcut models"); ax[1].invert_yaxis()
        for i, it in enumerate(items):
            ax[1].text(it[1] + 0.005, i, f"{it[1]:.3f}", va="center", fontsize=7)
        save_23(fig, "fig_cococo")


    def fig_labels_23():
        """Fig. label efficiency: selection regret vs number of labels, knowledge-stratified vs random, on three tasks."""
        src = [("s14d_v9_verdicts.json", "X11", "CUB birds", "KSE"), ("s17d_v12_verdicts.json", "X29", "GTSRB signs", "KSE"), ("s15d_v10_verdicts.json", "X16", "MNIST-Add", "KSE_prop")]
        have_any = False; fig, ax = plt_23.subplots(1, 3, figsize=(7.2, 2.3))
        for a, (f, k, title, key) in zip(ax, src):
            v = load_23(f)
            if v is None or k not in v or not isinstance(v[k][3], dict):
                a.set_axis_off(); continue
            have_any = True; x = v[k][3]; ms = sorted(x, key=lambda s: int(s))
            a.plot([int(m) for m in ms], [x[m]["random"] for m in ms], "o-", color=C_23["base"], ms=3, label="random labels")
            a.plot([int(m) for m in ms], [x[m][key] for m in ms], "s-", color=C_23["ours"], ms=3, label="knowledge-stratified")
            a.set_xscale("log"); a.set_title(title, fontsize=8); a.set_xlabel("labels")
        ax[0].set_ylabel("selection regret"); ax[0].legend(fontsize=6, frameon=False)
        if not have_any:
            plt_23.close(fig); return "label-efficiency verdicts missing"
        save_23(fig, "fig_labels")


    def runtime_23():
        """CPU time of brackets and the full pairwise certificate (numpy), for the sizes of the paper and larger."""
        rows = []
        for n, R, K, m in ((2000, 40, 12, 13), (10000, 40, 12, 13), (2000, 90, 20, 64), (50000, 40, 12, 13)):
            r = np_23.random.default_rng(0); P = r.random((R, n, K)) < 0.3; V = r.random((m, K)) < 0.3   # one fibre per input (worst case: the largest fibre)
            t0 = time_23.perf_counter()
            D = np_23.stack([(P[k][:, None, :] != V[None]).mean(-1) for k in range(R)])                  # (R, n, m) normalised Hamming distances
            lo, hi = D.min(-1).mean(1), D.max(-1).mean(1)
            t1 = time_23.perf_counter()
            UP = np_23.stack([(D[k][None] - D).max(-1).mean(1) for k in range(R)])                         # (R, R) sharp pairwise certificate
            t2 = time_23.perf_counter()
            rows.append({"inputs": n, "models": R, "facts": K, "legal_combinations": m, "brackets_s": round(t1 - t0, 3), "pairwise_certificate_s": round(t2 - t1, 3)})
            del P, D, UP
        return rows


    for f_23 in (fig_drift_23, fig_projection_23, fig_shortcut_23, fig_cococo_23, fig_labels_23):
        try:
            msg_23 = f_23()
            if msg_23:
                skipped_23.append(f"{f_23.__name__}: {msg_23}")
        except Exception as e_23:
            skipped_23.append(f"{f_23.__name__}: {type(e_23).__name__}: {str(e_23)[:150]}"); plt_23.close("all")
    RUNTIME_23 = runtime_23()
    with open(os_23.path.join(FIG_DIR_23, "runtime.json"), "w", encoding="utf-8") as fh_23:
        json_23.dump(RUNTIME_23, fh_23, indent=1)
    zip_path_23 = ckpt("paper_figures.zip")
    with zipfile_23.ZipFile(zip_path_23, "w", zipfile_23.ZIP_DEFLATED) as z_23:
        for fn_23 in sorted(os_23.listdir(FIG_DIR_23)):
            z_23.write(os_23.path.join(FIG_DIR_23, fn_23), fn_23)
        z_23.writestr("paper_numbers.txt", PAPER_NUMBERS_22)                    # from cell 22 (this also makes 23 run after 22)
    print(f"figures made: {made_23}")
    print(f"figures skipped: {skipped_23 if skipped_23 else 'none'}")
    print("runtime (CPU, numpy):")
    for row_23 in RUNTIME_23:
        print("  ", row_23)
    print(f"\nAll figures, runtime.json and paper_numbers.txt are in {zip_path_23}. Download it from the file panel and attach it in the chat.")
    return


@app.cell
def _(CFG, SMOKE):
    # ---- 24. BDD-OIA into <work dir>/data/bddoia, from the first source that works. molab blocks Zenodo and Google Drive and
    #      caps uploads at 100 MB, so download https://zenodo.org/records/11612556/files/BDD-OIA-original-dataset.zip?download=1
    #      in your browser and then EITHER
    #      (A) upload it to a Hugging Face dataset repo (huggingface.co/new-dataset, private is fine) and put its id in HF_REPO_24, OR
    #      (B) split it into parts under 100 MB (7-Zip: right-click > 7-Zip > Split file, 90M; macOS/Linux: split -b 90m <zip> bdd.zip.)
    #          and drag every part into molab's file panel; this cell joins them.
    #      When it prints "ready", re-run the two BDD-OIA cells (`if have("s9k_bdd.json")` and `V11_TEMPLATES = [`).
    import os as os_24, re as re_24, zipfile as zipfile_24, hashlib as hashlib_24, urllib.request as urlreq_24

    HF_REPO_24 = ""                                                     # route A: e.g. "your-name/bdd-oia" (dataset repo holding the zip)
    ROOT_24 = os_24.path.join(CFG["work_dir"], "data", "bddoia")
    ZEN_URLS_24 = ["https://zenodo.org/records/11612556/files/BDD-OIA-original-dataset.zip?download=1",
                   "https://zenodo.org/api/records/11612556/files/BDD-OIA-original-dataset.zip/content"]
    ZEN_MD5_24 = "5d34469f888aaffb358e838d864b688e"


    def bdd_ready_24(root):
        """True when the action/reason json files and images are under root."""
        if not os_24.path.isdir(root):
            return False
        names = [f for _, _, fs in os_24.walk(root) for f in fs]
        return any("reason" in f and f.endswith(".json") for f in names) and any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in names)


    def find_local_zip_24():
        """A whole BDD-OIA zip already on this machine: BDDOIA_ZIP, or the largest *.zip with 'bdd' in its name near the notebook."""
        env = os_24.environ.get("BDDOIA_ZIP", "")
        if env and os_24.path.exists(env) and zipfile_24.is_zipfile(env):
            return env
        found = [p for p in near_files_24() if os_24.path.basename(p).lower().endswith(".zip") and "bdd" in os_24.path.basename(p).lower() and zipfile_24.is_zipfile(p)]
        return max(found, key=os_24.path.getsize) if found else None


    def near_files_24():
        """Files in the notebook folder (3 levels) and the home folder (2 levels), skipping hidden folders and the target folder."""
        out = []
        for base, depth in ((os_24.getcwd(), 3), (os_24.path.expanduser("~"), 2)):
            base_depth = base.rstrip(os_24.sep).count(os_24.sep)
            for d, ds, fs in os_24.walk(base):
                if d.count(os_24.sep) - base_depth >= depth:
                    ds[:] = []
                ds[:] = [x for x in ds if not x.startswith(".") and x not in ("site-packages", "node_modules")]
                if os_24.path.abspath(d) == os_24.path.abspath(ROOT_24):
                    continue
                out += [os_24.path.join(d, f) for f in fs]
        return sorted(set(out))


    def join_parts_24(dest):
        """Join split parts of the zip (bdd...zip.001/.002..., or bdd...zip.aa/.ab... from `split`) into dest. Returns dest or None."""
        pat = re_24.compile(r"(bdd.*\.zip)\.(\d{3}|[a-z]{2})$", re_24.I)
        groups = {}
        for p in near_files_24():
            m = pat.search(os_24.path.basename(p))
            if m:
                groups.setdefault(os_24.path.join(os_24.path.dirname(p), m.group(1).lower()), []).append(p)
        if not groups:
            return None
        parts = max(groups.values(), key=lambda g: sum(os_24.path.getsize(x) for x in g))
        parts = sorted(parts, key=lambda x: os_24.path.basename(x).lower())
        print(f"joining {len(parts)} uploaded parts: {[os_24.path.basename(x) for x in parts]}")
        md5 = hashlib_24.md5()
        with open(dest + ".part", "wb") as out:
            for x in parts:
                with open(x, "rb") as fh:
                    while True:
                        b = fh.read(1 << 22)
                        if not b:
                            break
                        out.write(b); md5.update(b)
        os_24.replace(dest + ".part", dest)
        print("checksum matches the Zenodo file" if md5.hexdigest() == ZEN_MD5_24 else
              f"checksum {md5.hexdigest()} differs from the Zenodo file (fine if it unzips; otherwise a part is missing)")
        return dest if zipfile_24.is_zipfile(dest) else None


    def download_zenodo_24(dest):
        for url in ZEN_URLS_24:
            tmp = dest + ".part"
            try:
                md5 = hashlib_24.md5(); got = 0
                req = urlreq_24.Request(url, headers={"User-Agent": "curl/8.5.0", "Accept": "*/*"})
                with urlreq_24.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
                    while True:
                        b = r.read(1 << 22)
                        if not b:
                            break
                        fh.write(b); md5.update(b); got += len(b)
                if md5.hexdigest() != ZEN_MD5_24:
                    raise IOError(f"md5 mismatch after {got / 1e6:.0f} MB")
                os_24.replace(tmp, dest)
                return f"Zenodo ({got / 1e6:.0f} MB, checksum ok)"
            except Exception as e:
                if os_24.path.exists(tmp):
                    os_24.remove(tmp)
                print(f"   Zenodo {url[:60]}...: {repr(e)[:120]}")
        return None


    def download_hf_24(dest_dir):
        if not HF_REPO_24:
            return None
        try:
            from huggingface_hub import snapshot_download
            p = snapshot_download(HF_REPO_24, repo_type="dataset", local_dir=dest_dir, token=os_24.environ.get("HF_TOKEN") or None)
            return f"Hugging Face {HF_REPO_24} -> {p}"
        except Exception as e:
            print(f"   Hugging Face {HF_REPO_24}: {repr(e)[:160]}")
            return None


    def unpack_24(zp, root):
        with zipfile_24.ZipFile(zp) as z:
            z.extractall(root)
        for _ in range(2):                                                  # unpack zips stored inside the zip
            inner = [os_24.path.join(d, f) for d, _, fs in os_24.walk(root) for f in fs if f.lower().endswith(".zip") and os_24.path.abspath(os_24.path.join(d, f)) != os_24.path.abspath(zp)]
            for q in inner:
                with zipfile_24.ZipFile(q) as z:
                    z.extractall(os_24.path.dirname(q))
                os_24.remove(q)


    if SMOKE:
        print("SMOKE run: the BDD cells build their own synthetic stand-in; nothing to download.")
    elif bdd_ready_24(ROOT_24):
        print(f"BDD-OIA already present in {ROOT_24}: ready, re-run the two BDD-OIA cells if they were skipped")
    else:
        os_24.makedirs(ROOT_24, exist_ok=True)
        zp_24 = os_24.path.join(ROOT_24, "bddoia.zip")
        try:
            src_24 = zp_24 if (os_24.path.exists(zp_24) and zipfile_24.is_zipfile(zp_24)) else (find_local_zip_24() or join_parts_24(zp_24))
            if src_24:
                print(f"unpacking {src_24} ({os_24.path.getsize(src_24) / 1e6:.0f} MB)")
                unpack_24(src_24, ROOT_24)
            elif HF_REPO_24:
                print(f"downloading from Hugging Face {HF_REPO_24} ...")
                if download_hf_24(ROOT_24):
                    for z_24 in [os_24.path.join(d, f) for d, _, fs in os_24.walk(ROOT_24) for f in fs if f.lower().endswith(".zip")]:
                        unpack_24(z_24, ROOT_24)
            else:
                print("no uploaded zip or parts found and HF_REPO_24 is empty; trying Zenodo ...")
                if download_zenodo_24(zp_24):
                    unpack_24(zp_24, ROOT_24)
        except Exception as e_24:
            print(f"BDD-OIA preparation failed: {repr(e_24)[:300]}")
        n_img_24 = sum(f.lower().endswith((".jpg", ".jpeg", ".png")) for _, _, fs in os_24.walk(ROOT_24) for f in fs)
        n_json_24 = sorted(f for _, _, fs in os_24.walk(ROOT_24) for f in fs if f.endswith(".json"))
        print(f"BDD-OIA in {ROOT_24}: {n_img_24} images, json files: {n_json_24[:8]}")
        if bdd_ready_24(ROOT_24):
            print("ready: now re-run the two BDD-OIA cells")
        else:
            print("NOT ready. Download https://zenodo.org/records/11612556/files/BDD-OIA-original-dataset.zip?download=1 in your browser,\n"
                  "then (A) upload it to a Hugging Face dataset repo and set HF_REPO_24, or (B) split it into <100 MB parts and drag them\n"
                  "into molab's file panel. Re-run this cell. If it still says NOT ready, send this output.")
    return


@app.cell
def _(BDD_SPECS, RUN_NAME, V11R, V11_CVAL, V11_FR, V11_NAMES, V11_Z, V11_ZVAL, ckpt, cons_mask, hf_push, label_ids, load_json, save_json):
    # ---- 25. BDD-OIA final analysis on real dashcam frames. (i) The rules are sometimes wrong on real data: k-robust brackets and
    #      pairwise certificates with k certified from 100 labelled frames (hypergeometric, delta = .05), checked against the truth;
    #      (ii) select by any estimator, certify by the rules (Prop. 4) with ATC, DoC, average confidence and 100 source labels;
    #      (iii) score pre-registration 9. Uses the readers of cell 16c and the results of cell 16d (runs after them by itself).
    #      Writes s25_bdd_paper.json and s25_bdd_verdicts.json to the work dir ----
    import os as os_25
    import numpy as np_25
    from scipy.stats import hypergeom as hypergeom_25, spearmanr as spearmanr_25

    CFG_25 = dict(delta=0.05, m=100, reps=20, seed=2025)


    def blocks_25(acts):
        """Per-block (n, W) masks of the worlds consistent with each frame's logged action (forward/stop, left, right)."""
        lb = [label_ids(BDD_SPECS[0], [tuple(a[:2]) for a in acts]), label_ids(BDD_SPECS[1], [(a[2],) for a in acts]),
              label_ids(BDD_SPECS[2], [(a[3],) for a in acts])]
        return [cons_mask(s, l) for s, l in zip(BDD_SPECS, lb)]


    def topk_25(h, k):
        """Sum of the k largest entries along the last axis."""
        k = int(k)
        if k <= 0:
            return np_25.zeros(h.shape[:-1])
        if k >= h.shape[-1]:
            return h.sum(-1)
        return np_25.partition(h, h.shape[-1] - k, axis=-1)[..., -k:].sum(-1)


    def certify_k_25(x, m, n, delta):
        """Largest number of rule-violating frames among n that is still compatible (p > delta) with x violations in m frames
        drawn without replacement: with probability >= 1 - delta the true number is at most this."""
        if m >= n:
            return int(x)
        ks = np_25.arange(x, n - (m - x) + 1)
        ok = ks[hypergeom_25.cdf(x, n, ks, m) > delta]
        return int(ok.max()) if len(ok) else int(x)


    def spear_25(a, b):
        r = spearmanr_25(a, b)[0]
        return None if not np_25.isfinite(r) else float(r)


    def run_25():
        A = (np_25.asarray(V11_Z) > 0).astype(np_25.int8); acts = np_25.asarray(V11_FR["test"][2]); cons_all = blocks_25(acts)
        sat = np_25.stack([c.any(1) for c in cons_all], 1).all(1)              # label-free: the logged action is possible under the rules
        A = A[:, sat]; Y = np_25.asarray(V11_FR["test"][3])[sat].astype(np_25.int8); cons = [c[sat] for c in cons_all]
        R, n, K = A.shape; names = list(V11_NAMES); heads = [nm.split(":")[1] for nm in names]
        # per-frame floor, ceiling and sharp pairwise term, block by block (the rules factor over the three blocks)
        LOf = np_25.zeros((R, n)); HIf = np_25.zeros((R, n)); PER = np_25.zeros((R, R, n)); kin = np_25.ones(n, bool)
        for s, c in zip(BDD_SPECS, cons):
            ix = s["ix"]; kb = len(ix); used = c.any(0); V = s["V"][used].astype(np_25.float32); cu = c[:, used]
            a = A[:, :, ix].astype(np_25.float32)
            H = (a @ (1 - V).T + (1 - a) @ V.T) / K                             # (R, n, W_used) error contribution of this block
            LOf += np_25.where(cu[None], H, np_25.inf).min(-1); HIf += np_25.where(cu[None], H, -np_25.inf).max(-1)
            for r in range(R):
                PER[r] += np_25.where(cu[None], H[r][None] - H, -np_25.inf).max(-1)
            wid = (Y[:, ix].astype(np_25.int64) * (2 ** np_25.arange(kb)[::-1])).sum(1)     # index of the true block world in worlds_of order
            vid = (s["V"].astype(np_25.int64) * (2 ** np_25.arange(kb)[::-1])).sum(1)
            kin &= c[np_25.arange(n), np_25.argsort(vid)[wid]]                  # evaluation only: does the truth satisfy the rules?
            del H, a
        HAM = (A[:, None] != A[None]).mean(-1)                                  # (R, R, n): the most two readers can differ on a frame
        efr = (A != Y[None]).mean(-1); err = efr.mean(1); best = float(err.min()); off = ~np_25.eye(R, dtype=bool)
        n_viol = int((~kin).sum())

        def bounds(k):
            """k-robust bracket and certificate: the rules may be wrong on up to k unknown frames (truth then any 0/1 vector)."""
            lo = (LOf.sum(1) - topk_25(LOf, k)) / n; hi = (HIf.sum(1) + topk_25(1 - HIf, k)) / n
            up = (PER.sum(-1) + topk_25(HAM - PER, k)) / n; np_25.fill_diagonal(up, 0.0)
            return lo, hi, up

        def check(lo, hi, up):
            cr = np_25.maximum(up, 0).max(1); top = ~np_25.any(up < -1e-12, axis=0)
            return {"bracket_ok": bool(np_25.all((lo - 1e-9 <= err) & (err <= hi + 1e-9))),
                    "pairs_ok": bool(np_25.all(err[:, None] - err[None] <= up + 1e-9)),
                    "regret_le_CR_all": bool(np_25.all(err - best <= cr + 1e-9)),
                    "mean_width": float((hi - lo).mean()), "certified_pairs": float(((up < -1e-12) & off).sum() / max(1, off.sum())),
                    "top_set_size": int(top.sum()), "minimax_regret": float(err[int(np_25.lexsort((lo, cr))[0])] - best),
                    "minimax_CR": float(cr.min())}

        out = {"n_frames": int(len(sat)), "n_sat": n, "n_readers": R, "n_violations": n_viol, "violation_rate": n_viol / n,
               "best_err": best, "names": names, "heads": heads, "err": err.tolist()}
        lo0, hi0, up0 = bounds(0); out["k0"] = check(lo0, hi0, up0)
        lo_o, hi_o, up_o = bounds(n_viol); out["oracle_k"] = dict(check(lo_o, hi_o, up_o), k=n_viol)
        # (i) certify k from m labelled frames, repeated over independent draws
        rng = np_25.random.default_rng(CFG_25["seed"]); reps = []
        for rep in range(CFG_25["reps"]):
            idx = rng.choice(n, min(CFG_25["m"], n), replace=False); x = int((~kin[idx]).sum())
            kp = certify_k_25(x, len(idx), n, CFG_25["delta"]); ck = check(*bounds(kp))
            reps.append(dict(ck, x=x, k=kp, covered=bool(kp >= n_viol)))
        cov = [r_["covered"] for r_ in reps]
        out["certify_k"] = {"m": CFG_25["m"], "delta": CFG_25["delta"], "reps": CFG_25["reps"], "k": [r_["k"] for r_ in reps],
                            "violations_in_sample": [r_["x"] for r_ in reps], "coverage": float(np_25.mean(cov)),
                            "valid_when_covered": bool(all(r_["bracket_ok"] and r_["pairs_ok"] and r_["regret_le_CR_all"] for r_ in reps if r_["covered"])),
                            "n_covered": int(sum(cov)), "valid_any": float(np_25.mean([r_["bracket_ok"] and r_["pairs_ok"] for r_ in reps])),
                            "mean_k": float(np_25.mean([r_["k"] for r_ in reps])), "mean_width": float(np_25.mean([r_["mean_width"] for r_ in reps])),
                            "mean_minimax_regret": float(np_25.mean([r_["minimax_regret"] for r_ in reps])),
                            "mean_minimax_CR": float(np_25.mean([r_["minimax_CR"] for r_ in reps]))}
        k_dep = reps[0]["k"] if reps[0]["covered"] else n_viol                  # the first draw is the deployed audit
        out["deployed"] = dict(check(*bounds(k_dep)), k=k_dep, from_first_draw=bool(reps[0]["covered"]))
        LOk, HIk, UPk = bounds(k_dep); CRk = np_25.maximum(UPk, 0).max(1); top = ~np_25.any(UPk < -1e-12, axis=0)
        # (ii) select by any estimator, certify by the rules
        P = 1 / (1 + np_25.exp(-np_25.asarray(V11_Z, np_25.float64)[:, sat])); Ps = 1 / (1 + np_25.exp(-np_25.asarray(V11_ZVAL, np_25.float64)))
        Ys = np_25.asarray(V11_CVAL); conf = lambda Q: np_25.maximum(Q, 1 - Q)
        atc = np_25.empty(R)
        for r in range(R):
            acc = float(((Ps[r] > 0.5) == Ys).mean()); t = np_25.quantile(conf(Ps[r]).ravel(), 1 - acc) if 0 < acc < 1 else 0.5
            atc[r] = 1 - float((conf(P[r]).ravel() > t).mean())
        src = ((Ps > 0.5) != Ys[None]).mean((1, 2))
        ests = {"ATC": atc, "DoC": np_25.clip(src + conf(Ps).mean((1, 2)) - conf(P).mean((1, 2)), 0, 1),
                "avg_confidence": 1 - conf(P).mean((1, 2)), "source_err_100_labels": src}
        sel = {}
        for nm, e in ests.items():
            ep = np_25.clip(e, LOk, HIk); cand = np_25.flatnonzero(top); rp = int(cand[np_25.argmin(ep[cand])]); rr = int(np_25.argmin(e))
            sel[nm] = {"mae_raw": float(np_25.abs(e - err).mean()), "mae_projected": float(np_25.abs(ep - err).mean()),
                       "impossible": int(((e < LOk - 1e-12) | (e > HIk + 1e-12)).sum()), "of": R,
                       "regret_raw": float(err[rr] - best), "regret_projected": float(err[rp] - best), "CR_of_pick": float(CRk[rp]),
                       "pick_raw": names[rr], "pick_projected": names[rp], "spearman_raw": spear_25(e, err)}
        out["selection"] = sel
        out["per_reader"] = {"LO0": lo0.tolist(), "HI0": hi0.tolist(), "LOk": LOk.tolist(), "HIk": HIk.tolist(), "CRk": CRk.tolist(),
                             "estimates": {k_: v.tolist() for k_, v in ests.items()}}
        out["spearman_floor_err"] = spear_25(LOk, err); out["spearman_mid_err"] = spear_25((LOk + HIk) / 2, err)
        # label-free baselines of cell 16d and its KSE runs, kept here so the figure cell needs only this result
        out["baselines_16d"] = {"regret": V11R["regret"], "picked_head": V11R["picked_head"], "cert_selected": V11R["cert_selected"],
                                "by_head": V11R["by_head"]}
        ms = sorted({x_["m"] for x_ in V11R["kse"]})
        out["kse"] = {str(m_): {k_: float(np_25.mean([x_[k_] for x_ in V11R["kse"] if x_["m"] == m_])) for k_ in ("random", "KSE")} for m_ in ms}
        return out


    V25R = None
    if V11R is None or V11_FR is None:
        print("Cell 25 skipped: BDD-OIA results are not available yet. Run cell 24, then the BDD-OIA cells (16c, 16d).")
    else:
        V25R = run_25()
        pv_25 = load_json(ckpt("s16e_v11_verdicts.json")) if os_25.path.exists(ckpt("s16e_v11_verdicts.json")) else {}
        ok_25 = lambda *ks: all(bool(pv_25[k][1]) for k in ks) if all(k in pv_25 for k in ks) else None
        V25R["pre_registration_9"] = {"P30 G19-G22 hold": ok_25("G19", "G20", "G21", "G22"),
                                      "P31 shortcut signature (NeSy obeys rules more, reads concepts worse than probe)": ok_25("X18"),
                                      "P32 label consistency picks NeSy and test-then-select beats it": ok_25("X19"),
                                      "P33 test-then-select <= every label-free baseline (BDD) and <= .01 (MNIST)": ok_25("X20", "X25"),
                                      "P35 KSE beats random labelling at every budget (BDD)": ok_25("X22")}
        ck_25 = V25R["certify_k"]; dp_25 = V25R["deployed"]; sl_25 = V25R["selection"]
        V25V = {
            "GB1": ("GUAR", ck_25["n_covered"] > 0 and ck_25["valid_when_covered"],
                    "rules sometimes wrong: with k certified from 100 labelled frames, every k-robust bracket and pairwise certificate holds",
                    {"k": ck_25["k"], "true_violations": V25R["n_violations"], "n_covered": ck_25["n_covered"]}),
            "GB2": ("GUAR", dp_25["bracket_ok"] and dp_25["pairs_ok"] and all(v["regret_projected"] <= v["CR_of_pick"] + 1e-9 for v in sl_25.values()),
                    "Prop. 4 on real frames: every estimator's pick (after projection and veto) has regret at most its k-robust certificate",
                    {k: [round(v["regret_projected"], 4), round(v["CR_of_pick"], 4)] for k, v in sl_25.items()}),
            "GB3": ("GUAR", all(v["mae_projected"] <= v["mae_raw"] + 1e-12 for v in sl_25.values()),
                    "projection onto the k-robust bracket never increases any estimator's error",
                    {k: [round(v["mae_raw"], 4), round(v["mae_projected"], 4), f"{v['impossible']}/{v['of']} impossible"] for k, v in sl_25.items()}),
            "XB1": ("EXPL", ck_25["coverage"] >= 1 - CFG_25["delta"], "the certified k covers the true number of rule violations in at least 95% of draws",
                    {"coverage": ck_25["coverage"], "mean_k": ck_25["mean_k"], "true": V25R["n_violations"], "n": V25R["n_sat"]}),
            "XB2": ("EXPL", all(v["regret_projected"] <= v["regret_raw"] + 1e-12 for v in sl_25.values()),
                    "projecting and vetoing never makes an estimator pick a worse reader", {k: [round(v["regret_raw"], 4), round(v["regret_projected"], 4)] for k, v in sl_25.items()}),
            "XB3": ("EXPL", (V25R["spearman_floor_err"] or 0) >= 0.8, "the k-robust floor alone ranks the readers by true concept error (Spearman >= .8)",
                    {"floor": V25R["spearman_floor_err"], "midpoint": V25R["spearman_mid_err"]}),
            "XB4": ("EXPL", V25R["k0"]["bracket_ok"] and V25R["k0"]["pairs_ok"], "even the plain (k = 0) bounds hold on every frame, including the frames where the rules fail",
                    {"violation_rate": round(V25R["violation_rate"], 4), "k0": V25R["k0"]}),
        }
        save_json(ckpt("s25_bdd_paper.json"), V25R)
        save_json(ckpt("s25_bdd_verdicts.json"), {k: [v[0], bool(v[1]), v[2], v[3]] for k, v in V25V.items()}); hf_push(RUN_NAME, force=True)
        print(f"BDD-OIA: {V25R['n_readers']} readers on {V25R['n_sat']} test frames whose action the rules allow (of {V25R['n_frames']}); "
              f"the rules are wrong on {V25R['n_violations']} of them ({V25R['violation_rate']:.3f}); best reader error {V25R['best_err']:.4f}")
        print(f"plain bounds (k = 0): {V25R['k0']}")
        print(f"k certified from {ck_25['m']} labelled frames, {ck_25['reps']} draws: k = {ck_25['k']} (true {V25R['n_violations']}), "
              f"coverage {ck_25['coverage']:.2f}, all bounds valid whenever covered: {ck_25['valid_when_covered']}, mean bracket width {ck_25['mean_width']:.4f}")
        print(f"deployed audit (k = {dp_25['k']}): {dp_25}")
        print("select by any estimator, certify by the rules (regret raw -> projected, certificate of the pick, impossible estimates):")
        for k_25, v_25 in sl_25.items():
            print(f"   {k_25:22s} regret {v_25['regret_raw']:.4f} -> {v_25['regret_projected']:.4f}  CR {v_25['CR_of_pick']:.4f}  "
                  f"MAE {v_25['mae_raw']:.4f} -> {v_25['mae_projected']:.4f}  impossible {v_25['impossible']}/{v_25['of']}")
        print(f"Spearman with true error: floor {V25R['spearman_floor_err']}, midpoint {V25R['spearman_mid_err']}")
        for k_25, (kind_25, ok2_25, desc_25, val_25) in V25V.items():
            print(f"{('PASS' if ok2_25 else 'FAIL') if kind_25 == 'GUAR' else ('YES ' if ok2_25 else 'NO  '):8s} [{kind_25}] {k_25} {desc_25}\n         {val_25}")
        print("Pre-registration 9 score:", {k: ("confirmed" if v else ("not confirmed" if v is False else "not available")) for k, v in V25R["pre_registration_9"].items()})
    return (V25R,)


@app.cell
def _(CFG, RUN_NAME, V25R, ckpt, load_json):
    # ---- 26. Final bundle. Draws the BDD-OIA figure from cell 25, re-collects every verdict and result number of this run (now with
    #      BDD-OIA) into paper_numbers_final.txt, and zips it with every paper figure (from cell 23) to <work dir>/paper_final.zip.
    #      Download that zip from the file panel and attach it in the chat ----
    import os as os_26, glob as glob_26, json as json_26, zipfile as zipfile_26
    import numpy as np_26
    import matplotlib.pyplot as plt_26

    FIG_DIR_26 = ckpt("paper_figures"); os_26.makedirs(FIG_DIR_26, exist_ok=True)
    plt_26.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    HEAD_COL_26 = {"zs": "#8c8c8c", "probe50": "#1f6fb2", "nesy_rand": "#d1495b", "nesy_zs": "#e8a33d"}
    HEAD_LAB_26 = {"zs": "zero-shot", "probe50": "50-label probe", "nesy_rand": "action-only NeSy", "nesy_zs": "action-only NeSy (zs init)"}


    def fig_bdd_26(d):
        """Fig. BDD-OIA: (a) k-robust bracket vs true error per reader; (b) regret of each selector and the certificate of its pick;
        (c) knowledge-stratified vs random frame labelling."""
        fig, ax = plt_26.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1.25, 0.9]})
        err = np_26.array(d["err"]); lo = np_26.array(d["per_reader"]["LOk"]); hi = np_26.array(d["per_reader"]["HIk"]); hd = np_26.array(d["heads"])
        for h in [x for x in HEAD_COL_26 if x in set(hd)]:
            m = hd == h
            ax[0].errorbar(err[m], (lo[m] + hi[m]) / 2, yerr=[(lo[m] + hi[m]) / 2 - lo[m], hi[m] - (lo[m] + hi[m]) / 2], fmt="o", ms=3,
                           lw=0.8, color=HEAD_COL_26[h], label=HEAD_LAB_26[h])
        t = [0, float(max(hi.max(), err.max()))]; ax[0].plot(t, t, "k-", lw=0.6)
        ax[0].set_xlabel("true concept error"); ax[0].set_ylabel("certified bracket"); ax[0].legend(fontsize=5.5, frameon=False, loc="lower right")
        ax[0].set_title(f"k = {d['deployed']['k']} of {d['n_sat']} frames; all inside: {d['deployed']['bracket_ok']}", fontsize=7)
        b = d["baselines_16d"]["regret"]; s = d["selection"]
        rows = [("label consistency", b.get("label_consistency"), None, "#8c8c8c"), ("confidence", b.get("confidence"), None, "#8c8c8c"),
                ("majority agreement", b.get("majority_agreement"), None, "#8c8c8c"),
                ("minimax r† (rules only)", d["deployed"]["minimax_regret"], d["deployed"]["minimax_CR"], "#1f6fb2")]
        lab = {"ATC": "ATC", "DoC": "DoC", "avg_confidence": "avg. confidence", "source_err_100_labels": "100 labels"}
        for k, v in s.items():
            rows.append((f"{lab.get(k, k)}", v["regret_raw"], None, "#c9c9c9"))
            rows.append((f"{lab.get(k, k)} + rules", v["regret_projected"], v["CR_of_pick"], "#1f6fb2"))
        rows = [r for r in rows if r[1] is not None]; ys = np_26.arange(len(rows))
        ax[1].barh(ys, [r[1] for r in rows], color=[r[3] for r in rows])
        for y, r in zip(ys, rows):
            if r[2] is not None:
                ax[1].plot([r[2], r[2]], [y - 0.4, y + 0.4], "k-", lw=1.0)
        ax[1].set_yticks(ys); ax[1].set_yticklabels([r[0] for r in rows], fontsize=6); ax[1].invert_yaxis()
        ax[1].set_xlabel("selection regret (bar) and certificate (tick)", fontsize=7)
        ks = d["kse"]; ms = sorted(ks, key=int)
        ax[2].plot([int(m) for m in ms], [ks[m]["random"] for m in ms], "o-", color="#8c8c8c", ms=3, label="random frames")
        ax[2].plot([int(m) for m in ms], [ks[m]["KSE"] for m in ms], "s-", color="#1f6fb2", ms=3, label="knowledge-stratified")
        ax[2].set_xscale("log"); ax[2].set_xlabel("labelled frames"); ax[2].set_ylabel("selection regret"); ax[2].legend(fontsize=6, frameon=False)
        for ext in ("pdf", "png"):
            fig.savefig(os_26.path.join(FIG_DIR_26, f"fig_bdd.{ext}"), bbox_inches="tight")
        plt_26.close(fig)


    def flat_26(obj, path="", depth=0, out=None):
        """(path, value) for every number / bool / short string inside obj, down to depth 3; long lists are skipped."""
        out = [] if out is None else out
        if isinstance(obj, dict) and depth < 4:
            for k, v in obj.items():
                flat_26(v, f"{path}.{k}" if path else str(k), depth + 1, out)
        elif isinstance(obj, (list, tuple)) and len(obj) <= 24 and depth < 4 and all(not isinstance(x, (dict, list)) for x in obj):
            out.append((path, list(obj)))
        elif isinstance(obj, bool) or isinstance(obj, (int, float)) or (isinstance(obj, str) and len(obj) <= 40):
            out.append((path, round(obj, 4) if isinstance(obj, float) else obj))
        return out


    notes_26 = []
    if V25R is None:
        notes_26.append("fig_bdd not drawn: cell 25 has no BDD-OIA result yet")
    else:
        try:
            fig_bdd_26(V25R)
        except Exception as e_26:
            notes_26.append(f"fig_bdd failed: {type(e_26).__name__}: {str(e_26)[:150]}"); plt_26.close("all")

    lines_26 = [f"PAPER NUMBERS (final) | run {RUN_NAME} | seed {CFG['seed']}"]
    for f_26 in sorted(glob_26.glob(ckpt("s*verdict*.json"))):
        lines_26.append(f"## {os_26.path.basename(f_26)}")
        try:
            for k_26, v_26 in load_json(f_26).items():
                v_26 = list(v_26) if isinstance(v_26, (list, tuple)) else [None, v_26]
                ok_26 = bool(v_26[1]) if len(v_26) > 1 else None
                tag_26 = ("PASS" if ok_26 else "FAIL") if v_26[0] == "GUAR" else ("YES" if ok_26 else "NO")
                lines_26.append(f"{k_26} {tag_26} {str(v_26[2])[:110] if len(v_26) > 2 else ''} :: {json_26.dumps(v_26[3], default=str)[:900] if len(v_26) > 3 else ''}")
        except Exception as e_26:
            lines_26.append(f"(could not read: {type(e_26).__name__})")
    nverd_26 = len(lines_26)
    for f_26 in sorted(glob_26.glob(ckpt("s*.json"))):
        if "verdict" in f_26 or "confirmatory" in f_26:
            continue
        try:
            fl_26 = flat_26(load_json(f_26))
        except Exception as e_26:
            lines_26.append(f"## {os_26.path.basename(f_26)} (could not read: {type(e_26).__name__})"); continue
        if fl_26:
            lines_26.append(f"## {os_26.path.basename(f_26)} ({len(fl_26)} values{', first 160 shown' if len(fl_26) > 160 else ''})")
            lines_26 += [f"{p_26} = {x_26}" for p_26, x_26 in fl_26[:160]]
    PAPER_NUMBERS_26 = "\n".join(lines_26)
    with open(ckpt("paper_numbers_final.txt"), "w", encoding="utf-8") as fh_26:
        fh_26.write(PAPER_NUMBERS_26)

    want_26 = ["fig_drift", "fig_projection", "fig_shortcut", "fig_cococo", "fig_labels", "fig_bdd"]
    have_26 = [w for w in want_26 if os_26.path.exists(os_26.path.join(FIG_DIR_26, f"{w}.pdf"))]
    zip_26 = ckpt("paper_final.zip")
    with zipfile_26.ZipFile(zip_26, "w", zipfile_26.ZIP_DEFLATED) as z_26:
        for fn_26 in sorted(os_26.listdir(FIG_DIR_26)):
            z_26.write(os_26.path.join(FIG_DIR_26, fn_26), fn_26)
        z_26.write(ckpt("paper_numbers_final.txt"), "paper_numbers_final.txt")
        for fn_26 in ("s25_bdd_paper.json", "s16d_v11_bdd.json"):
            if os_26.path.exists(ckpt(fn_26)):
                z_26.write(ckpt(fn_26), fn_26)

    print("\n".join(lines_26[:nverd_26]))
    print(f"\nfigures in the bundle: {have_26}")
    if len(have_26) < len(want_26):
        print(f"missing: {[w for w in want_26 if w not in have_26]}. "
              + ("Run cell 23 (the one starting '# ---- 23. Paper figures') once, then this cell again." if any(w != "fig_bdd" for w in want_26 if w not in have_26) else ""))
    for n_26 in notes_26:
        print(n_26)
    print(f"\nEverything is in {zip_26}. Download it from the file panel and attach it in the chat.")
    return


if __name__ == "__main__":
    app.run()
