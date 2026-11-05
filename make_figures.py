"""Regenerate docs/figures/*.png from the saved outputs in this repo.

Usage: python3 make_figures.py --kf-dir <dir with kf_NNNN_left.jpg> --traj <s4_rep.tum> [--out docs/figures]

The keyframe JPEGs are the stage-1 rectified left images (keyframes.py writes them after G.rectify), which is the
pixel frame of every box in detections.json / findings.json (rect.json size 1600x1200), so boxes are drawn as is.
All numbers are read from run_bus*/ JSON files and labels.json; nothing is hard-coded except layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent
GREEN, ORANGE, GREY, DARK = "#009E73", "#D55E00", "#8C8C8C", "#333333"
BLUE = "#0072B2"
NOTE = "Single AI annotator, small labelled set (40 frames, 17 usable crops), murky lake scene."
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 150, "savefig.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def jl(p):
    return json.load(open(ROOT / p))


def save(fig, out, name):
    p = out / name
    kw = {"pil_kwargs": {"quality": 82}} if name.endswith(".jpg") else {}
    fig.savefig(p, facecolor="white", bbox_inches="tight", pad_inches=0.12, **kw)
    plt.close(fig)
    print(f"{p}  {p.stat().st_size / 1024:.0f} KB")


def hero(out, kf_dir):
    f = {x["id"]: x for x in jl("run_bus/findings.json")["findings"]}
    v1 = jl("run_bus_v1/vlm_outputs.json")["per_finding"]
    lab = jl("labels.json")["crops"]
    # chosen by data: confirmed+labelled true (tire, window, marine growth), the two highest-ranked crack crops
    # (unclear, labelled false), and F27 (confirmed by v1 but labelled false = the one v1 false positive)
    ids = ["F01", "F15", "F18", "F20", "F26", "F27"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.3))
    for ax, i in zip(axs.ravel(), ids):
        x = f[i]
        img = cv2.cvtColor(cv2.imread(str(kf_dir / f"{x['best_frame']}_left.jpg")), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
        conf = v1[i]["status"] == "confirmed"
        col = GREEN if conf else GREY
        x0, y0, x1, y1 = [c / 2 for c in x["best_box"]]
        ax.imshow(img)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=3))
        ax.text(x0 + 4, max(y0 - 6, 22), f"{i} {x['cls']} {x['score_max']:.2f}", color="white", fontsize=9,
                fontweight="bold", va="bottom", bbox=dict(fc=col, ec="none", pad=1.5))
        gt = lab[i]
        gts = {True: "true", False: "false"}.get(gt, "unlabelled")
        ax.set_title(f"v1: {'confirmed' if conf else 'unclear'}; annotator: crop is {gts}", fontsize=9.5,
                     color=ORANGE if (gt is False) else DARK)
        ax.axis("off")
    fig.suptitle("Frame-context verifier confirms tire, window and marine growth, but passes one debris false "
                 "positive and leaves crack candidates unclear", fontsize=11, y=0.995)
    fig.text(0.5, 0.0, "Box colour = Qwen3-VL-4B v1 status (green confirmed, grey unclear; v1 never answered 'no'). "
             "Orange title = annotator judged the crop false.\n" + NOTE, ha="center", va="top", fontsize=8.5, color=DARK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "hero_findings.jpg")


def pipeline(out):
    steps = [("ROS bag\n(stereo)", GREY), ("Keyframes\n+ VIO pose\n(uw_ct_vio)", BLUE),
             ("OWLv2\nopen-vocab\ndetection", BLUE), ("Stereo SGBM\ndepth", BLUE),
             ("3D findings\n+ cross-frame\nmerge", BLUE), ("Qwen3-VL-4B\nverification\n(crop + frame)", ORANGE),
             ("Report\n(md / html)", GREEN), ("Faithfulness\ncheck", GREEN)]
    fig, ax = plt.subplots(figsize=(13, 2.3))
    n, w, gap = len(steps), 1.5, 0.28
    for k, (t, c) in enumerate(steps):
        x = k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, 1, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec="none",
                                    alpha=0.9))
        ax.text(x + w / 2, 0.5, t, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        if k < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 0.5), xytext=(x + w + 0.02, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
    ax.set_xlim(-0.1, n * (w + gap) - gap + 0.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")
    ax.set_title("Pipeline: bag to a verified inspection report with 3D positions in the VIO frame",
                 fontsize=11)
    save(fig, out, "pipeline.png")


def mapfig(out, traj):
    a = np.loadtxt(traj, ndmin=2)
    a = a[np.all(np.isfinite(a), axis=1)]
    a = a[np.argsort(a[:, 0])]
    dur = a[-1, 0] - a[0, 0]
    length = np.linalg.norm(np.diff(a[:, 1:4], axis=0), axis=1).sum()
    fnd = jl("run_bus/findings.json")["findings"]
    v1 = jl("run_bus_v1/vlm_outputs.json")["per_finding"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(a[:, 1], a[:, 2], color=DARK, lw=1.2, label="VIO track (left camera)")
    ax.plot(*a[0, 1:3], "o", color=DARK, ms=7)
    ax.annotate("start", a[0, 1:3], xytext=(6, -12), textcoords="offset points", fontsize=9)
    mk = {"tire": "o", "window": "s", "marine_growth": "^", "fish": "D", "crack": "d", "debris": "v",
          "hole_damage": "P", "corrosion": "*"}
    placed = [x for x in fnd if x["xyz_world"]]
    for cls in mk:
        for conf in (True, False):
            pts = [x["xyz_world"] for x in placed if x["cls"] == cls and (v1[x["id"]]["status"] == "confirmed") == conf]
            if not pts:
                continue
            p = np.array(pts)
            ax.scatter(p[:, 0], p[:, 1], marker=mk[cls], s=70, zorder=3,
                       facecolors=GREEN if conf else "none", edgecolors=GREEN if conf else ORANGE, linewidths=1.6,
                       label=f"{cls} ({'confirmed' if conf else 'unclear'})")
    ax.set_xlabel("x in VIO world frame (m)")
    ax.set_ylabel("y in VIO world frame (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title(f"{len(placed)} of the top {len(fnd)} findings get a metric 3D position along a 64 m VIO loop", fontsize=11)
    fig.text(0.5, -0.01, f"Plan view (x-y; z spans only {np.ptp(a[:, 3]):.1f} m, taken as depth, gravity not verified), VIO world frame. Dive: {length:.1f} m "
             f"track, {dur:.0f} s. Filled green = confirmed by v1; hollow orange = unclear. {len(fnd) - len(placed)} "
             f"findings had no valid depth.\n" + NOTE, ha="center", va="top", fontsize=8.5)
    save(fig, out, "map_findings.png")


def verifier(out):
    m0, m1 = jl("run_bus_v0/metrics.json"), jl("run_bus_v1/metrics.json")
    ce0, ce1 = m0["crop_eval"], m1["crop_eval"]
    conf = [sum(1 for x in jl(f"{r}/vlm_outputs.json")["per_finding"].values() if x["status"] == "confirmed")
            for r in ("run_bus_v0", "run_bus_v1")]
    flags = [m0["faithfulness"]["n_hallucinated"], m1["faithfulness"]["n_hallucinated"]]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 1, 1]})
    names = ["recall of\ntrue crops", "precision\nafter filter", "accuracy\n(confirm vs not)"]
    keys = ["vlm_recall_of_true_crops", "precision_after_vlm_filter", "accuracy_confirm_vs_not"]
    xs = np.arange(3)
    for off, ce, c, lbl in ((-0.2, ce0, GREY, "v0 crop only"), (0.2, ce1, BLUE, "v1 frame context")):
        vals = [ce[k] for k in keys]
        a1.bar(xs + off, [0 if v is None else v for v in vals], 0.38, color=c, label=lbl)
        for xx, v in zip(xs + off, vals):
            a1.text(xx, 0.02 if v is None else v + 0.02, "n/a" if v is None else f"{v:.2f}", ha="center", fontsize=9)
    a1.set_xticks(xs, names)
    a1.set_ylim(0, 1.3)
    a1.set_ylabel(f"score ({ce1['n_labelled_crops']} labelled crops; 8 true)")
    a1.legend(frameon=False, loc="upper center", fontsize=9, ncol=2)
    a1.set_title("Quality on labelled crops")
    for ax, vals, t in ((a2, conf, "confirmed\n(of 30 crops)"), (a3, flags, "faithfulness\nflags")):
        ax.bar([0, 1], vals, color=[GREY, BLUE], width=0.6)
        for xx, v in enumerate(vals):
            ax.text(xx, v + 0.02 * max(vals), str(v), ha="center", fontsize=10)
        ax.set_xticks([0, 1], ["v0", "v1"])
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_title(t)
    fig.suptitle("Frame context turns a verifier that rejects everything into one that keeps all 8 true crops "
                 "(but also adds flags)", fontsize=11)
    fig.text(0.5, -0.02, "v0 precision undefined (kept none); v0 summary is fixed code text, hence 0 flags. v1 accuracy counts 'unclear' as not confirmed. "
             "Verifier chosen on this same small set.\n" + NOTE, ha="center", va="top", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "verifier_comparison.png")


def detector(out):
    m = jl("run_bus/metrics.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    thr = sorted(m["sweep"]["ens"], key=float)
    for mode, c, ls in (("base", GREY, "--"), ("ens", BLUE, "-")):
        a1.plot([float(t) for t in thr], [m["sweep"][mode][t]["precision"] for t in thr], ls, color=c, marker="o",
                label=f"precision ({mode})")
        a1.plot([float(t) for t in thr], [m["sweep"][mode][t]["recall"] for t in thr], ls, color=c, marker="s",
                alpha=0.6, label=f"recall ({mode})")
    a1.axvline(0.15, color=ORANGE, lw=1, ls=":")
    a1.text(0.152, 0.04, "operating\nthr 0.15", color=ORANGE, fontsize=8.5)
    a1.set_xlabel("OWLv2 score threshold")
    a1.set_ylabel("micro P / R (image level, 40 frames)")
    a1.set_ylim(0, 1.05)
    a1.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    a1.set_title("Micro precision / recall vs threshold")
    il = m["image_level"]["ens"]
    cls = [c for c in il if "support" in il[c] and c != "micro"]
    y = np.arange(len(cls))
    for off, k, c, lbl in ((-0.2, "precision", BLUE, "precision"), (0.2, "recall", GREEN, "recall")):
        vals = [il[k_][k] for k_ in cls]
        a2.barh(y + off, [0 if v is None else v for v in vals], 0.38, color=c, label=lbl)
        for yy, v in zip(y + off, vals):
            a2.text(0.01 if v is None else v + 0.01, yy, "n/a" if v is None else f"{v:.2f}", va="center", fontsize=8)
    a2.set_yticks(y, [f"{c}  (n={il[c]['support']})" + ("  no positives" if il[c]["support"] == 0 else "") for c in cls])
    a2.invert_yaxis()
    a2.set_xlim(0, 1.4)
    a2.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.98))
    a2.set_title("Per class at thr 0.15 (ens; n = frames with class)")
    fig.suptitle("Detector is precise on common classes but low-recall, and crack/fish prompts only fire false positives",
                 fontsize=11)
    fig.text(0.5, -0.01, "n/a = undefined (no detections or no positives). Crack and fish have zero labelled positives, "
             "so their precision is 0 by construction.\n" + NOTE, ha="center", va="top", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out, "detector_pr.png")


def report_shot(out):
    """Optional: cropped headless-Chrome screenshot of examples/report.html (skipped if google-chrome is absent)."""
    import shutil
    import subprocess
    import tempfile

    from PIL import Image

    chrome = shutil.which("google-chrome")
    if chrome is None:
        print("google-chrome not found: skipping example_report.jpg")
        return
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "r.png"
        subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", f"--user-data-dir={td}/p",
                        "--window-size=1100,1500", f"--screenshot={png}",
                        (ROOT / "examples" / "report.html").as_uri()], check=True, capture_output=True, timeout=120)
        im = Image.open(png).convert("RGB").crop((0, 0, 1085, 1440))
        p = out / "example_report.jpg"
        im.save(p, quality=82, optimize=True)
        print(f"{p}  {p.stat().st_size / 1024:.0f} KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kf-dir", required=True, help="dir with stage-1 rectified kf_NNNN_left.jpg")
    ap.add_argument("--traj", required=True, help="TUM trajectory (left camera), e.g. s4_rep.tum")
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hero(out, Path(a.kf_dir))
    pipeline(out)
    mapfig(out, a.traj)
    verifier(out)
    detector(out)
    report_shot(out)


if __name__ == "__main__":
    main()
