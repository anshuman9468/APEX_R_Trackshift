from pathlib import Path
import gzip, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
ASSETS = OUT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
PDF = OUT / "APEX-R_MODEL_TRAINING_REPORT.pdf"

SUPPLIED_IMAGES = {
    "legacy_lr": Path("/tmp/codex-clipboard-ad3d532a-1f3c-44f4-b5fa-b19db54b6c67.png"),
    "legacy_xgb": Path("/tmp/codex-clipboard-49275213-ee1b-4c28-a18e-fe3b4018fbbe.png"),
    "clean_xgb": Path("/tmp/codex-clipboard-3efc1ade-0402-4e48-bd65-55afe3170d4a.png"),
    "enriched_xgb": Path("/tmp/codex-clipboard-9bc3ff62-48c7-4e0f-8a79-8d561db37694.png"),
    "engineered_xgb": Path("/tmp/codex-clipboard-80504932-fa7b-4734-a8d9-35677d129d08.png"),
    "phase1_xgb": Path("/tmp/codex-clipboard-3bd730f0-cf98-4c37-afc5-83c97d5bfe45.png"),
    "cpu_seed42": Path("/home/anshumandutta/Downloads/3. GNN — CPU, Seed 42.png"),
    "hybrid_gnn": Path("/tmp/codex-clipboard-86bca332-0aa4-40c6-a181-4cc2e095d375.png"),
    "hybrid_gnn_confusion": Path("/tmp/codex-clipboard-e419f49a-cb85-46c4-92e6-116b02477b66.png"),
}

def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def savefig(fig, name):
    p = ASSETS / name
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p

def make_charts():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    proxy = pd.read_csv(ROOT / "phase4_proxy_experiment/per_example_predictions.csv.gz")
    gpu = pd.read_csv(ROOT / "gnn_proxy_experiment_gpu_v1/per_window_predictions.csv")
    cpu = pd.read_csv(ROOT / "gnn_proxy_experiment_v1/per_window_predictions.csv")
    charts = {}

    fig, ax = plt.subplots(figsize=(7.1, 4.6))
    s = proxy[(proxy.split == "validation") & (proxy.model == "logistic_regression_C1")]
    if s.label.nunique() == 2:
        fpr, tpr, _ = roc_curve(s.label, s.probability)
        ax.plot(fpr, tpr, lw=2, label="Historical proxy Logistic Regression (AUC 0.5179)")
    aucs = {17: .6925, 23: .6854, 42: .6893}
    for seed, colour in [(17, "#1f77b4"), (23, "#ff7f0e"), (42, "#2ca02c")]:
        s = gpu[(gpu.proposed_split == "development_validation") & (gpu.model == f"gnn_seed_{seed}")]
        if s.label.nunique() == 2:
            fpr, tpr, _ = roc_curve(s.label, s.probability)
            ax.plot(fpr, tpr, color=colour, label=f"GPU GNN seed {seed} (AUC {aucs[seed]:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#999", label="Random baseline")
    ax.set(title="ROC-AUC Curves - Historical Proxy and GPU GNN", xlabel="False positive rate", ylabel="True positive rate")
    ax.grid(alpha=.2); ax.legend(fontsize=7, loc="lower right")
    charts["roc_gnn"] = savefig(fig, "roc_historical_proxy_and_gnn.png")

    fig, ax = plt.subplots(figsize=(7.1, 4.6))
    aucs = {17: .7098, 23: .6612, 42: .6942}
    for seed, colour in [(17, "#1f77b4"), (23, "#ff7f0e"), (42, "#2ca02c")]:
        s = cpu[(cpu.proposed_split == "development_validation") & (cpu.model == f"gnn_seed_{seed}")]
        if s.label.nunique() == 2:
            fpr, tpr, _ = roc_curve(s.label, s.probability)
            ax.plot(fpr, tpr, color=colour, label=f"CPU GNN seed {seed} (AUC {aucs[seed]:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#999")
    ax.set(title="ROC History Curve - CPU GNN Validation", xlabel="False positive rate", ylabel="True positive rate")
    ax.grid(alpha=.2); ax.legend(fontsize=7, loc="lower right")
    charts["roc_cpu"] = savefig(fig, "roc_cpu_gnn.png")

    for source, label, aps, name in [(cpu, "CPU", {17:.1208,23:.1061,42:.1178}, "average_precision_cpu_gnn.png"), (gpu, "GPU", {17:.1236,23:.1081,42:.1276}, "average_precision_gpu_gnn.png")]:
        fig, ax = plt.subplots(figsize=(7.1, 4.6))
        for seed, colour in [(17, "#1f77b4"), (23, "#ff7f0e"), (42, "#2ca02c")]:
            s = source[(source.proposed_split == "development_validation") & (source.model == f"gnn_seed_{seed}")]
            if s.label.nunique() == 2:
                p, r, _ = precision_recall_curve(s.label, s.probability)
                ax.plot(r, p, color=colour, label=f"{label} seed {seed} (AP {aps[seed]:.4f})")
        ax.set(title=f"Average Precision Comparison Curves - {label} GNN", xlabel="Recall", ylabel="Precision")
        ax.grid(alpha=.2); ax.legend(fontsize=7, loc="upper right")
        charts["ap_" + label.lower()] = savefig(fig, name)

    fig, ax = plt.subplots(figsize=(7.1, 4.6))
    for seed, colour in [(17, "#1f77b4"), (23, "#ff7f0e"), (42, "#2ca02c")]:
        history = read_json(ROOT / f"gnn_proxy_experiment_gpu_v1/gnn_seed_{seed}_history.json")
        ax.plot([x["epoch"] for x in history], [x["validation_ap"] for x in history], color=colour, label=f"Seed {seed}")
    ax.set(title="Training History - Validation Average Precision", xlabel="Epoch", ylabel="Validation AP")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    charts["history"] = savefig(fig, "gnn_training_history.png")

    cm = np.array([[1994, 100], [91, 15]])
    fig, ax = plt.subplots(figsize=(4.5, 3.8)); im = ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=13, color="white" if value > 1100 else "black")
    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"]); ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title("GNN - Development-Test Confusion Matrix"); fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    charts["cm"] = savefig(fig, "gnn_seed42_confusion_matrix.png")

    # Compact result graph for the report section. Values are computed from
    # the saved per-window prediction files; best epoch comes from each
    # corresponding saved training history.
    result_rows = []
    for device, source_path, history_root in [("CPU", ROOT / "gnn_proxy_experiment_v1/per_window_predictions.csv", ROOT / "gnn_proxy_experiment_v1"),
                                              ("GPU", ROOT / "gnn_proxy_experiment_gpu_v1/per_window_predictions.csv", ROOT / "gnn_proxy_experiment_gpu_v1")]:
        source = pd.read_csv(source_path)
        for seed in (17, 23, 42):
            history = read_json(history_root / f"gnn_seed_{seed}_history.json")
            best = max(history, key=lambda x: x["validation_ap"])
            for split in ("development_validation", "development_test"):
                sample = source[(source.proposed_split == split) & (source.model == f"gnn_seed_{seed}")]
                result_rows.append({
                    "model": f"{device} {seed}", "epoch": int(best["epoch"]), "split": split,
                    "ap": average_precision_score(sample.label, sample.probability),
                    "auc": roc_auc_score(sample.label, sample.probability),
                })
    result = pd.DataFrame(result_rows)
    labels = [f"{device} {seed}\nE{int(result[(result.model == f'{device} {seed}') & (result.split == 'development_validation')].epoch.iloc[0])}"
              for device in ("CPU", "GPU") for seed in (17, 23, 42)]
    x = np.arange(len(labels)); width = .36
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35))
    for ax, metric, title in [(axes[0], "ap", "Average Precision"), (axes[1], "auc", "ROC-AUC")]:
        val = [result[(result.model == f"{device} {seed}") & (result.split == "development_validation")][metric].iloc[0]
               for device in ("CPU", "GPU") for seed in (17, 23, 42)]
        test = [result[(result.model == f"{device} {seed}") & (result.split == "development_test")][metric].iloc[0]
                for device in ("CPU", "GPU") for seed in (17, 23, 42)]
        ax.bar(x - width/2, val, width, label="Validation", color="#234f88")
        ax.bar(x + width/2, test, width, label="Development test", color="#8aaed1")
        for bars in ax.containers:
            ax.bar_label(bars, fmt="%.3f", padding=1, fontsize=4.8, rotation=90)
        ax.set_title(title); ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=6.4)
        ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.2); ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("GNN Result Summary - Saved Prediction Data", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .93])
    charts["gnn_result_summary"] = savefig(fig, "gnn_result_summary.png")
    return charts

ST = {}
def P(text, style):
    return Paragraph(str(text).replace("&", "&amp;"), ST[style])

def make_table(rows, widths, font=7.2):
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([("FONTNAME", (0,0), (-1,0), "Times-Bold"), ("FONTNAME", (0,1), (-1,-1), "Times-Roman"), ("FONTSIZE", (0,0), (-1,-1), font), ("LEADING", (0,0), (-1,-1), font+2), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dce8f5")), ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#7892ad")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0), (-1,-1), 4), ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4)]))
    return t

def picture(path, caption, width=165*mm, height=None):
    height = height or width * .57
    return [Image(str(path), width=width, height=height), P(caption, "caption")]

def supplied_picture(key, caption, width=115*mm, height=87.5*mm):
    path = SUPPLIED_IMAGES[key]
    if path.exists():
        return picture(path, caption, width=width, height=height)
    return [box("IMAGE UNAVAILABLE", f"The supplied image for {caption} was not found at {path}."), Spacer(1, 2*mm)]

def box(title, text, height=45*mm):
    t = Table([[P(f"<b>{title}</b><br/>{text}", "body")]], colWidths=[165*mm], rowHeights=[height])
    t.setStyle(TableStyle([("BOX", (0,0), (-1,-1), 1, colors.HexColor("#234f88")), ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#edf3f9")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (0,0), (-1,-1), "CENTER")]))
    return t

def on_page(canvas, doc):
    canvas.saveState(); canvas.setStrokeColor(colors.black); canvas.setLineWidth(.7); canvas.rect(13*mm, 13*mm, A4[0]-26*mm, A4[1]-26*mm)
    canvas.setFont("Times-Roman", 8); canvas.setFillColor(colors.HexColor("#555555")); canvas.drawCentredString(A4[0]/2, 8*mm, f"APEX-R Model Training Report  |  Page {doc.page}"); canvas.restoreState()

def build():
    global ST
    s = getSampleStyleSheet()
    ST = {
        "title": ParagraphStyle("title", parent=s["Title"], fontName="Times-Bold", fontSize=25, leading=30, alignment=TA_CENTER, textColor=colors.HexColor("#234f88"), spaceAfter=14),
        "subtitle": ParagraphStyle("subtitle", parent=s["Normal"], fontName="Times-Roman", fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=s["Heading1"], fontName="Times-Bold", fontSize=17, leading=21, spaceBefore=8, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=s["Heading2"], fontName="Times-Bold", fontSize=13, leading=16, spaceBefore=7, spaceAfter=5),
        "body": ParagraphStyle("body", parent=s["BodyText"], fontName="Times-Roman", fontSize=10.2, leading=14, spaceAfter=7),
        "small": ParagraphStyle("small", parent=s["BodyText"], fontName="Times-Roman", fontSize=8.2, leading=10.4, spaceAfter=4),
        "caption": ParagraphStyle("caption", parent=s["BodyText"], fontName="Times-Italic", fontSize=8.5, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=7),
        "bullet": ParagraphStyle("bullet", parent=s["BodyText"], fontName="Times-Roman", fontSize=10, leading=13, leftIndent=13, firstLineIndent=-8, spaceAfter=3),
    }
    c = make_charts()
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=22*mm, leftMargin=22*mm, topMargin=20*mm, bottomMargin=19*mm)
    st = []
    st += [Spacer(1,14*mm), P("APEX-R: Race Strategy Studio", "title"), P("TrackShift Hackathon 2025", "subtitle"), Spacer(1,4*mm), P("PROJECT TITLE", "h1"), P("Graph Neural Network and XGBoost experiments for Formula 1 race-energy decision support", "subtitle"), P("Team Devsez | VIPS-TC (GGSIPU)", "subtitle"), Spacer(1,8*mm), P("PROJECT DELIVERABLES", "h1")]
    for a,b in [("Historical overtake-opportunity models", "Legacy, clean-label, enriched and engineered XGBoost comparisons."), ("Graph learning proxy experiment", "GPU-trained GNN on 37 telemetry-coverage-passed races."), ("Race replay and strategy studio", "Historical telemetry reference, simulated energy and constrained action branches."), ("Evaluation report and visual analytics", "ROC-AUC, average-precision curves, confusion matrix and model limitations."), ("Working local application", "FastAPI backend with CUDA inference when available and an offline-first frontend.")]: st.append(P(f"- <b>{a}</b><br/>{b}", "bullet"))
    st += [Spacer(1,5*mm), P("PROJECT SNAPSHOT", "h1"), make_table([["Area", "Verified scope"], ["Data", "37 telemetry-coverage-passed races; historical public race telemetry"], ["Models", "Logistic Regression, XGBoost, and compact CPU/GPU GNN experiments"], ["Final demo model", "Hybrid GNN + GRU + Physics Model view; GNN GPU seed 42 is the frozen inference base"], ["Decision layer", "Constrained simulated energy strategy; GNN remains advisory"], ["Evidence boundary", "Proxy position-swap task; not verified overtaking or real battery telemetry"]], [38*mm,127*mm], 7.4), Spacer(1,5*mm), P("This report follows the supplied hackathon-report structure. Metrics remain separated by task and split; they are not one common benchmark.", "body"), PageBreak()]

    st += [P("EXECUTIVE SUMMARY", "h1"), P("APEX-R combines supervised race-state experiments with a constrained energy simulator. The final demo model is the Hybrid GNN + GRU + Physics Model view, while the GPU GNN seed 42 checkpoint is the second-last reproducible graph model and frozen inference base. The graph task remains a fixed-pair classified-order position-swap proxy and all model outputs remain advisory.", "body")]
    st += [make_table([["Family", "Target", "Best result", "Meaning"], ["Legacy OpenF1", "Overtake within 60 s", "XGB AUC 0.7879; AP 0.1939", "Opportunity proxy"], ["Clean OpenF1", "Clean overtake within 60 s", "AUC 0.9306; AP 0.2470", "Sealed baseline"], ["Engineered OpenF1", "Same clean target", "P 12.73% at R 70.04%", "Validation candidate"], ["GNN", "Next-lap boundary swap", "AUC 0.7009; AP 0.1043", "Experimental proxy"]], [31*mm,45*mm,43*mm,46*mm], 7.0), Spacer(1,4*mm), P("DATASET AND LIMITATIONS", "h1"), P("The OpenF1 models used sessions 7953, 7779, 7787 and session 9070 for evaluation. The GNN used 37 races, 15,404 supervised graphs, 822 positive proxy windows and 14,582 negative proxy windows. Nineteen windows were excluded because attacker or target telemetry was missing. Unknown/censored windows were not converted into negatives.", "body"), P("The audited public data does not contain actual ERS percentage, battery state of health, battery temperature, private power maps, fuel load or pit-wall strategy instructions. Dashboard energy values are simulated.", "body"), PageBreak()]

    st += [P("MODEL TRAINING METHODOLOGY", "h1"), P("The experiments were staged from simple baselines to engineered features and graph learning. Race/session-separated validation was used. The GNN used whole-race chronology partitions and a disk-backed graph cache to avoid host-RAM exhaustion.", "body"), P("EXPERIMENT MAP", "h2"), make_table([["Stage", "Primary input", "Evaluation meaning"], ["Legacy", "Eight baseline race-state features", "Initial overtake-opportunity benchmark"], ["Feature engineering", "Rolling telemetry and relative context", "Validation-only feature contribution"], ["GNN", "Car nodes, physical-neighbour edges, fixed driver pair", "Boundary position-swap proxy"], ["Dashboard", "Historical reference + simulated energy", "Offline demo; not causal race outcome evidence"]], [35*mm,67*mm,63*mm], 7.2), Spacer(1,3*mm), P("COMMON METRICS", "h2"), P("ROC-AUC measures ranking discrimination. Average Precision is more informative for rare positives. Brier score measures probability error. Accuracy is shown with its threshold because negative-class prevalence can make it look high.", "body"), P("ENVIRONMENT", "h2"), P("RTX 3050 Laptop GPU; PyTorch 2.11.0+cu128; CUDA 12.8; PyTorch Geometric 2.8.0.post1. GNN configuration: two GINEConv layers, hidden width 32, dropout 0.2, Adam lr 0.001, weight decay 0.0001, batch size 32, unweighted BCE-with-logits, maximum 100 epochs and patience 15.", "body"), PageBreak()]

    st += [P("LEGACY MODELS", "h1"), P("Legacy Logistic Regression", "h2"), P("StandardScaler, C=1.0, lbfgs, maximum 2,000 iterations, 12 actual iterations, no class weighting, random state 2026; eight baseline features.", "body"), make_table([["ROC-AUC", "AP", "Brier", "Accuracy at 0.50"], ["0.711676", "0.122823", "0.058589", "93.72%"]], [38*mm]*4), Spacer(1,2*mm)] + supplied_picture("legacy_lr", "Figure 6. Supplied confusion matrix - Legacy Logistic Regression.") + [P("Why it was not selected: it provided a useful linear baseline, but its ROC-AUC and Average Precision were lower than the later XGBoost candidates.", "small"), Spacer(1,3*mm), P("Legacy XGBoost", "h2"), P("300 trees, depth 4, learning rate 0.04, min child weight 8, subsample 0.9, column sample 0.9, L1 0.05, L2 5.0, binary logistic objective, random state 2026.", "body"), make_table([["ROC-AUC", "AP", "Brier", "Accuracy at 0.50"], ["0.787855", "0.193889", "0.055747", "93.39%"]], [38*mm]*4), Spacer(1,2*mm)] + supplied_picture("legacy_xgb", "Figure 7. Supplied confusion matrix - Legacy XGBoost.") + [P("Why it was not selected: it improved on Logistic Regression, but was superseded by the clean-label and feature-engineered XGBoost experiments.", "small"), Spacer(1,3*mm), P("CLEANED FEATURE XGBOOST ACTUAL", "h1"), P("Eight baseline features; maximum 1,500 trees, depth 4, learning rate 0.025, early stopping 100 rounds, min child weight 8, subsample 0.9, column sample 0.9, L1 0.05, L2 6.0, max_delta_step 1, random state 2026. Saved locked checkpoint: 75 trees.", "body"), make_table([["Evaluation", "ROC-AUC", "AP", "Brier", "Accuracy", "Precision", "Recall", "F1"], ["Session 9070 validation", "0.806429", "0.167154", "0.038128", "95.76%", "-", "-", "-"], ["Session 11353 holdout", "0.930631", "0.247008", "0.026587", "85.83%", "15.51%", "84.42%", "0.262097"]], [40*mm,20*mm,18*mm,19*mm,20*mm,20*mm,20*mm,16*mm], 6.5), P("Holdout confusion matrix: TP=65, FP=354, FN=12, TN=2152.", "small"), Spacer(1,2*mm)] + supplied_picture("clean_xgb", "Figure 8. Supplied confusion matrix - Clean 8-feature XGBoost Actual.") + [P("Why it was not selected for the GNN demo: it is the strongest locked OpenF1 baseline, but it solves a different 60-second overtake-opportunity task and is not the fixed-pair graph proxy used by the demo model.", "small"), PageBreak()]

    st += [P("ENRICHED AND ENGINEERED XGBOOST", "h1"), P("Enriched XGBoost", "h2"), P("Added 10-second rolling speed, speed delta, throttle, brake, RPM, gear and DRS-open features to the baseline. Same clean-label split and early stopping.", "body"), make_table([["ROC-AUC", "AP", "Brier", "Accuracy"], ["0.794901", "0.166549", "0.038203", "95.76%"]], [38*mm]*4), Spacer(1,2*mm)] + supplied_picture("enriched_xgb", "Figure 9. Supplied confusion matrix - Enriched XGBoost.") + [P("Why it was not selected: the additional rolling channels did not improve the validation ranking or probability metrics enough to displace the later engineered candidates.", "small"), Spacer(1,3*mm), P("Engineered XGBoost sweep", "h2"), make_table([["Configuration", "ROC-AUC", "AP", "Brier", "Precision @ recall >=70%"], ["Weight 1.0", "0.813796", "0.170905", "0.038155", "9.66%"], ["Weight 1.5", "0.820597", "0.201444", "0.038715", "10.98%"], ["Weight 2.0", "0.824588", "0.206596", "0.041459", "11.36%"], ["Weight 3.0", "0.823136", "0.203604", "0.046523", "11.25%"], ["Weight 2.0 + Platt", "0.824588", "0.206596", "0.037802", "11.36%"], ["MLP 64/32/16", "0.582879", "0.055759", "0.043294", "5.34%"], ["Explicit interactions", "0.819668", "0.199591", "0.041235", "11.24%"]], [42*mm,25*mm,23*mm,23*mm,45*mm], 6.8), P("Supplied confusion-matrix F1: 0.6667 (TP=2, FP=1, FN=1). This is image-derived and is not a replacement for the full validation sweep.", "small"), Spacer(1,2*mm)] + supplied_picture("engineered_xgb", "Figure 10. Supplied confusion matrix - Engineered XGBoost (weight 2.0).") + [P("Why it was not selected: it was a strong XGBoost validation candidate, but Phase 1 TracingInsights features produced the better same-split candidate on the selected precision-at-recall criterion.", "small"), Spacer(1,3*mm), P("PHASE 1 TRACINGINSIGHTS XGBOOST", "h1"), P("125 trees, depth 3, learning rate 0.03, scale_pos_weight 2.0, grouped race-level Platt calibration, 19 total features and native NaN handling.", "body"), make_table([["ROC-AUC", "AP", "Brier", "ECE", "Precision", "Recall", "Threshold"], ["0.818115", "0.188670", "0.037707", "0.020933", "12.73%", "70.04%", "0.07526282"]], [25*mm]*7, 7.0), P("Supplied confusion-matrix F1: 0.7500 (TP=3, FP=1, FN=1). This image-derived value should be read alongside, not merged into, the reported validation metrics.", "small"), Spacer(1,2*mm)] + supplied_picture("phase1_xgb", "Figure 11. Supplied confusion matrix - Phase 1 TracingInsights XGBoost.") + [P("Why it was not selected for the demo: it was the best same-split XGBoost candidate, but it was never evaluated on the sealed holdout and its task/features do not match the frozen graph-proxy demo contract.", "small"), PageBreak()]

    st += [P("PHASE 5 OPENF1 PIT CANDIDATE", "h1"), P("Winner: 75 trees, depth 3, learning rate 0.05, scale_pos_weight 1.0, raw probabilities and 23 total features. OpenF1 returned zero pit records for every approved session, so the four pit fields were 100% missing and contributed no learned signal.", "body"), make_table([["ROC-AUC", "AP", "Brier", "ECE", "Precision", "Recall", "Threshold"], ["0.819668", "0.186187", "0.037604", "0.011769", "13.25%", "71.60%", "0.06664582"]], [25*mm]*7, 7.0), P("Why it was not selected: its pit-derived columns were entirely missing in the approved sessions, so it did not add usable signal beyond the existing features.", "small"), Spacer(1,6*mm), P("GNN", "h1"), P("The selected GNN uses edge-aware GINEConv message passing over 37 telemetry-coverage-passed races. Target: fixed attacker/target classified-order reversal at the next lap boundary. Official demo checkpoint: GPU seed 42, best epoch 28; 43 epochs were run and validation AP selected the checkpoint.", "body"), make_table([["Split", "Graphs", "Positives", "Negatives", "AP", "ROC-AUC", "Brier"], ["Validation", "2,533", "103", "2,430", "0.127602", "0.689272", "0.038754"], ["Development test", "2,200", "106", "2,094", "0.104338", "0.700857", "0.045965"]], [37*mm,24*mm,23*mm,23*mm,21*mm,24*mm,22*mm], 7.1), P("Frozen GPU GNN development-test threshold result: accuracy 91.32%, precision 13.04%, recall 14.15%, F1 13.57%; TP=15, FP=100, FN=91, TN=1994.", "small"), Spacer(1,2*mm)] + picture(c["gnn_result_summary"], "Figure 12. GNN result summary from saved CPU/GPU prediction files; E labels show each run's best epoch.", width=150*mm, height=70*mm) + [P("Why this model was used: it was the best saved GNN configuration for the demo proxy by validation AP (0.1276), with a frozen reproducible checkpoint at GPU seed 42, epoch 28. It matches the graph-input contract and remains an advisory signal; it does not select the strategy action.", "small"), PageBreak(), P("FINAL MODEL: FROZEN HYBRID GNN + GRU + PHYSICS MODEL", "h1"), P("Physics-degradation and multi-signal final model view", "h2")] + supplied_picture("hybrid_gnn", "Figure 13. Hybrid GNN prediction-probability distribution supplied for the final physics-degradation view.", width=155*mm, height=92*mm) + [P("Why this is our final model: the Hybrid GNN + GRU + Physics Model is the strongest integrated model view for the demo because it combines graph-based race context, temporal modelling, tyre degradation, pit-stop and safety-constraint signals. This is the final demo-model designation; the supplied plot alone is not a new accuracy claim.", "small"), P("The supplied graph reports 4,242 valid classification samples, but no standalone checkpoint, full metric table or reproducible Physics-Degrade experiment is present in the project artifacts. Therefore its independent F1 remains N/A. The frozen reproducible inference base is the GPU GNN seed 42 checkpoint at epoch 28.", "small"), Spacer(1,3*mm), P("FROZEN HYBRID GNN RESULTS", "h2"), Image(str(SUPPLIED_IMAGES["hybrid_gnn_confusion"]), width=165*mm, height=56*mm), P("Figure 14. Supplied Frozen Hybrid GNN + GRU + Physics Model classification summary.", "caption"), P("The supplied figure reports 4,242 valid classification rows and presents separate results for tyre degradation, pit-stop observed and safety constraint. It is included as supplied visual evidence and is not independently recomputed from a retained project checkpoint.", "small"), PageBreak()]

    st += [P("ROC-AUC CURVES AND TRAINING HISTORY", "h1")] + picture(c["roc_gnn"], "Figure 1. Historical proxy Logistic Regression and GPU GNN ROC curves.") + [Spacer(1,3*mm)] + picture(c["roc_cpu"], "Figure 2. CPU GNN ROC comparison for seeds 17, 23 and 42.") + [PageBreak()]
    st += [P("PHASE 5 OPENF1 ROC CURVE", "h1"), box("ROC CURVE DATA PLACEHOLDER", "The saved Phase 5 artifact retains the summary AUC of 0.819668 but does not retain raw per-window probabilities and labels required to reconstruct an exact ROC curve. The curve is not fabricated."), P("Reported Phase 5 metrics: AP 0.186187, Brier 0.037604, ECE 0.011769. Pit records were unavailable in all approved sessions.", "body"), Spacer(1,4*mm)] + picture(c["history"], "Figure 3. GPU GNN validation AP history used for checkpoint selection.") + [PageBreak()]
    st += [P("AVERAGE PRECISION COMPARISONS", "h1")] + picture(c["ap_cpu"], "Figure 4. Average Precision curves for CPU GNN seeds 17, 23 and 42.") + [Spacer(1,3*mm)] + picture(c["ap_gpu"], "Figure 5. Average Precision curves for GPU GNN seeds 17, 23 and 42.") + [PageBreak()]
    st += [P("CONFUSION MATRIX AND PREDICTION RESULTS", "h1")] + picture(c["cm"], "Figure 6. GNN seed 42 development-test confusion matrix at threshold 0.18466353.") + [P("The 91.32% accuracy is not the primary success measure because the set contains 2,094 negative proxy windows and only 106 positive proxy windows. AP, ROC-AUC and the confusion matrix are needed together.", "body"), PageBreak()]

    screenshots = [
        ("/tmp/codex-clipboard-4b0173a0-2a9a-423f-ad33-b6b88088871b.png", "Screenshot 1 - Initial strategy dashboard", "The dense simulator layout makes the model signal difficult to notice; this is a presentation issue, not proof of checkpoint failure."),
        ("/tmp/codex-clipboard-c84a2eaa-3d55-4467-908d-3836695aeb52.png", "Screenshot 2 - Historical replay reference", "Observed telemetry and simulated branches appear together; missing speed fields can make the stream look incomplete."),
        ("/tmp/codex-clipboard-62e7bd4e-68cd-42b7-a205-c54ad1d61b58.png", "Screenshot 3 - Replay with advisory score", "The GNN is visible but has zero action influence, so DEFEND can remain the simulator choice even when the score changes."),
        ("/tmp/codex-clipboard-23f3896a-3e44-4a6c-a7ef-e12a67e74b62.png", "Screenshot 4 - Fixed historical decision window", "The fixed pair is correct for the proxy task, but the display can be mistaken for a continuously changing overtake predictor."),
    ]
    st += [P("APPLICATION SCREENSHOTS AND OBSERVED ISSUES", "h1")]
    for path, cap, issue in screenshots:
        if Path(path).exists(): st += [Image(path, width=160*mm, height=88*mm), P(cap, "caption"), P(f"<b>Observed issue:</b> {issue}", "small"), Spacer(1,2*mm)]
        else: st += [box(cap, "Image file is unavailable; reserved space kept for replacement."), P(f"<b>Observed issue:</b> {issue}", "small")]
    st += [PageBreak(), P("CHALLENGES AND SOLUTIONS", "h1"), P("Challenge 1: Rare positive events", "h2"), P("Positive windows are uncommon, so accuracy can look high while positive detection is weak. AP, ROC-AUC, precision, recall and confusion counts are reported together.", "body"), P("Challenge 2: Different targets", "h2"), P("The GNN uses a boundary position-swap proxy while the original XGBoost uses a cleaned 60-second opportunity target. Their scores must remain separate.", "body"), P("Challenge 3: Missing energy channels", "h2"), P("Actual battery percentage, ERS deployment, SOH and battery temperature were not available in the approved public historical data. Dashboard energy is simulated.", "body"), P("Challenge 4: Host memory", "h2"), P("The disk-backed graph store and lazy loading reduced the measured load/preprocessing/one-batch peak to approximately 498.5 MiB while preserving graph values.", "body"), P("Challenge 5: Persistent DEFEND display", "h2"), P("DEFEND may be selected because the supplied rear-gap and energy state make it best, or because input freshness fails. The action must not be forced to make the model appear successful.", "body"), PageBreak()]
    st += [P("CONCLUSION AND FUTURE WORK", "h1"), P("APEX-R now has a reproducible progression from baseline XGBoost to engineered features, proxy Logistic Regression and GPU GNN. The clean 8-feature XGBoost remains the only model with a locked final holdout score. The final demo model is the Hybrid GNN + GRU + Physics Model view, with the frozen seed-42 GNN retained as the second-last reproducible inference base.", "body"), P("Next priorities: improve causal live-input coverage, obtain independently verified timestamped pass labels, retain raw probability/label files for every requested curve, and obtain licensed energy telemetry before training an energy model.", "body"), P("FINAL STATUS", "h2"), P("Final demo model: Hybrid GNN + GRU + Physics Model. Frozen reproducible inference base: GNN GPU seed 42, best epoch 28. Output remains advisory; the energy simulator selects the final action.", "body"), P("The saved experiment reports remain the source of truth for exact values and configurations.", "small")]
    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Created: {PDF}")
    print(f"Bytes: {PDF.stat().st_size}")

if __name__ == "__main__": build()
