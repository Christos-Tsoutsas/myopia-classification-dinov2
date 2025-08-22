import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# === INPUT DATA ===
overall = pd.DataFrame([
    ["EfficientNet-B3", 0.88, 0.8846, 0.8592],
    ["ResNet50",       0.92, 0.9247, 0.9534],
    ["VGG16",          0.94, 0.9396, 0.9523],
    ["DINO-v2 (ViT)",  0.97, 0.9667, 0.9632],
], columns=["Model", "Accuracy", "Macro F1", "Peak Avg Val Acc"])

per_class = pd.DataFrame([
    ["EfficientNet-B3", "High_Myopia", 0.90, 0.76, 0.82, 0.95, 0.91],
    ["EfficientNet-B3", "Normal",      0.98, 1.00, 0.99, 1.00, 1.00],
    ["EfficientNet-B3", "Pathological_Myopia", 0.79, 0.90, 0.84, 0.96, 0.93],

    ["ResNet50", "High_Myopia", 0.93, 0.87, 0.90, 0.98, 0.96],
    ["ResNet50", "Normal",      1.00, 0.98, 0.99, 1.00, 1.00],
    ["ResNet50", "Pathological_Myopia", 0.85, 0.92, 0.89, 0.98, 0.98],

    ["VGG16", "High_Myopia", 0.94, 0.89, 0.91, 0.99, 0.97],
    ["VGG16", "Normal",      1.00, 1.00, 1.00, 1.00, 1.00],
    ["VGG16", "Pathological_Myopia", 0.88, 0.93, 0.91, 0.99, 0.98],

    ["DINO-v2 (ViT)", "High_Myopia", 0.96, 0.95, 0.95, 1.00, 0.99],
    ["DINO-v2 (ViT)", "Normal",      1.00, 1.00, 1.00, 1.00, 1.00],
    ["DINO-v2 (ViT)", "Pathological_Myopia", 0.94, 0.95, 0.95, 1.00, 0.99],
], columns=["Model", "Class", "Precision", "Recall", "F1", "ROC AUC", "PR AUC (AP)"])

# === UTILS ===
def star_best(df, cols, group_col=None):
    df = df.copy()
    if group_col is None:
        for c in cols:
            max_val = df[c].max()
            df[c] = df[c].map(lambda x: f"{x:.4f} ★" if x == max_val else f"{x:.4f}")
    else:
        for g in df[group_col].unique():
            mask = df[group_col] == g
            for c in cols:
                max_val = df.loc[mask, c].max()
                df.loc[mask, c] = df.loc[mask, c].map(lambda x: f"{x:.2f} ★" if x == max_val else f"{x:.2f}")
    return df

def plotly_table(df, title, filename_prefix):
    header_color = "#1f2d3d"
    cell_color = "#f8f9fa"
    header = dict(values=list(df.columns),
                  fill_color=header_color,
                  font_color="white",
                  align='center',
                  line_color='white')
    cells = dict(values=[df[c] for c in df.columns],
                 fill_color=[[cell_color]*len(df)],
                 align='center',
                 line_color='white',
                 font=dict(size=12))
    fig = go.Figure(data=[go.Table(header=header, cells=cells)])
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=20)),
        margin=dict(l=20, r=20, t=60, b=20),
        width=900,
        height=400 + 25*len(df)
    )
    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_dir / f"{filename_prefix}.html"))
    # PNG export (optional, requires kaleido)
    try:
        fig.write_image(str(out_dir / f"{filename_prefix}.png"), scale=2)
    except Exception:
        pass
    return fig

if __name__ == "__main__":
    overall_star = star_best(overall, ["Accuracy", "Macro F1", "Peak Avg Val Acc"])
    per_class_star = star_best(per_class, ["Precision", "Recall", "F1", "ROC AUC", "PR AUC (AP)"], group_col="Class")

    plotly_table(overall_star, "Overall Model Comparison", "overall_table_plotly")
    plotly_table(per_class_star, "Per-Class Metrics Comparison", "per_class_table_plotly")

    # Also export CSVs for record-keeping
    out_dir = Path("./outputs")
    overall.to_csv(out_dir / "overall_comparison.csv", index=False)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    print("Tables and CSVs saved in ./outputs")
