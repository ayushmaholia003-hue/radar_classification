import html
from pathlib import Path

import joblib


MODEL_PATH = "models/xgboost_classifier.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
OUTPUT_PATH = "feature_importance.svg"


def main():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    feature_names = preprocessor.get_feature_names()
    importances = model.estimator.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)

    width = 980
    height = 560
    left_margin = 240
    right_margin = 70
    top_margin = 60
    bottom_margin = 40
    bar_height = 24
    bar_gap = 10
    max_bar_width = width - left_margin - right_margin
    max_value = max(importances) if len(importances) else 1.0

    rows = []
    for index, (name, value) in enumerate(ranked):
        y = top_margin + index * (bar_height + bar_gap)
        bar_width = 0 if max_value == 0 else (value / max_value) * max_bar_width
        safe_name = html.escape(name)
        rows.append(
            f'<text x="{left_margin - 12}" y="{y + 17}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="14" fill="#1f2937">{safe_name}</text>'
        )
        rows.append(
            f'<rect x="{left_margin}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" '
            f'rx="4" fill="#2563eb" />'
        )
        rows.append(
            f'<text x="{left_margin + bar_width + 10:.2f}" y="{y + 17}" '
            f'font-family="Arial, sans-serif" font-size="13" fill="#111827">{value:.4f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc" />
  <text x="{left_margin}" y="30" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#0f172a">
    Radar Classifier Feature Importance
  </text>
  <text x="{left_margin}" y="48" font-family="Arial, sans-serif" font-size="13" fill="#475569">
    Higher bars indicate stronger contribution in the trained gradient boosting model.
  </text>
  <line x1="{left_margin}" y1="{top_margin - 8}" x2="{left_margin}" y2="{height - bottom_margin}" stroke="#94a3b8" stroke-width="1" />
  {"".join(rows)}
</svg>
"""

    Path(OUTPUT_PATH).write_text(svg, encoding="utf-8")

    print("Feature importances:")
    for name, value in ranked:
        print(f"  {name:22s} {value:.6f}")
    print(f"\nSaved chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
