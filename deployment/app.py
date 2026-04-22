import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Poverty Prediction Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS / Theme ──────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0D1117;
    color: #E6EDF3;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #12243A 100%);
    border-right: 1px solid #1F6FEB44;
}
[data-testid="stSidebar"] * { color: #C9D1D9 !important; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #161B22 0%, #1C2333 100%);
    border: 1px solid #1F6FEB55;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px #00000055;
}
[data-testid="stMetricValue"] {
    color: #58A6FF !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 0.8rem !important; }
[data-testid="stMetricDelta"] { color: #3FB950 !important; }
h1 { color: #58A6FF !important; border-bottom: 1px solid #1F6FEB44; padding-bottom: 0.3rem; }
h2, h3 { color: #79C0FF !important; }
.stAlert { border-radius: 8px !important; border-left: 4px solid #1F6FEB !important; }
[data-testid="stDataFrame"] { border: 1px solid #1F6FEB33; border-radius: 8px; }
hr { border-color: #1F6FEB44 !important; margin: 1.5rem 0 !important; }
label { color: #8B949E !important; }
.stCaption, small { color: #8B949E !important; }
[data-testid="stMainBlockContainer"] { background-color: #0D1117; }
section.main > div { background-color: #0D1117; }
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib dark theme ───────────────────────────────────────────────────
# Define colors as constants so they can be reused throughout the file
CHART_BG     = "#0D1117"
CHART_FIG_BG = "#161B22"
CHART_TEXT   = "#C9D1D9"
CHART_GRID   = "#21262D"
CHART_EDGE   = "#30363D"

# Set rcParams FIRST before seaborn so seaborn inherits them
mpl.rcParams.update({
    "figure.facecolor":   CHART_FIG_BG,
    "axes.facecolor":     CHART_BG,
    "axes.edgecolor":     CHART_EDGE,
    "axes.labelcolor":    CHART_TEXT,
    "axes.titlecolor":    CHART_TEXT,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "xtick.color":        CHART_TEXT,
    "ytick.color":        CHART_TEXT,
    "xtick.labelcolor":   CHART_TEXT,
    "ytick.labelcolor":   CHART_TEXT,
    "text.color":         CHART_TEXT,
    "grid.color":         CHART_GRID,
    "grid.linewidth":     0.8,
    "legend.facecolor":   CHART_FIG_BG,
    "legend.edgecolor":   CHART_EDGE,
    "legend.labelcolor":  CHART_TEXT,
    "figure.titlesize":   13,
    "figure.titleweight": "bold",
    "font.size":          10,
    "patch.edgecolor":    CHART_BG,
})

# Apply seaborn AFTER rcParams and pass rc dict to prevent overrides
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor":   CHART_BG,
    "figure.facecolor": CHART_FIG_BG,
    "grid.color":       CHART_GRID,
    "axes.edgecolor":   CHART_EDGE,
    "text.color":       CHART_TEXT,
    "axes.labelcolor":  CHART_TEXT,
    "xtick.color":      CHART_TEXT,
    "ytick.color":      CHART_TEXT,
    "axes.titlecolor":  CHART_TEXT,
})


def _suptitle(fig, text, **kwargs):
    """suptitle with explicit color so no override can hide it."""
    kwargs.setdefault("fontsize", 13)
    kwargs.setdefault("fontweight", "bold")
    fig.suptitle(text, color=CHART_TEXT, **kwargs)


def _fix_fig(fig):
    """Force every text element in every axes to be visible after rendering."""
    for ax in fig.get_axes():
        ax.title.set_color(CHART_TEXT)
        ax.xaxis.label.set_color(CHART_TEXT)
        ax.yaxis.label.set_color(CHART_TEXT)
        ax.tick_params(colors=CHART_TEXT, which="both", labelcolor=CHART_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(CHART_EDGE)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_color(CHART_TEXT)
            leg.get_frame().set_facecolor(CHART_FIG_BG)
            leg.get_frame().set_edgecolor(CHART_EDGE)
    if fig._suptitle:
        fig._suptitle.set_color(CHART_TEXT)


# ─── Constants ───────────────────────────────────────────────────────────────
FEATURES = [
    "gdp_per_capita", "urban_population_percent", "rural_population_percent",
    "population_density", "total_population", "renewable_energy_percent",
    "government_effectiveness", "elec_access_change", "gdp_growth", "urban_change",
]
RISK_ORDER  = ["Severe", "Moderate", "Minimal"]
RISK_COLORS = ["#F85149", "#E3B341", "#3FB950"]


# ─── Data loaders ────────────────────────────────────────────────────────────
def _find(filename):
    candidates = [
        filename,
        os.path.join("data", "processed", os.path.basename(filename)),
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", os.path.basename(filename)),
        os.path.join(os.path.dirname(__file__), os.path.basename(filename)),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner="Loading analysis_ready.csv …")
def load_main():
    p = _find("analysis_ready.csv")
    if p:
        return pd.read_csv(p)
    st.error("Cannot find data/processed/analysis_ready.csv. Run data_wrangling.ipynb first.")
    st.stop()


@st.cache_data(show_spinner="Loading test predictions …")
def load_test():
    p = _find("test_predictions.csv")
    if p:
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner="Loading feature importance …")
def load_fi():
    p = _find("feature_importance.csv")
    if p:
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner="Loading forecast …")
def load_forecast():
    p = _find("forecast_2024_2027.csv")
    if p:
        return pd.read_csv(p)
    return None


# ─── Fallback model training ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Output CSVs not found — training models from scratch …")
def train_fallback(df):
    le = LabelEncoder()
    le.fit(RISK_ORDER)

    train = df[df["year"] <= 2017].copy()
    test  = df[df["year"] >= 2021].copy()
    med   = train[FEATURES].median()

    X_tr     = train[FEATURES].fillna(med)
    y_clf_tr = le.transform(train["risk_category"])
    y_reg_tr = train["electricity_access"]
    X_te     = test[FEATURES].fillna(med)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    reg = RandomForestRegressor(n_estimators=800, max_depth=16, min_samples_split=6,
                                min_samples_leaf=2, max_features=0.6, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_clf_tr)
    reg.fit(X_tr, y_reg_tr)

    t = test.copy()
    t["pred_electricity_access"] = reg.predict(X_te).clip(0, 100)
    t["pred_risk_category"]      = le.inverse_transform(clf.predict(X_te))
    test_df = t[["country", "year", "electricity_access", "risk_category",
                 "pred_electricity_access", "pred_risk_category"]].copy()

    fi_rows = []
    for feat, imp in zip(FEATURES, clf.feature_importances_):
        fi_rows.append({"feature": feat, "importance": imp, "model": "classifier"})
    for feat, imp in zip(FEATURES, reg.feature_importances_):
        fi_rows.append({"feature": feat, "importance": imp, "model": "regressor"})
    fi_df = pd.DataFrame(fi_rows)

    latest = df.sort_values("year").groupby("country").last().reset_index()
    rows = []
    prev = latest.copy()
    for yr in range(2024, 2028):
        fX     = prev[FEATURES].fillna(med)
        pred_r = reg.predict(fX).clip(0, 100)
        pred_c = le.inverse_transform(clf.predict(fX))
        for i, (_, row) in enumerate(prev.iterrows()):
            rows.append({"country": row["country"], "year": yr,
                         "pred_electricity_access": pred_r[i],
                         "pred_risk_category": pred_c[i]})
        prev = prev.copy()
        prev["electricity_access"] = pred_r
        prev["elec_access_change"] = pred_r - latest["electricity_access"].values
    fore_df = pd.DataFrame(rows)

    return test_df, fi_df, fore_df


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.markdown(
        "<div style='padding:10px 0 4px 0'>"
        "<span style='font-size:1.5rem'>⚡</span> "
        "<span style='font-size:1.1rem; font-weight:700; color:#58A6FF'>Energy Poverty</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("CAP5771 - Introduction to DataScience")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate to",
        [
            "🌍 Overview",
            "📊 Views 1–3: Trends & Progress",
            "🔬 Views 4–6: Features & Models",
            "✅ Views 7–8: Evaluation",
            "🔮 View 9: 2024–2027 Forecast",
            "🔍 Widget A: Country Explorer",
            "🌐 Widget B: Year Snapshot",
            "↔️ Widget C: Risk Transitions",
            "ℹ️ About",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:0.75rem; color:#8B949E; padding-top:4px'>"
        "📦 World Bank WDI · 218 Countries · 1990–2023"
        "</div>",
        unsafe_allow_html=True,
    )
    return page


# ─── Overview ────────────────────────────────────────────────────────────────
def page_overview(df):
    st.title("⚡ Energy Poverty Prediction Dashboard")
    st.markdown(
        "<p style='color:#8B949E; font-size:0.95rem; margin-top:-0.5rem'>"
        "Use the sidebar to navigate."
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    latest_yr = int(df[df["electricity_access"].notna()]["year"].max())
    latest    = df[df["year"] == latest_yr]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Countries",                 df["country"].nunique())
    c2.metric("Year range",                f"{int(df['year'].min())}–{int(df['year'].max())}")
    c3.metric(f"Global avg ({latest_yr})", f"{latest['electricity_access'].mean():.1f}%")
    c4.metric("Countries < 50 %",          int((latest["electricity_access"] < 50).sum()))
    c5.metric("Countries ≥ 90 %",          int((latest["electricity_access"] >= 90).sum()))

    st.divider()
    st.info(
        "**Key takeaway:** Global electricity access rose from ~72 % (1990) to ~90 % (2023), "
        "but ~700 M people remain without access — concentrated in Sub-Saharan Africa. "
        "`renewable_energy_percent` and `gdp_per_capita` are the strongest predictors. "
        "RF Baseline classifier achieved **92.97 % validation accuracy**."
    )

    st.markdown("#### Model Performance at a Glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Classifier Val Accuracy", "92.97%", "RF Baseline")
    m2.metric("Regressor Val R²",        "0.9178",  "RF Improved")
    m3.metric("Test Accuracy",           "87.77%",  "2021–2023")
    m4.metric("Test R²",                 "0.8633",  "MAE = 4.91 pp")


# ─── Views 1–3 ───────────────────────────────────────────────────────────────
def page_views_1_3(df):
    st.title("📊 Views 1–3: Trends & Progress")

    # View 1
    st.subheader("View 1 — Global Electricity Access (1990–2023)")
    st.caption(
        "Left: global average rising from ~72 % (1990) to ~90 % (2023). "
        "Right: bimodal distribution — many near 100 %, many stuck below 50 %."
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _suptitle(fig, "View 1 — Global Electricity Access (1990–2023)")

    global_avg = df.groupby("year")["electricity_access"].mean()
    axes[0].plot(global_avg.index, global_avg.values, color="#58A6FF", linewidth=2.5)
    axes[0].fill_between(global_avg.index, global_avg.values, alpha=0.15, color="#58A6FF")
    axes[0].axhline(90, color="#3FB950", linestyle="--", linewidth=1.5, label="90% — Minimal threshold")
    axes[0].axhline(50, color="#F85149", linestyle="--", linewidth=1.5, label="50% — Severe threshold")
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Average Electricity Access (%)")
    axes[0].set_title("Global Average Over Time"); axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.4)

    latest    = int(df["year"].max())
    latest_df = df[df["year"] == latest]["electricity_access"].dropna()
    axes[1].hist(latest_df, bins=25, color="#F78166", edgecolor=CHART_BG)
    axes[1].axvline(latest_df.mean(), color="#E3B341", linestyle="--", linewidth=2,
                    label=f"Mean = {latest_df.mean():.1f}%")
    axes[1].axvline(50, color="#F85149", linestyle=":", linewidth=1.5, label="50% boundary")
    axes[1].axvline(90, color="#3FB950", linestyle=":", linewidth=1.5, label="90% boundary")
    axes[1].set_xlabel(f"Electricity Access (%) in {latest}")
    axes[1].set_ylabel("Number of Countries")
    axes[1].set_title(f"Distribution Across Countries ({latest})")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.4)

    _fix_fig(fig)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.divider()

    # View 2
    st.subheader("View 2 — Risk Category Distribution")
    st.caption("Left: overall counts | Right: year-by-year trend with train/val/test split markers")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _suptitle(fig, "View 2 — Risk Category Distribution")

    counts = df["risk_category"].value_counts()
    vals   = [counts.get(c, 0) for c in RISK_ORDER]
    bars   = axes[0].bar(RISK_ORDER, vals, color=RISK_COLORS, edgecolor=CHART_BG, width=0.5)
    axes[0].set_title("Overall Risk Category Counts (1990–2023)")
    axes[0].set_ylabel("Number of Country-Year Rows"); axes[0].grid(True, axis="y", alpha=0.4)
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f"{v:,}", ha="center", fontweight="bold", color=CHART_TEXT)

    risk_by_year = df.groupby(["year", "risk_category"]).size().unstack(fill_value=0)
    for cat, col in zip(RISK_ORDER, RISK_COLORS):
        if cat in risk_by_year.columns:
            axes[1].plot(risk_by_year.index, risk_by_year[cat], label=cat, color=col, linewidth=2)
    axes[1].axvline(2017, color=CHART_TEXT, linestyle="--", linewidth=1, label="Train/Val split (2017)")
    axes[1].axvline(2020, color="#8B949E",  linestyle="--", linewidth=1, label="Val/Test split (2020)")
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Number of Countries")
    axes[1].set_title("Risk Categories Over Time"); axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.4)

    _fix_fig(fig)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("Sharp drop in Severe (red) after 2010 reflects major electrification pushes in South Asia and Sub-Saharan Africa.")
    st.divider()

    # View 3
    st.subheader("View 3 — Electricity Access Progress (1990–2023)")
    st.caption("Left: top 10 most improved | Right: 15 countries with lowest access today.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _suptitle(fig, "View 3 — Electricity Access Progress (1990–2023)")

    first_year  = df.groupby("country")["electricity_access"].first()
    last_year   = df.groupby("country")["electricity_access"].last()
    improvement = (last_year - first_year).sort_values(ascending=False).head(10)

    colors_imp = ["#1F6FEB" if i == 0 else "#58A6FF" for i in range(len(improvement))]
    axes[0].barh(improvement.index[::-1], improvement.values[::-1],
                 color=colors_imp[::-1], edgecolor=CHART_BG)
    axes[0].set_xlabel("Percentage Point Improvement")
    axes[0].set_title("Top 10 Most Improved Countries\n(1990–2023)")
    axes[0].grid(True, axis="x", alpha=0.4)
    for i, (country, val) in enumerate(zip(improvement.index[::-1], improvement.values[::-1])):
        axes[0].text(val + 0.3, i, f"+{val:.1f}pp", va="center", fontsize=8, color=CHART_TEXT)

    latest_year = int(df["year"].max())
    severe_now  = (df[df["year"] == latest_year]
                   .sort_values("electricity_access").head(15)[["country", "electricity_access"]])
    colors_sev  = ["#F85149" if v < 30 else "#FF7B72" for v in severe_now["electricity_access"]]
    axes[1].barh(severe_now["country"], severe_now["electricity_access"],
                 color=colors_sev, edgecolor=CHART_BG)
    axes[1].axvline(50, color=CHART_TEXT, linestyle="--", linewidth=1.5, label="50% Severe threshold")
    axes[1].set_xlabel("Electricity Access (%)")
    axes[1].set_title(f"15 Countries with Lowest Access ({latest_year})")
    axes[1].legend(fontsize=9); axes[1].grid(True, axis="x", alpha=0.4)
    for i, (_, row) in enumerate(severe_now.iterrows()):
        axes[1].text(row["electricity_access"] + 0.5, i,
                     f"{row['electricity_access']:.1f}%", va="center", fontsize=8, color=CHART_TEXT)

    _fix_fig(fig)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ─── Views 4–6 ───────────────────────────────────────────────────────────────
def page_views_4_6(df, feat_imp):
    st.title("🔬 Views 4–6: Features & Models")

    if feat_imp is None:
        st.warning("feature_importance.csv not found.")
        return

    # View 4
    st.subheader("View 4 — Feature Importance — Best Models")
    st.caption("Dark blue = single most important feature in each model")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _suptitle(fig, "View 4 — Feature Importance — Best Models")

    cls_imp = feat_imp[feat_imp["model"] == "classifier"].sort_values("importance")
    reg_imp = feat_imp[feat_imp["model"] == "regressor"].sort_values("importance")

    for ax, imp, title in zip(
        axes,
        [reg_imp, cls_imp],
        ["RF Regressor (Improved) — Best Regressor\n(Val R²=0.9178)",
         "RF Classifier (Baseline) — Best Classifier\n(Val Accuracy=0.9297)"],
    ):
        colors_bar = ["#1F6FEB" if v == imp["importance"].max() else "#388BFD"
                      for v in imp["importance"]]
        ax.barh(imp["feature"], imp["importance"], color=colors_bar, edgecolor=CHART_BG)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Importance Score"); ax.grid(True, axis="x", alpha=0.4)
        for i, (feat, val) in enumerate(zip(imp["feature"], imp["importance"])):
            ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=8, color=CHART_TEXT)

    _fix_fig(fig)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown(
        "**`renewable_energy_percent` dominates both models.** "
        "It is negatively correlated — high renewable % reflects traditional biomass in energy-poor countries."
    )
    st.divider()

    # View 5
    st.subheader("View 5 — Correlation with Electricity Access")
    st.caption("Blue = positive correlation | Red = negative correlation")

    numeric_cols = ["electricity_access", "gdp_per_capita", "urban_population_percent",
                    "rural_population_percent", "renewable_energy_percent",
                    "government_effectiveness", "population_density",
                    "elec_access_change", "gdp_growth", "urban_change"]
    corr = (df[numeric_cols].corr()["electricity_access"]
            .drop("electricity_access").sort_values())
    colors_corr = ["#F85149" if v < 0 else "#58A6FF" for v in corr.values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(corr.index, corr.values, color=colors_corr, edgecolor=CHART_BG)
    ax.axvline(0, color=CHART_TEXT, linewidth=0.8)
    ax.set_title("View 5 — Correlation with Electricity Access\n(All features vs target)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation"); ax.grid(True, axis="x", alpha=0.4)
    for i, (feat, val) in enumerate(zip(corr.index, corr.values)):
        offset = 0.01 if val >= 0 else -0.01
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"{val:.2f}", va="center", fontsize=9, ha=ha, color=CHART_TEXT)

    _fix_fig(fig)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("`rural_population_percent` is the strongest negative driver. `gdp_per_capita` and governance are strong positive drivers.")
    st.divider()

    # View 6
    st.subheader("View 6 — Model Performance Comparison (Validation Set)")
    st.caption("Dark blue bar = best model in each task")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _suptitle(fig, "View 6 — Model Performance Comparison (Validation Set)")

    cls_models = ["RF Baseline", "RF Improved", "XGBoost", "XGBoost Imp"]
    cls_scores = [0.9297, 0.9021, 0.9083, 0.9052]
    cls_colors = ["#1F6FEB", "#388BFD", "#E3B341", "#F0C862"]
    bars = axes[0].bar(cls_models, cls_scores, color=cls_colors, edgecolor=CHART_BG, width=0.5)
    axes[0].set_ylim(0.85, 0.955); axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_title("Classification — Validation Accuracy\nBest: RF Baseline (0.9297)")
    axes[0].grid(True, axis="y", alpha=0.4); axes[0].tick_params(axis="x", labelsize=9)
    for bar, val in zip(bars, cls_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                     f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9,
                     color=CHART_TEXT)

    reg_models = ["RF Baseline", "RF Improved", "XGBoost", "XGBoost Imp"]
    reg_scores = [0.9122, 0.9178, 0.9064, 0.9101]
    reg_colors = ["#388BFD", "#1F6FEB", "#F0C862", "#E3B341"]
    bars2 = axes[1].bar(reg_models, reg_scores, color=reg_colors, edgecolor=CHART_BG, width=0.5)
    axes[1].set_ylim(0.88, 0.930); axes[1].set_ylabel("Validation R²")
    axes[1].set_title("Regression — Validation R²\nBest: RF Improved (0.9178)")
    axes[1].grid(True, axis="y", alpha=0.4); axes[1].tick_params(axis="x", labelsize=9)
    for bar, val in zip(bars2, reg_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                     f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9,
                     color=CHART_TEXT)

    _fix_fig(fig)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("**RF Baseline outperformed all tuned classifiers** — simpler models generalise better when class distribution shifts over time.")


# ─── Views 7–8 ───────────────────────────────────────────────────────────────
def page_views_7_8(df, test_df):
    st.title("✅ Views 7–8: Test Evaluation & GDP Analysis")

    if test_df is None:
        st.warning("test_predictions.csv not found. Run data_modeling.ipynb and save the CSV.")
        return

    # View 7
    st.subheader("View 7 — Test Set Performance (2021–2023)")
    st.caption("Left: actual vs predicted regression | Right: confusion matrix")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _suptitle(fig, "View 7 — Test Set Performance (2021–2023)")

    axes[0].scatter(test_df["electricity_access"], test_df["pred_electricity_access"],
                    alpha=0.4, s=20, color="#58A6FF")
    lo = test_df["electricity_access"].min(); hi = test_df["electricity_access"].max()
    axes[0].plot([lo, hi], [lo, hi], color="#F85149", linestyle="--", linewidth=1.5,
                 label="Perfect prediction")
    axes[0].set_xlabel("Actual Electricity Access (%)")
    axes[0].set_ylabel("Predicted Electricity Access (%)")
    axes[0].set_title("Regression: Actual vs Predicted\nTest R²=0.8633  MAE=4.91pp")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.4)

    le_dash = LabelEncoder(); le_dash.fit(["Minimal", "Moderate", "Severe"])
    y_true = le_dash.transform(test_df["risk_category"])
    y_pred = le_dash.transform(test_df["pred_risk_category"])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_dash.classes_, yticklabels=le_dash.classes_, ax=axes[1],
                linewidths=0.5, linecolor=CHART_BG)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].set_title("Classification: Confusion Matrix\nTest Accuracy=0.8777")

    _fix_fig(fig)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("**Moderate class is hardest to predict (F1 ≈ 0.59)** — transition-zone countries are sensitive to one-off investments or political changes.")
    st.divider()

    # View 8
    st.subheader("View 8 — GDP vs Electricity Access")
    st.caption("Each dot = one country in that reference year. Log10 GDP on x-axis.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("View 8 — GDP per Capita vs Electricity Access\n(each dot = one country)",
                 fontsize=12, fontweight="bold")

    years_to_show = [1995, 2005, 2015, 2023]
    palette       = ["#388BFD", "#3FB950", "#E3B341", "#F85149"]
    for yr, col in zip(years_to_show, palette):
        sub = df[df["year"] == yr].dropna(subset=["gdp_per_capita", "electricity_access"])
        ax.scatter(np.log10(sub["gdp_per_capita"] + 1), sub["electricity_access"],
                   alpha=0.6, s=30, color=col, label=str(int(yr)))

    ax.axhline(90, color="#3FB950", linestyle="--", linewidth=1, alpha=0.7, label="90% Minimal boundary")
    ax.axhline(50, color="#F85149", linestyle="--", linewidth=1, alpha=0.7, label="50% Severe boundary")
    ax.set_xlabel("Log10 GDP per Capita (USD)"); ax.set_ylabel("Electricity Access (%)")
    ax.legend(title="Year", fontsize=9); ax.grid(True, alpha=0.4)

    _fix_fig(fig)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("Countries moved **upward and rightward** over time — richer and better electrified.")


# ─── View 9 ──────────────────────────────────────────────────────────────────
def page_view_9(df, forecast_df):
    st.title("🔮 View 9 — 2024–2027 Energy Poverty Forecast")

    if forecast_df is None:
        st.warning("forecast_2024_2027.csv not found. Run data_modeling.ipynb and save the CSV.")
        return

    st.markdown("""
| Year | Avg Access | Severe | Moderate | Minimal |
|------|-----------|--------|----------|---------| 
| 2024 | 85.8%     | 30     | 28       | 160     |
| 2025 | 85.9%     | 30     | 27       | 161     |
| 2026 | 86.1%     | 28     | 29       | 161     |
| 2027 | 86.3%     | 27     | 30       | 161     |

**27 countries** remain in Severe risk by 2027 — almost entirely in Sub-Saharan Africa.
""")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _suptitle(fig, "View 9 — 2024–2027 Energy Poverty Forecast")

    years = [2024, 2025, 2026, 2027]; x = np.arange(len(years)); width = 0.25
    for i, (cat, col) in enumerate(zip(RISK_ORDER, RISK_COLORS)):
        vals = [forecast_df[forecast_df["year"] == yr]["pred_risk_category"]
                .value_counts().get(cat, 0) for yr in years]
        axes[0].bar(x + i * width, vals, width, label=cat, color=col, edgecolor=CHART_BG, alpha=0.9)
    axes[0].set_xticks(x + width); axes[0].set_xticklabels(years)
    axes[0].set_xlabel("Forecast Year"); axes[0].set_ylabel("Number of Countries")
    axes[0].set_title("Predicted Risk Category\n2024–2027")
    axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.4)

    hist_avg = df.groupby("year")["electricity_access"].mean()
    fore_avg = forecast_df.groupby("year")["pred_electricity_access"].mean()
    axes[1].plot(hist_avg.index, hist_avg.values,
                 color="#58A6FF", linewidth=2, label="Historical (actual)")
    axes[1].plot(fore_avg.index, fore_avg.values,
                 color="#F78166", linewidth=2, linestyle="--", marker="o", label="Forecast (predicted)")
    axes[1].axvline(2023, color=CHART_TEXT, linestyle="--", linewidth=1.5, label="Forecast start (2023)")
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Average Electricity Access (%)")
    axes[1].set_title("Global Average: Historical + Forecast")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.4)

    _fix_fig(fig)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("🚨 Countries Projected SEVERE by 2027")
    severe_2027 = (forecast_df[
        (forecast_df["year"] == 2027) & (forecast_df["pred_risk_category"] == "Severe")
    ][["country", "pred_electricity_access"]].sort_values("pred_electricity_access")
      .reset_index(drop=True))
    severe_2027.columns = ["Country", "Predicted Access (%)"]
    severe_2027["Predicted Access (%)"] = severe_2027["Predicted Access (%)"].round(1)
    st.dataframe(severe_2027, use_container_width=True)


# ─── Widget A ─────────────────────────────────────────────────────────────────
def page_widget_a(df, test_df, forecast_df):
    st.title("🔍 Widget A — Country Explorer")
    st.caption(
        "Blue = actual history | Coral = model test predictions (2021–2023) | Purple = forecast (2024–2027)"
    )

    country_list = sorted(df["country"].unique().tolist())
    c = st.selectbox("Select a country", country_list,
                     index=country_list.index("India") if "India" in country_list else 0)

    sub      = df[df["country"] == c].sort_values("year")
    test_sub = test_df[test_df["country"] == c].sort_values("year") if test_df is not None else pd.DataFrame()
    fore_sub = forecast_df[forecast_df["country"] == c].sort_values("year") if forecast_df is not None else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    _suptitle(fig, f"{c} — Electricity Access & GDP per Capita", fontsize=12)

    axes[0].plot(sub["year"], sub["electricity_access"], "o-", color="#58A6FF",
                 label="Actual", linewidth=2)
    if len(test_sub) > 0:
        axes[0].plot(test_sub["year"], test_sub["pred_electricity_access"], "s--",
                     color="#F78166", label="Predicted (test 2021–2023)", linewidth=2)
    if len(fore_sub) > 0:
        axes[0].plot(fore_sub["year"], fore_sub["pred_electricity_access"], "^--",
                     color="#BC8CFF", label="Forecast (2024–2027)", linewidth=2)
    axes[0].axhline(90, color="#3FB950", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].axhline(50, color="#F85149", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Electricity Access (%)")
    axes[0].set_title("Historical + Predictions + Forecast")
    axes[0].legend(fontsize=8); axes[0].set_ylim(0, 105); axes[0].grid(True, alpha=0.4)

    axes[1].plot(sub["year"], sub["gdp_per_capita"] / 1000, "o-", color="#3FB950", linewidth=2)
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("GDP per Capita (USD thousands)")
    axes[1].set_title("GDP per Capita Trend"); axes[1].grid(True, alpha=0.4)

    _fix_fig(fig)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    if len(fore_sub) > 0:
        st.subheader(f"Forecast Summary for {c}")
        disp = fore_sub[["year", "pred_electricity_access", "pred_risk_category"]].copy()
        disp.columns = ["Year", "Predicted Access (%)", "Risk Category"]
        disp["Predicted Access (%)"] = disp["Predicted Access (%)"].round(1)
        st.dataframe(disp.reset_index(drop=True), use_container_width=True)


# ─── Widget B ─────────────────────────────────────────────────────────────────
def page_widget_b(df):
    st.title("🌐 Widget B — Global Year Snapshot")
    st.caption(
        "X axis = Log GDP per capita · Y axis = Electricity access % · "
        "Colour = Renewable energy % — the two strongest predictors in one chart."
    )

    yr  = st.slider("Select Year", int(df["year"].min()), int(df["year"].max()), 2015)
    sub = df[df["year"] == yr].dropna(
        subset=["gdp_per_capita", "electricity_access", "renewable_energy_percent"])

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(
        np.log10(sub["gdp_per_capita"] + 1),
        sub["electricity_access"],
        c=sub["renewable_energy_percent"],
        cmap="RdYlGn", alpha=0.8, s=55, vmin=0, vmax=100,
        edgecolors=CHART_BG, linewidths=0.3,
    )
    cbar = plt.colorbar(sc, ax=ax, label="Renewable Energy %")
    cbar.ax.yaxis.label.set_color(CHART_TEXT)
    cbar.ax.tick_params(colors=CHART_TEXT, labelcolor=CHART_TEXT)
    ax.axhline(90, color="#3FB950", linestyle="--", linewidth=1.5, alpha=0.7, label="90% Minimal")
    ax.axhline(50, color="#F85149", linestyle="--", linewidth=1.5, alpha=0.7, label="50% Severe")
    ax.set_xlabel("Log10 GDP per Capita (USD)"); ax.set_ylabel("Electricity Access (%)")
    ax.set_title(
        f"GDP vs Electricity Access ({yr})\n"
        f"Colour = Renewable Energy % | {len(sub)} countries shown"
    )
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    _fix_fig(fig)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.caption(
        "Green = high renewable energy. Red = low. The renewable energy paradox: "
        "high renewable % (traditional biomass) often coincides with low electricity access."
    )


# ─── Widget C ─────────────────────────────────────────────────────────────────
def page_widget_c(df):
    st.title("↔️ Widget C — Risk Category Transition Explorer")
    st.caption("Compare any two years. Diagonal = stayed the same. Off-diagonal = changed category.")

    col1, col2 = st.columns(2)
    with col1:
        start_yr = st.slider("Start year", 1990, 2022, 2000)
    with col2:
        end_yr = st.slider("End year", start_yr + 1, 2023, 2023)

    start  = df[df["year"] == start_yr][["country", "risk_category"]].rename(
        columns={"risk_category": "start_category"})
    end    = df[df["year"] == end_yr][["country", "risk_category"]].rename(
        columns={"risk_category": "end_category"})
    merged = start.merge(end, on="country").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _suptitle(fig, f"Risk Category: {start_yr} → {end_yr}", fontsize=12)

    transition = (pd.crosstab(merged["start_category"], merged["end_category"])
                  .reindex(index=RISK_ORDER, columns=RISK_ORDER, fill_value=0))
    sns.heatmap(transition, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=RISK_ORDER, yticklabels=RISK_ORDER,
                linewidths=0.5, linecolor=CHART_BG)
    axes[0].set_xlabel(f"Category in {end_yr}")
    axes[0].set_ylabel(f"Category in {start_yr}")
    axes[0].set_title("Transition Matrix\n(rows=start, cols=end)")

    improved  = merged[(merged["start_category"] == "Severe") &
                       (merged["end_category"] != "Severe")]["country"].tolist()
    no_change = merged[(merged["start_category"] == "Severe") &
                       (merged["end_category"] == "Severe")]["country"].tolist()

    axes[1].axis("off")
    axes[1].set_title(f"Severe in {start_yr} — What happened by {end_yr}?", fontweight="bold")
    text  = f"Improved out of Severe ({len(improved)}):\n"
    text += "\n".join(f"  • {c}" for c in sorted(improved)[:15])
    if len(improved) > 15:
        text += f"\n  ... and {len(improved)-15} more"
    text += f"\n\nStill Severe ({len(no_change)}):\n"
    text += "\n".join(f"  • {c}" for c in sorted(no_change)[:15])
    axes[1].text(0.02, 0.95, text, transform=axes[1].transAxes, fontsize=8, va="top",
                 family="monospace", color=CHART_TEXT,
                 bbox=dict(boxstyle="round", facecolor="#1C2333", alpha=0.9, edgecolor="#1F6FEB55"))

    _fix_fig(fig)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ─── About ────────────────────────────────────────────────────────────────────
def page_about():
    st.title("ℹ️ About This Project")
    st.markdown("""
## Energy Poverty Prediction — CAP5771 - Introduction to Data Science

**Author:** Rishitha Pydipati, Nikhitha Pydipati | **University:** University of Florida | **Course:** CAP5771 — Introduction to Data Science

---

### Project Summary
Predicts **energy poverty risk** (Severe / Moderate / Minimal) for countries through 2027
using supervised machine learning trained on World Bank Development Indicators (1990–2023).

### Data
- **Source:** World Bank World Development Indicators — [data.worldbank.org](https://data.worldbank.org)
- **Scope:** 8 indicators · 218 countries · 1990–2023 · Fully public

### Models

| Model | Task | Val Score |
|---|---|---|
| RF Baseline | Classification (risk category) | 92.97% accuracy |
| RF Improved | Regression (electricity access %) | R² = 0.9178 |

### Full Data Science Cycle

| Stage | Notebook / File |
|---|---|
| Data Acquisition | `energy_poverty_prediction.ipynb` |
| Data Wrangling | `data_wrangling.ipynb` → `analysis_ready.csv` |
| Modeling | `data_modeling.ipynb` → `test_predictions.csv`, `feature_importance.csv`, `forecast_2024_2027.csv` |
| Visualization | `data_visualization_static.ipynb` (9 views + 3 widgets) |
| Deployment | `deployment/app.py` — this Streamlit app |

### Key Findings
1. `renewable_energy_percent` is #1 predictor — but **negatively** correlated (traditional biomass paradox)
2. `gdp_per_capita` and `government_effectiveness` are strong positive drivers
3. Global access grew from ~72% (1990) to ~90% (2023) but progress is slowing
4. **27 countries** projected Severe by 2027 — nearly all Sub-Saharan Africa
5. Moderate class (transition zone) is hardest to classify: F1 ≈ 0.59

### Repository
[github.com/npydipati/CAP5771_npydipati](https://github.com/npydipati/CAP5771_npydipati)
""")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    page = sidebar()

    df       = load_main()
    test_df  = load_test()
    feat_imp = load_fi()
    fore_df  = load_forecast()

    if test_df is None or feat_imp is None or fore_df is None:
        test_df, feat_imp, fore_df = train_fallback(df)

    if page == "🌍 Overview":
        page_overview(df)
    elif page == "📊 Views 1–3: Trends & Progress":
        page_views_1_3(df)
    elif page == "🔬 Views 4–6: Features & Models":
        page_views_4_6(df, feat_imp)
    elif page == "✅ Views 7–8: Evaluation":
        page_views_7_8(df, test_df)
    elif page == "🔮 View 9: 2024–2027 Forecast":
        page_view_9(df, fore_df)
    elif page == "🔍 Widget A: Country Explorer":
        page_widget_a(df, test_df, fore_df)
    elif page == "🌐 Widget B: Year Snapshot":
        page_widget_b(df)
    elif page == "↔️ Widget C: Risk Transitions":
        page_widget_c(df)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()