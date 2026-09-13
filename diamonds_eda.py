import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import pickle

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 110

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

charts = {}

df = pd.read_csv("diamonds.csv")

# ---- Note data quality quirks (a few physically impossible zero/extreme dimensions) ----
bad_rows = df[(df[['x','y','z']] == 0).any(axis=1) | (df['y'] > 20) | (df['z'] > 20)]
n_bad = len(bad_rows)
df_clean = df.drop(index=bad_rows.index)

# ============ STATISTICAL SUMMARY ============
summary_stats = df_clean.describe().round(2)

# ============ DISTRIBUTIONS ============
fig, axes = plt.subplots(2, 2, figsize=(11,8))
sns.histplot(df_clean["price"], bins=50, kde=True, ax=axes[0,0], color="#3d5a80")
axes[0,0].set_title("Price Distribution")
sns.histplot(df_clean["carat"], bins=50, kde=True, ax=axes[0,1], color="#e07a5f")
axes[0,1].set_title("Carat Distribution")
sns.histplot(df_clean["depth"], bins=50, kde=True, ax=axes[1,0], color="#81b29a")
axes[1,0].set_title("Depth % Distribution")
sns.histplot(df_clean["table"], bins=50, kde=True, ax=axes[1,1], color="#f2cc8f")
axes[1,1].set_title("Table % Distribution")
plt.tight_layout()
charts["distributions"] = fig_to_b64(fig)

# ============ CATEGORICAL BREAKDOWN ============
fig, axes = plt.subplots(1, 3, figsize=(14,4))
cut_order = ["Fair","Good","Very Good","Premium","Ideal"]
sns.countplot(data=df_clean, x="cut", order=cut_order, ax=axes[0], color="#3d5a80")
axes[0].set_title("Count by Cut")
color_order = sorted(df_clean['color'].unique())
sns.countplot(data=df_clean, x="color", order=color_order, ax=axes[1], color="#e07a5f")
axes[1].set_title("Count by Color Grade")
clarity_order = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
sns.countplot(data=df_clean, x="clarity", order=clarity_order, ax=axes[2], color="#81b29a")
axes[2].set_title("Count by Clarity Grade")
axes[2].tick_params(axis='x', rotation=45)
plt.tight_layout()
charts["categorical_counts"] = fig_to_b64(fig)

# ============ CORRELATION HEATMAP ============
num_cols = ["carat","depth","table","price","x","y","z"]
corr = df_clean[num_cols].corr()
fig, ax = plt.subplots(figsize=(6.5,5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap (Numeric Features)")
charts["correlation"] = fig_to_b64(fig)

# ============ PRICE vs CARAT (key relationship) ============
fig, ax = plt.subplots(figsize=(7,5))
sample = df_clean.sample(5000, random_state=42)  # sample for readable scatter
sns.scatterplot(data=sample, x="carat", y="price", hue="cut", hue_order=cut_order,
                 palette="viridis", alpha=0.5, ax=ax, s=20)
ax.set_title("Price vs Carat, colored by Cut")
charts["price_vs_carat"] = fig_to_b64(fig)

# ============ PRICE BY CATEGORICAL FACTORS ============
fig, axes = plt.subplots(1, 3, figsize=(15,4.5))
sns.boxplot(data=df_clean, x="cut", y="price", order=cut_order, ax=axes[0], color="#3d5a80")
axes[0].set_title("Price by Cut")
sns.boxplot(data=df_clean, x="color", y="price", order=color_order, ax=axes[1], color="#e07a5f")
axes[1].set_title("Price by Color")
sns.boxplot(data=df_clean, x="clarity", y="price", order=clarity_order, ax=axes[2], color="#81b29a")
axes[2].set_title("Price by Clarity")
axes[2].tick_params(axis='x', rotation=45)
plt.tight_layout()
charts["price_by_category"] = fig_to_b64(fig)

# ============ AVG PRICE PER CARAT BY CUT (surprising insight) ============
avg_price_cut = df_clean.groupby("cut", observed=True)["price"].mean().reindex(cut_order)
fig, ax = plt.subplots(figsize=(6,4))
avg_price_cut.plot(kind="bar", ax=ax, color="#f2cc8f")
ax.set_title("Average Price by Cut Quality")
ax.set_ylabel("Average Price ($)")
plt.xticks(rotation=0)
charts["avg_price_cut"] = fig_to_b64(fig)

# ============ PAIRWISE RELATIONSHIPS (x,y,z vs price) ============
fig, ax = plt.subplots(figsize=(6,4.5))
sns.scatterplot(data=sample, x="x", y="price", alpha=0.4, ax=ax, color="#3d5a80", s=15)
ax.set_title("Price vs Length (x dimension)")
charts["price_vs_x"] = fig_to_b64(fig)

# Key correlation values for text
corr_price_carat = corr.loc["price","carat"]
corr_price_x = corr.loc["price","x"]
corr_price_depth = corr.loc["price","depth"]
corr_price_table = corr.loc["price","table"]

with open("eda_results.pkl","wb") as f:
    pickle.dump({
        "charts": charts,
        "summary_stats": summary_stats,
        "n_bad": n_bad,
        "bad_rows": bad_rows,
        "corr_price_carat": corr_price_carat,
        "corr_price_x": corr_price_x,
        "corr_price_depth": corr_price_depth,
        "corr_price_table": corr_price_table,
        "avg_price_cut": avg_price_cut,
        "shape_before": df.shape,
        "shape_after": df_clean.shape,
    }, f)

print("Rows removed (data errors):", n_bad)
print("Shape before/after:", df.shape, df_clean.shape)
print("Correlation price-carat:", round(corr_price_carat,3))
print("Correlation price-depth:", round(corr_price_depth,3))
print("Correlation price-table:", round(corr_price_table,3))
print(avg_price_cut)
