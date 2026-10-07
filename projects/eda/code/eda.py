"""EDA do CarDekho (Car details v3). Gera as figuras em ../figures e imprime os números do relatório.

Rodar a partir da raiz do repositório:
    python docs/projects/eda/code/eda.py
"""
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import (CAT, NUM, SEED, TARGET, build_pipeline, clean,  # noqa: E402
                           load_raw, split)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

# azul para os dados, laranja só para destaque (categoria rara, mediana) e rampa azul para magnitude
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#256abf", "#f0efec", "#c23b3a"])
sns.set_theme(style="whitegrid", rc={
    "figure.dpi": 110, "savefig.dpi": 130, "axes.edgecolor": "#c8c7c2",
    "grid.color": "#e6e5e1", "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


def h(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# =============================================================================
# 1. Inspeção inicial
# =============================================================================
raw = load_raw()
h("1A. Dicionário (bruto)")
print("shape bruto:", raw.shape)
print(raw.dtypes)
print(raw.head(3).T)
print("modelos distintos (name):", raw["name"].nunique())
print("marcas distintas:", raw["name"].str.split().str[0].nunique())
print("unidades de mileage por fuel:")
print(pd.crosstab(raw["fuel"], raw["mileage"].str.split().str[1]))

h("1B. Qualidade (bruto)")
na = raw.isna().sum()
print(pd.DataFrame({"n": na, "%": (100 * na / len(raw)).round(2)}))
spec = ["mileage", "engine", "max_power", "torque", "seats"]
print("linhas com alguma spec ausente:", raw[spec].isna().any(axis=1).sum())
print("linhas com todas as specs ausentes:", raw[spec].isna().all(axis=1).sum())
print("max_power == '0' ou ' bhp':", raw["max_power"].isin(["0", " bhp"]).sum())
print("duplicatas exatas:", raw.duplicated().sum(),
      f"({100 * raw.duplicated().mean():.2f}%)")
print("km_driven: top 5", raw["km_driven"].nlargest(5).tolist(),
      "| < 1000 km:", (raw["km_driven"] < 1000).sum())
print("owner == Test Drive Car:", (raw["owner"] == "Test Drive Car").sum())
print("ano min/max:", raw["year"].min(), raw["year"].max())
mis = raw[spec].isna().all(axis=1)
print("ano mediano com/sem specs:", raw.loc[mis, "year"].median(), raw.loc[~mis, "year"].median())
print("preço mediano com/sem specs:", raw.loc[mis, TARGET].median(), raw.loc[~mis, TARGET].median())
print("seller_type nas linhas sem specs:\n", raw.loc[mis, "seller_type"].value_counts(normalize=True).round(3))

df = clean(raw)
print("\nshape após limpeza (dedup + parse):", df.shape)
print(df.isna().sum())
print("mileage == 0 no bruto:", (raw["mileage"].str.startswith("0.0 ", na=False) | raw["mileage"].str.startswith("0 ", na=False)).sum())
print("valores impossíveis após limpeza, mileage == 0:", (df["mileage"] == 0).sum(),
      "| seats < 2:", (df["seats"] < 2).sum())
print("torque_nm min/max:", df["torque_nm"].min(), df["torque_nm"].max())

h("1B. Vazamento: correlação de Spearman de cada numérica com o preço")
print(df[NUM + [TARGET]].corr(method="spearman")[TARGET].round(3).sort_values())

h("1C. Alvo")
y_all = df[TARGET]
print(y_all.describe().round(0))
print("assimetria:", round(y_all.skew(), 3), "| curtose:", round(y_all.kurt(), 3))
logy = np.log(y_all)
print("assimetria log:", round(logy.skew(), 3), "| curtose log:", round(logy.kurt(), 3))
print("% acima de 1.5*IQR (bruto):",
      round(100 * (y_all > y_all.quantile(.75) + 1.5 * (y_all.quantile(.75) - y_all.quantile(.25))).mean(), 2))
print("baseline (prever a mediana), MAE ₹:", round((y_all - y_all.median()).abs().mean(), 0))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(y_all / 1e5, bins=60, color=BLUE, edgecolor="white", linewidth=0.5)
ax[0].axvline(y_all.median() / 1e5, color=ORANGE, lw=2, label=f"mediana {y_all.median() / 1e5:.2f}")
ax[0].axvline(y_all.mean() / 1e5, color="#0b0b0b", lw=1.5, ls="--", label=f"média {y_all.mean() / 1e5:.2f}")
ax[0].set(title=f"Preço bruto (assimetria {y_all.skew():.2f})",
          xlabel="selling_price (lakh ₹ = 100 mil rúpias)", ylabel="carros")
ax[0].legend(frameon=False)
ax[1].hist(logy, bins=60, color=BLUE, edgecolor="white", linewidth=0.5)
ax[1].set(title=f"log(preço) (assimetria {logy.skew():.2f})", xlabel="log(selling_price)", ylabel="carros")
fig.suptitle("Figura 1. Distribuição do alvo antes e depois do log", fontweight="bold")
save(fig, "fig01-alvo.png")

h("1D. Split")
X_tr, X_te, y_tr, y_te = split(df)
print("treino:", X_tr.shape, "teste:", X_te.shape)
print("log-preço treino média/mediana/dp:", round(y_tr.mean(), 3), round(y_tr.median(), 3), round(y_tr.std(), 3))
print("log-preço teste  média/mediana/dp:", round(y_te.mean(), 3), round(y_te.median(), 3), round(y_te.std(), 3))
print("preço mediano treino/teste:", round(np.exp(y_tr.median())), round(np.exp(y_te.median())))

# =============================================================================
# 2. Univariada, sobre o treino (os dados que o modelo verá)
# =============================================================================
tr = X_tr.assign(price=np.exp(y_tr), log_price=y_tr)

h("2A. Numéricas (treino)")
desc = tr[NUM].describe().T
desc["skew"] = tr[NUM].skew()
desc["%NaN"] = 100 * tr[NUM].isna().mean()
print(desc.round(2))
q1, q3 = tr[NUM].quantile(.25), tr[NUM].quantile(.75)
iqr_out = ((tr[NUM] < q1 - 1.5 * (q3 - q1)) | (tr[NUM] > q3 + 1.5 * (q3 - q1))).sum()
print("outliers por IQR 1.5:\n", iqr_out)
print("seats value_counts:\n", tr["seats"].value_counts().sort_index())
print("mileage por fuel (mediana):\n", tr.groupby("fuel")["mileage"].median())

units = {"year": "ano", "mileage": "kmpl (km/kg p/ CNG/LPG)", "seats": "assentos",
         "km_driven": "km", "engine": "cc", "max_power": "bhp", "torque_nm": "Nm"}
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for a, c in zip(axes.flat, NUM):
    v = tr[c].dropna()
    if c == "km_driven":  # com cauda até 2,4 milhões de km, só dá para ler em escala log
        a.hist(v, bins=np.logspace(0, np.log10(v.max()), 50), color=BLUE, edgecolor="white", linewidth=0.4)
        a.set_xscale("log")
    else:
        a.hist(v, bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
    a.axvline(v.median(), color=ORANGE, lw=1.6)
    a.set(title=f"{c}  (assim. {v.skew():.2f})", ylabel="carros",
          xlabel=units[c] + (" (escala log)" if c == "km_driven" else ""))
axes.flat[-1].axis("off")
axes.flat[-1].text(0.05, 0.5, "linha laranja = mediana\nn = treino (5525 carros)",
                   fontsize=10, color="#52514e", va="center")
fig.suptitle("Figura 2. Distribuição das numéricas no treino", fontweight="bold")
save(fig, "fig02-numericas-hist.png")

fig, axes = plt.subplots(1, 7, figsize=(14, 3.6))
for a, c in zip(axes, NUM):
    a.boxplot(tr[c].dropna(), widths=0.55, patch_artist=True,
              boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE),
              medianprops=dict(color=ORANGE, lw=2),
              flierprops=dict(marker="o", ms=3, mfc=BLUE, mec="none", alpha=.35))
    a.set(title=c, xticks=[], ylabel=units[c])
    a.text(0.5, -0.06, f"{iqr_out[c]} fora de 1,5·IQR", transform=a.transAxes,
           ha="center", va="top", fontsize=8.5, color="#52514e")
fig.suptitle("Figura 3. Boxplots das numéricas no treino (outliers por 1,5·IQR)", fontweight="bold")
save(fig, "fig03-numericas-box.png")

h("2B. Categóricas (treino)")
for c in CAT:
    vc = tr[c].value_counts()
    print(f"\n{c}: cardinalidade {vc.size}")
    print(pd.DataFrame({"n": vc, "%": (100 * vc / len(tr)).round(2)}).head(40))
rare_brands = tr["brand"].value_counts()
print("marcas com < 20 linhas no treino:", (rare_brands < 20).sum(),
      "| linhas cobertas:", rare_brands[rare_brands < 20].sum())
unseen = set(X_te["brand"]) - set(X_tr["brand"])
print("marcas no teste ausentes do treino:", unseen)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for a, c in zip(axes, ["fuel", "seller_type", "transmission", "owner"]):
    vc = tr[c].value_counts()
    pct = 100 * vc / len(tr)
    colors = [ORANGE if p < 1 else BLUE for p in pct]
    a.barh(vc.index[::-1], vc.values[::-1], color=colors[::-1], height=0.65)
    for i, (n, p) in enumerate(zip(vc.values[::-1], pct.values[::-1])):
        a.text(n, i, f" {p:.1f}%", va="center", fontsize=8.5, color="#52514e")
    a.set(title=f"{c} (card. {vc.size})", xlabel="carros (escala log)", xscale="log",
          xlim=(1, vc.max() * 4))
    a.grid(axis="y", visible=False)
fig.suptitle("Figura 4. Frequência das categóricas de baixa cardinalidade (laranja = categoria rara, < 1%)",
             fontweight="bold")
save(fig, "fig04-categoricas.png")

vc = tr["brand"].value_counts()
fig, a = plt.subplots(figsize=(13, 4))
colors = [ORANGE if n < 20 else BLUE for n in vc.values]
a.bar(vc.index, vc.values, color=colors, width=0.7)
a.axhline(20, color="#52514e", lw=1, ls="--")
a.text(len(vc) - 0.5, 22, "min_frequency = 20", ha="right", fontsize=9, color="#52514e")
a.set(title=f"brand: cardinalidade {vc.size}; {(vc < 20).sum()} marcas raras (laranja) somam {vc[vc < 20].sum()} carros",
      ylabel="carros (escala log)", yscale="log")
a.tick_params(axis="x", rotation=60)
a.grid(axis="x", visible=False)
fig.suptitle("Figura 5. Frequência das marcas no treino (alta cardinalidade)", fontweight="bold")
save(fig, "fig05-marcas.png")

# =============================================================================
# 3. Bivariada e multivariada (treino)
# =============================================================================
h("3A. Numérica x numérica: Spearman e Pearson (treino)")
cols = NUM + ["price"]
sp = tr[cols].corr(method="spearman")
pe = tr[cols].corr(method="pearson")
print("Spearman:\n", sp.round(3))
print("Pearson:\n", pe.round(3))
pairs = (sp.where(np.triu(np.ones(sp.shape, bool), 1)).stack()
         .rename("rho").reset_index().assign(abs=lambda d: d.rho.abs())
         .sort_values("abs", ascending=False))
print("pares mais correlacionados (Spearman):\n", pairs.head(10).round(3))

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for a, m, t in [(ax[0], sp, "Spearman (ρ)"), (ax[1], pe, "Pearson (r)")]:
    sns.heatmap(m, annot=True, fmt=".2f", cmap=DIV, vmin=-1, vmax=1, ax=a, square=True,
                cbar_kws={"shrink": .75}, annot_kws={"size": 8}, linewidths=.5, linecolor="white")
    a.set_title(t)
fig.suptitle("Figura 6. Correlação entre numéricas e o preço (treino)", fontweight="bold")
save(fig, "fig06-correlacao.png")

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for a, (x, yv, logx) in zip(ax, [("engine", "max_power", False), ("max_power", "torque_nm", False),
                                 ("year", "km_driven", False)]):
    sc = a.scatter(tr[x], tr[yv], c=tr["log_price"], cmap=SEQ, s=8, alpha=.6, edgecolors="none")
    a.set(xlabel=f"{x} ({units[x]})", ylabel=f"{yv} ({units[yv]})",
          title=f"{x} × {yv}: ρ = {sp.loc[x, yv]:.2f}")
    if yv == "km_driven":
        a.set_yscale("log")
cb = fig.colorbar(sc, ax=ax, shrink=.85, pad=.01)
cb.set_label("log(preço)")
fig.suptitle("Figura 7. Pares mais correlacionados, coloridos pelo log do preço", fontweight="bold", x=.45)
fig.savefig(FIG_DIR / "fig07-dispersao.png", bbox_inches="tight")
plt.close(fig)
print("[fig] fig07-dispersao.png")

h("3B. Categórica x alvo (treino)")
for c in CAT:
    g = tr.groupby(c)["price"].agg(["count", "median", "mean"]).sort_values("median")
    print(f"\n{c}:\n", g.round(0))

fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
for a, c in zip(axes, ["fuel", "transmission", "seller_type", "owner"]):
    order = tr.groupby(c)["log_price"].median().sort_values().index
    sns.boxplot(data=tr, x=c, y="log_price", order=order, ax=a, color="#cde2fb",
                linecolor=BLUE, fliersize=2, width=.6,
                medianprops=dict(color=ORANGE, lw=2))
    med = tr.groupby(c)["price"].median()
    a.set_xticks(range(len(order)))
    short = {"Fourth & Above Owner": "4º+", "Third Owner": "3º", "Second Owner": "2º",
             "First Owner": "1º", "Test Drive Car": "Test Drive", "Trustmark Dealer": "Trustmark"}
    a.set_xticklabels([f"{short.get(o, o)}\n{med[o] / 1e5:.1f} lakh" for o in order], fontsize=8, rotation=0)
    a.set(title=c, xlabel="", ylabel="log(preço)" if c == "fuel" else "")
fig.suptitle("Figura 8. log(preço) por categoria (rótulo = preço mediano)", fontweight="bold")
save(fig, "fig08-categoria-alvo.png")

bm = tr.groupby("brand")["price"].agg(["count", "median"])
bm = bm[bm["count"] >= 20].sort_values("median")
fig, a = plt.subplots(figsize=(12, 4.2))
order = bm.index
sns.boxplot(data=tr[tr["brand"].isin(order)], x="brand", y="log_price", order=order, ax=a,
            color="#cde2fb", linecolor=BLUE, fliersize=2, width=.6,
            medianprops=dict(color=ORANGE, lw=2))
a.tick_params(axis="x", rotation=45)
a.set(xlabel="", ylabel="log(preço)",
      title=f"Marcas com ≥ 20 carros no treino ({len(order)}), ordenadas pela mediana")
fig.suptitle("Figura 9. log(preço) por marca", fontweight="bold")
save(fig, "fig09-marca-alvo.png")

h("3C. Numérica x categórica (treino)")
for num, cat in [("max_power", "transmission"), ("km_driven", "owner"), ("year", "seller_type")]:
    g = tr.groupby(cat)[num].describe()[["count", "25%", "50%", "75%"]]
    g["IQR"] = g["75%"] - g["25%"]
    print(f"\n{num} por {cat}:\n", g.round(1))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for a, (num, cat, logy_) in zip(axes, [("max_power", "transmission", False),
                                        ("km_driven", "owner", True),
                                        ("year", "seller_type", False)]):
    order = tr.groupby(cat)[num].median().sort_values().index
    sns.boxplot(data=tr, x=cat, y=num, order=order, ax=a, color="#cde2fb", linecolor=BLUE,
                fliersize=2, width=.6, medianprops=dict(color=ORANGE, lw=2))
    if logy_:
        a.set_yscale("log")
    a.tick_params(axis="x", rotation=15, labelsize=8)
    a.set(xlabel="", ylabel=f"{num} ({units[num]})", title=f"{num} por {cat}")
fig.suptitle("Figura 10. Numéricas agrupadas por categóricas (treino)", fontweight="bold")
save(fig, "fig10-num-cat.png")

# =============================================================================
# 4. Pré-processamento
# =============================================================================
h("4A. Estratégias")
print("ausentes no treino por coluna:\n", X_tr.isna().sum())
spec_c = ["mileage", "engine", "max_power", "torque_nm", "seats"]
print("treino: linhas com specs todas ausentes:", X_tr[spec_c].isna().all(axis=1).sum(),
      "| alguma:", X_tr[spec_c].isna().any(axis=1).sum())
print("escala bruta (min/max no treino):\n", X_tr[NUM].agg(["min", "max", "std"]).T.round(2))

prep = build_pipeline()
t0 = time.time()
prep.fit(X_tr)
Z_tr, Z_te = prep.transform(X_tr), prep.transform(X_te)
print(f"fit em {time.time() - t0:.2f}s")
for block, cols_ in [("num", ["year", "mileage", "seats"]),
                     ("log", ["km_driven", "engine", "max_power", "torque_nm"])]:
    w = prep.named_transformers_[block].named_steps["win"]
    imp = prep.named_transformers_[block].named_steps["imp"]
    for i, c in enumerate(cols_):
        v = X_tr[c]
        n = ((v < w.lo_[i]) | (v > w.hi_[i])).sum()
        print(f"winsor {c:10s} [{w.lo_[i]:.1f}, {w.hi_[i]:.1f}]  linhas cortadas treino: {n}"
              f"  | mediana imputada: {imp.statistics_[i]:.1f}")
lo = {**dict(zip(["year", "mileage", "seats"], prep.named_transformers_["num"].named_steps["win"].lo_)),
      **dict(zip(["km_driven", "engine", "max_power", "torque_nm"], prep.named_transformers_["log"].named_steps["win"].lo_))}
hi = {**dict(zip(["year", "mileage", "seats"], prep.named_transformers_["num"].named_steps["win"].hi_)),
      **dict(zip(["km_driven", "engine", "max_power", "torque_nm"], prep.named_transformers_["log"].named_steps["win"].hi_))}
affected = pd.Series(False, index=X_tr.index)
for c in NUM:
    affected |= (X_tr[c] < lo[c]) | (X_tr[c] > hi[c])
print("linhas do treino com ao menos 1 valor cortado:", affected.sum(),
      f"({100 * affected.mean():.2f}%)")
aff_te = pd.Series(False, index=X_te.index)
for c in NUM:
    aff_te |= (X_te[c] < lo[c]) | (X_te[c] > hi[c])
print("linhas do teste cortadas com limites do treino:", aff_te.sum())
print("assimetria após winsor+log (treino):")
names = list(prep.get_feature_names_out())
Zdf = pd.DataFrame(Z_tr, columns=names)
print(Zdf.iloc[:, :7].skew().round(3))
cat_enc = prep.named_transformers_["cat"].named_steps["cod"]
print("categorias infrequentes:", {c: list(v) for c, v in zip(CAT, cat_enc.infrequent_categories_) if v is not None})

h("4C. Pipeline")
print("Z_tr", Z_tr.shape, "Z_te", Z_te.shape)
print("NaN treino/teste:", int(np.isnan(Z_tr).sum()), int(np.isnan(Z_te).sum()))
print("média numéricas treino (max |.|):", np.abs(Z_tr[:, :7].mean(0)).max().round(6))
print("média numéricas teste:", Z_te[:, :7].mean(0).round(3))
print("dp numéricas treino:", Z_tr[:, :7].std(0).round(3))
print("n features:", len(names))
print(names)

# =============================================================================
# 4B. Redução de dimensionalidade
# =============================================================================
h("4B. PCA")
pca = PCA(random_state=SEED).fit(Z_tr)
evr = pca.explained_variance_ratio_
print("variância explicada:", evr.round(4)[:12])
print("acumulada:", np.cumsum(evr).round(4)[:12])
print("componentes p/ 80%/90%/95%:", [int(np.searchsorted(np.cumsum(evr), t) + 1) for t in (.8, .9, .95)])
print("autovalores ~0:", int((pca.explained_variance_ < 1e-10).sum()))
load = pd.DataFrame(pca.components_[:2].T, index=names, columns=["PC1", "PC2"])
for pc in ["PC1", "PC2"]:
    print(f"\n{pc}, maiores |loadings|:\n", load[pc].reindex(load[pc].abs().sort_values(ascending=False).index).head(8).round(3))
A_tr = pca.transform(Z_tr)[:, :2]
print("Spearman PC1 x log_price:", round(pd.Series(A_tr[:, 0]).corr(pd.Series(y_tr.values), method="spearman"), 3))
print("Spearman PC2 x log_price:", round(pd.Series(A_tr[:, 1]).corr(pd.Series(y_tr.values), method="spearman"), 3))

# amostra para os métodos não lineares: o t-SNE é O(n log n) e não tem .transform
rng = np.random.default_rng(SEED)
idx = rng.choice(len(Z_tr), size=3000, replace=False)
Zs, ys = Z_tr[idx], y_tr.values[idx]


def avalia(E, nome):
    tw5 = trustworthiness(Zs, E, n_neighbors=5)
    tw30 = trustworthiness(Zs, E, n_neighbors=30)
    r2 = cross_val_score(KNeighborsRegressor(15), E, ys, cv=5, scoring="r2").mean()
    print(f"{nome:26s} trust k5 {tw5:.3f}  k30 {tw30:.3f}  R² kNN(log preço) {r2:.3f}")
    return dict(metodo=nome, tw5=tw5, tw30=tw30, r2=r2)


res = [avalia(A_tr[idx], "PCA (2 comp.)")]
print("referência: R² kNN no espaço completo (40 dim):",
      round(cross_val_score(KNeighborsRegressor(15), Zs, ys, cv=5, scoring="r2").mean(), 3))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
k = np.arange(1, len(evr) + 1)
ax[0].bar(k, evr, color=BLUE, width=.7)
ax[0].plot(k, np.cumsum(evr), color=ORANGE, marker="o", ms=3, lw=1.6, label="acumulada")
ax[0].axhline(.9, color="#52514e", lw=1, ls="--")
ax[0].text(len(k), .92, "90%", ha="right", fontsize=8.5, color="#52514e")
ax[0].set(title=f"Variância explicada (PC1+PC2 = {100 * evr[:2].sum():.1f}%)",
          xlabel="componente", ylabel="fração da variância")
ax[0].legend(frameon=False, loc="center right")
sc = ax[1].scatter(A_tr[:, 0], A_tr[:, 1], c=y_tr.values, cmap=SEQ, s=6, alpha=.6, edgecolors="none")
ax[1].set(title="Treino projetado em PC1 × PC2", xlabel=f"PC1 ({100 * evr[0]:.1f}%)",
          ylabel=f"PC2 ({100 * evr[1]:.1f}%)")
fig.colorbar(sc, ax=ax[1], label="log(preço)", shrink=.85)
top = load.loc[load.abs().max(axis=1).sort_values(ascending=False).index[:10]]
yy = np.arange(len(top))
ax[2].barh(yy - .2, top["PC1"], height=.38, color=BLUE, label="PC1")
ax[2].barh(yy + .2, top["PC2"], height=.38, color=ORANGE, label="PC2")
ax[2].set_yticks(yy)
ax[2].set_yticklabels([n.split("__")[1] for n in top.index], fontsize=8.5)
ax[2].invert_yaxis()
ax[2].axvline(0, color="#52514e", lw=.8)
ax[2].set(title="Maiores loadings de PC1 e PC2", xlabel="peso")
ax[2].legend(frameon=False)
fig.suptitle("Figura 11. PCA ajustada no treino", fontweight="bold")
save(fig, "fig11-pca.png")

h("4B. t-SNE e UMAP (amostra de 3000 linhas do treino)")
import umap  # noqa: E402

maps = {}
for perp in [5, 30, 50]:
    t0 = time.time()
    E = TSNE(2, perplexity=perp, random_state=SEED, init="pca").fit_transform(Zs)
    maps[f"t-SNE perp={perp}"] = E
    r = avalia(E, f"t-SNE perplexity={perp}")
    r["tempo"] = time.time() - t0
    res.append(r)
for nn in [5, 15, 50]:
    t0 = time.time()
    E = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=SEED).fit_transform(Zs)
    maps[f"UMAP nn={nn}"] = E
    r = avalia(E, f"UMAP n_neighbors={nn}")
    r["tempo"] = time.time() - t0
    res.append(r)
print(pd.DataFrame(res).round(3).to_string(index=False))

# controle: o mesmo t-SNE sobre colunas embaralhadas uma a uma, o que desfaz a relação entre elas
Zr = np.column_stack([rng.permutation(Zs[:, j]) for j in range(Zs.shape[1])])
Er = TSNE(2, perplexity=30, random_state=SEED, init="pca").fit_transform(Zr)
print("controle t-SNE perp=30 em dados embaralhados, R² kNN:",
      round(cross_val_score(KNeighborsRegressor(15), Er, ys, cv=5, scoring="r2").mean(), 3))

# o que forma as ilhas do UMAP: pureza dos clusters k-means em cada categórica, comparada à taxa base
E = maps["UMAP nn=15"]
from sklearn.cluster import KMeans  # noqa: E402
lab = KMeans(8, n_init=10, random_state=SEED).fit_predict(E)
Xs = X_tr.iloc[idx].reset_index(drop=True)
for c in CAT:
    ct = pd.crosstab(lab, Xs[c], normalize="index")
    base = Xs[c].value_counts(normalize=True).iloc[0]
    print(f"pureza média dos 8 clusters UMAP em {c}: {ct.max(axis=1).mean():.3f}  (taxa base {base:.3f})")

for method, keys in [("t-SNE", ["t-SNE perp=5", "t-SNE perp=30", "t-SNE perp=50"]),
                     ("UMAP", ["UMAP nn=5", "UMAP nn=15", "UMAP nn=50"])]:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, key in zip(ax, keys):
        E = maps[key]
        r = next(x for x in res if x["metodo"].replace("perplexity", "perp").replace("n_neighbors", "nn") == key)
        sc = a.scatter(E[:, 0], E[:, 1], c=ys, cmap=SEQ, s=5, alpha=.7, edgecolors="none")
        a.set(title=f"{key}\ntrust(k=5) {r['tw5']:.3f} · R² kNN {r['r2']:.3f}",
              xticks=[], yticks=[], xlabel="dim 1", ylabel="dim 2")
    fig.colorbar(sc, ax=ax, label="log(preço)", shrink=.85, pad=.01)
    n = 12 if method == "t-SNE" else 13
    fig.suptitle(f"Figura {n}. {method} com três valores do parâmetro de vizinhança "
                 "(3000 carros do treino)", fontweight="bold", x=.45, y=1.06)
    fig.savefig(FIG_DIR / f"fig{n}-{method.lower().replace('-', '')}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] fig{n}-{method.lower().replace('-', '')}.png")
