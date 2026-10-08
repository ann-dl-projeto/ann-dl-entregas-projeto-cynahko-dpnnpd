"""EDA do CarDekho (Car details v3): gera as figuras em ../figures e imprime os números do relatório.

Rodar a partir da raiz do repositório:
    python docs/projects/eda/code/eda.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import CAT, NUM, SEED, TARGET, build_pipeline, clean, load_raw, split

# Configuração de exibição
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# Pasta onde as figuras são salvas
FIGURAS = Path(__file__).resolve().parents[1] / "figures"

# Paleta: azul para os dados, laranja para destaque
AZUL, AZUL_CLARO, LARANJA, CINZA = "#2a78d6", "#cde2fb", "#eb6834", "#52514e"
ESCALA_AZUL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
ESCALA_DIVERGENTE = LinearSegmentedColormap.from_list("div", ["#256abf", "#f0efec", "#c23b3a"])
sns.set_theme(style="whitegrid", rc={
    "figure.dpi": 110, "savefig.dpi": 130, "axes.edgecolor": "#c8c7c2",
    "grid.color": "#e6e5e1", "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})

# Unidades dos eixos e rótulos curtos das categorias nos gráficos
UNIDADES = {"year": "ano", "mileage": "kmpl (km/kg p/ CNG/LPG)", "seats": "assentos",
            "km_driven": "km", "engine": "cc", "max_power": "bhp", "torque_nm": "Nm"}
ROTULOS_CURTOS = {"Fourth & Above Owner": "4º+", "Third Owner": "3º", "Second Owner": "2º",
                  "First Owner": "1º", "Test Drive Car": "Test Drive", "Trustmark Dealer": "Trustmark"}


# Funções auxiliares
def titulo(texto):
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def salvar(fig, nome, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(FIGURAS / nome, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {nome}")


def tabela_ausentes(df):
    ausentes = df.isna().sum()
    tabela = pd.DataFrame({"ausentes": ausentes, "%": (100 * ausentes / len(df)).round(2)})
    return tabela[ausentes > 0]


def fora_do_iqr(x):
    q1, q3 = x.quantile(.25), x.quantile(.75)
    return (x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))


def caixas(dados, x, y, ordem, ax):
    sns.boxplot(data=dados, x=x, y=y, order=ordem, ax=ax, color=AZUL_CLARO, linecolor=AZUL,
                fliersize=2, width=.6, medianprops=dict(color=LARANJA, lw=2))


# Carrega o CSV bruto
bruto = load_raw()

# 1A. Dicionário: tamanho, modelos, marcas e unidades
titulo("1A. Dicionário de dados")
print("linhas x colunas:", bruto.shape)
print("ano de fabricação:", bruto["year"].min(), "a", bruto["year"].max())
print("modelos distintos:", bruto["name"].nunique())
print("marcas distintas:", bruto["name"].str.split().str[0].nunique())
print("unidade de mileage por combustível:\n", pd.crosstab(bruto["fuel"], bruto["mileage"].str.split().str[1]))

# 1B. Qualidade: ausentes, valores impossíveis e duplicatas no bruto
titulo("1B. Qualidade")
print(tabela_ausentes(bruto))
especificacoes = ["mileage", "engine", "max_power", "torque", "seats"]
sem_ficha = bruto[especificacoes].isna().all(axis=1)
print("linhas sem nenhuma especificação:", sem_ficha.sum(),
      "| com alguma ausente:", bruto[especificacoes].isna().any(axis=1).sum())
print("ano mediano sem/com ficha:", bruto.loc[sem_ficha, "year"].median(), bruto.loc[~sem_ficha, "year"].median())
print("preço mediano sem/com ficha:", bruto.loc[sem_ficha, TARGET].median(), bruto.loc[~sem_ficha, TARGET].median())
print("% de particulares entre os sem ficha:",
      round(100 * (bruto.loc[sem_ficha, "seller_type"] == "Individual").mean(), 1))
print("max_power '0' ou ' bhp':", bruto["max_power"].isin(["0", " bhp"]).sum())
print("mileage = 0:", pd.to_numeric(bruto["mileage"].str.split().str[0]).eq(0).sum())
print("km_driven, 5 maiores:", bruto["km_driven"].nlargest(5).tolist(), "| menor:", bruto["km_driven"].min())
print("duplicatas exatas:", bruto.duplicated().sum(), f"({100 * bruto.duplicated().mean():.2f}%)")

# Limpeza (parse das unidades, marca, duplicatas) e o que sobra depois dela
dados = clean(bruto)
print("\nlinhas removidas na limpeza:", len(bruto) - len(dados), "| restam:", len(dados))
print(tabela_ausentes(dados))
print("maiores torques:\n", dados.nlargest(3, "torque_nm")[["brand", "engine", "max_power", "torque_nm"]])

# 1B. Vazamento: nenhuma numérica deve prever o preço sozinha
titulo("1B. Vazamento: Spearman de cada numérica com o preço")
print(dados[NUM + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET).round(3).sort_values())

# 1C. Alvo: distribuição do preço antes e depois do log
titulo("1C. Alvo")
preco = dados[TARGET]
log_preco = np.log(preco)
resumo_alvo = pd.DataFrame({"preço": preco.describe(), "log(preço)": log_preco.describe()})
resumo_alvo.loc["assimetria"] = [preco.skew(), log_preco.skew()]
resumo_alvo.loc["curtose"] = [preco.kurt(), log_preco.kurt()]
print(resumo_alvo.round(2))
print("% acima de Q3 + 1,5·IQR:", round(100 * fora_do_iqr(preco).mean(), 2))

# Figura 1: histograma do preço bruto e do log
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(preco / 1e5, bins=60, color=AZUL, edgecolor="white", linewidth=0.5)
ax[0].axvline(preco.median() / 1e5, color=LARANJA, lw=2, label=f"mediana {preco.median() / 1e5:.2f}")
ax[0].axvline(preco.mean() / 1e5, color="#0b0b0b", lw=1.5, ls="--", label=f"média {preco.mean() / 1e5:.2f}")
ax[0].set(title=f"Preço bruto (assimetria {preco.skew():.2f})",
          xlabel="selling_price (lakh = 100 mil rúpias)", ylabel="carros")
ax[0].legend(frameon=False)
ax[1].hist(log_preco, bins=60, color=AZUL, edgecolor="white", linewidth=0.5)
ax[1].set(title=f"log(preço) (assimetria {log_preco.skew():.2f})", xlabel="log(selling_price)", ylabel="carros")
fig.suptitle("Figura 1. Distribuição do alvo antes e depois do log", fontweight="bold")
salvar(fig, "fig01-alvo.png")

# 1D. Split 80/20 estratificado e baseline da mediana
titulo("1D. Treino e teste")
X_tr, X_te, y_tr, y_te = split(dados)
print(pd.DataFrame({"treino": y_tr.describe(), "teste": y_te.describe()}).round(3))
print("preço mediano treino/teste:", round(np.exp(y_tr.median())), round(np.exp(y_te.median())))
print("baseline (mediana do treino), MAE no teste:", round((np.exp(y_te) - np.exp(y_tr.median())).abs().mean()))

# Daqui em diante, toda análise usa só o treino
treino = X_tr.assign(price=np.exp(y_tr), log_price=y_tr)

# 2A. Numéricas: estatísticas, assimetria e outliers por IQR
titulo("2A. Numéricas (treino)")
fora_iqr = fora_do_iqr(treino[NUM]).sum()
estatisticas = treino[NUM].describe().T
estatisticas["assimetria"] = treino[NUM].skew()
estatisticas["% NaN"] = 100 * treino[NUM].isna().mean()
estatisticas["fora de 1,5·IQR"] = fora_iqr
print(estatisticas.round(2))
print("cilindradas mais comuns:", treino["engine"].value_counts().head(3).to_dict())

# Figura 2: histogramas das numéricas
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, coluna in zip(axes.flat, NUM):
    valores = treino[coluna].dropna()
    if coluna == "km_driven":
        ax.hist(valores, bins=np.logspace(0, np.log10(valores.max()), 50), color=AZUL, edgecolor="white", linewidth=0.4)
        ax.set_xscale("log")
    else:
        ax.hist(valores, bins=40, color=AZUL, edgecolor="white", linewidth=0.4)
    ax.axvline(valores.median(), color=LARANJA, lw=1.6)
    ax.set(title=f"{coluna}  (assim. {valores.skew():.2f})", ylabel="carros",
           xlabel=UNIDADES[coluna] + (" (escala log)" if coluna == "km_driven" else ""))
axes.flat[-1].axis("off")
axes.flat[-1].text(0.05, 0.5, "linha laranja = mediana\nn = treino (5525 carros)",
                   fontsize=10, color=CINZA, va="center")
fig.suptitle("Figura 2. Distribuição das numéricas no treino", fontweight="bold")
salvar(fig, "fig02-numericas-hist.png")

# Figura 3: boxplots das numéricas
fig, axes = plt.subplots(1, 7, figsize=(14, 3.6))
for ax, coluna in zip(axes, NUM):
    ax.boxplot(treino[coluna].dropna(), widths=0.55, patch_artist=True,
               boxprops=dict(facecolor=AZUL_CLARO, edgecolor=AZUL),
               medianprops=dict(color=LARANJA, lw=2),
               flierprops=dict(marker="o", ms=3, mfc=AZUL, mec="none", alpha=.35))
    ax.set(title=coluna, xticks=[], ylabel=UNIDADES[coluna])
    ax.text(0.5, -0.06, f"{fora_iqr[coluna]} fora de 1,5·IQR", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color=CINZA)
fig.suptitle("Figura 3. Boxplots das numéricas no treino (outliers por 1,5·IQR)", fontweight="bold")
salvar(fig, "fig03-numericas-box.png")

# 2B. Categóricas: frequência, cardinalidade e categorias raras
titulo("2B. Categóricas (treino)")
for coluna in CAT:
    contagem = treino[coluna].value_counts()
    print(f"\n{coluna}: cardinalidade {contagem.size}")
    print(pd.DataFrame({"n": contagem, "%": (100 * contagem / len(treino)).round(2)}))
marcas = treino["brand"].value_counts()
marcas_raras = marcas[marcas < 20]
print("\nmarcas com < 20 carros:", marcas_raras.size, "| carros nessas marcas:", marcas_raras.sum())
print("marcas que só aparecem no teste:", set(X_te["brand"]) - set(X_tr["brand"]))

# Figura 4: frequência das categóricas de baixa cardinalidade
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, coluna in zip(axes, ["fuel", "seller_type", "transmission", "owner"]):
    contagem = treino[coluna].value_counts()
    pct = 100 * contagem / len(treino)
    cores = [LARANJA if p < 1 else AZUL for p in pct]
    ax.barh(contagem.index[::-1], contagem.values[::-1], color=cores[::-1], height=0.65)
    for i, (n, p) in enumerate(zip(contagem.values[::-1], pct.values[::-1])):
        ax.text(n, i, f" {p:.1f}%", va="center", fontsize=8.5, color=CINZA)
    ax.set(title=f"{coluna} (card. {contagem.size})", xlabel="carros (escala log)", xscale="log",
           xlim=(1, contagem.max() * 4))
    ax.grid(axis="y", visible=False)
fig.suptitle("Figura 4. Frequência das categóricas de baixa cardinalidade (laranja = categoria rara, < 1%)",
             fontweight="bold")
salvar(fig, "fig04-categoricas.png")

# Figura 5: frequência das marcas
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(marcas.index, marcas.values, color=[LARANJA if n < 20 else AZUL for n in marcas.values], width=0.7)
ax.axhline(20, color=CINZA, lw=1, ls="--")
ax.text(len(marcas) - 0.5, 22, "min_frequency = 20", ha="right", fontsize=9, color=CINZA)
ax.set(title=f"brand: cardinalidade {marcas.size}; {marcas_raras.size} marcas raras (laranja) somam {marcas_raras.sum()} carros",
       ylabel="carros (escala log)", yscale="log")
ax.tick_params(axis="x", rotation=60)
ax.grid(axis="x", visible=False)
fig.suptitle("Figura 5. Frequência das marcas no treino (alta cardinalidade)", fontweight="bold")
salvar(fig, "fig05-marcas.png")

# 3A. Numérica x numérica: correlações e pares redundantes
titulo("3A. Numérica x numérica (treino)")
colunas = NUM + ["price"]
spearman = treino[colunas].corr(method="spearman")
pearson = treino[colunas].corr(method="pearson")
print("Spearman:\n", spearman.round(3))
print("Pearson:\n", pearson.round(3))
pares = pd.DataFrame([(a, b, spearman.loc[a, b]) for i, a in enumerate(colunas) for b in colunas[i + 1:]],
                     columns=["coluna 1", "coluna 2", "spearman"])
print("pares mais correlacionados:\n", pares.sort_values("spearman", key=abs, ascending=False).head(10).round(3))
nm_por_bhp = treino["torque_nm"] / treino["max_power"]
print("Nm por bhp (mediana) por combustível:", nm_por_bhp.groupby(treino["fuel"]).median().round(2).to_dict())

# Figura 6: matrizes de Spearman e Pearson
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, matriz, nome in [(axes[0], spearman, "Spearman (ρ)"), (axes[1], pearson, "Pearson (r)")]:
    sns.heatmap(matriz, annot=True, fmt=".2f", cmap=ESCALA_DIVERGENTE, vmin=-1, vmax=1, ax=ax, square=True,
                cbar_kws={"shrink": .75}, annot_kws={"size": 8}, linewidths=.5, linecolor="white")
    ax.set_title(nome)
fig.suptitle("Figura 6. Correlação entre numéricas e o preço (treino)", fontweight="bold")
salvar(fig, "fig06-correlacao.png")

# Figura 7: dispersão dos pares mais correlacionados
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (x, y) in zip(axes, [("engine", "torque_nm"), ("max_power", "torque_nm"), ("year", "km_driven")]):
    pontos = ax.scatter(treino[x], treino[y], c=treino["log_price"], cmap=ESCALA_AZUL, s=8, alpha=.6, edgecolors="none")
    ax.set(xlabel=f"{x} ({UNIDADES[x]})", ylabel=f"{y} ({UNIDADES[y]})",
           title=f"{x} × {y}: ρ = {spearman.loc[x, y]:.2f}")
    if y == "km_driven":
        ax.set_yscale("log")
fig.colorbar(pontos, ax=axes, shrink=.85, pad=.01).set_label("log(preço)")
fig.suptitle("Figura 7. Pares mais correlacionados, coloridos pelo log do preço", fontweight="bold", x=.45)
salvar(fig, "fig07-dispersao.png", tight=False)

# 3B. Categórica x alvo: preço por categoria
titulo("3B. Categórica x alvo (treino)")
for coluna in CAT:
    por_categoria = treino.groupby(coluna)["price"].agg(["count", "median", "mean"]).sort_values("median")
    print(f"\n{coluna}:\n", por_categoria.round(0))

# Figura 8: log(preço) por categoria
fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
for ax, coluna in zip(axes, ["fuel", "transmission", "seller_type", "owner"]):
    ordem = treino.groupby(coluna)["log_price"].median().sort_values().index
    caixas(treino, coluna, "log_price", ordem, ax)
    mediana = treino.groupby(coluna)["price"].median()
    ax.set_xticks(range(len(ordem)))
    ax.set_xticklabels([f"{ROTULOS_CURTOS.get(o, o)}\n{mediana[o] / 1e5:.1f} lakh" for o in ordem],
                       fontsize=8, rotation=0)
    ax.set(title=coluna, xlabel="", ylabel="log(preço)" if coluna == "fuel" else "")
fig.suptitle("Figura 8. log(preço) por categoria (rótulo = preço mediano)", fontweight="bold")
salvar(fig, "fig08-categoria-alvo.png")

# Figura 9: log(preço) por marca (só marcas com 20 carros ou mais)
por_marca = treino.groupby("brand")["price"].agg(["count", "median"])
marcas_grandes = por_marca[por_marca["count"] >= 20].sort_values("median").index
fig, ax = plt.subplots(figsize=(12, 4.2))
caixas(treino[treino["brand"].isin(marcas_grandes)], "brand", "log_price", marcas_grandes, ax)
ax.tick_params(axis="x", rotation=45)
ax.set(xlabel="", ylabel="log(preço)",
       title=f"Marcas com ≥ 20 carros no treino ({len(marcas_grandes)}), ordenadas pela mediana")
fig.suptitle("Figura 9. log(preço) por marca", fontweight="bold")
salvar(fig, "fig09-marca-alvo.png")

# 3C. Numérica x categórica: posição e dispersão por grupo
titulo("3C. Numérica x categórica (treino)")
pares_num_cat = [("max_power", "transmission"), ("km_driven", "owner"), ("year", "seller_type")]
for num, cat in pares_num_cat:
    quartis = treino.groupby(cat)[num].describe()[["count", "25%", "50%", "75%"]]
    quartis["IQR"] = quartis["75%"] - quartis["25%"]
    print(f"\n{num} por {cat}:\n", quartis.round(1))

# Figura 10: boxplots agrupados
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (num, cat) in zip(axes, pares_num_cat):
    ordem = treino.groupby(cat)[num].median().sort_values().index
    caixas(treino, cat, num, ordem, ax)
    if num == "km_driven":
        ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    ax.set(xlabel="", ylabel=f"{num} ({UNIDADES[num]})", title=f"{num} por {cat}")
fig.suptitle("Figura 10. Numéricas agrupadas por categóricas (treino)", fontweight="bold")
salvar(fig, "fig10-num-cat.png")

# 4A. Estratégias: o que o pipeline aprende no treino
titulo("4A. Estratégias")
print("ausentes no treino:\n", tabela_ausentes(X_tr))

# Ajusta o pipeline só no treino e transforma treino e teste
pipeline = build_pipeline().fit(X_tr)
Z_tr, Z_te = pipeline.transform(X_tr), pipeline.transform(X_te)
nomes = list(pipeline.get_feature_names_out())
Z_tr_df, Z_te_df = pd.DataFrame(Z_tr, columns=nomes), pd.DataFrame(Z_te, columns=nomes)
numericas = nomes[:len(NUM)]

# Limites da winsorização e medianas de imputação aprendidos
blocos_numericos = [pipeline.named_transformers_["num"], pipeline.named_transformers_["log"]]
corte_min = pd.Series(np.concatenate([b["win"].lo_ for b in blocos_numericos]), index=NUM)
corte_max = pd.Series(np.concatenate([b["win"].hi_ for b in blocos_numericos]), index=NUM)
mediana_imputada = pd.Series(np.concatenate([b["imp"].statistics_ for b in blocos_numericos]), index=NUM)
print(pd.DataFrame({"corte min": corte_min, "corte max": corte_max, "mediana imputada": mediana_imputada}).round(1))


# Quantas linhas a winsorização afeta
def tem_valor_cortado(X):
    return ((X[NUM] < corte_min) | (X[NUM] > corte_max)).any(axis=1)


print("linhas com algum valor cortado, treino:", tem_valor_cortado(X_tr).sum(),
      f"({100 * tem_valor_cortado(X_tr).mean():.2f}%) | teste:", tem_valor_cortado(X_te).sum())
print("assimetria depois do pipeline (treino):\n", Z_tr_df[numericas].skew().round(3))
codificador = pipeline.named_transformers_["cat"]["cod"]
print("categorias infrequentes:",
      {c: list(v) for c, v in zip(CAT, codificador.infrequent_categories_) if v is not None})

# 4B. PCA: variância explicada, loadings e relação com o preço
titulo("4B. PCA (treino inteiro)")
pca = PCA(random_state=SEED).fit(Z_tr)
variancia = pca.explained_variance_ratio_
acumulada = np.cumsum(variancia)
print("variância explicada:", variancia[:12].round(4))
print("acumulada:", acumulada[:12].round(4))
print("componentes para 80%, 90% e 95%:", [int(np.searchsorted(acumulada, t)) + 1 for t in (.8, .9, .95)])
print("autovalores nulos:", int((pca.explained_variance_ < 1e-10).sum()))
loadings = pd.DataFrame(pca.components_[:2].T, index=nomes, columns=["PC1", "PC2"])
for pc in ["PC1", "PC2"]:
    print(f"\n{pc}, maiores |loadings|:\n", loadings[pc].sort_values(key=abs, ascending=False).head(8).round(3))
pcs = pd.DataFrame(pca.transform(Z_tr)[:, :2], columns=["PC1", "PC2"])
print("\nSpearman com log(preço):", pcs.corrwith(pd.Series(y_tr.values), method="spearman").round(3).to_dict())

# Figura 11: variância, projeção PC1 x PC2 e loadings
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
componentes = np.arange(1, len(variancia) + 1)
axes[0].bar(componentes, variancia, color=AZUL, width=.7)
axes[0].plot(componentes, acumulada, color=LARANJA, marker="o", ms=3, lw=1.6, label="acumulada")
axes[0].axhline(.9, color=CINZA, lw=1, ls="--")
axes[0].text(len(componentes), .92, "90%", ha="right", fontsize=8.5, color=CINZA)
axes[0].set(title=f"Variância explicada (PC1+PC2 = {100 * variancia[:2].sum():.1f}%)",
            xlabel="componente", ylabel="fração da variância")
axes[0].legend(frameon=False, loc="center right")
pontos = axes[1].scatter(pcs["PC1"], pcs["PC2"], c=y_tr.values, cmap=ESCALA_AZUL, s=6, alpha=.6, edgecolors="none")
axes[1].set(title="Treino projetado em PC1 × PC2", xlabel=f"PC1 ({100 * variancia[0]:.1f}%)",
            ylabel=f"PC2 ({100 * variancia[1]:.1f}%)")
fig.colorbar(pontos, ax=axes[1], label="log(preço)", shrink=.85)
maiores = loadings.loc[loadings.abs().max(axis=1).sort_values(ascending=False).index[:10]]
posicao = np.arange(len(maiores))
axes[2].barh(posicao - .2, maiores["PC1"], height=.38, color=AZUL, label="PC1")
axes[2].barh(posicao + .2, maiores["PC2"], height=.38, color=LARANJA, label="PC2")
axes[2].set_yticks(posicao)
axes[2].set_yticklabels([nome.split("__")[1] for nome in maiores.index], fontsize=8.5)
axes[2].invert_yaxis()
axes[2].axvline(0, color=CINZA, lw=.8)
axes[2].set(title="Maiores loadings de PC1 e PC2", xlabel="peso")
axes[2].legend(frameon=False)
fig.suptitle("Figura 11. PCA ajustada no treino", fontweight="bold")
salvar(fig, "fig11-pca.png")

# 4B. t-SNE e UMAP numa amostra de 3000 carros
titulo("4B. t-SNE e UMAP (amostra de 3000 carros do treino)")
rng = np.random.default_rng(SEED)
amostra = rng.choice(len(Z_tr), size=3000, replace=False)
Z_amostra, y_amostra = Z_tr[amostra], y_tr.values[amostra]


# Métricas: vizinhança preservada (trustworthiness) e preço previsível pelo mapa (R² kNN)
def r2_knn(projecao):
    return cross_val_score(KNeighborsRegressor(15), projecao, y_amostra, cv=5, scoring="r2").mean()


def avaliar(projecao):
    return {"trust k=5": trustworthiness(Z_amostra, projecao, n_neighbors=5),
            "trust k=30": trustworthiness(Z_amostra, projecao, n_neighbors=30),
            "R² kNN": r2_knn(projecao)}


# Projeções: PCA, t-SNE e UMAP com três valores de vizinhança cada
projecoes = {"PCA (2 comp.)": pcs.values[amostra]}
for perplexidade in [5, 30, 50]:
    tsne = TSNE(2, perplexity=perplexidade, random_state=SEED, init="pca")
    projecoes[f"t-SNE perp={perplexidade}"] = tsne.fit_transform(Z_amostra)
for vizinhos in [5, 15, 50]:
    mapa = umap.UMAP(n_neighbors=vizinhos, min_dist=0.1, random_state=SEED)
    projecoes[f"UMAP nn={vizinhos}"] = mapa.fit_transform(Z_amostra)

# Tabela 3 e referência no espaço completo
avaliacao = pd.DataFrame({nome: avaliar(projecao) for nome, projecao in projecoes.items()}).T
print(avaliacao.round(3))
print("referência: R² kNN no espaço completo (40 dim):", round(r2_knn(Z_amostra), 3))

# Controle: t-SNE em colunas embaralhadas não deve prever o preço
Z_embaralhado = np.column_stack([rng.permutation(coluna) for coluna in Z_amostra.T])
controle = TSNE(2, perplexity=30, random_state=SEED, init="pca").fit_transform(Z_embaralhado)
print("controle: R² kNN do t-SNE (perp=30) em colunas embaralhadas:", round(r2_knn(controle), 3))

# O que define as ilhas do UMAP: pureza dos grupos em cada categórica
grupos = KMeans(8, n_init=10, random_state=SEED).fit_predict(projecoes["UMAP nn=15"])
X_amostra = X_tr.iloc[amostra].reset_index(drop=True)
for coluna in CAT:
    pureza = pd.crosstab(grupos, X_amostra[coluna], normalize="index").max(axis=1).mean()
    taxa_base = X_amostra[coluna].value_counts(normalize=True).iloc[0]
    print(f"pureza dos 8 grupos do UMAP em {coluna}: {pureza:.3f} (taxa base {taxa_base:.3f})")

# Figuras 12 e 13: mapas do t-SNE e do UMAP
for numero, metodo, arquivo in [(12, "t-SNE", "fig12-tsne.png"), (13, "UMAP", "fig13-umap.png")]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    nomes_do_metodo = [nome for nome in projecoes if nome.startswith(metodo)]
    for ax, nome in zip(axes, nomes_do_metodo):
        projecao = projecoes[nome]
        pontos = ax.scatter(projecao[:, 0], projecao[:, 1], c=y_amostra, cmap=ESCALA_AZUL, s=5, alpha=.7,
                            edgecolors="none")
        ax.set(title=f"{nome}\ntrust(k=5) {avaliacao.loc[nome, 'trust k=5']:.3f} · "
                     f"R² kNN {avaliacao.loc[nome, 'R² kNN']:.3f}",
               xticks=[], yticks=[], xlabel="dim 1", ylabel="dim 2")
    fig.colorbar(pontos, ax=axes, label="log(preço)", shrink=.85, pad=.01)
    fig.suptitle(f"Figura {numero}. {metodo} com três valores do parâmetro de vizinhança "
                 "(3000 carros do treino)", fontweight="bold", x=.45, y=1.06)
    salvar(fig, arquivo, tight=False)

# 4C. Verificações do pipeline: shape, NaN, escala e categoria nova
titulo("4C. Pipeline")
print("shape treino/teste:", Z_tr.shape, Z_te.shape)
print("NaN treino/teste:", int(np.isnan(Z_tr).sum()), int(np.isnan(Z_te).sum()))
print("numéricas no treino:\n", Z_tr_df[numericas].agg(["mean", "std"]).round(3))
print("média das numéricas no teste:\n", Z_te_df[numericas].mean().round(3))
lexus = X_te["brand"].eq("Lexus").to_numpy()
print("Lexus em cat__brand_infrequent_sklearn:", Z_te_df.loc[lexus, "cat__brand_infrequent_sklearn"].tolist())
print(len(nomes), "colunas:", nomes)
