# Perguntas prováveis da prova de projeto: entrega 1 (EDA)

Material de estudo baseado em `docs/projects/eda/index.md`. Cada pergunta traz uma resposta
curta e o lugar do relatório onde está a evidência. A prova costuma cobrar o **porquê** das
decisões, não só o número.

---

## 1. Dataset e tarefa

**1.1. Qual é o dataset, qual a tarefa e qual o alvo?**
Vehicle dataset from CarDekho, arquivo `Car details v3.csv`, com anúncios de carros usados na Índia. A tarefa é regressão do preço de revenda (`selling_price`, em ₹), modelado como `log(selling_price)`.

**1.2. Por que o `v3` e não os outros três arquivos do pacote?**
`car data.csv` tem 301 linhas, abaixo do mínimo de 1.000. `CAR DETAILS FROM CAR DEKHO.csv` não tem ficha técnica. `car details v4.csv` tem 2.059 linhas, cerca de 4 vezes menos que o `v3` (8.128).

**1.3. O que é uma linha do dataset? Tem data?**
Uma linha é um anúncio de um carro usado, observado uma vez. Não tem data do anúncio, só o ano de fabricação (`year`).

**1.4. Por que regressão e não classificação?**
O alvo é um preço, ou seja, um valor contínuo. Transformar o preço em faixas jogaria informação fora e criaria limites arbitrários.

**1.5. Quantas linhas e colunas, antes e depois da limpeza?**
Bruto: 8.128 × 13 (12 features + alvo). Depois de remover duplicatas: 6.907 linhas, com 7 features numéricas e 5 categóricas. Depois do pipeline: 40 colunas.

---

## 2. Limpeza e qualidade

**2.1. Quais colunas chegam como texto e precisam de parse?**
`mileage` ("23.4 kmpl"), `engine` ("1248 CC"), `max_power` ("74 bhp") e `torque` ("190Nm@ 2000rpm"). O tipo que o pandas atribui (`object`) não diz o tipo real da variável.

**2.2. Como foi tratado o torque com duas unidades (Nm e kgm)?**
kgm × 9,80665 = Nm. A exceção são valores acima de 60 marcados como "kgm": esses são lidos como Nm, porque convertê-los daria absurdos (um Tata Sumo com 1.128 Nm, sendo que o maior torque real é 640 Nm, de um Volvo). O rpm é descartado porque vem em formatos irreconciliáveis.

**2.3. E o `mileage` com kmpl e km/kg na mesma coluna?**
Os valores não são comparáveis entre si, mas a unidade acompanha exatamente o combustível (km/kg só em CNG/LPG). O one-hot de `fuel` absorve essa diferença.

**2.4. Quantos valores faltam e qual o padrão?**
Cerca de 2,7% em cada uma das 5 colunas de ficha técnica (`mileage`, `engine`, `max_power`, `torque`, `seats`). Faltam **em bloco**: 215 linhas não têm nenhuma das 5.

**2.5. A ausência é aleatória? Por que isso importa?**
Não. As linhas sem ficha técnica são de carros mais velhos (ano mediano 2008 contra 2015) e mais baratos (₹175 mil contra ₹450 mil), e 92% são vendas de particulares. A ausência carrega informação sobre o preço, por isso criamos um `MissingIndicator` além de imputar a mediana.

**2.6. Por que um indicador só, e não um por coluna?**
As 5 especificações faltam juntas, então um indicador basta (foi usado o de `engine`).

**2.7. Que valores impossíveis vocês encontraram?**
`max_power = "0"` ou `" bhp"` (7 linhas) e `mileage = 0` (17 linhas), que viraram NaN. Um carro com 1 km e outro com 2.360.457 km (cerca de 60 voltas na Terra). Um Maruti de 58 bhp com 789 Nm, provavelmente erro de digitação.

**2.8. Quantas duplicatas e por que removê-las antes do split?**
1.221 (14,8%): 1.202 cópias exatas mais 19 que ficam iguais depois do parse. Se ficassem, a mesma linha poderia cair no treino e no teste, e o erro de teste mediria memorização em vez de generalização.

**2.9. Quais colunas foram descartadas e por quê?**
`name`: 2.058 valores distintos em 8.128 linhas, quase um identificador. A parte útil vira `brand` (primeira palavra). `torque` em texto: substituída por `torque_nm`.

**2.10. Existe vazamento de dados (leakage) nas colunas?**
Nenhuma coluna é posterior à venda. A numérica mais associada ao preço é `year` (ρ = 0,706), longe do patamar suspeito (> 0,95). Os riscos de vazamento vêm do processo: duplicatas nos dois lados do split e estatísticas calculadas fora do treino. Os dois foram tratados.

**2.11. A extração de `brand` pela primeira palavra tem algum problema?**
Trunca alguns nomes ("Land" é Land Rover, "Ashok" é Ashok Leyland). Como essas marcas são raras, caem em `infrequent` e isso não afeta o modelo.

---

## 3. Alvo

**3.1. Como é a distribuição do preço?**
Muito assimétrica à direita: assimetria 5,57, curtose 52,7, média ₹517 mil contra mediana ₹400 mil. 4,7% dos carros passam de Q3 + 1,5·IQR.

**3.2. Por que modelar log(preço)?**
(a) A assimetria cai de 5,57 para −0,16. (b) Com MSE no preço bruto, os ~5% de carros de luxo dominariam o gradiente. (c) O erro vira relativo: errar ₹50 mil num carro de ₹2 lakh pesa mais do que num de ₹50 lakh. (d) A depreciação é exponencial e fica quase linear em log.

**3.3. Qual é a baseline?**
Prever sempre a mediana dá MAE de ₹278.084. Qualquer modelo da entrega 2 precisa bater esse valor.

**3.4. O que é 1 lakh?**
₹100.000.

---

## 4. Split treino/teste

**4.1. Como foi feito o split?**
80/20 (5.525 / 1.382), `random_state=42`, estratificado pelos decis de log(preço).

**4.2. Por que estratificar se regressão não tem classes?**
Para garantir que os carros caros, que são poucos, apareçam nos dois lados. As médias de treino e teste coincidem até a 3ª casa decimal.

**4.3. Por que não um split temporal?**
O anúncio não tem data. `year` é o ano de fabricação, não o da venda.

**4.4. Em que ordem as coisas acontecem e por quê?**
Parse → remoção de duplicatas → split → estatísticas aprendidas só no treino (`fit` no treino, `transform` no teste). Assim nenhuma informação do teste vaza para o modelo.

---

## 5. Análise univariada

**5.1. Qual a coluna numérica mais problemática?**
`km_driven`, com assimetria 12,63 vinda quase toda do carro de 2,36 milhões de km. Em escala log ela fica unimodal em torno de 70 mil km.

**5.2. Por que a regra do 1,5·IQR não serve para remover outliers aqui?**
Em `seats`, Q1 = Q3 = 5, então o IQR é zero e todo carro que não tem 5 lugares (1.176) vira "outlier". Em `engine`, os 965 marcados são SUVs e sedãs grandes de verdade.

**5.3. Por que `engine` tem várias modas em degraus?**
Os motores se concentram em poucas cilindradas de catálogo (1.248 cc em 739 carros, 1.197 cc em 561, 796 cc em 330).

**5.4. Por que padronizar?**
O desvio de `km_driven` (57.195) é cerca de 57.000 vezes o de `seats` (0,99). Sem padronização, os gradientes ficam em escalas muito diferentes e o treino por gradiente sofre.

**5.5. Como estão as categóricas?**
Muito desbalanceadas: 91,6% manuais, 89,5% vendidos por particulares, 98,6% diesel ou gasolina. Só 463 automáticos e 80 carros a GNV/GLP no treino.

**5.6. Por que `owner` foi tratada como nominal, se parece ordinal?**
Por causa de `Test Drive Car`: 4 seminovos de concessionária com preço mediano de ₹60,7 lakh, que não cabem na ordem como "categoria 0".

**5.7. Como lidaram com as 31 marcas?**
`OneHotEncoder(min_frequency=20, handle_unknown="infrequent_if_exist")`: marcas com menos de 20 carros e marcas nunca vistas vão para uma coluna `infrequent`. Ficam 17 marcas mais `infrequent`.

**5.8. Por que `handle_unknown` importa? Dê o exemplo.**
A Lexus aparece só no teste (1 carro). Um one-hot comum quebraria no `transform`. Com `infrequent_if_exist`, ela vira `cat__brand_infrequent_sklearn = 1`.

---

## 6. Análise bivariada e multivariada

**6.1. Por que Spearman em vez de Pearson?**
As numéricas têm caudas longas, valores em degraus e relações monotônicas não lineares (como a depreciação). Spearman usa postos, então um único outlier não domina o resultado.

**6.2. Qual o par mais correlacionado?**
`engine` × `torque_nm`, ρ = 0,845. `engine`, `max_power` e `torque_nm` formam um bloco redundante (0,73 a 0,85): as três medem o tamanho do motor.

**6.3. A colinearidade atrapalha?**
Numa rede neural, não impede o treino, já que não há inversão de matriz. Nenhum par passa de 0,95, então nenhuma coluna saiu por redundância. Na entrega 2: weight decay e importância por permutação do bloco inteiro.

**6.4. O que significa ρ = 0,71 e r = 0,43 entre `year` e preço?**
A relação é monotônica forte, mas curva (depreciação exponencial). A distância entre os dois coeficientes é mais um argumento para usar log no alvo.

**6.5. O que são as duas faixas paralelas em `max_power` × `torque_nm`?**
Uma faixa por combustível: o diesel entrega 2,35 Nm por bhp e a gasolina 1,36. É uma **interação**, que uma MLP consegue aprender e um modelo linear sem termos de interação não.

**6.6. Quais categóricas mais mexem no preço?**
`brand` é a mais forte: mediana de ₹2,2 lakh (Chevrolet) a ₹25,5 lakh (BMW), 11,7×. Também: `owner` (2,9×), `fuel` (2,6×), `transmission` (2,2×), `seller_type` (1,6×).

**6.7. O efeito de `transmission` (2,2×) é só do câmbio?**
Não. Os automáticos têm mediana de 126 bhp contra 82 dos manuais, então parte do efeito é potência (carro maior). Da mesma forma, `seller_type` carrega idade. Por isso o modelo precisa ver as variáveis em conjunto.

**6.8. Como `km_driven` varia com `owner`?**
Só muda a posição (50k, 80k, 90k, 99k km), com dispersão constante: cada troca de dono soma km.

---

## 7. Pré-processamento

**7.1. Descreva o pipeline.**
`ColumnTransformer` com 4 ramos:
- `num` [year, mileage, seats]: Winsorizer → mediana → StandardScaler
- `log` [km_driven, engine, max_power, torque_nm]: Winsorizer → mediana → log1p → StandardScaler
- `cat` [fuel, seller_type, transmission, owner, brand]: moda → OneHot(min_frequency=20)
- `miss` [engine]: MissingIndicator

**7.2. Por que mediana e não média na imputação?**
A mediana resiste às caudas. A média de `max_power` (87,8) fica 7% acima da mediana (81,9).

**7.3. O que é winsorização e por que usá-la em vez de remover outliers?**
É cortar os valores nos quantis de 0,5% e 99,5%: um valor acima do limite vira o limite. A linha é mantida e só o extremo é limitado. Remover linhas descartaria SUVs e carros de 7 lugares legítimos. Afeta 201 linhas do treino (3,64%) e 49 do teste.

**7.4. Por que winsorizar antes do log?**
Para que valores impossíveis (1 km, 2,36 milhões de km) não distorçam a escala. Depois do corte, o log reduz a cauda que sobra: a assimetria de `km_driven` vai de 12,6 para −1,05.

**7.5. Por que `log1p` e não `log`?**
`log1p(x) = log(1 + x)` é definido em zero.

**7.6. Por que não usar `drop="first"` no one-hot?**
Com `drop="first"`, a categoria de referência fica mais perto de todas as outras do que elas entre si, o que distorce a geometria que t-SNE e UMAP usam. A colinearidade exata do one-hot não atrapalha uma rede neural.

**7.7. Por que redes neurais precisam de entradas padronizadas?**
Com ReLU/tanh treinadas por gradiente, os pesos ligados a `km_driven` receberiam gradientes da ordem de 10⁵ e os de `seats`, da ordem de 1. O treino fica instável e lento.

**7.8. Quais são as 40 colunas de saída?**
7 numéricas + 4 `fuel` + 3 `seller_type` + 2 `transmission` + 5 `owner` (4 + infrequent) + 18 `brand` (17 + infrequent) + 1 indicador = 40.

**7.9. Como vocês provam que não há vazamento no pipeline?**
A média das numéricas no treino é exatamente 0 (dp 1), e no teste fica entre −0,021 e 0,053. É diferente de zero porque o scaler não viu o teste. Treino e teste têm 0 NaN, porque o teste é imputado com as medianas do treino.

**7.10. Por que `fit` só no treino?**
Se a média, mediana ou quantil vierem da base inteira, informação do teste entra no modelo e a avaliação fica otimista.

---

## 8. Redução de dimensionalidade

**8.1. Quanto da variância PC1 + PC2 explicam?**
57,1% (37,7% + 19,4%). São necessárias 5 componentes para 80% e 9 para 90%.

**8.2. O que representam PC1 e PC2?**
PC1 é o **porte** do carro (engine, torque, max_power, seats positivos; mileage negativo). PC2 é **idade e uso** (year positivo, km_driven negativo).

**8.3. Qual componente acompanha mais o preço?**
A PC2 (Spearman 0,724), mais que a PC1 (0,472). Carro novo e grande é caro: a cor escurece na diagonal.

**8.4. Por que 5 autovalores são exatamente zero?**
Um para cada bloco de one-hot (fuel, seller_type, transmission, owner, brand). As colunas de cada bloco somam 1 em toda linha, o que gera dependência linear.

**8.5. Diferença entre PCA, t-SNE e UMAP?**
PCA é linear, preserva variância global, é rápida e tem `.transform()`. t-SNE é não linear, preserva vizinhança local e não projeta dados novos. UMAP é não linear, preserva melhor a estrutura local e global e pode projetar dados novos.

**8.6. Por que t-SNE e UMAP usaram só 3.000 carros?**
O t-SNE não tem `.transform()` e seu custo cresce mais rápido que o número de linhas.

**8.7. Como compararam as projeções?**
- **Trustworthiness** (k = 5 local, k = 30 global): o mapa preserva os vizinhos do espaço original?
- **R² de um kNN** (k = 15, CV 5 folds) prevendo log(preço) a partir das 2 coordenadas: vizinhos no mapa têm preço parecido?

**8.8. Por que não usar silhueta?**
Silhueta precisa de rótulos de grupo, e o alvo é contínuo.

**8.9. O que t-SNE e UMAP mostram que a PCA não mostra?**
Entre 5 e 7 ilhas definidas por **combustível** (pureza 0,96 contra taxa base de 0,54) e **famílias de marca** (0,54 contra 0,32). Na PCA esses grupos se sobrepõem. A trustworthiness local sobe de 0,907 (PCA) para 0,99.

**8.10. Os métodos não lineares ajudam a prever o preço?**
Quase nada: R² de 0,832 na PCA contra 0,837 no melhor t-SNE. Idade e porte já capturam quase tudo o que 2 dimensões dizem sobre o preço.

**8.11. Qual o efeito dos hiperparâmetros (perplexidade, n_neighbors)?**
Valores pequenos pioram: UMAP com n_neighbors = 5 fragmenta em ilhotas (R² 0,762) e t-SNE com perplexidade 5 vira uma nuvem uniforme. Os resultados estabilizam a partir de 30 (t-SNE) e 15 (UMAP). Só confiamos nas ilhas que aparecem nos dois valores maiores.

**8.12. Para que serviu o controle com colunas embaralhadas?**
O t-SNE desenha grupos até em dados sem estrutura. Com as colunas embaralhadas, o R² do preço cai para −0,07, o que prova que a estrutura dos mapas reais vem dos dados e não do método.

**8.13. Vão usar a redução como entrada do modelo?**
Não. Com 2 componentes, a PCA perde 43% da variância, e o t-SNE não projeta dados novos. A MLP recebe as 40 colunas padronizadas.

**8.14. O que significa o kNN chegar a R² = 0,875 nas 40 dimensões?**
Que há bastante estrutura para aprender. Também sugere um teto: falta informação como estado de conservação, acidentes e cidade.

---

## 9. Síntese, riscos e próximos passos

**9.1. Resuma o EDA em três frases.**
O preço segue aproximadamente uma log-normal e depende sobretudo de idade e porte. Combustível e marca criam saltos discretos. Os dados exigiram muita limpeza (duplicatas, texto com unidades, valores impossíveis, ausência em bloco não aleatória).

**9.2. Quais os riscos para a entrega 2 e como mitigar?**

| Risco | Mitigação |
|-------|-----------|
| Erro alto em grupos pequenos (automáticos, CNG/LPG, marcas premium) | Reportar o MAE por subgrupo |
| Métrica dominada por carros caros | Treinar em log; reportar MAE em ₹ e MAPE |
| Duplicatas inflando o teste | Já removidas antes do split |
| Vazamento de estatísticas | Pipeline inteiro dentro do `fit` de cada fold |
| Colinearidade engine/power/torque | Weight decay; importância por permutação do bloco |
| Ausência informativa | O indicador já está no pipeline |
| Falta de informação (estado, acidentes) | Aceitar o teto; comparar com a baseline da mediana e com um modelo linear |

**9.3. Quais os limites do dataset?**
Poucos exemplos de luxo e de GNV/GLP. Não dá para fazer validação temporal (sem data), e a inflação entre anúncios fica como ruído. Faltam estado de conservação, histórico de acidentes e cidade.

**9.4. Por que uma MLP faz sentido para esse problema?**
O preço tem interações (combustível × torque), relações não lineares e grupos discretos (marca, combustível) que um modelo linear não capta sem engenharia de features. As entradas já estão padronizadas e sem NaN.

---

## 10. Perguntas "pegadinha" e conceituais

**10.1. Se eu mudar o `random_state`, os resultados mudam?**
Os números exatos do split, do t-SNE e do UMAP mudam um pouco. As conclusões (log do alvo, ilhas por combustível, PC1 = porte, PC2 = idade) devem se manter. Por isso só consideramos estruturas estáveis entre hiperparâmetros.

**10.2. Por que não removeram o carro de 2,36 milhões de km?**
Removê-lo seria uma decisão manual caso a caso. A winsorização trata esse e qualquer outro extremo de forma sistemática e reprodutível, com limites aprendidos no treino.

**10.3. Por que o MissingIndicator marca 162 linhas no treino, e não 215?**
215 é o número na base bruta. Depois de remover duplicatas e separar 80% para treino, sobram 162 linhas sem ficha técnica no treino. As 11 linhas a mais com NaN em `mileage` vêm de `mileage = 0` e são erros pontuais, sem indicador.

**10.4. Como sabem que os dados de teste não influenciaram nenhuma figura?**
Da seção 2 em diante, todas as figuras e todos os parâmetros vêm só do treino. A única análise na base inteira é a do alvo (seção 1C), feita antes do split, para justificar o log.

**10.5. Qual a diferença entre MAE, MSE e MAPE, e qual usar?**
MAE é o erro absoluto médio (em ₹, fácil de interpretar). MSE eleva o erro ao quadrado e pune muito os erros grandes (seria dominado pelos carros de luxo). MAPE é o erro percentual médio, relativo ao preço. O plano é treinar em log e reportar MAE em ₹ e MAPE.

**10.6. Como cada integrante contribuiu?**
Combinem a resposta antes da prova. Ela precisa bater com o histórico do Git.
