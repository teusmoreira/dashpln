import json, glob, os, re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from PIL import Image
import io, base64

# ── Paleta & configuração ────────────────────────────────
CORES_SENTIMENTO = {
    "positive": "#2ecc71",
    "neutral":  "#f39c12",
    "negative": "#e74c3c",
}

st.set_page_config(
    page_title="Consumidor.gov · Dashboard de Sentimento",
    page_icon="🛒",
    layout="wide",
)

# ── CSS personalizado ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

[data-testid="stMetric"] {
    background: #0f0f1a;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8888aa !important; font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #e8e8ff !important; font-size: 2rem; font-weight: 800; }

.bloco-titulo {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 60%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: .2rem;
}
.bloco-sub { color: #8888aa; font-size: .95rem; margin-bottom: 2rem; }
.secao-header {
    font-size: 1.1rem; font-weight: 700;
    color: #a78bfa;
    letter-spacing: .06em;
    text-transform: uppercase;
    border-left: 3px solid #a78bfa;
    padding-left: .7rem;
    margin: 2rem 0 .8rem;
}
stDataFrame { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────
def nota_para_sentimento(nota):
    try:
        n = int(nota)
        if n >= 4: return "positive"
        if n == 3: return "neutral"
        return "negative"
    except:
        return None

def vader_sentimento(texto):
    if pd.isna(texto) or str(texto).strip() == "": return None
    score = SentimentIntensityAnalyzer().polarity_scores(str(texto))["compound"]
    if score >= 0.05:   return "positive"
    if score <= -0.05:  return "negative"
    return "neutral"

def gerar_wordcloud(textos, titulo, cor):
    stopwords_pt = {"de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você", "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo", "pois", "já", "daí", "dia", "xxx"}
    texto_junto = " ".join(textos.dropna().astype(str).tolist())
    wc = WordCloud(
        width=700, height=350,
        background_color="#0f0f1a",
        colormap="cool",
        max_words=80,
        collocations=False,
        stopwords=stopwords_pt
    ).generate(texto_junto)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()

@st.cache_data(show_spinner="🔄 Carregando dataset...")
def carregar_dados():
    # Lê diretamente o arquivo parquet que estará junto com o app.py
    return pd.read_parquet("dataset_consumidor.parquet")

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.markdown("### 🛒 Filtros & Controles")
n_amostras = st.sidebar.slider("Registros para análise de modelo", 20, 500, 100, step=20)
usar_vader  = st.sidebar.checkbox("Exibir análise VADER", value=True)
usar_tfidf  = st.sidebar.checkbox("Exibir análise TF-IDF + LR", value=True)
st.sidebar.markdown("---")
st.sidebar.info("Dataset: [Kaggle](https://www.kaggle.com/datasets/beatrizmsarmento/relatos-de-consumidores-do-site-consumidor-gov-br)")

# ── Cabeçalho ────────────────────────────────────────────
st.markdown('<div class="bloco-titulo">📊 Análise de Sentimento</div>', unsafe_allow_html=True)
st.markdown('<div class="bloco-sub">Relatos de consumidores · consumidor.gov.br</div>', unsafe_allow_html=True)

# ── Carregar dados ────────────────────────────────────────
df_raw = carregar_dados()

# ── Garantir coluna texto unificada ──────────────────────
col_texto = None
for c in ["relato", "comentario", "texto", "descricao"]:
    if c in df_raw.columns:
        col_texto = c
        break
if col_texto is None and len(df_raw.columns) > 0:
    col_texto = df_raw.columns[-1]

df_raw["_texto"] = df_raw[col_texto]

# Nota → sentimento esperado
if "nota" in df_raw.columns:
    df_raw["sentimento_esperado"] = df_raw["nota"].apply(nota_para_sentimento)
else:
    df_raw["sentimento_esperado"] = None

# ═══════════════════════════════════════════════════════
#  SEÇÃO 1 · Visão geral do dataset
# ═══════════════════════════════════════════════════════
st.markdown('<div class="secao-header">1 · Visão geral do Dataset</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de registros",  f"{len(df_raw):,}")
c2.metric("Colunas",             len(df_raw.columns))
c3.metric("Empresas únicas",     df_raw["empresa"].nunique() if "empresa" in df_raw.columns else "–")
c4.metric("% sem texto",
          f"{df_raw['_texto'].isna().mean()*100:.1f}%" if col_texto else "–")

with st.expander("🔍 Amostra dos dados brutos"):
    st.dataframe(df_raw.head(10), use_container_width=True)

# ═══════════════════════════════════════════════════════
#  SEÇÃO 2 · Distribuição de notas
# ═══════════════════════════════════════════════════════
if "nota" in df_raw.columns:
    st.markdown('<div class="secao-header">2 · Distribuição de Notas</div>', unsafe_allow_html=True)

    notas_num = pd.to_numeric(df_raw["nota"], errors="coerce").dropna()
    col_a, col_b = st.columns(2)

    # Histograma
    fig_hist = px.histogram(
        notas_num, x=notas_num,
        nbins=5, color_discrete_sequence=["#a78bfa"],
        labels={"x": "Nota", "y": "Frequência"},
        title="Distribuição de notas",
    )
    fig_hist.update_layout(
        paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
        font_color="#c0c0e0", bargap=.1
    )
    col_a.plotly_chart(fig_hist, use_container_width=True)

    # Pizza sentimentos esperados
    dist = df_raw["sentimento_esperado"].value_counts().reset_index()
    dist.columns = ["sentimento", "count"]
    fig_pie = px.pie(
        dist, names="sentimento", values="count",
        color="sentimento",
        color_discrete_map=CORES_SENTIMENTO,
        title="Sentimento esperado (por nota)",
        hole=.45,
    )
    fig_pie.update_layout(
        paper_bgcolor="#0f0f1a", font_color="#c0c0e0"
    )
    col_b.plotly_chart(fig_pie, use_container_width=True)

# ═══════════════════════════════════════════════════════
#  SEÇÃO 3 · Top empresas
# ═══════════════════════════════════════════════════════
if "empresa" in df_raw.columns:
    st.markdown('<div class="secao-header">3 · Top Empresas com mais Reclamações</div>', unsafe_allow_html=True)

    top_n = st.slider("Número de empresas", 5, 30, 15)
    top_emp = df_raw["empresa"].value_counts().head(top_n).reset_index()
    top_emp.columns = ["empresa", "total"]

    fig_bar = px.bar(
        top_emp.sort_values("total"),
        x="total", y="empresa", orientation="h",
        color="total",
        color_continuous_scale="Purples",
        labels={"total": "Registros", "empresa": ""},
        title=f"Top {top_n} empresas",
    )
    fig_bar.update_layout(
        paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
        font_color="#c0c0e0", coloraxis_showscale=False,
        height=max(350, top_n * 28),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Sentimento médio por empresa
    if "sentimento_esperado" in df_raw.columns:
        mapa_num = {"positive": 1, "neutral": 0, "negative": -1}
        df_raw["_sent_num"] = df_raw["sentimento_esperado"].map(mapa_num)
        sent_emp = (
            df_raw.groupby("empresa")["_sent_num"]
            .mean()
            .reindex(top_emp["empresa"])
            .reset_index()
            .fillna(0)
        )
        sent_emp.columns = ["empresa", "score"]

        fig_sent = px.bar(
            sent_emp.sort_values("score"),
            x="score", y="empresa", orientation="h",
            color="score",
            color_continuous_scale=[(0,"#e74c3c"),(0.5,"#f39c12"),(1,"#2ecc71")],
            range_color=[-1, 1],
            labels={"score": "Score médio", "empresa": ""},
            title="Score de sentimento por empresa (−1 = neg, +1 = pos)",
        )
        fig_sent.update_layout(
            paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
            font_color="#c0c0e0",
            height=max(350, top_n * 28),
        )
        st.plotly_chart(fig_sent, use_container_width=True)

# ═══════════════════════════════════════════════════════
#  SEÇÃO 4 · Evolução temporal
# ═══════════════════════════════════════════════════════
if "data" in df_raw.columns:
    st.markdown('<div class="secao-header">4 · Evolução Temporal</div>', unsafe_allow_html=True)
    try:
        df_raw["_data"] = pd.to_datetime(df_raw["data"], errors="coerce", dayfirst=True)
        df_tempo = (
            df_raw.dropna(subset=["_data"])
            .set_index("_data")
            .resample("ME")["_texto"]
            .count()
            .reset_index()
        )
        df_tempo.columns = ["mes", "registros"]

        fig_line = px.area(
            df_tempo, x="mes", y="registros",
            color_discrete_sequence=["#a78bfa"],
            title="Volume mensal de relatos",
            labels={"mes": "", "registros": "Registros"},
        )
        fig_line.update_layout(
            paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
            font_color="#c0c0e0"
        )
        st.plotly_chart(fig_line, use_container_width=True)
    except Exception as e:
        st.warning(f"Não foi possível processar datas: {e}")

# ═══════════════════════════════════════════════════════
#  SEÇÃO 5 · Nuvem de palavras
# ═══════════════════════════════════════════════════════
if col_texto:
    st.markdown('<div class="secao-header">5 · Nuvem de Palavras por Sentimento</div>', unsafe_allow_html=True)
    sentimentos_wc = ["positive", "negative", "neutral"]
    cols_wc = st.columns(3)
    for i, sent in enumerate(sentimentos_wc):
        sub = df_raw[df_raw["sentimento_esperado"] == sent]["_texto"]
        if len(sub) < 5:
            cols_wc[i].warning(f"Poucos registros: {sent}")
            continue
        img_bytes = gerar_wordcloud(sub, sent, CORES_SENTIMENTO[sent])
        cols_wc[i].markdown(f"**{sent.capitalize()}**")
        cols_wc[i].image(img_bytes, use_container_width=True)

# ═══════════════════════════════════════════════════════
#  SEÇÃO 6 · VADER
# ═══════════════════════════════════════════════════════
if usar_vader and col_texto:
    st.markdown('<div class="secao-header">6 · Modelo VADER (léxico)</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner="🔄 Rodando VADER...")
    def rodar_vader(df, n):
        df_v = df.head(n).copy()
        df_v["pred_vader"] = df_v["_texto"].apply(vader_sentimento)
        return df_v

    df_vader = rodar_vader(df_raw, n_amostras)

    col_v1, col_v2 = st.columns(2)

    dist_v = df_vader["pred_vader"].value_counts().reset_index()
    dist_v.columns = ["sentimento", "count"]
    fig_v = px.bar(
        dist_v, x="sentimento", y="count",
        color="sentimento",
        color_discrete_map=CORES_SENTIMENTO,
        title="Distribuição VADER",
        labels={"count": "Registros", "sentimento": ""},
    )
    fig_v.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", font_color="#c0c0e0", showlegend=False)
    col_v1.plotly_chart(fig_v, use_container_width=True)

    if "sentimento_esperado" in df_vader.columns:
        mask = df_vader["pred_vader"].notna() & df_vader["sentimento_esperado"].notna()
        y_true = df_vader.loc[mask, "sentimento_esperado"]
        y_pred = df_vader.loc[mask, "pred_vader"]
        if len(y_true) > 0:
            labels = ["positive", "neutral", "negative"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig_cm = px.imshow(
                cm, x=labels, y=labels,
                color_continuous_scale="Purp",
                labels={"x": "Predito", "y": "Real"},
                title="Matriz de Confusão VADER",
                text_auto=True,
            )
            fig_cm.update_layout(paper_bgcolor="#0f0f1a", font_color="#c0c0e0")
            col_v2.plotly_chart(fig_cm, use_container_width=True)

            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
            col_v1.metric("Acurácia VADER", f"{acc:.2%}")
            col_v1.metric("F1-macro VADER", f"{f1:.2%}")

# ═══════════════════════════════════════════════════════
#  SEÇÃO 7 · TF-IDF + Logistic Regression
# ═══════════════════════════════════════════════════════
if usar_tfidf and col_texto and "sentimento_esperado" in df_raw.columns:
    st.markdown('<div class="secao-header">7 · Modelo TF-IDF + Logistic Regression</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner="🔄 Treinando TF-IDF + LR...")
    def rodar_tfidf(df, n):
        df_ml = df.head(n).copy()
        df_ml = df_ml[df_ml["_texto"].notna() & df_ml["sentimento_esperado"].notna()]
        df_ml["_texto"] = df_ml["_texto"].astype(str)
        if len(df_ml) < 10:
            return None
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
            ("clf",   LogisticRegression(max_iter=1000, random_state=42))
        ])
        cv = min(5, len(df_ml["sentimento_esperado"].unique()))
        preds = cross_val_predict(pipeline, df_ml["_texto"], df_ml["sentimento_esperado"], cv=cv)
        df_ml["pred_tfidf"] = preds
        return df_ml

    df_ml = rodar_tfidf(df_raw, n_amostras)

    if df_ml is None:
        st.warning("Registros insuficientes para treinar o modelo. Aumente o slider de amostras.")
    else:
        col_t1, col_t2 = st.columns(2)

        labels = ["positive", "neutral", "negative"]
        cm_t = confusion_matrix(df_ml["sentimento_esperado"], df_ml["pred_tfidf"], labels=labels)
        fig_cm_t = px.imshow(
            cm_t, x=labels, y=labels,
            color_continuous_scale="Blues",
            labels={"x": "Predito", "y": "Real"},
            title="Matriz de Confusão TF-IDF + LR",
            text_auto=True,
        )
        fig_cm_t.update_layout(paper_bgcolor="#0f0f1a", font_color="#c0c0e0")
        col_t1.plotly_chart(fig_cm_t, use_container_width=True)

        acc_t = accuracy_score(df_ml["sentimento_esperado"], df_ml["pred_tfidf"])
        f1_t  = f1_score(df_ml["sentimento_esperado"], df_ml["pred_tfidf"], average="macro", labels=labels, zero_division=0)
        col_t2.metric("Acurácia TF-IDF + LR", f"{acc_t:.2%}")
        col_t2.metric("F1-macro TF-IDF + LR", f"{f1_t:.2%}")

        # Top features
        try:
            pipe_fit = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
                ("clf",   LogisticRegression(max_iter=1000, random_state=42))
            ])
            pipe_fit.fit(df_ml["_texto"], df_ml["sentimento_esperado"])
            features = pipe_fit.named_steps["tfidf"].get_feature_names_out()
            coefs    = pipe_fit.named_steps["clf"].coef_
            classes  = pipe_fit.named_steps["clf"].classes_

            fig_feat_list = []
            for idx, cls in enumerate(classes):
                top_idx = coefs[idx].argsort()[-15:][::-1]
                fig_feat_list.append(pd.DataFrame({"token": features[top_idx], "coef": coefs[idx][top_idx], "classe": cls}))
            df_feat = pd.concat(fig_feat_list)

            fig_feat = px.bar(
                df_feat, x="coef", y="token", color="classe",
                facet_col="classe",
                orientation="h",
                color_discrete_map={"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"},
                title="Top tokens por classe (TF-IDF + LR)",
            )
            fig_feat.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", font_color="#c0c0e0", showlegend=False)
            st.plotly_chart(fig_feat, use_container_width=True)
        except Exception:
            pass

# ═══════════════════════════════════════════════════════
#  SEÇÃO 8 · Comparação de Modelos
# ═══════════════════════════════════════════════════════
if usar_vader and usar_tfidf and col_texto and "sentimento_esperado" in df_raw.columns:
    st.markdown('<div class="secao-header">8 · Comparação de Modelos</div>', unsafe_allow_html=True)

    labels = ["positive", "neutral", "negative"]
    rows = []

    if "df_vader" in dir() and df_vader is not None:
        mask = df_vader["pred_vader"].notna() & df_vader["sentimento_esperado"].notna()
        if mask.sum() > 0:
            yt, yp = df_vader.loc[mask, "sentimento_esperado"], df_vader.loc[mask, "pred_vader"]
            rows.append({
                "Modelo": "VADER",
                "Acurácia": accuracy_score(yt, yp),
                "F1-macro": f1_score(yt, yp, average="macro", labels=labels, zero_division=0),
                "F1-neg":   f1_score(yt, yp, labels=["negative"], average="micro", zero_division=0),
                "F1-pos":   f1_score(yt, yp, labels=["positive"], average="micro", zero_division=0),
            })

    if "df_ml" in dir() and df_ml is not None:
        yt, yp = df_ml["sentimento_esperado"], df_ml["pred_tfidf"]
        rows.append({
            "Modelo": "TF-IDF + LR",
            "Acurácia": accuracy_score(yt, yp),
            "F1-macro": f1_score(yt, yp, average="macro", labels=labels, zero_division=0),
            "F1-neg":   f1_score(yt, yp, labels=["negative"], average="micro", zero_division=0),
            "F1-pos":   f1_score(yt, yp, labels=["positive"], average="micro", zero_division=0),
        })

    if rows:
        df_comp = pd.DataFrame(rows).set_index("Modelo")
        df_comp_pct = df_comp.map(lambda v: f"{v:.2%}")
        st.dataframe(df_comp_pct, use_container_width=True)

        df_melt = df_comp.reset_index().melt(id_vars="Modelo", var_name="Métrica", value_name="Score")
        fig_comp = px.bar(
            df_melt, x="Métrica", y="Score", color="Modelo",
            barmode="group",
            color_discrete_sequence=["#a78bfa", "#60a5fa"],
            title="Comparação de desempenho dos modelos",
            labels={"Score": "Score", "Métrica": ""},
        )
        fig_comp.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", font_color="#c0c0e0", yaxis_tickformat=".0%")
        st.plotly_chart(fig_comp, use_container_width=True)



st.markdown("<br><br><center style='color:#444;font-size:.8rem;'>Dashboard gerado por Claude · Anthropic</center>", unsafe_allow_html=True)
