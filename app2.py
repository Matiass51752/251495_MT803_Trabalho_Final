# app_cafe.py  ─ Dashboard de Vendas do Café
# ------------------------------------------------------------
# Requisitos: streamlit, streamlit-option-menu, pandas, plotly
# Arquivos necessários: dirty_cafe_sales_cleaned_clusters.csv
# ------------------------------------------------------------

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from scipy import stats
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

# ─── 1) CONFIGURAÇÃO DA PÁGINA ────────────────────────────────
st.set_page_config(
    page_title="PC315 Coffee Insights",
    page_icon="☕",
    layout="wide"
)

# ─── 2) CSS GLOBAL ────────────────────────────────────────────
style_css = """
<style>

/* 0) Remover fundo cinza padrão da sidebar */
[data-testid="stSidebar"] > div:first-child {
    background-color: #F3ECE0 !important;  /* mesmo bege claro do menu */
    padding: 0;                            /* remove espaçamento extra */
}

/* 1) Tabela de inconsistências */
#inconsistencias thead th {
    background-color: #6F4E37;  /* marrom escuro */
    color: white;
    font-weight: 600;
}
#inconsistencias tbody tr:nth-of-type(odd) {
    background-color: #F3ECE0;  /* bege claro */
}
#inconsistencias {
    width: 100%;
    font-size: 0.9rem;
}

/* 2) Cabeçalhos principais */
h1, h2, h3, h4 {
    color: #4A3726;  /* marrom médio */
}

/* 3) Texto padrão */
body, p, li {
    color: #333333;  /* cinza escuro para legibilidade */
    background-color: #FFFFFF;
}

/* 4) Cards e info boxes */
.stAlert, .stSuccess, .stInfo {
    border-left: 4px solid #6F4E37;
}
</style>
"""
st.markdown(style_css, unsafe_allow_html=True)

# ─── 3) SIDEBAR NAVIGAÇÃO ────────────────────────────────────
with st.sidebar:
    #logo = Image.open("pc315.png")
    st.image("pc315.png", width=250) 
    sidebar_style = {
        "container": {"padding": "0!important", "background-color": "#F3ECE0"},
        "icon": {"color": "#4A3726", "font-size": "1.2rem"},
        "nav-link": {
            "font-size": "1rem", "text-align": "left", "margin": "0px", "padding": "10px",
            "--hover-color": "#D9C9B8",
            "border-radius": "8px", "color": "#4A3726"
        },
        "nav-link-selected": {
            "background-color": "#6F4E37",
            "color": "#FFFFFF",
            "font-weight": "600"
        },
        "icon-selected": {"color": "#FFFFFF"},
        "menu-title": {
            "font-size": "1.2rem", "font-weight": "700",
            "color": "#4A3726", "margin-bottom": "1rem"
        }
    }

    selected_page = option_menu(
        menu_title="Navegação",
        options=[
            "Visão Geral",
            "Qualidade dos Dados",
            "Segmentos de Clientes",
            "Receita & Oportunidades",
            "Simulação"
        ],
        icons=[
            "house-door-fill",
            "graph-up",
            "people-fill",
            "cash-stack",
            "activity"
        ],
        menu_icon="cast",
        default_index=0,
        styles=sidebar_style
    )

# … resto do seu código …

# 3 ─ CARREGA DADOS ----------------------------------------------------------
@st.cache_resource
def load_clean_data():
    path = "dirty_cafe_sales_cleaned_clusters.csv"
    return pd.read_csv(path)

df = load_clean_data()

# ─── 0)  GARANTIR COLUNA Month  (faça ISSO antes de df0/df1) ─────────────
if "Month" not in df.columns:
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    df["Month"] = df["Transaction Date"].dt.month

# ─── 1)  Agora sim, filtre os clusters com Month já existente ────────────
df0 = df[df["Cluster"] == 0]
df1 = df[df["Cluster"] == 1]


df_orig = pd.read_csv("dirty_cafe_sales.csv")  # Carrega dados originais para referência


# 2. LIMPEZA BÁSICA ----------------------------------------------------------
invalid_tokens = ['ERROR', 'UNKNOWN', 'None', 'none', 'NONE', ' ', '', np.nan]
df_orig.replace(invalid_tokens, np.nan, inplace=True)

# 3. CRIAR COLUNA Quarter ANTES de USAR -------------------------------------
df_orig['Transaction Date'] = pd.to_datetime(df_orig['Transaction Date'], errors='coerce')
df_orig['Quarter'] = df_orig['Transaction Date'].dt.quarter          # <<<<< garante existência

# ... resto do tratamento ...

# 4. SEQUÊNCIA DE LISTAS para imputação
numeric_cols  = ['Quantity', 'Price Per Unit', 'Total Spent']
numeric_full  = numeric_cols + ['AvgSpendingPerItem', 'Quarter']

# 5. CONTAR TOKENS INVÁLIDOS (usa df_orig)
tokens_count = int(
    sum(df_orig[col].isin(invalid_tokens).sum() for col in df_orig.columns)
)

# 6. Nº de linhas removidas ao descartar Item ausente
rows_removed = int((df_orig['Item'].isna()).sum())          # antes do dropna
df.dropna(subset=['Item'], inplace=True)



cluster_stats = (
    df.groupby('Cluster')
      .agg(Ticket_Médio   = ('Total Spent', 'mean'),
           Receita_Total  = ('Total Spent', 'sum'),
           Volume_Itens   = ('Quantity',    'sum'))   # ❶ novo campo
      .reset_index()
)

# Paleta opcional (teal e laranja, por exemplo)
cluster_palette = {0: "#14b8a6", 1: "#f97316"}  # ajuste às suas cores


# ---------- números de base por cluster ----------
base_stats = (
    df.groupby('Cluster')
      .agg(Vendas=('Transaction ID', 'count'),
           Receita_Atual=('Total Spent', 'sum'))
      .reset_index()
)

# 4 ─ PÁGINAS ---------------------------------------------------------------

# --- 0. VISÃO GERAL ---------------------------------------------------------
# --- VISÃO GERAL ------------------------------------------------------------
if selected_page == "Visão Geral":


    # 1) Logo + título
    #st.image("logo_pc315.png", width=100)  # substitua pelo caminho do logo se tiver
    st.title("PC315 – Premium Coffee 315")
    st.subheader("Dashboard de Vendas & Insights 2023 → Estratégias 2025")

    st.markdown("---")

    st.subheader("Quem somos?")

    st.image("premium_coffee_fachada.png", width=450)
    # 2) Quem somos?
    st.markdown(
        """
        A **PC315 – Premium Coffee 315** é uma cafeteria artesanal em Campinas,  
        fundada em 2019. Servimos cafés, sanduíches artesanais  
        e saladas frescas, tudo preparado na hora, com ingredientes selecionados.
        """
    )

    st.subheader("Objetivo deste Dashboard:")

    # 3) Objetivo do Dashboard
    st.markdown(
        """ 
        - **Entender** o comportamento de compra dos clientes via “agrupamento” (clusters)  
        - **Monitorar** a qualidade dos dados e principais KPIs de vendas  
        - **Identificar** oportunidades de receita e definir estratégias de fidelização  
        - **Simular** o impacto financeiro de promoções antes de implementá-las
        """
    )

    st.subheader("O que você vai encontrar:")

    # 4) Sumário Executivo
    st.markdown(
        """  
        1. **Qualidade dos Dados:** anomalias, faltantes e correções aplicadas  
        2. **Segmentos de Clientes:** perfis “Pega-e-Leva” vs “Refeição Completa”  
        3. **Perfil de Consumo:** mix de produtos e métodos de pagamento  
        4. **Receita & Oportunidades:** KPIs comparativos e recomendações  
        5. **Simulação:** teste cenários de upsell e estime ganhos
        """
    )

    st.subheader("Para a Diretoria:")

    st.markdown(
    """
    Este painel une **dados reais de 2023** e **simulações de upsell** para apoiar decisões táticas.  
    O módulo de simulação mostra, em minutos:

    • Quanto o ticket médio pode crescer ao incluir ofertas de snack ou combos.  
    • Qual o impacto financeiro de diferentes faixas de adesão (*uptake*) — do cenário conservador ao otimista.  
    • O efeito dessas ações ao longo dos meses, permitindo prever sazonalidade e fluxo de caixa.  
    

    Ajuste preços, custos e níveis de adesão conforme a realidade de cada unidade
    e veja, em tempo real, o potencial de incremento de receita e margem antes de investir.
    """
)
#• Como a qualificação de dados (redução de “Unknown”) melhora a precisão das projeções.
    # 6) Navegação rápida
    st.markdown(
        """
        > Use o menu lateral para explorar cada seção:  
        > **Qualidade dos Dados**, **Segmentos**, **Perfil de Consumo**,  
        > **Receita & Oportunidades** e **Simulação**.
        """
    )

    # 7) Rodapé / Créditos
    st.markdown(
        """
        <small>Analista: Matheus Queiroz Mota • Data: 22/06/2025 • © PC315</small>
        """,
        unsafe_allow_html=True
    )



    



# --- 1. QUALIDADE & EDA -----------------------------------------------------
# --- 3. QUALIDADE DOS DADOS -------------------------------------------------
# ---------------- QUALIDADE DOS DADOS ---------------------------------
# ---------------- QUALIDADE DOS DADOS ---------------------------------
elif selected_page == "Qualidade dos Dados":
    st.header("🔍 Qualidade dos Dados & Tratamento")

    st.subheader("📌 Por que manter dados faltantes?")
    st.markdown(
    """
    - **Location (39,6 %) e Payment Method (31,8 %)** faltantes gerariam perda significativa dos dados. Além de que substituir esses valores faltantes pelas classes mais frequentes (imputação) não é adequado, pois causaria viés em uma das classes.   
    - **“Unknown”** sinaliza falha de captura (PDV/app) e, em vez de apagar, deve ser monitorado como indicador de qualidade.
    """,
    unsafe_allow_html=True
)


    # 1 ────────────────────────────────────────────────────────────────
    # Percentual de faltantes (dados crus)
    # ───────────────── PERCENTUAL DE FALTANTES (HTML BONITO) ─────────────────
    miss_df = (
        df_orig.isna()
            .mean()
            .mul(100)
            .round(1)
            .reset_index()
            .rename(columns={'index': 'Variável', 0: 'Irregulares (%)'})
            .sort_values('Irregulares (%)', ascending=False)
    )

    # cria HTML da tabela com Styler
    html_miss = (
        miss_df
        .style
        .format({"Irregulares (%)": "{:.1f}%"})
        .hide(axis="index")
        .set_table_attributes("id='Irregulares' class='table table-striped table-hover'")
        .to_html(escape=False)
    )

    # remove o bloco <style> interno do Styler
    html_miss = re.sub(r"<style.*?</style>", "", html_miss, flags=re.S)

    st.subheader("Percentual de dados irregulares por coluna (dados brutos)",help="Dados irregulares constam os dados do tipo NaN, None, ' ', 'ERROR', etc.")
    st.markdown(
        f"""
        <style>
        /* cores e fontes da tabela de Irregulares */
        #Irregulares {{ width:100%; font-size:0.9rem; }}
        #Irregulares thead th {{ background-color:#0d9488; color:white; }}
        #Irregulares tbody tr:nth-of-type(odd) {{ background-color:#f6f6f6; }}
        </style>
        {html_miss}
        """,
        unsafe_allow_html=True
    )


    # 2 ────────────────────────────────────────────────────────────────
    # Tabela de inconsistências detectadas
    invalid_tokens = ['ERROR', 'UNKNOWN', 'None', 'none', 'NONE', ' ', '', np.nan]
    df_orig.replace(invalid_tokens, np.nan, inplace=True)
    tokens_count = int(
        sum(df_orig[col].isin(invalid_tokens).sum() for col in df_orig.columns)
    )

    # Falha na conversão numérica (ocorre durante a limpeza)
    num_coerce_na = int(df_orig[['Quantity', 'Price Per Unit', 'Total Spent']]
                        .apply(pd.to_numeric, errors='coerce')
                        .isna()
                        .sum()
                        .sum())

    # Incoerências em Total Spent (determinadas após limpeza parcial)
    qty  = pd.to_numeric(df_orig['Quantity'], errors='coerce')
    ppu  = pd.to_numeric(df_orig['Price Per Unit'], errors='coerce')
    tot  = pd.to_numeric(df_orig['Total Spent'], errors='coerce')
    diff_raw = qty * ppu - tot
    incoerentes_raw = int((diff_raw.abs() > 1e-2).sum())

    # Datas inválidas no bruto (transformadas em NaT)
    bad_dates = int(
        pd.to_datetime(df_orig['Transaction Date'], errors='coerce').isna().sum()
    )

    # Contagem de linhas removidas por Item ausente (antes do dropna)
    rows_removed = int(df_orig['Item'].isna().sum())

    # Cria a classe "Unknown" para colunas categóricas
    cat_cols = ["Location", "Payment Method"]
    irreg_cat = {c: int(df_orig[c].isna().sum()) for c in cat_cols}
    for c in cat_cols:
        df_orig[c].fillna("Unknown", inplace=True)

    # ------------------------------------------------------------------
    # Monta o DataFrame de inconsistências
    # ------------------------------------------------------------------
    inconsist = pd.DataFrame({
        "Problema": [
            "Tokens inválidos ('ERROR', 'UNKNOWN', ...)",
            "Falha na conversão numérica",
            "`Total Spent` incoerente",
            "Datas inválidas (NaT)",
            "`Item` ausente"
        ],
        "Ocorrências nos dados": [
            tokens_count,
            num_coerce_na,
            incoerentes_raw,
            bad_dates,
            rows_removed
        ],
        "Ação corretiva": [
            "Substituir por NaN",
            "Imputar mediana",
            "Recalcular Qtd × Preço",
            "Remover linhas com data inválida",
            "Descartar linhas sem 'Item'"
        ]
    })

    # ▸ adiciona Location e Payment Method faltantes
    for col in cat_cols:
        inconsist.loc[len(inconsist)] = [
            f"{col} faltante",        # Problema
            irreg_cat[col],           # Ocorrências
            "Preencher com 'Unknown'" # Ação corretiva
        ]

    # 3. Exibir tabela formatada (Styler) — permanece igual
    html_table = (
        inconsist
        .style
        .format({"Ocorrências nos dados": "{:,.0f}"})
        .hide(axis="index")
        .set_table_attributes("id='inconsistencias' class='table table-striped table-hover'")
        .to_html(escape=False)
    )
    html_table = re.sub(r"<style.*?</style>", "", html_table, flags=re.S)

    st.subheader("Inconsistências detectadas e tratamento aplicado")
    st.markdown(
        f"""
        <style>
        .table {{ width: 100%; font-size: 0.9rem; }}
        .table thead th {{ background-color:#14b8a6; color:white; }}
        .table-striped tbody tr:nth-of-type(odd) {{ background-color:#f6f6f6; }}
        </style>
        {html_table}
        """,
        unsafe_allow_html=True
    )




    # 3 ────────────────────────────────────────────────────────────────
    # Distribuição de Total Spent após tratamento
    st.markdown("### Distribuição de `Total Spent` (após limpeza)")
    fig_spent = px.histogram(
        df, x="Total Spent",           # <<< usa df limpo
        nbins=50,
        color_discrete_sequence=["#14b8a6"],
        labels={"Total Spent": "Total Spent (U$)"},
        height=350
    )
    st.plotly_chart(fig_spent, use_container_width=True)

    st.markdown("## Relação entre variáveis originais")

    st.markdown(
    """
    ### Como as variáveis se relacionam?
    Antes de treinar qualquer modelo, avaliamos **se e como** nossas métricas básicas se movem juntas.  
    - Na matriz à esquerda, vemos a correlação das variáveis **numéricas** (−1 → 1).  
    - Na matriz à direita, usamos **Cramér’s V** (0 → 1) para medir associação entre variáveis **categóricas**.  
    Isso nos mostra, por exemplo, se preço alto anda junto com maior gasto total ou se forma de pagamento tem ligação com local de consumo.
    """
)
    # ---------- prepara figuras ----------
    # 4.1 Numéricas
    numeric_original_cols = ['Quantity', 'Price Per Unit', 'Total Spent']
    corr_matrix = df[numeric_original_cols].corr()
    fig_num, ax_num = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1,
        ax=ax_num, square=True, cbar_kws={"shrink": .8}
    )
    ax_num.set_title("Correlação (Numéricas)")

    # 4.2 Categóricas (Cramér's V)
    categorical_original_cols = ['Item', 'Payment Method', 'Location']
    def cramers_v(x, y):
        ct = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(ct)[0]
        n = ct.sum().sum()
        phi2 = chi2 / n
        r, k = ct.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / max((kcorr-1), (rcorr-1)))

    n_cat = len(categorical_original_cols)
    cramer_mat = np.ones((n_cat, n_cat))
    for i in range(n_cat):
        for j in range(i+1, n_cat):
            v = cramers_v(df[categorical_original_cols[i]], df[categorical_original_cols[j]])
            cramer_mat[i, j] = cramer_mat[j, i] = v
    cramer_df = pd.DataFrame(cramer_mat, index=categorical_original_cols, columns=categorical_original_cols)

    fig_cat, ax_cat = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cramer_df, annot=True, fmt=".02f",
        cmap="viridis", vmin=0, vmax=1,
        ax=ax_cat, square=True, cbar_kws={"shrink": .8}
    )
    ax_cat.set_title("Associação (Cramér's V) – Categóricas")

    # ---------- exibe lado a lado ----------
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_num, use_container_width=True)
    with col2:
        st.pyplot(fig_cat, use_container_width=True)

    # ---------- cards de interpretação ----------
    c1, c2 = st.columns(2)
    with c1:
        st.info(
            "**Números-chave**\n\n"
            "- `Quantity` × `Total Spent`: **0,69** → vender mais itens aumenta gasto.\n"
            "- `Price Per Unit` × `Total Spent`: **0,63** → itens caros elevam total.\n"
            "- `Quantity` × `Price Per Unit`: **0,00** → preço não influi na quantidade."
        )
    with c2:
        st.info(
            "**Categóricas quase independentes**\n\n"
            "- Máximo Cramér’s V ≈ **0,03** → escolha do item, pagamento e localização\n"
            "  não estão casadas entre si.\n"
            "- Confirma que variáveis categóricas trazem sinais distintos ao modelo."
        )


    st.header("⚖️  Unknown × Informado – Location & Payment")

    # ---------- Seleção da métrica ----------
    metric = st.selectbox(
        "Escolha a métrica numérica a comparar:",
        ("Total Spent", "Quantity", "Price Per Unit"),
        index=0
    )

    # ---------- Função de teste (Mann–Whitney) ----------
    def test_unknown(df, col, metric):
        is_unk = df[col] == "Unknown"
        g_unk  = df.loc[is_unk,  metric].dropna()
        g_inf  = df.loc[~is_unk, metric].dropna()
        stat, p = stats.mannwhitneyu(g_unk, g_inf, alternative="two-sided")
        return {
            "Variável": col,
            "# Unknown": len(g_unk),
            "# Informado": len(g_inf),
            "Mediana Unknown": g_unk.median(),
            "Mediana Informado": g_inf.median(),
            "p-valor": p
        }

    results = pd.DataFrame([
        test_unknown(df, "Location",       metric),
        test_unknown(df, "Payment Method", metric)
    ])

    st.dataframe(
        results.style.format({
            "Mediana Unknown":   "U$ {:,.2f}" if metric == "Total Spent" else "{:,.2f}",
            "Mediana Informado": "U$ {:,.2f}" if metric == "Total Spent" else "{:,.2f}",
            "p-valor":           "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # ---------- Explicação para a diretoria ----------
    with st.expander("ℹ️  O que é o teste Mann–Whitney e como ler o p-valor?"):
        st.markdown(
            """
            **Teste usado**  
            *Mann–Whitney U* (não paramétrico) compara se duas amostras independentes
            têm a **mesma distribuição** sem assumir normalidade.  
            Aplicamos aos grupos **“Unknown”** e **“Informado”**.

            **p-valor**  
            Probabilidade de observar uma diferença **igual ou maior** que a dos dados
            **se** as duas distribuições fossem, na verdade, iguais.  
            * Regra prática: **p < 0,05** → diferença estatisticamente significativa.  
            * p ≥ 0,05 → não há evidência de diferença; 
            * Mantemos “Unknown” como flag de qualidade,
              pois apagar reduziria ~40 % da base e não traria ganho analítico.
            """
        )        



elif selected_page == "Segmentos de Clientes":

    # ---------- Expander • Metodologia de clusterização ----------
    with st.expander("🔍 Como construímos os clusters? (clique para ver)", expanded=False):
        st.markdown(
            """
            **Algoritmo:** `KPrototypes` (mistura K-Means + K-Modes)  
            • Lida bem com variáveis **numéricas** e **categóricas** ao mesmo tempo.  
            • Avaliamos k de 2 a 8 usando o índice de *silhouette* (quanto maior, melhor).  

            | k | Silhouette |
            |---|------------|
            | 2 | **0.554**  |
            | 3 | 0.444      |
            | 4 | 0.403      |
            | 5 | 0.391      |
            | 6 | 0.373      |
            | 7 | 0.348      |
            | 8 | 0.377      |

            **Resultado:** melhor separação em **2 clusters**  
            1. **Cluster 0 – Pega-e-Leva** (alto volume de bebidas; ticket baixo)  
            2. **Cluster 1 – Refeição Completa** (menor volume; receita e ticket altos)
            """
        )

        # gráfico rápido Silhouette vs k
        k_vals   = [2,3,4,5,6,7,8]
        sil_vals = [0.554,0.444,0.403,0.391,0.373,0.348,0.377]
        fig_sil = px.bar(
            x=k_vals, y=sil_vals, text=[f"{v:.3f}" for v in sil_vals],
            labels={"x":"k","y":"Silhouette"},
            title="Silhouette por k"
        )
        fig_sil.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig_sil, use_container_width=True)


    st.header("🍰 Perfil de Consumo")

    # --- Top 5 itens vendidos ---
    st.markdown("##### Top 5 itens vendidos por cluster")
    top5 = (
        df.groupby(["Cluster", "Item"])
          .size()
          .reset_index(name="Count")
          .sort_values(["Cluster", "Count"], ascending=[True, False])
          .groupby("Cluster").head(5)
    )
    fig_items = px.bar(
        top5, x="Cluster", y="Count", color="Item", barmode="group",
        facet_col="Cluster",
        category_orders={"Cluster": sorted(df['Cluster'].unique())},
        height=450
    )
    st.plotly_chart(fig_items, use_container_width=True)

    # descrição em texto
    st.markdown(
        """
        **Padrão de consumo de itens**  
        - **Cluster 0 (Pega-e-Leva):** domina o ranking *Coffee, Cookie, Tea, Juice, Cake* – vendas rápidas de bebidas e lanches individuais.  
        - **Cluster 1 (Refeição Completa):** lideram *Sandwich, Salad, Smoothie, Juice, Coffee* – refeições compostas e sobremesas.
        """
    )

    # --- Método de pagamento ---
    st.markdown("##### Método de pagamento por cluster")
    pay_counts = (
        df.groupby(["Cluster", "Payment Method"])
          .size()
          .reset_index(name="Count")
    )
    fig_pay = px.bar(
        pay_counts, x="Cluster", y="Count", color="Payment Method",
        barmode="group", height=350,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_pay, use_container_width=True)

    st.markdown(
        """
        **Método de pagamento predominante (desconsiderando "Unknown")**  
        - **Cluster 0:** maior uso de *Digital Wallet* e *Credit Card*, seguido de *Cash*.  
        - **Cluster 1:** padrão similar, mas com participação relativa maior de *Cash*.
        """
    )

    # --- Local de consumo ---
    st.markdown("##### Local de consumo por cluster")
    loc_counts = (
        df.groupby(["Cluster", "Location"])
          .size()
          .reset_index(name="Count")
    )
    fig_loc = px.bar(
        loc_counts, x="Cluster", y="Count", color="Location",
        barmode="group", height=350,
        color_discrete_sequence=["#14b8a6", "#f97316", "#999999"]
    )
    fig_loc.update_layout(legend_title_text="Local")
    st.plotly_chart(fig_loc, use_container_width=True)

    st.markdown(
        """
        **Distribuição por local (desconsiderando "Unknown")**  
        - **Cluster 0:** maioria das vendas em *Takeaway* seguido de *In-store*.  
        - **Cluster 1:** quantidade maior em *In-store* seguido de *Takeaway*.
        """
    )



# --- 3. PERFIL DE CONSUMO ---------------------------------------------------
#elif selected_page == "Perfil de Consumo":


# --- 4. RECEITA & OPORTUNIDADES --------------------------------------------
elif selected_page == "Receita & Oportunidades":
    st.header("💰 Receita & Oportunidades")

    # --- paleta única de azuis para todos os gráficos ---
    cluster_palette = {
        0: "#6baed6",  # azul claro
        1: "#2171b5"   # azul escuro
    }
    category_palette = {
        "Bebida": cluster_palette[0],
        "Comida": cluster_palette[1]
    }

    # --- mapeamento das opções de rádio para o ID do cluster ---
    radio_map = {
        "Todos": None,
        "Cluster 0 (Pega-e-Leva)": 0,
        "Cluster 1 (Refeição Completa)": 1
    }
    selected = st.radio("Mostrar dados de:", list(radio_map.keys()), horizontal=True)
    cid = radio_map[selected]  # None, 0 ou 1

    # --- extrai stats de acordo com a escolha ---
    if cid is None:
        stats_view  = cluster_stats.copy()
        ticket_val  = stats_view["Ticket_Médio"].mean()
        receita_val = stats_view["Receita_Total"].sum()
        vol_val     = stats_view["Volume_Itens"].sum()
    else:
        stats_view  = cluster_stats[cluster_stats["Cluster"] == cid]
        ticket_val  = stats_view["Ticket_Médio"].iat[0]
        receita_val = stats_view["Receita_Total"].iat[0]
        vol_val     = stats_view["Volume_Itens"].iat[0]

    # --- KPI cards ------------------------------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Ticket médio",    f"U$ {ticket_val:,.2f}")
    k2.metric("Receita total",   f"U$ {receita_val:,.0f}")
    k3.metric("Volume de itens", f"{vol_val:,.0f}")

    st.markdown("---")

    # --- gráficos lado a lado -------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ticket Médio (U$)")
        fig_ticket = px.bar(
            stats_view,
            x="Cluster", y="Ticket_Médio",
            color="Cluster", color_discrete_map=cluster_palette,
            text_auto=".2f", height=320
        ).update_layout(showlegend=False)
        st.plotly_chart(fig_ticket, use_container_width=True)

    with col2:
        st.subheader("Receita Total (U$)")
        fig_rev = px.bar(
            stats_view,
            x="Cluster", y="Receita_Total",
            color="Cluster", color_discrete_map=cluster_palette,
            text_auto=",.0f", height=320
        ).update_layout(showlegend=False)
        st.plotly_chart(fig_rev, use_container_width=True)

    # --- Mix de produtos por receita -----------------
    st.subheader("Mix de produtos por receita")
    treemap_df = df if cid is None else df[df["Cluster"] == cid]
    fig_tree = px.treemap(
        treemap_df,
        path=['Cluster', 'Item'],
        values='Total Spent',
        color='Cluster', color_discrete_map=cluster_palette,
        height=450
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # --- Volume de Bebidas vs. Comidas ---------------
    st.subheader("Volume de Bebidas vs. Comidas")
    bebidas = ["Coffee", "Tea", "Juice", "Smoothie"]
    df["Categoria"] = np.where(df["Item"].isin(bebidas), "Bebida", "Comida")
    vol_cat = (
        (df if cid is None else df[df["Cluster"] == cid])
        .groupby(["Cluster", "Categoria"])["Quantity"]
        .sum()
        .reset_index()
    )
    fig_vol = px.bar(
        vol_cat,
        x="Cluster", y="Quantity",
        color="Categoria", barmode="stack",
        text_auto=True,
        color_discrete_map=category_palette,
        labels={"Quantity": "Volume"},
        height=320
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # --- Receita mensal (área empilhada) --------------
    st.subheader("Receita ao longo de 2023")
    if 'Month' not in df.columns:
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
        df['Month'] = df['Transaction Date'].dt.month

    df_month = (
        df.dropna(subset=['Month'])
          .groupby(['Cluster', 'Month'])['Total Spent']
          .sum()
          .reset_index()
          .sort_values('Month')
    )
    if cid is not None:
        df_month = df_month[df_month["Cluster"] == cid]

    fig_area = px.area(
        df_month,
        x="Month", y="Total Spent",
        color="Cluster", color_discrete_map=cluster_palette,
        labels={"Month": "Mês", "Total Spent": "Receita"},
        height=350
    )
    fig_area.update_traces(stackgroup="one")
    st.plotly_chart(fig_area, use_container_width=True)

    # --- Recomendações -------------------------------
    st.markdown("### Recomendações")
    st.success(
        """
        **Cluster 0 – Pega-e-Leva**  
        • Cartão “10 cafés = 1 grátis”  
        • Combo café + snack para subir ticket

        **Cluster 1 – Refeição Completa**  
        • Combo premium (sanduíche + salada + bebida)  
        • Programa de pontos / sobremesa de cortesia
        """
    )


# =============================================================
#  📈  SIMULADOR DE UPSELL POR CLUSTER  – duas opções
# =============================================================
elif selected_page == "Simulação":
    # ------------------------------------------------------------------
# EXPLICADOR DA SIMULAÇÃO
# ------------------------------------------------------------------
    with st.expander("❔  O que esta simulação faz e de onde vêm os números?", expanded=False):

        st.markdown(
            """
            ### 1. Fonte de dados  
            • Base: **Base de Dados de 2023** já tratada e segmentada em 2 clusters  
            • Total de transações usadas: **{:,}**  
            • Variáveis-chave: *Item, Quantity, Price Per Unit, Total Spent, Month, Cluster*

            ### 2. Regras de elegibilidade  
            | Cluster | Critério para entrar na simulação | Oferta de Upsell |
            |---------|-----------------------------------|------------------|
            | **0 – Pega-e-Leva** | Venda contém pelo menos 1 bebida (Coffee / Tea / Juice / Smoothie) | **Snack** unitário (cookie, brownie…) |
            | **1 – Refeição Completa** | Venda contém Sandwich **ou** Salad | **Combo** (bebida + sobremesa) |

            ### 3. Fórmula financeira  
            ```text
            Lucro extra
                = (Preço da oferta – Custo) × Vendas elegíveis × Uptake
            ```
            *Uptake* = % estimado de clientes que aceitarão a oferta  
            (ajustável no **slider** de cada cluster).

            ### 4. Como o Uptake é aplicado  
            • A cada mudança no slider geramos uma **amostra aleatória** do tamanho  
            `round(vendas_elegíveis × uptake)` para refletir aceitação parcial.  
            • Isso produz uma **projeção** — não altera o dataset original.  
            • Use diferentes cenários para ver limites **conservador (≥ 20 %)**  
            até **agressivo (≤ 80 %)**.

            ### 5. Gráficos e métricas exibidos  
            • **KPIs instantâneos** de Receita & Lucro extra (caixas verdes)  
            • **Ticket médio projetado por mês** (linha preta) vs. histórico original (cinza)  
            • **Barra de Lucro Extra** para comparar Cluster 0 × Cluster 1  

            ---
            **Interprete assim:** se a área preta ultrapassa a linha cinza, significa que — dado  
            o uptake selecionado — a política de upsell aumenta o ticket médio naquele mês.
            """.format(len(df))  # df já está carregado no seu script
        )

    
    st.header("📈 Simulador de Estratégias de Upsell")


    # --- escolha do cenário -------------------------------
    cenário = st.selectbox(
        "💡 O que você quer testar?",
        ("Upsell de Snacks (Cluster 0)", "Combo Refeição Plus (Cluster 1)")
    )

    # listas–chave
    bebidas   = ["Coffee", "Tea", "Juice", "Smoothie"]
    refeicoes = ["Sandwich", "Salad"]
    df0, df1  = df[df["Cluster"] == 0], df[df["Cluster"] == 1]

    # métricas de base
    base_ticket0 = df0["Total Spent"].mean()
    base_ticket1 = df1["Total Spent"].mean()

    # ======================================================
    #  ⓐ  UPSALE PROGRESSIVO DE SNACKS – Cluster 0
    # ======================================================
    if cenário.startswith("Upsell"):
        st.subheader("Cluster 0 – Upsell de Snacks")

        st.info(
            "Calculamos sobre **vendas que já incluem bebida**. "
            "Você escolhe **quantos snacks** oferecer, preço, custo e taxa de aceitação."
        )

        elig0      = df0[df0["Item"].isin(bebidas)]
        n_elig0    = len(elig0)

        cols = st.columns(4)
        n_snack   = cols[0].number_input("Snacks por upsell", 1, 3, 1, key="sn")
        snack_p   = cols[1].number_input("Preço unit. U$", 2.0, 15.0, 6.0, 0.5, key="sp")
        snack_c   = cols[2].number_input("Custo unit. U$", 0.5, 10.0, 3.0, 0.5, key="sc")
        uptake0   = cols[3].slider("Uptake %", 0, 100, 60, 5, key="su") / 100

        vend_extra0   = n_elig0 * uptake0
        receita_extra = vend_extra0 * n_snack * snack_p
        lucro_extra   = vend_extra0 * n_snack * (snack_p - snack_c)

        new_ticket0   = (df0["Total Spent"].sum() + receita_extra) / len(df0)

        k1, k2, k3 = st.columns(3)
        k1.metric("Ticket médio (antes)",  f"U$ {base_ticket0:,.2f}")
        k2.metric("Ticket médio (depois)", f"U$ {new_ticket0:,.2f}")
        k3.metric("Δ Ticket %",            f"{(new_ticket0/base_ticket0-1)*100:.1f}%")

        st.markdown("#### Receita e lucro incrementais")
        st.write(f"• Vendas elegíveis: **{n_elig0:,}**  &nbsp;&nbsp;• Uptake: **{uptake0*100:.0f}%**")
        st.markdown(
    f"""
    <div style="border-left:4px solid #28a745;
                background:#eaf7ec;
                padding:0.4rem 0.8rem;
                border-radius:4px;">
        <strong>Receita extra</strong> U$ {receita_extra:,.0f}
        &emsp;|&emsp;
        <strong>Lucro extra</strong> U$ {lucro_extra:,.0f}
    </div>
    """,
    unsafe_allow_html=True
)
    # ======================================================
    #  ⓑ  COMBO “REFEIÇÃO COMPLETA PLUS” – Cluster 1
    # ======================================================
    else:
        st.subheader("Cluster 1 – Combo “Refeição Completa Plus”")

        st.info(
            "Consideramos vendas que já incluem **sanduíche ou salada**. "
            "O combo adiciona **bebida + sobremesa** por um preço definido."
        )

        elig1      = df1[df1["Item"].isin(refeicoes)]
        n_elig1    = len(elig1)

        cols = st.columns(4)
        combo_p   = cols[0].number_input("Preço do combo U$", 4.0, 30.0, 12.0, 0.5, key="cp")
        combo_c   = cols[1].number_input("Custo do combo U$", 2.0, 20.0, 6.0, 0.5, key="cc")
        uptake1   = cols[3].slider("Uptake %", 0, 100, 40, 5, key="cu") / 100

        vend_extra1   = n_elig1 * uptake1
        receita_extra = vend_extra1 * combo_p
        lucro_extra   = vend_extra1 * (combo_p - combo_c)

        new_ticket1   = (df1["Total Spent"].sum() + receita_extra) / len(df1)

        k1, k2, k3 = st.columns(3)
        k1.metric("Ticket médio (antes)",  f"U$ {base_ticket1:,.2f}")
        k2.metric("Ticket médio (depois)", f"U$ {new_ticket1:,.2f}")
        k3.metric("Δ Ticket %",            f"{(new_ticket1/base_ticket1-1)*100:.1f}%")

        st.markdown("#### Receita e lucro incrementais")
        st.write(
    f"• Vendas elegíveis: **{n_elig1:,}**  "
    f"• Uptake: **{uptake1*100:.0f}%**"
)
        st.markdown(
    f"""
    <div style="border-left:4px solid #28a745;
                background:#eaf7ec;
                padding:0.5rem 1rem;
                border-radius:4px;">
        <strong>Receita extra:</strong> U$ {receita_extra:,.0f}
        &emsp;|&emsp;
        <strong>Lucro extra:</strong> U$ {lucro_extra:,.0f}
    </div>
    """,
    unsafe_allow_html=True
)
    # ----------------------------------------------------------
    # MOSTRAR EVOLUÇÃO MENSAL DO TICKET MÉDIO
    # ----------------------------------------------------------
    show_monthly = st.checkbox("📅 Mostrar ticket médio mês-a-mês", value=True)
    if show_monthly:

        # ─── 1) Garante coluna Month ───────────────────────────
        if "Month" not in df.columns:
            df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
            df["Month"] = df["Transaction Date"].dt.month

        # ─── 2) Seleciona qual cluster estamos simulando ───────
        if cenário.startswith("Upsell"):        # Cluster 0
            cluster_id    = 0
            df_cluster    = df0
            n_elig_func   = lambda d: d[d["Item"].isin(bebidas)].shape[0]
            extra_revenue = vend_extra0 * n_snack * snack_p   # total proj.
        else:                                   # Cluster 1
            cluster_id    = 1
            df_cluster    = df1
            n_elig_func   = lambda d: d[d["Item"].isin(refeicoes)].shape[0]
            extra_revenue = vend_extra1 * combo_p

        # ─── 3) Base mensal (cluster) ──────────────────────────
        base_month = (
            df_cluster
            .groupby("Month")
            .agg( tot_spent   = ("Total Spent",   "sum"),
                n_trans     = ("Transaction ID","count"))
            .reset_index()
        )
        base_month["Ticket Base"] = base_month["tot_spent"] / base_month["n_trans"]

        # ─── 4) Receita extra por mês (valor esperado) ─────────
        elig_by_month = (
            df_cluster
            .groupby("Month")
            .apply(n_elig_func)
            .rename("n_elig")
            .reset_index()
        )

        # taxa uptake & receita por venda vem do cenário
        if cenário.startswith("Upsell"):
            receita_venda = n_snack * snack_p
            uptake        = uptake0
        else:
            receita_venda = combo_p
            uptake        = uptake1

        elig_by_month["rec_extra"] = elig_by_month["n_elig"] * uptake * receita_venda

        # ─── 5) Ticket Projetado ───────────────────────────────
        df_month_sim = base_month.merge(elig_by_month, on="Month")
        df_month_sim["Ticket Proj"] = (
            (df_month_sim["tot_spent"] + df_month_sim["rec_extra"])
            / df_month_sim["n_trans"]
        )

        # ─── 6) Gráfico linha dupla ────────────────────────────
        fig_ticket = px.line(
            df_month_sim,
            x="Month",
            y=["Ticket Base", "Ticket Proj"],
            markers=True,
            labels={"value": "Ticket (U$)", "variable": ""},
            title="Ticket médio mensal – antes × depois"
        )
        fig_ticket.update_traces(line_width=3)
        st.plotly_chart(fig_ticket, use_container_width=True)

# --------- Rodapé -----------------------------------------------------------
st.markdown(
    "<small>© 2025 — Dashboard Cafeteria • Desenvolvido com Streamlit</small>",
    unsafe_allow_html=True
)