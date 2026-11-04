import json
from pathlib import Path
from urllib.request import urlopen

import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_resource
def carregar_modelo():
    caminho_modelo = Path(__file__).resolve().parent / "model" / "model_price"
    return mlflow.sklearn.load_model(str(caminho_modelo))

try:
    modelo = carregar_modelo()
except Exception as erro:
    st.error(f"Não foi possível carregar o modelo do MLflow: {erro}")
    st.stop()

@st.cache_data(ttl=3600)
def obter_cotacao_usd_brl():
    with urlopen("https://api.frankfurter.dev/v2/rate/usd/brl", timeout=10) as resposta:
        cotacao = json.load(resposta)

    return cotacao["rate"], cotacao["date"]

def formatar_reais(valor):
    valor_formatado = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {valor_formatado}"

st.title("Predição de Preço de Imóveis")
st.markdown(
    " - Este aplicativo estima o preço de um imóvel com base em suas características. "
    "Forneça os detalhes do imóvel na aba 'Dados do imóvel' e clique em 'Prever preço' para obter a estimativa."
)

tab_modelo, tab_dados, tab_previsao = st.tabs(["Modelo", "Dados do imóvel", "Previsão"])


with tab_dados:
    st.header("Dados do imóvel")
    exp1 = st.expander("Sobre o modelo")
    exp1.write(
        "Este modelo foi treinado com dados de imóveis de Seattle, nos Estados Unidos. "
        "Ele usa número de quartos e banheiros, áreas da casa e do terreno, quantidade de andares, "
        "proximidade da água, condição do imóvel e anos de construção e reforma para estimar o preço. "
        "Como os dados de treinamento são de Seattle, os resultados podem não representar bem imóveis "
        "de outras regiões."
    )
    st.header("Características do imóvel")

    bedrooms = st.number_input(
        "Quartos",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
    )

    bathrooms = st.number_input(
        "Banheiros",
        min_value=0.0,
        max_value=10.0,
        value=1.5,
        step=0.5,
    )

    area_living_m2 = st.number_input(
        "Área interna da casa (m²)",
        min_value=0.0,
        max_value=1500.0,
        value=124.5,
        step=5.0,
    )

    area_lot_m2 = st.number_input(
        "Área do terreno (m²)",
        min_value=0.0,
        max_value=140000.0,
        value=735.1,
        step=10.0,
    )

    area_above_m2 = st.number_input(
        "Área acima do solo (m²)",
        min_value=0.0,
        max_value=1500.0,
        value=124.5,
        step=5.0,
    )

    floors = st.number_input(
        "Andares",
        min_value=0.0,
        max_value=4.0,
        value=1.5,
        step=0.5,
    )

    opcao_waterfront = st.selectbox(
        "O imóvel fica de frente para a água?",
        options=["Não", "Sim"],
    )
    waterfront = 1 if opcao_waterfront == "Sim" else 0

    condition = st.number_input(
        "Condição geral do imóvel (1 a 5)",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

    yr_renovated = st.number_input(
        "Ano da última reforma (0 = sem registro)",
        min_value=0,
        max_value=2014,
        value=0,
        step=1,
    )

    fator_m2_para_sqft = 10.7639
    sqft_living = area_living_m2 * fator_m2_para_sqft
    sqft_lot = area_lot_m2 * fator_m2_para_sqft
    sqft_above = area_above_m2 * fator_m2_para_sqft


with tab_modelo:
    st.header("Modelo de previsão")
    st.write(
        "O modelo de previsão é um Random Forest Regressor treinado com dados de imóveis de Seattle," \
        " nos Estados Unidos. Ele utiliza as características do imóvel fornecidas na aba 'Dados do imóvel'" \
        " para estimar o preço."
        )
    pipeline = modelo.best_estimator_
    nomes = {
        "bedrooms": "Quartos", "bathrooms": "Banheiros", "floors": "Andares",
        "waterfront": "Frente para a água", "condition": "Condição do imóvel",
        "yr_renovated": "Ano da reforma", "sqft_living": "Área interna da casa",
        "sqft_lot": "Área do terreno", "sqft_above": "Área acima do solo",
    }
    importancias = pd.Series(
        pipeline.named_steps["modelo"].feature_importances_,
        index=pipeline[:-1].get_feature_names_out(),
    )
    importancias.index = [
        next((pt for en, pt in nomes.items()
              if nome == en or nome.startswith(f"{en}_")), nome)
        for nome in importancias.index
    ]
    importancias = importancias.groupby(level=0).sum().nlargest(10).sort_values()
    figura, eixo = plt.subplots()
    eixo.barh(importancias.index, importancias.values)
    eixo.set_xlabel("Importância")
    eixo.set_title("10 características mais importantes")
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

with tab_previsao:
    st.header("Previsão")

    dados_imovel = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "condition": condition,
        "sqft_above": sqft_above,
        "yr_renovated": yr_renovated,
    }])

    if st.button("Prever preço", type="primary"):
        if area_lot_m2 <= 5:
            st.error("A área do terreno deve ser maior que 5 m².")
        elif area_living_m2 <= 5:
            st.error("A área interna da casa deve ser maior que 5 m².")
        else:
            st.subheader("Resultado da previsão")
            st.write("Com base nas características fornecidas, o preço estimado do imóvel é:")
            
        preco_previsto = modelo.predict(dados_imovel)[0]
        st.metric(
            label="Preço estimado em dólares",
            value=f"US$ {preco_previsto:,.2f}",
        )

        try:
            cotacao_usd_brl, data_cotacao = obter_cotacao_usd_brl()
        except Exception:
            st.warning("Não foi possível consultar a cotação USD/BRL agora.")
        else:
            preco_previsto_brl = preco_previsto * cotacao_usd_brl

            st.metric(
                label="Equivalente aproximado em reais",
                value=formatar_reais(preco_previsto_brl),
            )
            st.caption(
                f"Cotação de referência: 1 USD = {formatar_reais(cotacao_usd_brl)} "
                f"(atualizada em {data_cotacao})."
            )
