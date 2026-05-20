import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Predição de Obesidade", layout="wide", page_icon="🏥")

# Carregamento do modelo (com cache para ser instantâneo)
@st.cache_resource(show_spinner="Carregando modelo preditivo...")
def carregar_modelo():
    try:
        return joblib.load('modelo_obesidade_campeao.joblib')
    except Exception as e:
        st.error(f"Erro ao carregar o modelo. Verifique a versão do scikit-learn. Detalhes: {e}")
        return None

modelo = carregar_modelo()

# Dicionários de Tradução para Interface e Gráficos
dict_resultados = {
    0: 'Abaixo do Peso', 1: 'Peso Normal', 2: 'Sobrepeso Nível I', 
    3: 'Sobrepeso Nível II', 4: 'Obesidade Tipo I', 
    5: 'Obesidade Tipo II', 6: 'Obesidade Tipo III'
}

target_translation = {
    "Insufficient_Weight": "Abaixo do Peso", "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso Nível I", "Overweight_Level_II": "Sobrepeso Nível II",
    "Obesity_Type_I": "Obesidade Tipo I", "Obesity_Type_II": "Obesidade Tipo II",
    "Obesity_Type_III": "Obesidade Tipo III"
}

# ==========================================
# 2. BARRA LATERAL (MENU DE PREENCHIMENTO)
# ==========================================
st.sidebar.header("📋 Dados do Paciente")
st.sidebar.markdown("Preencha os campos para análise:")

gender_pt = st.sidebar.selectbox("Gênero", ["Feminino", "Masculino"])
age = st.sidebar.number_input("Idade", 10, 100, 25)
height = st.sidebar.number_input("Altura (m)", 1.0, 2.5, 1.70, step=0.01)
weight = st.sidebar.number_input("Peso (kg)", 30.0, 250.0, 70.0, step=0.1)

family_history_pt = st.sidebar.selectbox("Histórico familiar de obesidade?", ["Sim", "Não"])
favc_pt = st.sidebar.selectbox("Consome fast-food/doces?", ["Sim", "Não"])
fcvc_pt = st.sidebar.selectbox("Consumo de vegetais", ["Raramente ou nunca", "Em algumas refeições", "Em todas as refeições"], index=1)
ncp = st.sidebar.slider("Refeições principais por dia", 1, 4, 3)

caec_pt = st.sidebar.selectbox("Come entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
smoke_pt = st.sidebar.selectbox("Fumante?", ["Sim", "Não"])
ch2o_pt = st.sidebar.selectbox("Consumo de água diário", ["Menos de 1 litro", "De 1 a 2 litros", "Mais de 2 litros"], index=1)

scc_pt = st.sidebar.selectbox("Monitora calorias?", ["Sim", "Não"])
faf_pt = st.sidebar.selectbox("Atividade física", ["Nenhuma", "1-2 vezes na semana", "3-4 vezes na semana", "5+ vezes na semana"], index=1)
tue = st.sidebar.slider("Horas diárias em telas", 0, 2, 1)
calc_pt = st.sidebar.selectbox("Consumo de álcool", ["Não", "Às vezes", "Frequentemente", "Sempre"])
mtrans_pt = st.sidebar.selectbox("Transporte principal", ["Transporte Público", "Automóvel", "Caminhada", "Motocicleta", "Bicicleta"])

# ==========================================
# 3. TELA PRINCIPAL (RESULTADOS E GRÁFICOS)
# ==========================================
st.title("🏥 Sistema de Diagnóstico de Obesidade")
st.markdown("Plataforma de triagem preditiva e visão analítica baseada em Machine Learning.")
st.divider()

if st.sidebar.button("🧠 Realizar Predição", type="primary", use_container_width=True):
    if modelo is not None:
        # Mapeamentos para o Algoritmo
        map_genero = {"Feminino": "Female", "Masculino": "Male"}
        map_sim_nao = {"Sim": 1, "Não": 0}
        map_freq = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
        map_transporte = {"Transporte Público": "Public_Transportation", "Automóvel": "Automobile", "Caminhada": "Walking", "Motocicleta": "Motorbike", "Bicicleta": "Bike"}
        map_agua = {"Menos de 1 litro": 1, "De 1 a 2 litros": 2, "Mais de 2 litros": 3}
        map_atividade = {"Nenhuma": 0, "1-2 vezes na semana": 1, "3-4 vezes na semana": 2, "5+ vezes na semana": 3}
        map_vegetais = {"Raramente ou nunca": 1, "Em algumas refeições": 2, "Em todas as refeições": 3}
        
        dados = {
            'Gender': map_genero[gender_pt], 'Age': age, 'Height': height, 'Weight': weight,
            'family_history': map_sim_nao[family_history_pt], 'FAVC': map_sim_nao[favc_pt],
            'FCVC': map_vegetais[fcvc_pt], 'NCP': ncp, 'CAEC': map_freq[caec_pt],
            'SMOKE': map_sim_nao[smoke_pt], 'CH2O': map_agua[ch2o_pt], 'SCC': map_sim_nao[scc_pt],
            'FAF': map_atividade[faf_pt], 'TUE': tue, 'CALC': map_freq[calc_pt], 'MTRANS': map_transporte[mtrans_pt]
        }
        
        df_input = pd.DataFrame([dados])
        df_input['BMI'] = df_input['Weight'] / (df_input['Height'] ** 2) # O Segredo do nosso modelo!
        
        ordem_colunas = ['Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'BMI']
        df_input = df_input[ordem_colunas]

        try:
            predicao_bruta = modelo.predict(df_input)[0] 
            grau, resultado_legivel = dict_resultados.get(predicao_bruta, (1, 'Peso Normal'))
            
            st.success("Análise preditiva concluída com sucesso!")
            
            c1, c2 = st.columns(2)
            c1.metric("IMC Calculado", f"{df_input['BMI'].iloc[0]:.2f} kg/m²")
            c2.metric("Diagnóstico da Inteligência Artificial", resultado_legivel)
            
            if grau >= 4:
                st.error("🚨 **Alerta Clínico:** Padrão indica Obesidade. Recomenda-se acompanhamento médico.")
            elif grau >= 2:
                st.warning("⚠️ **Atenção:** Padrão indica Sobrepeso. Recomendado monitoramento.")
            else:
                st.info("✅ **Padrão Normal:** O paciente não apresenta quadro de obesidade.")
                
        except Exception as e:
            st.error(f"Erro na execução preditiva: {e}")

# ==========================================
# 4. VISÃO ANALÍTICA (PLOTLY)
# ==========================================
st.divider()
st.subheader("📊 Visão Analítica de Negócio")

try:
    # Lendo os dados para os gráficos
    df_view = pd.read_csv('Obesity.csv')
    df_view['Obesity'] = df_view['Obesity'].map(target_translation)
    df_view['Gender'] = df_view['Gender'].map({"Male" : "Masculino", "Female" : "Feminino"})
    
    ordem_clinica = [
        "Abaixo do Peso", "Peso Normal", "Sobrepeso Nível I", 
        "Sobrepeso Nível II", "Obesidade Tipo I", "Obesidade Tipo II", "Obesidade Tipo III"
    ]
    
    colA, colB = st.columns(2)
    
    with colA:
        fig1 = px.histogram(df_view, x="Obesity", color="Gender", title="Distribuição de Diagnósticos por Gênero", color_discrete_sequence=['#3b82f6', '#ec4899'])
        fig1.update_xaxes(categoryorder='array', categoryarray=ordem_clinica, title="Diagnóstico")
        fig1.update_yaxes(title="Número de Pacientes")
        st.plotly_chart(fig1, use_container_width=True)
        
    with colB:
        fig2 = px.box(df_view, x="Obesity", y="Age", title="Idade vs Diagnóstico", color="Obesity")
        fig2.update_xaxes(categoryorder='array', categoryarray=ordem_clinica, title="Diagnóstico")
        fig2.update_yaxes(title="Idade (anos)")
        st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.warning("Para visualizar os gráficos analíticos, certifique-se de que o arquivo 'Obesity.csv' está na mesma pasta do projeto.")
