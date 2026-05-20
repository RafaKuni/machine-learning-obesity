import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Predição de Obesidade", layout="centered", page_icon="🏥")

st.title("🏥 Classificador de Obesidade")
st.markdown("Preencha os dados abaixo para verificar se o padrão indica risco de obesidade.")
st.divider()

# ==========================================
# 2. CARREGAMENTO DO MODELO (CACHE)
# ==========================================
@st.cache_resource(show_spinner="Carregando Inteligência Artificial...")
def carregar_modelo():
    try:
        # Carrega o modelo treinado (certifique-se de que o arquivo .joblib está no GitHub)
        return joblib.load('modelo_obesidade_campeao.joblib')
    except Exception as e:
        st.error(f"Erro ao carregar o modelo. Detalhes: {e}")
        return None

modelo = carregar_modelo()

# ==========================================
# 3. INTERFACE DO FORMULÁRIO (2 COLUNAS)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    gender_pt = st.selectbox("Gênero", ["Feminino", "Masculino"])
    age = st.number_input("Idade", min_value=10, max_value=100, value=25)
    height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)
    family_history_pt = st.checkbox("Histórico familiar de obesidade")
    favc_pt = st.checkbox("Consumo frequente de fast-food/doces")
    fcvc_pt = st.selectbox("Consumo de vegetais", ["Raramente ou nunca", "Em algumas refeições", "Em todas as refeições"], index=1)
    ncp = st.slider("Refeições principais por dia", 1, 4, 3)

with col2:
    caec_pt = st.selectbox("Come entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    scc_pt = st.checkbox("Monitora calorias diárias")
    ch2o_pt = st.selectbox("Consumo de água", ["Menos de 1 litro", "De 1 a 2 litros", "Mais de 2 litros"], index=1)
    smoke_pt = st.checkbox("Fumante")
    faf_pt = st.selectbox("Atividade física", ["Nenhuma", "1-2 vezes na semana", "3-4 vezes na semana", "5+ vezes na semana"], index=1)
    tue = st.slider("Horas diárias em telas", 0, 2, 1)
    calc_pt = st.selectbox("Consumo de álcool", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    mtrans_pt = st.selectbox("Transporte principal", ["Transporte Público", "Automóvel", "Caminhada", "Motocicleta", "Bicicleta"])

st.divider()

# ==========================================
# 4. PROCESSAMENTO E PREDIÇÃO
# ==========================================
# Utilizando width="stretch" atualizado para as novas versões do Streamlit
if st.button("Analisar Paciente", type="primary", width="stretch"):
    if modelo is not None:
        
        # Dicionários de Tradução para o formato que a IA entende
        map_genero = {"Feminino": "Female", "Masculino": "Male"}
        map_freq = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
        map_transporte = {"Transporte Público": "Public_Transportation", "Automóvel": "Automobile", "Caminhada": "Walking", "Motocicleta": "Motorbike", "Bicicleta": "Bike"}
        map_agua = {"Menos de 1 litro": 1, "De 1 a 2 litros": 2, "Mais de 2 litros": 3}
        map_atividade = {"Nenhuma": 0, "1-2 vezes na semana": 1, "3-4 vezes na semana": 2, "5+ vezes na semana": 3}
        map_vegetais = {"Raramente ou nunca": 1, "Em algumas refeições": 2, "Em todas as refeições": 3}
        
        # Montagem do Dicionário de Dados
        dados = {
            'Gender': map_genero[gender_pt], 
            'Age': age, 
            'Height': height, 
            'Weight': weight,
            'family_history': 1 if family_history_pt else 0, 
            'FAVC': 1 if favc_pt else 0,
            'FCVC': map_vegetais[fcvc_pt], 
            'NCP': ncp, 
            'CAEC': map_freq[caec_pt],
            'SMOKE': 1 if smoke_pt else 0, 
            'CH2O': map_agua[ch2o_pt], 
            'SCC': 1 if scc_pt else 0,
            'FAF': map_atividade[faf_pt], 
            'TUE': tue, 
            'CALC': map_freq[calc_pt], 
            'MTRANS': map_transporte[mtrans_pt]
        }
        
        # Criação do DataFrame e Feature Engineering do IMC
        df_input = pd.DataFrame([dados])
        df_input['BMI'] = df_input['Weight'] / (df_input['Height'] ** 2)
        
        # Ordenação exata das colunas exigida pelo modelo
        ordem_colunas = [
            'Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 
            'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 
            'CALC', 'MTRANS', 'BMI'
        ]
        df_input = df_input[ordem_colunas]

        try:
            # Realiza a predição
            predicao_bruta = modelo.predict(df_input)[0] 
            
            # Categorias que representam obesidade no dataset original
            categorias_obesidade = ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
            
            st.write("") # Espaçamento
            
            # Exibe o veredito final
            if predicao_bruta in categorias_obesidade:
                st.error("### 🚨 Resultado: POSITIVO para Obesidade")
                st.write("O padrão de dados inserido indica quadro de obesidade. Recomenda-se avaliação clínica detalhada.")
            else:
                st.success("### ✅ Resultado: NEGATIVO para Obesidade")
                st.write("O padrão de dados inserido indica peso normal, abaixo do peso ou leve sobrepeso.")
                
        except Exception as e:
            st.error(f"Erro interno na predição. Detalhes: {e}")
