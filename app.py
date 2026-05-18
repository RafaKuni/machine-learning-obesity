import streamlit as st
import pandas as pd
import joblib

# 1. Configuração da página
st.set_page_config(page_title="Predição de Obesidade - FIAP", layout="wide", page_icon="🏥")

st.title("🏥 Sistema de Triagem Preditiva de Obesidade")
st.markdown("Insira os dados clínicos e os hábitos do paciente para prever o nível de obesidade com base em Machine Learning.")
st.divider()

# 2. Carregando o Modelo Treinado
@st.cache_resource
def load_model():
    return joblib.load('modelo_obesidade_campeao.joblib')

modelo = load_model()

# 3. Mapeamento Reverso
diagnosticos = {
    0: 'Abaixo do Peso (Insufficient Weight)', 
    1: 'Peso Normal (Normal Weight)', 
    2: 'Sobrepeso Nível I (Overweight Level I)', 
    3: 'Sobrepeso Nível II (Overweight Level II)', 
    4: 'Obesidade Tipo I (Obesity Type I)', 
    5: 'Obesidade Tipo II (Obesity Type II)', 
    6: 'Obesidade Tipo III (Obesity Type III)'
}

# 4. Construindo o Formulário
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Físico e Genética")
    gender = st.selectbox("Gênero", ["Female", "Male"])
    age = st.number_input("Idade", min_value=10, max_value=100, value=25)
    height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)
    family_history = st.selectbox("Histórico Familiar de Excesso de Peso?", ["yes", "no"])

with col2:
    st.subheader("Hábitos Alimentares")
    favc = st.selectbox("Consome alimentos calóricos com frequência?", ["yes", "no"])
    fcvc = st.slider("Frequência de vegetais nas refeições (1-3)", 1, 3, 2)
    ncp = st.slider("Número de refeições principais (1-4)", 1, 4, 3)
    caec = st.selectbox("Come entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
    scc = st.selectbox("Monitora as calorias que ingere?", ["yes", "no"])

with col3:
    st.subheader("Estilo de Vida")
    smoke = st.selectbox("Fumante?", ["yes", "no"])
    ch2o = st.slider("Consumo diário de água (litros 1-3)", 1, 3, 2)
    faf = st.slider("Frequência de atividade física (dias/semana 0-3)", 0, 3, 1)
    tue = st.slider("Tempo de uso de telas (nível 0-2)", 0, 2, 1)
    calc = st.selectbox("Consumo de álcool?", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Meio de transporte", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

st.divider()

# 5. Processamento e Predição
if st.button("🧠 Gerar Diagnóstico Preditivo", type="primary"):
    
    dados = {
        'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
        'family_history': family_history, 'FAVC': favc, 'FCVC': fcvc, 'NCP': ncp,
        'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o, 'SCC': scc, 'FAF': faf,
        'TUE': tue, 'CALC': calc, 'MTRANS': mtrans
    }
    df_input = pd.DataFrame([dados])

    # Feature Engineering (idêntico ao treino)
    df_input['BMI'] = df_input['Weight'] / (df_input['Height'] ** 2)
    
    binarias = ['family_history', 'FAVC', 'SMOKE', 'SCC']
    for col in binarias:
        df_input[col] = df_input[col].map({'no': 0, 'yes': 1})
        
    map_freq = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    df_input['CAEC'] = df_input['CAEC'].map(map_freq)
    df_input['CALC'] = df_input['CALC'].map(map_freq)

    # Predição pelo Pipeline
    predicao_num = modelo.predict(df_input)[0]
    resultado_texto = diagnosticos[predicao_num]
    
    st.success("Análise concluída com sucesso!")
    
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("IMC Calculado do Paciente", f"{df_input['BMI'].iloc[0]:.2f}")
    res_col2.metric("Diagnóstico do Modelo", resultado_texto)
    
    if predicao_num >= 4:
        st.error("🚨 Atenção: Paciente classificado em estágio de Obesidade. Recomenda-se acompanhamento clínico.")
    elif predicao_num >= 2:
        st.warning("⚠️ Alerta: Paciente em faixa de sobrepeso.")
    else:
        st.info("✅ Paciente dentro de parâmetros de peso mais saudáveis.")
