import streamlit as st
import pandas as pd
import joblib

# 1. Configuração da página
st.set_page_config(page_title="Predição de Obesidade", layout="wide", page_icon="🏥")

st.title("🏥 Sistema de Triagem Preditiva de Obesidade")
st.markdown("Insira os dados clínicos e os hábitos do paciente para prever o nível de risco.")
st.divider()

# 2. Carregando o Modelo Treinado DIRETAMENTE (Sem cache para evitar travamentos na nuvem)
try:
    modelo = joblib.load('modelo_obesidade_campeao.joblib')
except Exception as e:
    st.error(f"⚠️ Erro crítico ao carregar o modelo. Verifique o arquivo .joblib. Detalhe: {e}")
    modelo = None

# Mapeamento do diagnóstico (De String para Nível e Texto amigável)
dict_resultados = {
    'Insufficient_Weight': (0, 'Abaixo do Peso'),
    'Normal_Weight': (1, 'Peso Normal'),
    'Overweight_Level_I': (2, 'Sobrepeso Nível I'),
    'Overweight_Level_II': (3, 'Sobrepeso Nível II'),
    'Obesity_Type_I': (4, 'Obesidade Tipo I'),
    'Obesity_Type_II': (5, 'Obesidade Tipo II'),
    'Obesity_Type_III': (6, 'Obesidade Tipo III')
}

# 3. Interface do Formulário 100% em Português
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Físico e Genética")
    gender_pt = st.selectbox("Gênero biológico", ["Feminino", "Masculino"])
    age = st.number_input("Idade (anos)", min_value=10, max_value=100, value=25)
    height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)
    family_history_pt = st.selectbox("Histórico familiar de excesso de peso?", ["Sim", "Não"])

with col2:
    st.subheader("Hábitos Alimentares")
    favc_pt = st.selectbox("Consome alimentos calóricos com frequência?", ["Sim", "Não"])
    fcvc = st.slider("Frequência de consumo de vegetais nas refeições (1 a 3)", 1, 3, 2)
    ncp = st.slider("Número de refeições principais por dia (1 a 4)", 1, 4, 3)
    caec_pt = st.selectbox("Costuma comer entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    scc_pt = st.selectbox("Monitora as calorias que ingere diariamente?", ["Sim", "Não"])

with col3:
    st.subheader("Estilo de Vida")
    smoke_pt = st.selectbox("O paciente é fumante?", ["Sim", "Não"])
    ch2o = st.slider("Consumo diário de água (litros de 1 a 3)", 1, 3, 2)
    faf = st.slider("Frequência de atividade física (dias por semana de 0 a 3)", 0, 3, 1)
    tue = st.slider("Tempo diário de uso de telas (escala de 0 a 2)", 0, 2, 1)
    calc_pt = st.selectbox("Frequência de consumo de álcool?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    mtrans_pt = st.selectbox("Principal meio de transporte utilizado", ["Transporte Público", "Automóvel", "Caminhada", "Motocicleta", "Bicicleta"])

st.divider()

# 4. Processamento Preditivo
if st.button("🧠 Gerar Diagnóstico", type="primary"):
    
    if modelo is None:
        st.error("O modelo não foi carregado corretamente. O sistema não pode fazer predições.")
    else:
        # Dicionários de conversão: Da tela (Português) para o Algoritmo (Inglês/Numérico)
        map_genero = {"Feminino": "Female", "Masculino": "Male"}
        map_sim_nao = {"Sim": 1, "Não": 0}
        map_freq = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
        map_transporte = {
            "Transporte Público": "Public_Transportation", 
            "Automóvel": "Automobile", 
            "Caminhada": "Walking", 
            "Motocicleta": "Motorbike", 
            "Bicicleta": "Bike"
        }
        
        # Estrutura de dados idêntica à que o modelo estudou
        dados = {
            'Gender': map_genero[gender_pt],
            'Age': age,
            'Height': height,
            'Weight': weight,
            'family_history': map_sim_nao[family_history_pt],
            'FAVC': map_sim_nao[favc_pt],
            'FCVC': fcvc,
            'NCP': ncp,
            'CAEC': map_freq[caec_pt],
            'SMOKE': map_sim_nao[smoke_pt],
            'CH2O': ch2o,
            'SCC': map_sim_nao[scc_pt],
            'FAF': faf,
            'TUE': tue,
            'CALC': map_freq[calc_pt],
            'MTRANS': map_transporte[mtrans_pt]
        }
        
        df_input = pd.DataFrame([dados])

        # Criação da feature matadora
        df_input['BMI'] = df_input['Weight'] / (df_input['Height'] ** 2)

        # Imposição rigorosa da ordem das colunas
        ordem_colunas = [
            'Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 
            'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 
            'CALC', 'MTRANS', 'BMI'
        ]
        df_input = df_input[ordem_colunas]

        # Execução da predição
        try:
            # Retorna uma string como 'Obesity_Type_II'
            predicao_bruta = modelo.predict(df_input)[0] 
            
            # Converte a string usando o nosso dicionário
            grau, resultado_legivel = dict_resultados.get(predicao_bruta, (1, 'Peso Normal (Normal Weight)'))
            
            # Exibição dos resultados
            st.success("Análise concluída com sucesso!")
            c1, c2 = st.columns(2)
            c1.metric("IMC Calculado", f"{df_input['BMI'].iloc[0]:.2f} kg/m²")
            c2.metric("Diagnóstico do Algoritmo", resultado_legivel)
            
            # Lógica de Alertas Clínicos
            if grau >= 4:
                st.error("🚨 Alerta Clínico: Estágio de Obesidade. Recomenda-se acompanhamento médico imediato.")
            elif grau >= 2:
                st.warning("⚠️ Atenção Preventiva: Faixa de Sobrepeso. Recomendado monitoramento da saúde.")
            else:
                st.success("✅ Paciente apresenta índices dentro dos padrões de normalidade clínica.")
                
        except Exception as e:
            st.error(f"Erro na execução preditiva: {e}")
