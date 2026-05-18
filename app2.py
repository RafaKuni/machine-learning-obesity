import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Configuração da página
st.set_page_config(page_title="Predição de Obesidade", layout="wide", page_icon="🏥")

st.title("🏥 Sistema de Triagem Preditiva de Obesidade")
st.markdown("Insira os dados clínicos e os hábitos do paciente para prever o nível de risco.")
st.divider()

# 2. O Segredo: Treinar o modelo na Nuvem (Executa apenas uma vez e guarda na cache)
@st.cache_resource(show_spinner="A treinar a Inteligência Artificial médica... (Isto só acontece uma vez)")
def treinar_modelo_agora():
    try:
        # Carregar a base de dados
        df = pd.read_csv('Obesity.csv')
        
        # Limpeza Inicial
        df.drop_duplicates(inplace=True)
        colunas_ruidosas = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        df[colunas_ruidosas] = df[colunas_ruidosas].round().astype(int)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()

        # Feature Engineering Biológico
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        binarias = ['family_history', 'FAVC', 'SMOKE', 'SCC']
        for col in binarias:
            df[col] = df[col].map({'no': 0, 'yes': 1})
        map_freq = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        df['CAEC'] = df['CAEC'].map(map_freq)
        df['CALC'] = df['CALC'].map(map_freq)

        # Separação de Variáveis
        X = df.drop(columns=['Obesity'])
        y = df['Obesity']

        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]

        # Pipeline idêntico ao nosso Megazord
        preprocess = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ])
        
        modelo_rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
        
        clf = Pipeline([('preprocess', preprocess), ('model', modelo_rf)])
        
        # O treino acontece aqui em 2 segundos
        clf.fit(X, y)
        
        return clf
    except Exception as e:
        st.error(f"Erro ao ler os dados ou treinar o modelo: {e}")
        return None

# Carrega o modelo recém-treinado
modelo = treinar_modelo_agora()

# Mapeamento do diagnóstico
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
    mtrans_pt = st.selectbox("Meio de transporte utilizado", ["Transporte Público", "Automóvel", "Caminhada", "Motocicleta", "Bicicleta"])

st.divider()

# 4. Processamento Preditivo
if st.button("🧠 Gerar Diagnóstico", type="primary"):
    
    if modelo is None:
        st.error("O modelo não conseguiu treinar. Verifique se o ficheiro Obesity.csv está no GitHub.")
    else:
        # Dicionários de conversão
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

        # Criação do IMC
        df_input['BMI'] = df_input['Weight'] / (df_input['Height'] ** 2)

        # Ordem das colunas
        ordem_colunas = [
            'Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 
            'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 
            'CALC', 'MTRANS', 'BMI'
        ]
        df_input = df_input[ordem_colunas]

        # Execução
        try:
            predicao_bruta = modelo.predict(df_input)[0] 
            grau, resultado_legivel = dict_resultados.get(predicao_bruta, (1, 'Peso Normal'))
            
            st.success("Análise concluída com sucesso!")
            c1, c2 = st.columns(2)
            c1.metric("IMC Calculado", f"{df_input['BMI'].iloc[0]:.2f} kg/m²")
            c2.metric("Diagnóstico do Algoritmo", resultado_legivel)
            
            if grau >= 4:
                st.error("🚨 Alerta Clínico: Estágio de Obesidade. Recomenda-se acompanhamento médico.")
            elif grau >= 2:
                st.warning("⚠️ Atenção Preventiva: Faixa de Sobrepeso. Recomendado monitoramento.")
            else:
                st.success("✅ Paciente apresenta índices dentro dos padrões normais.")
                
        except Exception as e:
            st.error(f"Erro na execução preditiva: {e}")
