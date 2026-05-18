import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Configuração da página
st.set_page_config(page_title="Predição de Obesidade", layout="wide", page_icon="🏥")

st.title("🏥 Sistema de Triagem Preditiva de Obesidade")
st.markdown("Rafael Kuniyoshi")
st.markdown("Insira os dados clínicos e os hábitos do paciente para prever o nível de risco.")
st.divider()

# 2. Treinamento do modelo acoplado na Nuvem (Garante compatibilidade total de versões)
@st.cache_resource(show_spinner="A treinar a Inteligência Artificial médica... (Isto só acontece uma vez)")
def treinar_modelo_agora():
    try:
        df = pd.read_csv('Obesity.csv')
        df.drop_duplicates(inplace=True)
        colunas_ruidosas = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        df[colunas_ruidosas] = df[colunas_ruidosas].round().astype(int)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()

        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        binarias = ['family_history', 'FAVC', 'SMOKE', 'SCC']
        for col in binarias:
            df[col] = df[col].map({'no': 0, 'yes': 1})
        map_freq = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        df['CAEC'] = df['CAEC'].map(map_freq)
        df['CALC'] = df['CALC'].map(map_freq)

        X = df.drop(columns=['Obesity'])
        y = df['Obesity']

        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]

        preprocess = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ])
        
        modelo_rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
        clf = Pipeline([('preprocess', preprocess), ('model', modelo_rf)])
        clf.fit(X, y)
        return clf
    except Exception as e:
        st.error(f"Erro ao ler os dados ou treinar o modelo: {e}")
        return None

modelo = treinar_modelo_agora()

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

# 3. Interface do Formulário Reformulada (Checkboxes e Listas Suspensas)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Físico e Genética")
    gender_pt = st.selectbox("Gênero biológico", ["Feminino", "Masculino"])
    age = st.number_input("Idade (anos)", min_value=10, max_value=100, value=25)
    height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)
    
    st.write("")
    st.markdown("**Histórico familiar**")
    family_history_pt = st.checkbox("Alguém na minha família tem ou já teve excesso de peso ou obesidade?")

with col2:
    st.subheader("Hábitos Alimentares")
    st.markdown("**Alimentos calóricos**")
    favc_pt = st.checkbox("Consome alimentos calóricos com frequência (fast-food, fritos, doces)?")
    
    fcvc = st.slider("Qual sua frequência de consumo de vegetais nas refeições (1 a 3)", 1, 3, 2)
    ncp = st.slider("Número de refeições principais por dia (1 a 4)", 1, 4, 3)
    caec_pt = st.selectbox("Com que frequência você se alimenta entre as refeições principais?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    
    st.markdown("**Monitoramento de calorias**")
    scc_pt = st.checkbox("Monitora sua ingestão diária de calorias?")
    
    ch2o_pt = st.selectbox("Qual seu consumo diário de água?", [
        "Menos de 1 litro por dia", 
        "De 1 a 2 litros por dia", 
        "Mais de 2 litros por dia"
    ], index=1)

with col3:
    st.subheader("Estilo de Vida")
    st.markdown("**Tabagismo**")
    smoke_pt = st.checkbox("Possui o hábito de fumar?")
    st.markdown("**Tecnologia e Outros**")
    tue = st.slider("Tempo diário de uso de telas/dispositivos (escala de 0 a 2)", 0, 2, 1)    
    faf_pt = st.selectbox("Com que frequência você pratica alguma atividade física?", [
        "Nenhuma vez", 
        "1-2 vezes por semana", 
        "3-4 vezes por semana", 
        "5 vezes ou mais por semana"
    ], index=1)
    
    calc_pt = st.selectbox("Frequência de consumo de álcool?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    mtrans_pt = st.selectbox("Qual o seu meio de transporte mais utilizado?", ["Transporte Público", "Automóvel", "Caminhada", "Motocicleta", "Bicicleta"])

st.divider()

# 4. Processamento Preditivo e Conversões
if st.button("🧠 Gerar Diagnóstico", type="primary"):
    
    if modelo is None:
        st.error("O modelo não conseguiu treinar. Verifique se o arquivo Obesity.csv está no GitHub.")
    else:
        map_genero = {"Feminino": "Female", "Masculino": "Male"}
        map_freq = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
        map_transporte = {
            "Transporte Público": "Public_Transportation", 
            "Automóvel": "Automobile", 
            "Caminhada": "Walking", 
            "Motocicleta": "Motorbike", 
            "Bicicleta": "Bike"
        }
        
        # Mapeamentos internos das novas listas suspensas para os valores matemáticos correspondentes
        map_agua = {
            "Menos de 1 litro por dia": 1,
            "De 1 a 2 litros por dia": 2,
            "Mais de 2 litros por dia": 3
        }
        
        map_atividade = {
            "Nenhuma vez": 0,
            "1-2 vezes por semana": 1,
            "3-4 vezes por semana": 2,
            "5 vezes ou mais por semana": 3
        }
        
        # Montagem da estrutura convertendo os booleanos (True/False) dos checkboxes para 1/0
        dados = {
            'Gender': map_genero[gender_pt],
            'Age': age,
            'Height': height,
            'Weight': weight,
            'family_history': 1 if family_history_pt else 0,
            'FAVC': 1 if favc_pt else 0,
            'FCVC': fcvc,
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
        
        df_input = pd.DataFrame([dados])
        df_input['BMI'] = df_input['Weight'] / (df_input['Height'] ** 2)

        ordem_colunas = [
            'Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 
            'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 
            'CALC', 'MTRANS', 'BMI'
        ]
        df_input = df_input[ordem_colunas]

        try:
            predicao_bruta = modelo.predict(df_input)[0] 
            grau, resultado_legivel = dict_resultados.get(predicao_bruta, (1, 'Peso Normal'))
            
            st.success("Análise concluída com sucesso!")
            c1, c2 = st.columns(2)
            c1.metric("IMC Calculado", f"{df_input['BMI'].iloc[0]:.2f} kg/m²")
            c2.metric("Diagnóstico do Algoritmo", resultado_traduzido if 'resultado_traduzido' in locals() else resultado_legivel)
            
            # Lógica de Alertas Clínicos com a inclusão de Baixo Peso
            if grau >= 4:
                st.error("🚨 Alerta Clínico: Estágio de Obesidade. Recomenda-se acompanhamento médico imediato.")
            elif grau >= 2:
                st.warning("⚠️ Atenção Preventiva: Faixa de Sobrepeso. Recomendado monitoramento da saúde.")
            elif grau == 1:
                st.success("✅ Paciente apresenta índices dentro dos padrões de normalidade clínica.")
            elif grau == 0:
                st.warning("⚠️ Alerta Clínico: Paciente abaixo do peso saudável. Recomenda-se avaliação nutricional e clínica especializada.")
                
        except Exception as e:
            st.error(f"Erro na execução preditiva: {e}")
