import streamlit as st
import pandas as pd
import joblib

# 1. Configuração da página
st.set_page_config(page_title="Predição de Obesidade", layout="wide", page_icon="🏥")

st.title("🏥 Sistema de Triagem Preditiva de Obesidade")
st.markdown("Insira os dados clínicos e os hábitos do paciente para prever o nível de risco metabólico")
st.markdown("**by Rafael Kuniyoshi**")
st.divider()

# 2. Carregamento do Modelo Salvo 
@st.cache_resource(show_spinner="Carregando a Inteligência Artificial...")
def carregar_modelo():
    try:
        return joblib.load('modelo_obesidade_campeao.joblib')
    except Exception as e:
        st.error(f"Erro ao carregar o modelo. Verifique se o arquivo .joblib está no GitHub. Detalhes: {e}")
        return None

modelo = carregar_modelo()

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

# 3. Interface do Formulário Reformulada (3 Colunas)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Físico e Genética")
    gender_pt = st.selectbox("Gênero biológico", ["Feminino", "Masculino"])
    age = st.number_input("Idade (anos)", min_value=10, max_value=100, value=25)
    height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)
    
    st.write("")
    st.markdown("**Histórico familiar**")
    family_history_pt = st.checkbox("Alguém na minha família tem ou já teve excesso de peso ou obesidade")

with col2:
    st.markdown("#### Hábitos Alimentares")
    st.markdown("**Alimentos calóricos e Vegetais**")
    favc_pt = st.checkbox("Consumo de alimentos calóricos com frequência (fast-food, fritos, doces)")
    
    fcvc_pt = st.selectbox("Frequência de consumo de vegetais nas refeições", [
        "Raramente ou nunca", 
        "Em algumas refeições", 
        "Em todas as refeições"
    ], index=1)
    
    ncp = st.slider("Número de refeições principais por dia (1 a 4)", 1, 4, 3)
    caec_pt = st.selectbox("Costuma comer entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    
    st.markdown("**Monitoramento e Hidratação**")
    scc_pt = st.checkbox("Monitoro minha ingestão diária de calorias")
    
    ch2o_pt = st.selectbox("Consumo diário de água", [
        "Menos de 1 litro por dia", 
        "De 1 a 2 litros por dia", 
        "Mais de 2 litros por dia"
    ], index=1)

with col3:
    st.markdown("#### Estilo de Vida")
    st.markdown("**Tabagismo e Exercício**")
    smoke_pt = st.checkbox("O paciente possui o hábito de fumar")
    
    faf_pt = st.selectbox("Com que frequência você pratica alguma atividade física?", [
        "Nenhuma vez", 
        "1-2 vezes por semana", 
        "3-4 vezes por semana", 
        "5 vezes ou mais por semana"
    ], index=1)
    
    st.markdown("**Tecnologia e Outros**")
    tue = st.slider("Tempo diário de uso de telas/dispositivos (escala de 0 a 2)", 0, 2, 1)
    calc_pt = st.selectbox("Frequência de consumo de álcool?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    mtrans_pt = st.selectbox("Qual o seu meio de transporte mais utilizado?", ["Transporte Público", "Automóvel", "Caminhada", "Motocicleta", "Bicicleta"])

st.divider()

# 4. Processamento Preditivo e Conversões
if st.button("🧠 Gerar Diagnóstico", type="primary"):
    
    if modelo is None:
        st.error("O modelo não pôde ser carregado. Tente recarregar a página.")
    else:
        # Dicionários de conversão interna
        map_genero = {"Feminino": "Female", "Masculino": "Male"}
        map_freq = {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
        map_transporte = {
            "Transporte Público": "Public_Transportation", 
            "Automóvel": "Automobile", 
            "Caminhada": "Walking", 
            "Motocicleta": "Motorbike", 
            "Bicicleta": "Bike"
        }
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
        map_vegetais = {
            "Raramente ou nunca": 1,
            "Em algumas refeições": 2,
            "Em todas as refeições": 3
        }
        
        # Montagem dos dados
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
        
        df_input = pd.DataFrame([dados])
        df_input['BMI'] = df_input['Weight'] / (df_input['Height'] ** 2)

        ordem_colunas = [
            'Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 
            'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 
            'CALC', 'MTRANS', 'BMI'
        ]
        df_input = df_input[ordem_colunas]

        try:
            # Predição Bruta
            predicao_bruta = modelo.predict(df_input)[0] 
            grau, resultado_legivel = dict_resultados.get(predicao_bruta, (1, 'Peso Normal'))
            
            # Exibição das Métricas
            st.success("Análise preditiva e heurística concluída com sucesso!")
            c1, c2 = st.columns(2)
            c1.metric("IMC Calculado", f"{df_input['BMI'].iloc[0]:.2f} kg/m²")
            c2.metric("Diagnóstico do Algoritmo", resultado_legivel)
            
            st.divider()
            
            # ---------------------------------------------------------
            # Diagnóstico Principal (IA)
            # ---------------------------------------------------------
            st.markdown("### Parecer Clínico Geral")
            if grau >= 4:
                st.error(f"🚨 **Alerta Máximo:** Classificação de {resultado_legivel}. Recomenda-se acompanhamento médico e nutricional imediato.")
            elif grau >= 2:
                st.warning(f"⚠️ **Atenção Preventiva:** Classificação de {resultado_legivel}. Recomendado monitoramento da saúde e ajustes na rotina.")
            elif grau == 1:
                st.success(f"✅ **Diagnóstico:** {resultado_legivel}. Índices físicos e metabólicos gerais dentro do padrão esperado.")
            elif grau == 0:
                st.warning(f"⚠️ **Alerta Nutricional:** Classificação de {resultado_legivel}. Recomenda-se avaliação nutricional para descartar deficiências crônicas.")

            # ---------------------------------------------------------
            # MOTOR DE INSIGHTS HEURÍSTICOS
            # ---------------------------------------------------------
            is_atleta = ((faf_pt == "5 vezes ou mais por semana" or faf_pt == "3-4 vezes por semana" or faf_pt == "1-2 vezes por semana") and scc_pt == True and ch2o_pt == "Mais de 2 litros por dia")
            is_falso_magro = (grau == 1 and faf_pt == "Nenhuma vez" and favc_pt == True and fcvc_pt == "Raramente ou nunca")
            is_adolescente = (age < 18)
            is_idoso_sedentario = (age >= 60 and faf_pt == "Nenhuma vez")
            
            if any([is_atleta, is_falso_magro, is_adolescente, is_idoso_sedentario]):
                st.markdown("### 🔍 Insights Contextuais do Paciente")
                
            if is_atleta and grau >= 2:
                st.info("💪 O modelo aponta excesso de peso, mas a descrição sugere alta probabilidade de peso concentrado em massa muscular. O cálculo de IMC tradicional pode ser impreciso neste cenário. Recomenda-se exame de bioimpedância.")
                
            if is_falso_magro:
                st.error("🕵️ Embora o peso geral esteja normal, o alto sedentarismo combinado à má alimentação indica um forte risco de acúmulo de gordura visceral e síndrome metabólica.")
                
            if is_adolescente:
                st.warning("👶 O paciente é menor de 18 anos. As predições baseadas em IMC de adultos devem ser analisadas com cautela. O diagnóstico oficial deve utilizar as Curvas de Percentil da OMS.")
                
            if is_idoso_sedentario and grau <= 2:
                st.warning("👴 Em pacientes idosos sedentários, um IMC considerado 'normal' ou 'baixo' pode mascarar a perda severa de massa muscular substituída por tecido adiposo. Recomenda-se avaliar força e mobilidade de forma preventiva.")

        except Exception as e:
            st.error(f"Erro na execução preditiva: {e}")
