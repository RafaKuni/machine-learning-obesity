import streamlit as st
import joblib

st.title("Teste de Ligação - Fase 1 🏥")

st.write("Se está a ler isto, o servidor da Streamlit está a funcionar perfeitamente e o problema é o ficheiro do modelo!")

try:
    st.write("A tentar carregar o modelo...")
    modelo = joblib.load('modelo_obesidade_campeao.joblib')
    st.success("✅ Modelo carregado com sucesso! O problema estava nas versões.")
except Exception as e:
    st.error(f"❌ Erro ao carregar o modelo: {e}")
