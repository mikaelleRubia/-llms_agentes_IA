import streamlit as st
from utils import load_llm, format_res, build_prompt
from dotenv import load_dotenv

load_dotenv()

# Configurações iniciais
st.set_page_config(page_title="Gerador de Exercícios", layout="centered", page_icon="📖")
st.title("Gerador de Exercícios 📖")



st.sidebar.header("Configurações do modelo")
id_model = st.sidebar.text_input("ID do modelo", value = "deepseek-r1-distill-llama-70b")
temperature = st.sidebar.slider("Temperatura", 0.1, 1.5, 0.7, 0.1)

with st.form("formulario"):
  level = st.selectbox("Nível", ['Iniciante', 'Intermediário', 'Avançado'], index = 1)
  topic = st.text_input("Tema", placeholder="Matemática, Inglês, Física, etc.")
  quantity = st.slider("Quantidade de Exercícios", 1, 10, 5)
  interests = st.text_input("Interesses ou Preferências", placeholder="Ex: Filmes, Música, etc.")
  gerar = st.form_submit_button("Gerar Exercícios")

if gerar:
  with st.spinner("Gerando exercícios..."):
    llm = load_llm(id_model, temperature)
    prompt = build_prompt(topic, quantity, level, interests)
    res = llm.invoke(prompt)
    res_formatado = format_res(res.content)
    st.markdown(res_formatado)