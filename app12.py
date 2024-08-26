import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import random
import uuid

@st.cache_data
def carregar_dataframe(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        return None

@st.cache_data
def carregar_modelo(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except (FileNotFoundError, pickle.UnpicklingError):
        st.error(f"Erro: O modelo '{file_path}' não pôde ser carregado.")
        return None

@st.cache_data
def carregar_cuidados(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        return None

def processa_sintomas(sintomas, atributos, best_tree, label_encoder):
    diagnosticos_associados = []
    limiar_probabilidade = 0.1
    for sintoma in sintomas:
        dados_paciente_dict = {coluna: 0 for coluna in atributos}
        dados_paciente_dict[sintoma] = 1
        dados_paciente_np = [list(dados_paciente_dict.values())]
        probas = best_tree.predict_proba(dados_paciente_np)[0]
        diagnosticos_associados.extend([(label_encoder.inverse_transform([i])[0], sintoma)
                                        for i, p in enumerate(probas) if p > limiar_probabilidade])
    return diagnosticos_associados

def main():
    st.markdown("<h3 style='text-align: center; color: #FF5733;'>Sistema de Previsão de Diagnóstico de Enfermagem</h3>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: right; color: #ff0015;'>por Renato Douglas</h1>", unsafe_allow_html=True)


    df = carregar_dataframe('dataset_30outubroV3.csv')
    best_tree = carregar_modelo('modelAgosto.pkl')
    cuidados_df = carregar_cuidados('cuidados_diags.csv')

    if df is None or best_tree is None or cuidados_df is None:
        st.stop()

    atributos = [coluna for coluna in df.columns if coluna != 'diagnostico_de_Enfermagem']
    label_encoder = LabelEncoder()
    label_encoder.fit(df['diagnostico_de_Enfermagem'])

    sintomas = st.multiselect("Selecione os sintomas:", atributos)
    diagnostico_personalizado = st.text_area("Se desejar, escreva um diagnóstico personalizado:", "")

    if st.button("Processar"):
        if not sintomas and not diagnostico_personalizado:
            st.warning("Por favor, selecione ao menos um sintoma ou forneça um diagnóstico personalizado.")
            return

        st.session_state.diagnosticos_sugeridos = processa_sintomas(sintomas, atributos, best_tree, label_encoder)

    if 'diagnosticos_selecionados' not in st.session_state:
        st.session_state.diagnosticos_selecionados = []

    if 'diagnosticos_sugeridos' in st.session_state and st.session_state.diagnosticos_sugeridos:
        st.write("Diagnósticos sugeridos com base nos sintomas selecionados:")

        for diagnostico, sintoma in st.session_state.diagnosticos_sugeridos:
            sintoma_selecionado = sintoma.replace("_", " ")
            checkbox_key = f"{diagnostico}_{sintoma_selecionado}"
            if st.checkbox(f"Diagnóstico: {diagnostico}. Característica Definidora: {sintoma_selecionado}", 
                           key=checkbox_key):
                if diagnostico not in st.session_state.diagnosticos_selecionados:
                    st.session_state.diagnosticos_selecionados.append(diagnostico)
            else:
                if diagnostico in st.session_state.diagnosticos_selecionados:
                    st.session_state.diagnosticos_selecionados.remove(diagnostico)

    if st.session_state.diagnosticos_selecionados:
        st.subheader("Cuidados Relacionados aos Diagnósticos Selecionados:")
        for diagnostico_selecionado in st.session_state.diagnosticos_selecionados:
            st.markdown(f"**{diagnostico_selecionado}**")
            if diagnostico_selecionado in cuidados_df.columns:
                cuidados_relacionados = cuidados_df[diagnostico_selecionado].iloc[0]
                if cuidados_relacionados:
                    cuidados_list = cuidados_relacionados.split('\t')
                    for cuidado in cuidados_list:
                        st.checkbox(cuidado, key=f"{diagnostico_selecionado}_{cuidado}_{uuid.uuid4()}")

    st.text_area("Observações gerais sobre os diagnósticos selecionados:", key="observacoes_gerais")

if __name__ == "__main__":
    main()
