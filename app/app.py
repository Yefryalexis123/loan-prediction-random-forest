import streamlit as st
import joblib
import pandas as pd
import sys
import os

# Permitir importar desde src/
sys.path.append(os.path.abspath("."))

from src.preprocessing import preprocess_input

# =========================
# CARGAR MODELO
# =========================
model = joblib.load('models/modelo.pkl')

# =========================
# CONFIGURACIÓN PÁGINA
# =========================
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📊 Información del Modelo")

st.sidebar.write("""
Modelo de Machine Learning entrenado con:

- Algoritmo: Random Forest
- Dataset: Titanic Dataset
- Accuracy: 82.1%
- Framework: Streamlit
""")

# =========================
# TÍTULO PRINCIPAL
# =========================
st.title("🚢 Titanic Survival Predictor")

st.markdown("""
Aplicación web para predecir si un pasajero del Titanic
probablemente sobreviviría usando un modelo Random Forest.
""")

st.divider()

# =========================
# INPUTS
# =========================
col1, col2 = st.columns(2)

with col1:

    pclass = st.selectbox(
        "Clase del pasajero",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Sexo",
        ["male", "female"]
    )

    age = st.slider(
        "Edad",
        0,
        80,
        25
    )

with col2:

    sibsp = st.number_input(
        "Número de hermanos/esposos",
        0,
        10,
        0
    )

    parch = st.number_input(
        "Número de padres/hijos",
        0,
        10,
        0
    )

    fare = st.number_input(
        "Tarifa pagada",
        0.0,
        600.0,
        50.0
    )

# Puerto de embarque
embarked = st.selectbox(
    "Puerto de embarque",
    ["Q", "S"]
)

st.divider()

# =========================
# BOTÓN PREDICCIÓN
# =========================
if st.button("🔍 Predecir Supervivencia"):

    # Crear dataframe
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    # Preprocesamiento
    input_data = preprocess_input(input_data)

    # Predicción
    prediction = model.predict(input_data)

    # Resultado
    st.subheader("Resultado")

    if prediction[0] == 1:

        st.success(
            "✅ El pasajero probablemente SOBREVIVIRÍA"
        )

        st.balloons()

    else:

        st.error(
            "❌ El pasajero probablemente NO sobreviviría"
        )

# =========================
# FOOTER
# =========================
st.divider()

st.caption(
    "Proyecto de Machine Learning - Random Forest + Streamlit"
)