import streamlit as st
import joblib
import pandas as pd

# Cargar modelo
model = joblib.load('models/modelo.pkl')

# Configuración página
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Título
st.title("🚢 Titanic Survival Predictor")

st.markdown("Predicción de supervivencia usando Random Forest")

st.divider()

# Inputs usuario
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

sibsp = st.number_input(
    "Número de hermanos/esposos a bordo",
    0,
    10,
    0
)

parch = st.number_input(
    "Número de padres/hijos a bordo",
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

# Convertir sexo
sex = 0 if sex == "male" else 1

st.divider()

# Botón predicción
if st.button("🔍 Predecir Supervivencia"):

    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ El pasajero probablemente SOBREVIVIRÍA")
        st.balloons()

    else:
        st.error("❌ El pasajero probablemente NO sobreviviría")