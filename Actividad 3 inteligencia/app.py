import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# TITULO
# -----------------------------
st.title("Predicción de Supervivencia - Titanic")
st.write("Aplicación básica usando Random Forest")

# -----------------------------
# CARGAR DATASET
# -----------------------------
df = pd.read_csv("titanic.csv")

# -----------------------------
# PREPROCESAMIENTO SIMPLE
# -----------------------------
df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]

# eliminar nulos
df = df.dropna()

# convertir sexo
df['Sex'] = df['Sex'].map({
    'male': 0,
    'female': 1
})

# -----------------------------
# VARIABLES
# -----------------------------
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# -----------------------------
# DIVISION DATOS
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# MODELO RANDOM FOREST
# -----------------------------
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# INPUTS USUARIO
# -----------------------------
st.subheader("Ingrese los datos del pasajero")

pclass = st.selectbox(
    "Clase",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sexo",
    ["Hombre", "Mujer"]
)

age = st.slider(
    "Edad",
    1,
    80,
    25
)

fare = st.number_input(
    "Tarifa pagada",
    min_value=0.0,
    value=30.0
)

# convertir sexo
sex_value = 0

if sex == "Mujer":
    sex_value = 1

# -----------------------------
# PREDICCION
# -----------------------------
if st.button("Predecir"):

    data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_value],
        'Age': [age],
        'Fare': [fare]
    })

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("El pasajero probablemente sobreviviría")
    else:
        st.error("El pasajero probablemente no sobreviviría")

# -----------------------------
# METRICA SIMPLE
# -----------------------------
accuracy = model.score(X_test, y_test)

st.write("Accuracy aproximado del modelo:")
st.write(round(accuracy, 2))