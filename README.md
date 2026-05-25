# Random Forest — Predicción de Supervivencia en Titanic

**Estudiante:** Juan Fernando Bueno Torres  
**Asignatura:** Inteligencia Artificial I  
**Actividad:** Despliegue de modelo de Machine Learning en la nube  

---

# Descripción del proyecto

Este proyecto implementa un modelo de Machine Learning utilizando el algoritmo **Random Forest** para predecir la supervivencia de pasajeros del Titanic a partir de características como edad, sexo, clase y tarifa pagada.

La aplicación fue desarrollada en **Python** utilizando **Streamlit** como framework web, permitiendo que cualquier usuario pueda ingresar datos manualmente y obtener una predicción en tiempo real.

El proyecto busca demostrar el flujo completo de un sistema de Machine Learning, desde el entrenamiento del modelo hasta su integración en una aplicación web funcional.

---

# Problema a resolver

El objetivo principal es determinar si un pasajero del Titanic tenía probabilidades de sobrevivir o no, utilizando información histórica del dataset.

Este problema corresponde a un caso de **clasificación supervisada**, ya que el modelo aprende a partir de datos previamente etiquetados.

La variable objetivo es:

```text
Survived

loan-prediction-random-forest/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── titanic.csv
│   └── processed/
│
├── notebooks/
│   └── Grupo4_Notebook2_Librerias_FINAL_ipynb.ipynb
│
├── models/
│   └── modelo.pkl
│
├── src/
│   ├── __init__.py
│   └── preprocessing.py
│
├── app/
│   └── app.py
│
└── docs/

Instalación local en Linux

1. Clonar repositorio
git clone https://github.com/Yefryalexis123/loan-prediction-random-forest.git
cd loan-prediction-random-forest
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
http://localhost:8501

Instalación local en Windows
1. Clonar repositorio
git clone https://github.com/Yefryalexis123/loan-prediction-random-forest.git
cd loan-prediction-random-forest
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py
http://localhost:8501




Uso de la aplicación
Ingresar datos del pasajero.
Presionar el botón
🔍 Predecir Supervivencia
La aplicación mostrará si el pasajero probablemente sobreviviría o no.


Autores
Yefry Córdoba — Integración y despliegue
Integrante 2 — Notebook y preprocessing
Integrante 3 — Diseño Streamlit
