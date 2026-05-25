# loan-prediction-random-forest
random-forest-ml-app trabajo final--- Machine Learning web application using Random Forest

Descripción

Este proyecto consiste en el desarrollo de una aplicación web de Machine Learning utilizando el algoritmo Random Forest para predecir la supervivencia de pasajeros del Titanic.

La aplicación permite al usuario ingresar características de un pasajero, como clase, sexo, edad y tarifa pagada, para determinar si probablemente habría sobrevivido o no al desastre del Titanic.

El proyecto fue desarrollado como parte de la asignatura de Inteligencia Artificial I, aplicando el ciclo completo de un proyecto de Machine Learning:

Preprocesamiento de datos
Entrenamiento del modelo
Evaluación de métricas
Serialización del modelo
Desarrollo de aplicación web
Versionamiento con Git y GitHub
Algoritmo utilizado
Random Forest Classifier

Se utilizó el algoritmo Random Forest debido a su:

alta precisión en problemas de clasificación,
capacidad para manejar datos categóricos y numéricos,
robustez frente a overfitting,
buen desempeño en datasets medianos.
Métricas obtenidas
Métrica	Resultado
Accuracy	82.12%
Precision	80.0%
Recall	75.67%
F1 Score	77.78%
Dataset
Titanic Dataset

Dataset utilizado para predecir la supervivencia de pasajeros del Titanic.

Fuente
Kaggle Titanic Dataset
Features utilizadas
Pclass
Sex
Age
SibSp
Parch
Fare
Embarked
Target
Survived
Tecnologías utilizadas
Python
Pandas
NumPy
Scikit-learn
Streamlit
Joblib
Matplotlib
Seaborn


Estructura del proyecto

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
