# Random Forest — Predicción de Supervivencia en Titanic

**Estudiantes:** Juan Fernando Bueno Torres · Yefry Alexis Muñetón Córdoba · Jonathan Pedroza Bernal

**Asignatura:** Inteligencia Artificial I

**Actividad:** Despliegue de modelo de Machine Learning en la nube

**Institución:** Fundación Universitaria Los Libertadores

---

## Descripción

Este proyecto implementa un modelo de Machine Learning utilizando el algoritmo **Random Forest** para predecir si un pasajero del Titanic habría sobrevivido al naufragio, a partir de características como edad, sexo, clase y tarifa pagada.

La aplicación fue desarrollada en **Python** con **Streamlit** como framework web, permitiendo que cualquier usuario ingrese datos y obtenga una predicción en tiempo real.

El proyecto demuestra el flujo completo de un sistema de Machine Learning, desde el entrenamiento del modelo hasta su integración en una aplicación web funcional.

---

## Demostración

> App en desarrollo local — despliegue en la nube próximamente.

Capturas de pantalla disponibles en la carpeta `docs/`.

---

## Algoritmo utilizado

- **Algoritmo:** Random Forest Classifier
- **Por qué:** Es robusto ante datos ruidosos, maneja bien variables categóricas y numéricas, y reduce el sobreajuste gracias al conjunto de árboles de decisión.
- **Parámetros:** `n_estimators=100`, `random_state=42`, `criterion=gini`

### Métricas obtenidas

| Métrica | Valor |
|---|---|
| Accuracy | 93.0% |
| Precision | 92.1% |
| Recall | 92.1% |
| Dataset de entrenamiento | 80% (569 registros) |
| Dataset de prueba | 20% (143 registros) |

---

## Dataset

- **Fuente:** Titanic Dataset — Kaggle
- **Tamaño:** 891 registros totales · 712 registros válidos (sin nulos)
- **Features utilizadas:** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
- **Variable objetivo:** `Survived` (0 = No sobrevivió · 1 = Sobrevivió)

---

## Instalación local

### Requisitos

- Python 3.10+
- pip

### Pasos

```bash
git clone https://github.com/Yefryalexis123/loan-prediction-random-forest.git
cd loan-prediction-random-forest
pip install -r requirements.txt
```

### Ejecutar la aplicación

```bash
python -m streamlit run app/app.py
```

---

## Uso

1. Ingresa los datos del pasajero en el formulario (clase, sexo, edad, tarifa, puerto de embarque)
2. Presiona **Predecir supervivencia**
3. El modelo muestra si el pasajero habría sobrevivido o no, junto con el porcentaje de confianza

---

## Estructura del proyecto

```
loan-prediction-random-forest/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── raw/
│       └── titanic.csv
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
    ├── captura_inicio.png
    └── captura_resultado.png
```

---

## Despliegue planificado

- **Servicio:** Streamlit Cloud
- **Estado:** En proceso
- **URL:** Próximamente

---

## Autores

| Nombre | Rol |
|---|---|
| Juan Fernando Bueno Torres | Entrenamiento del modelo y notebook |
| Yefry Alexis Muñetón Córdoba | Desarrollo de la aplicación web |
| Jonathan Pedroza Bernal | Documentación y despliegue |

---

## Licencia

MIT License
