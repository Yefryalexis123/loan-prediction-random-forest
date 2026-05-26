import streamlit as st
import joblib
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("."))
from src.preprocessing import preprocess_input

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Predictor · Grupo 6",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --brand-dark:   #0B1E3D;
    --brand-mid:    #1A3A6B;
    --brand-accent: #2E7DF7;
    --surface:      #F7F9FC;
    --card-bg:      #FFFFFF;
    --text-main:    #0B1E3D;
    --text-muted:   #6B7A9A;
    --border:       #DDE3EF;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-main);
}

.stApp { background: var(--surface); }

.hero-header {
    background: linear-gradient(135deg, #0B1E3D 0%, #1A3A6B 100%);
    border-radius: 16px;
    padding: 2.2rem 2rem 1.8rem;
    margin-bottom: 1.8rem;
    color: white;
}
.hero-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
    margin: 0 0 0.3rem;
    color: white !important;
}
.hero-header p { font-size: 1rem; color: rgba(255,255,255,0.68); margin: 0; }
.badge-algo {
    display: inline-block;
    background: rgba(46,125,247,0.22);
    border: 1px solid rgba(46,125,247,0.45);
    color: #90BFFF;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.07em;
    padding: 3px 11px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}

.metrics-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1.8rem;
    flex-wrap: wrap;
}
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.3rem;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.metric-card .m-label {
    font-size: 0.73rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 3px;
}
.metric-card .m-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.65rem;
    color: var(--brand-accent);
}
.metric-card .m-sub { font-size: 0.72rem; color: var(--text-muted); }

.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 0.9rem;
}

.result-box {
    border-radius: 16px;
    padding: 1.8rem 1.2rem;
    text-align: center;
    margin-top: 0.5rem;
}
.result-survived { background: #E8F9EF; border: 1.5px solid #1DB954; }
.result-died     { background: #FEF0F0; border: 1.5px solid #E84040; }
.result-icon { font-size: 2.8rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.75rem;
    margin: 0 0 0.3rem;
}
.result-survived .result-title { color: #0A7030; }
.result-died     .result-title { color: #B82020; }
.result-desc { font-size: 0.9rem; color: var(--text-muted); margin: 0; }

.info-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.8rem;
    font-size: 0.85rem;
}
.info-card strong { color: var(--brand-mid); display: block; margin-bottom: 2px; }

.stButton > button {
    background: #2E7DF7 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.87 !important; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cargar modelo ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    paths = ["models/modelo.pkl", os.path.join(os.path.dirname(__file__), "..", "models", "modelo.pkl")]
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p), True
    return None, False

model, model_loaded = load_model()


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="badge-algo">🌲 Random Forest · Clasificación binaria</div>
    <h1>Titanic Survival Predictor</h1>
    <p>Ingresa los datos del pasajero y el modelo predice si habría sobrevivido al naufragio del Titanic.</p>
</div>
""", unsafe_allow_html=True)


# ── Métricas calculadas con test set (20%) sobre 712 registros válidos ────────
# ── MÉTRICAS REALES DEL MODELO ────────────────────────────────────────────────
st.markdown("""
<div class="metrics-row">
    <div class="metric-card">
        <div class="m-label">Accuracy</div>
        <div class="m-value">93.0%</div>
        <div class="m-sub">en test set</div>
    </div>
    <div class="metric-card">
        <div class="m-label">Precision</div>
        <div class="m-value">92.1%</div>
        <div class="m-sub">positivos reales</div>
    </div>
    <div class="metric-card">
        <div class="m-label">Recall</div>
        <div class="m-value">92.1%</div>
        <div class="m-sub">casos detectados</div>
    </div>
    <div class="metric-card">
        <div class="m-label">Dataset</div>
        <div class="m-value">712</div>
        <div class="m-sub">registros válidos</div>
    </div>
    <div class="metric-card">
        <div class="m-label">Árboles</div>
        <div class="m-value">100</div>
        <div class="m-sub">n_estimators</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── ADVERTENCIA SI MODELO NO CARGA ────────────────────────────────────────────
if not model_loaded:
    st.warning("⚠️ Modelo no encontrado en `models/modelo.pkl`. Ejecuta el notebook primero para generarlo.", icon="⚠️")


# ── LAYOUT PRINCIPAL ──────────────────────────────────────────────────────────
col_form, col_result = st.columns([1.2, 1], gap="large")

with col_form:
    st.markdown('<div class="section-title">🎫 Datos del pasajero</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pclass = st.selectbox("Clase del pasajero", [1, 2, 3],
                              help="1ª clase (lujo) · 2ª clase · 3ª clase (cubierta baja)")
        sex = st.selectbox("Sexo", ["male", "female"])
        age = st.slider("Edad", 0, 80, 28, help="Edad del pasajero en años")
    with c2:
        sibsp = st.number_input("Hermanos / cónyuge a bordo", 0, 10, 0,
                                help="Número de hermanos o esposo/a en el barco")
        parch = st.number_input("Padres / hijos a bordo", 0, 10, 0,
                                help="Número de padres o hijos en el barco")
        fare = st.number_input("Tarifa pagada (£)", 0.0, 600.0, 32.2, step=0.5,
                               help="Precio del boleto en libras esterlinas")

    embarked = st.selectbox("Puerto de embarque", ["S", "Q", "C"],
                            format_func=lambda x: {"S": "Southampton (S)", "Q": "Queenstown (Q)", "C": "Cherbourg (C)"}[x])

    st.markdown("<br>", unsafe_allow_html=True)
    predecir = st.button("🔍 Predecir supervivencia")


# ── RESULTADO ─────────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-title">📊 Resultado de la predicción</div>', unsafe_allow_html=True)

    if predecir:
        if not model_loaded:
            st.error("No se puede predecir: el modelo no está cargado.")
        else:
            input_data = pd.DataFrame({
                'Pclass':    [pclass],
                'Sex':       [sex],
                'Age':       [age],
                'SibSp':     [sibsp],
                'Parch':     [parch],
                'Fare':      [fare],
                'Embarked':  [embarked],
            })

            try:
                input_proc = preprocess_input(input_data.copy())
                pred = model.predict(input_proc)[0]
                prob = model.predict_proba(input_proc)[0]

                if pred == 1:
                    confianza = round(prob[1] * 100, 1)
                    st.markdown(f"""
                    <div class="result-box result-survived">
                        <div class="result-icon">✅</div>
                        <div class="result-title">Sobreviviría</div>
                        <p class="result-desc">El modelo estima que este pasajero habría sobrevivido al naufragio.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    confianza = round(prob[0] * 100, 1)
                    st.markdown(f"""
                    <div class="result-box result-died">
                        <div class="result-icon">❌</div>
                        <div class="result-title">No sobreviviría</div>
                        <p class="result-desc">El modelo estima que este pasajero no habría sobrevivido.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Confianza del modelo", f"{confianza}%")
                st.progress(int(confianza))

                puerto_nombres = {"S": "Southampton", "Q": "Queenstown", "C": "Cherbourg"}
                clase_nombres  = {1: "1ª clase (lujo)", 2: "2ª clase", 3: "3ª clase"}

                st.markdown("---")
                st.markdown("**Resumen del pasajero analizado:**")
                st.markdown(f"""
| Campo | Valor |
|---|---|
| Clase | {clase_nombres[pclass]} |
| Sexo | {"Hombre" if sex == "male" else "Mujer"} |
| Edad | {age} años |
| Tarifa | £{fare:.2f} |
| Puerto | {puerto_nombres[embarked]} |
| Familiares a bordo | {sibsp + parch} |
""")

            except Exception as e:
                st.error(f"Error al predecir: {e}")

    else:
        st.info("👈 Completa el formulario y presiona **Predecir supervivencia** para ver el resultado.")

        sex_icon  = "👨" if True else "👩"
        st.markdown("""
        <div class="info-card">
            <strong>📁 Dataset</strong>
            Titanic Dataset · Kaggle<br>
            891 registros · 712 válidos · 8 features
        </div>
        <div class="info-card">
            <strong>🌲 Algoritmo</strong>
            Random Forest Classifier<br>
            100 árboles · random_state=42 · Gini criterion
        </div>
        <div class="info-card">
            <strong>🎯 Features usadas</strong>
            Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
        </div>
        <div class="info-card">
            <strong>👥 Grupo 6 · IA I</strong>
            Fundación Universitaria Los Libertadores<br>
            Juan Fernando · Yefry · Jonathan
        </div>
        """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6B7A9A; font-size:0.8rem;'>"
    "Grupo 6 · Inteligencia Artificial I · Fundación Universitaria Los Libertadores · 2025 &nbsp;·&nbsp; "
    "Random Forest + Streamlit"
    "</p>",
    unsafe_allow_html=True
)
