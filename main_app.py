import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, learning_curve)
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc,
                             ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
import joblib
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🤖 ML Classification Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; margin: 5px;
    }
    .metric-card h2 { font-size: 2.2em; margin: 0; }
    .metric-card p  { font-size: 0.9em; margin: 0; opacity: 0.85; }
    .deploy-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; font-size: 1.2em; font-weight: bold;
    }
    .warn-box {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 16px; border-radius: 12px; color: #333;
        text-align: center; font-size: 1.1em; font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-weight: bold; font-size: 1em;
    }
    .stButton>button:hover { opacity: 0.9; transform: scale(1.02); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
DATASETS = {
    "Iris (flores, 3 clases)": "iris",
    "Wine (vinos, 3 clases)": "wine",
    "Breast Cancer (tumores, 2 clases)": "breast_cancer",
    "Digits (dígitos 0-9, 10 clases)": "digits",
}

MODELS = {
    "LDA (Análisis Discriminante Lineal)": "lda",
    "Naive Bayes (Gaussiano)": "bayes",
    "KNN (K-Vecinos Más Cercanos)": "knn",
    "Decision Tree (Árbol de Decisión)": "tree",
}

@st.cache_data
def load_dataset(name):
    loaders = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "digits": datasets.load_digits,
    }
    data = loaders[name]()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, data.target_names, data.DESCR

def build_model(model_key, params):
    if model_key == "lda":
        return LinearDiscriminantAnalysis()
    elif model_key == "bayes":
        return GaussianNB()
    elif model_key == "knn":
        return KNeighborsClassifier(n_neighbors=params.get("k", 5),
                                    metric=params.get("metric", "euclidean"))
    elif model_key == "tree":
        return DecisionTreeClassifier(max_depth=params.get("max_depth", None),
                                      criterion=params.get("criterion", "gini"),
                                      random_state=42)

def performance_label(acc):
    if acc >= 0.90: return "🟢 ALTO", "deploy-box"
    elif acc >= 0.75: return "🟡 MEDIO-ALTO", "deploy-box"
    elif acc >= 0.60: return "🟠 MEDIO", "warn-box"
    else:             return "🔴 BAJO", "warn-box"

def ready_to_deploy(acc):
    return acc >= 0.75

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png", width=150)
    st.title("⚙️ Configuración")

    st.subheader("1. Dataset")
    ds_label = st.selectbox("Selecciona dataset", list(DATASETS.keys()))
    ds_key   = DATASETS[ds_label]

    st.subheader("2. Preproceso")
    do_scale = st.checkbox("Normalizar (StandardScaler)", value=True)

    st.subheader("3. Feature Selection")
    feat_method = st.radio("Método",
        ["Ninguno", "PCA", "SelectKBest (ANOVA-F)", "LDA Embeddings"])
    n_components = st.slider("Componentes / Features a retener", 1, 20, 5)

    st.subheader("4. Modelo")
    model_label  = st.selectbox("Algoritmo", list(MODELS.keys()))
    model_key    = MODELS[model_label]

    params = {}
    if model_key == "knn":
        params["k"]      = st.slider("Vecinos (k)", 1, 20, 5)
        params["metric"] = st.selectbox("Distancia", ["euclidean","manhattan","minkowski"])
    elif model_key == "tree":
        params["max_depth"] = st.slider("Profundidad máxima", 1, 20, 4)
        params["criterion"] = st.selectbox("Criterio", ["gini","entropy"])

    st.subheader("5. Validación")
    test_size = st.slider("Tamaño del test (%)", 10, 40, 20) / 100
    cv_folds  = st.slider("Folds Cross-Validation", 3, 10, 5)

    run_btn = st.button("🚀 Entrenar Modelo")

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.title("🤖 Plataforma de Clasificación con Machine Learning")
st.markdown("Entrena, valida y despliega modelos de clasificación sobre datasets de scikit-learn.")

tabs = st.tabs(["📊 Datos", "🔧 Preproceso & Features", "🧠 Entrenamiento", "📈 Desempeño", "🚀 Despliegue", "🧪 Uso Real"])

# ═══════════════════════════════════════════
# TAB 0 – DATOS
# ═══════════════════════════════════════════
with tabs[0]:
    X, y, target_names, descr = load_dataset(ds_key)
    n_classes = len(np.unique(y))

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1,c2,c3,c4],
        ["Muestras","Features","Clases","Balance"],
        [len(X), X.shape[1], n_classes,
         f"{y.value_counts().min()}/{y.value_counts().max()}"]
    ):
        col.markdown(f"""<div class='metric-card'><h2>{val}</h2><p>{label}</p></div>""",
                     unsafe_allow_html=True)

    st.subheader("Vista previa del dataset")
    df_preview = X.copy()
    df_preview["Clase"] = [target_names[i] for i in y]
    st.dataframe(df_preview.head(20), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribución de clases")
        class_counts = pd.Series([target_names[i] for i in y]).value_counts()
        fig = px.bar(class_counts, color=class_counts.index,
                     labels={"value":"Muestras","index":"Clase"},
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Correlación de features")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        corr = X.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", ax=ax2,
                    annot=(X.shape[1] <= 8), fmt=".1f",
                    linewidths=0.5, square=True)
        ax2.set_title("Mapa de correlación")
        st.pyplot(fig2, use_container_width=True)

    with st.expander("📋 Descripción oficial del dataset"):
        st.text(descr[:2000])

# ═══════════════════════════════════════════
# TAB 1 – PREPROCESO & FEATURES
# ═══════════════════════════════════════════
with tabs[1]:
    st.subheader("Estadísticas antes del preproceso")
    st.dataframe(X.describe().T.style.background_gradient(cmap="Blues"),
                 use_container_width=True)

    X_proc = X.values.copy()
    scaler = None
    if do_scale:
        scaler  = StandardScaler()
        X_proc  = scaler.fit_transform(X_proc)
        st.success("✅ StandardScaler aplicado: media≈0, std≈1")

        fig_s, axes = plt.subplots(1, 2, figsize=(12, 3))
        axes[0].boxplot(X.values, vert=True)
        axes[0].set_title("Antes de escalar")
        axes[1].boxplot(X_proc, vert=True)
        axes[1].set_title("Después de escalar")
        for ax in axes:
            ax.set_xticklabels(X.columns, rotation=45, ha="right", fontsize=7)
        st.pyplot(fig_s, use_container_width=True)

    # Feature reduction / selection
    reducer = None
    X_feat  = X_proc.copy()
    k_use   = min(n_components, X_proc.shape[1])

    if feat_method == "PCA":
        reducer = PCA(n_components=k_use, random_state=42)
        X_feat  = reducer.fit_transform(X_proc)
        exp_var = reducer.explained_variance_ratio_
        st.info(f"PCA: {k_use} componentes → {exp_var.sum()*100:.1f}% varianza explicada")

        fig_pca = px.bar(x=[f"PC{i+1}" for i in range(k_use)],
                         y=exp_var * 100,
                         labels={"x":"Componente","y":"Varianza (%)"},
                         color=exp_var * 100,
                         color_continuous_scale="Viridis",
                         title="Varianza explicada por componente")
        st.plotly_chart(fig_pca, use_container_width=True)

        if k_use >= 2:
            df_pca = pd.DataFrame(X_feat[:, :2], columns=["PC1", "PC2"])
            df_pca["Clase"] = [target_names[i] for i in y]
            fig2d = px.scatter(df_pca, x="PC1", y="PC2", color="Clase",
                               title="Proyección PCA 2D",
                               color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig2d, use_container_width=True)

    elif feat_method == "SelectKBest (ANOVA-F)":
        selector = SelectKBest(f_classif, k=k_use)
        X_feat   = selector.fit_transform(X_proc, y)
        scores   = selector.scores_
        sel_idx  = selector.get_support(indices=True)
        sel_names = [X.columns[i] for i in sel_idx]
        st.info(f"SelectKBest: top-{k_use} features seleccionadas")

        fig_kb = px.bar(x=X.columns, y=scores,
                        color=scores, color_continuous_scale="Oranges",
                        title="Puntuación ANOVA-F por feature",
                        labels={"x":"Feature","y":"F-score"})
        fig_kb.add_hline(y=min(scores[sel_idx]),
                         line_dash="dash", line_color="red",
                         annotation_text="Umbral de selección")
        st.plotly_chart(fig_kb, use_container_width=True)
        st.write("**Features seleccionadas:**", ", ".join(sel_names))

    elif feat_method == "LDA Embeddings":
        n_lda = min(k_use, n_classes - 1)
        lda_red = LinearDiscriminantAnalysis(n_components=n_lda)
        X_feat  = lda_red.fit_transform(X_proc, y)
        st.info(f"LDA Embedding: {n_lda} componentes discriminantes")
        if n_lda >= 2:
            df_lda = pd.DataFrame(X_feat[:, :2], columns=["LD1","LD2"])
            df_lda["Clase"] = [target_names[i] for i in y]
            fig_ld = px.scatter(df_lda, x="LD1", y="LD2", color="Clase",
                                title="Proyección LDA 2D",
                                color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig_ld, use_container_width=True)
    else:
        st.info("No se aplicó reducción de features.")

    st.write(f"**Shape final del dataset:** {X_feat.shape}")

# ═══════════════════════════════════════════
# TAB 2 – ENTRENAMIENTO
# ═══════════════════════════════════════════
with tabs[2]:
    if not run_btn:
        st.info("👈 Configura los parámetros en el sidebar y presiona **Entrenar Modelo**.")
    else:
        st.subheader("Entrenando el modelo...")

        # Split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_feat, y, test_size=test_size, random_state=42, stratify=y)

        model = build_model(model_key, params)

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_feat, y, cv=cv, scoring="accuracy")

        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_te)
        test_acc = accuracy_score(y_te, y_pred)
        cv_mean  = cv_scores.mean()
        cv_std   = cv_scores.std()

        # Store in session state
        st.session_state["model"]       = model
        st.session_state["scaler"]      = scaler
        st.session_state["reducer"]     = (reducer if feat_method != "Ninguno" else None)
        st.session_state["feat_method"] = feat_method
        st.session_state["X_feat"]      = X_feat
        st.session_state["X_te"]        = X_te
        st.session_state["y_te"]        = y_te
        st.session_state["y_pred"]      = y_pred
        st.session_state["cv_scores"]   = cv_scores
        st.session_state["test_acc"]    = test_acc
        st.session_state["cv_mean"]     = cv_mean
        st.session_state["cv_std"]      = cv_std
        st.session_state["target_names"]= target_names
        st.session_state["model_label"] = model_label
        st.session_state["n_classes"]   = n_classes
        st.session_state["X_columns"]   = list(X.columns)
        st.session_state["y_all"]       = y
        st.session_state["trained"]     = True

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class='metric-card'><h2>{test_acc:.1%}</h2><p>Accuracy (Test)</p></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class='metric-card'><h2>{cv_mean:.1%}</h2><p>CV Mean Accuracy</p></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class='metric-card'><h2>±{cv_std:.3f}</h2><p>CV Std Deviation</p></div>""", unsafe_allow_html=True)

        st.subheader("Reporte de clasificación")
        report = classification_report(y_te, y_pred,
                                       target_names=target_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.style.background_gradient(cmap="RdYlGn"),
                     use_container_width=True)

        # CV box
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Box(y=cv_scores, name="CV Folds",
                                boxpoints="all", jitter=0.3,
                                marker_color="#667eea"))
        fig_cv.add_hline(y=cv_mean, line_dash="dash",
                         annotation_text=f"Mean: {cv_mean:.3f}")
        fig_cv.update_layout(title="Distribución de accuracy en Cross-Validation",
                             yaxis_title="Accuracy", height=350)
        st.plotly_chart(fig_cv, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 3 – DESEMPEÑO
# ═══════════════════════════════════════════
with tabs[3]:
    if not st.session_state.get("trained"):
        st.info("Primero entrena un modelo en la pestaña 🧠.")
    else:
        model_s  = st.session_state["model"]
        X_te_s   = st.session_state["X_te"]
        y_te_s   = st.session_state["y_te"]
        y_pred_s = st.session_state["y_pred"]
        tn_s     = st.session_state["target_names"]
        X_f_s    = st.session_state["X_feat"]
        y_s      = st.session_state["y_all"]

        # Confusion matrix
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Matriz de Confusión")
            cm = confusion_matrix(y_te_s, y_pred_s)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(cm, display_labels=tn_s).plot(
                ax=ax_cm, cmap="Blues", colorbar=False)
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm, use_container_width=True)

        with col2:
            st.subheader("Curva de Aprendizaje")
            train_sz, train_sc, val_sc = learning_curve(
                model_s, X_f_s, y_s, cv=5, scoring="accuracy",
                train_sizes=np.linspace(0.1, 1.0, 8), random_state=42)
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(x=train_sz,
                y=train_sc.mean(axis=1), mode="lines+markers",
                name="Train", line=dict(color="#667eea", width=2)))
            fig_lc.add_trace(go.Scatter(x=train_sz,
                y=val_sc.mean(axis=1), mode="lines+markers",
                name="Validación", line=dict(color="#f7971e", width=2)))
            fig_lc.update_layout(title="Curva de Aprendizaje",
                xaxis_title="Muestras de entrenamiento",
                yaxis_title="Accuracy", height=350, legend=dict(x=0.7, y=0.3))
            st.plotly_chart(fig_lc, use_container_width=True)

        # ROC curves
        st.subheader("Curvas ROC")
        n_c = st.session_state["n_classes"]
        if hasattr(model_s, "predict_proba"):
            y_prob = model_s.predict_proba(X_te_s)
            if n_c == 2:
                fpr, tpr, _ = roc_curve(y_te_s, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                    name=f"AUC = {roc_auc:.3f}",
                    line=dict(color="#764ba2", width=2)))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1],
                    line=dict(dash="dash", color="gray"), name="Aleatorio"))
                fig_roc.update_layout(title="ROC Curve (Binario)",
                    xaxis_title="FPR", yaxis_title="TPR", height=380)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                y_bin = label_binarize(y_te_s, classes=np.unique(y_te_s))
                fig_roc = go.Figure()
                colors = px.colors.qualitative.Vivid
                for i in range(n_c):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                        name=f"{tn_s[i]} (AUC={roc_auc:.2f})",
                        line=dict(color=colors[i % len(colors)], width=2)))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1],
                    line=dict(dash="dash", color="gray"), name="Aleatorio"))
                fig_roc.update_layout(title="ROC Curvas (Multiclase)",
                    xaxis_title="FPR", yaxis_title="TPR", height=400)
                st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("Este modelo no soporta probabilidades → ROC no disponible.")

        # Feature importance (Decision Tree)
        if st.session_state["model_label"].startswith("Decision") and hasattr(model_s, "feature_importances_"):
            st.subheader("Importancia de Features (Árbol de Decisión)")
            imp = model_s.feature_importances_
            n_feat = len(imp)
            feat_labels = (
                [f"PC{i+1}" for i in range(n_feat)]
                if st.session_state["feat_method"] == "PCA"
                else st.session_state["X_columns"][:n_feat]
            )
            fig_imp = px.bar(x=imp, y=feat_labels, orientation="h",
                             color=imp, color_continuous_scale="Teal",
                             labels={"x":"Importancia","y":"Feature"},
                             title="Feature Importance")
            fig_imp.update_layout(height=max(300, n_feat * 28), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 4 – DESPLIEGUE
# ═══════════════════════════════════════════
with tabs[4]:
    if not st.session_state.get("trained"):
        st.info("Primero entrena un modelo en la pestaña 🧠.")
    else:
        test_acc = st.session_state["test_acc"]
        cv_mean  = st.session_state["cv_mean"]
        label, box_cls = performance_label(test_acc)

        st.subheader("Evaluación de desempeño")
        st.markdown(f"""<div class='{box_cls}'>
            Desempeño del modelo: {label}<br>
            Test Accuracy: {test_acc:.1%} | CV Mean: {cv_mean:.1%}
        </div>""", unsafe_allow_html=True)

        st.markdown("")

        if ready_to_deploy(test_acc):
            st.success("✅ **El modelo supera el umbral del 75% — LISTO PARA DESPLIEGUE.**")

            # Download model
            buf = io.BytesIO()
            joblib.dump({
                "model":       st.session_state["model"],
                "scaler":      st.session_state["scaler"],
                "reducer":     st.session_state["reducer"],
                "feat_method": st.session_state["feat_method"],
                "target_names":st.session_state["target_names"],
            }, buf)
            buf.seek(0)
            st.download_button(
                "⬇️ Descargar modelo (.pkl)",
                buf, file_name="clasificador.pkl",
                mime="application/octet-stream"
            )

            st.subheader("📋 Ficha técnica del modelo")
            info = {
                "Algoritmo": st.session_state["model_label"],
                "Test Accuracy": f"{test_acc:.4f}",
                "CV Mean Accuracy": f"{cv_mean:.4f}",
                "CV Std": f"{st.session_state['cv_std']:.4f}",
                "Preproceso": "StandardScaler" if st.session_state["scaler"] else "Ninguno",
                "Feature extraction": st.session_state["feat_method"],
                "Shape de entrada": str(st.session_state["X_feat"].shape),
            }
            st.table(pd.DataFrame(info.items(), columns=["Parámetro","Valor"]))

        else:
            st.error("⚠️ El modelo NO alcanza el umbral mínimo de desempeño (75%). Prueba otro algoritmo, más features o ajusta los hiperparámetros.")

        # Benchmark comparativo
        st.subheader("🏆 Benchmark: todos los modelos")
        st.markdown("*Comparativa rápida con configuración por defecto y los mismos datos preprocesados.*")

        bench_results = []
        X_b = st.session_state["X_feat"]
        y_b = st.session_state["y_all"]
        bench_models = {
            "LDA": LinearDiscriminantAnalysis(),
            "Naive Bayes": GaussianNB(),
            "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        }
        for name, m in bench_models.items():
            scores = cross_val_score(m, X_b, y_b, cv=5, scoring="accuracy")
            bench_results.append({"Modelo": name,
                                   "CV Accuracy": scores.mean(),
                                   "Std": scores.std()})
        df_bench = pd.DataFrame(bench_results).sort_values("CV Accuracy", ascending=False)
        fig_b = px.bar(df_bench, x="Modelo", y="CV Accuracy",
                       error_y="Std", color="CV Accuracy",
                       color_continuous_scale="Viridis",
                       title="Comparativa de modelos (CV 5-fold)")
        fig_b.add_hline(y=0.75, line_dash="dash", line_color="red",
                        annotation_text="Umbral despliegue (75%)")
        fig_b.update_layout(height=380)
        st.plotly_chart(fig_b, use_container_width=True)
        st.dataframe(df_bench.style.background_gradient(subset=["CV Accuracy"], cmap="RdYlGn"),
                     use_container_width=True)

# ═══════════════════════════════════════════
# TAB 5 – USO REAL
# ═══════════════════════════════════════════
with tabs[5]:
    if not st.session_state.get("trained"):
        st.info("Primero entrena un modelo en la pestaña 🧠.")
    elif not ready_to_deploy(st.session_state["test_acc"]):
        st.warning("El modelo no alcanza el umbral mínimo. Mejora el modelo antes de usar en producción.")
    else:
        model_s    = st.session_state["model"]
        scaler_s   = st.session_state["scaler"]
        reducer_s  = st.session_state["reducer"]
        tn_s       = st.session_state["target_names"]
        cols_s     = st.session_state["X_columns"]
        feat_meth  = st.session_state["feat_method"]

        st.subheader("🧪 Predicción sobre nuevos datos")
        st.markdown("Ingresa los valores de las features para obtener una predicción en tiempo real.")

        # Example values from dataset mean
        X_raw, y_raw, _, _ = load_dataset(ds_key)
        means = X_raw.mean()

        user_inputs = {}
        n_cols_ui = 3
        ui_cols = st.columns(n_cols_ui)
        for i, col in enumerate(cols_s):
            with ui_cols[i % n_cols_ui]:
                user_inputs[col] = st.number_input(
                    col, value=float(f"{means[col]:.4f}"), format="%.4f", key=f"inp_{col}")

        if st.button("🔍 Predecir clase"):
            x_new = np.array([[user_inputs[c] for c in cols_s]])

            if scaler_s:
                x_new = scaler_s.transform(x_new)

            if reducer_s is not None and feat_meth != "Ninguno":
                x_new = reducer_s.transform(x_new)

            pred_class  = model_s.predict(x_new)[0]
            pred_label  = tn_s[pred_class]

            st.markdown(f"""
            <div class='deploy-box'>
                🎯 Clase predicha: <b>{pred_label}</b>
            </div>""", unsafe_allow_html=True)

            if hasattr(model_s, "predict_proba"):
                probs = model_s.predict_proba(x_new)[0]
                df_probs = pd.DataFrame({"Clase": tn_s, "Probabilidad": probs})
                fig_p = px.bar(df_probs, x="Clase", y="Probabilidad",
                               color="Probabilidad",
                               color_continuous_scale="Teal",
                               title="Probabilidades por clase")
                fig_p.update_layout(height=300)
                st.plotly_chart(fig_p, use_container_width=True)

        # ── Batch examples ──────────────────────────────────────
        st.divider()
        st.subheader("📦 Ejemplos en lote (muestras del test set)")
        X_te_s = st.session_state["X_te"]
        y_te_s = st.session_state["y_te"]
        y_pred_s = st.session_state["y_pred"]

        n_examples = min(10, len(X_te_s))
        df_examples = pd.DataFrame(
            X_te_s[:n_examples],
            columns=[f"F{i+1}" for i in range(X_te_s.shape[1])]
        )
        df_examples.insert(0, "Real", [tn_s[v] for v in y_te_s[:n_examples]])
        df_examples.insert(1, "Predicho", [tn_s[v] for v in y_pred_s[:n_examples]])
        df_examples["✅ OK"] = df_examples["Real"] == df_examples["Predicho"]

        st.dataframe(
            df_examples.style.applymap(
                lambda v: "background-color: #d4edda" if v is True
                          else ("background-color: #f8d7da" if v is False else ""),
                subset=["✅ OK"]
            ),
            use_container_width=True
        )

        # ── Code snippet ──────────────────────────────────────
        st.divider()
        st.subheader("📄 Código de inferencia (Python)")
        code_snippet = '''
import joblib, numpy as np

# Cargar el artefacto
artifact = joblib.load("clasificador.pkl")
model    = artifact["model"]
scaler   = artifact["scaler"]
reducer  = artifact["reducer"]

# Datos nuevos (shape: [n_muestras, n_features_originales])
x_new = np.array([[...]])  # <- rellena con tus valores

if scaler:  x_new = scaler.transform(x_new)
if reducer: x_new = reducer.transform(x_new)

clase_idx = model.predict(x_new)[0]
clases    = artifact["target_names"]
print("Clase predicha:", clases[clase_idx])
'''
        st.code(code_snippet, language="python")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center>🤖 ML Classification Platform · Built with Streamlit + scikit-learn</center>",
    unsafe_allow_html=True
)
