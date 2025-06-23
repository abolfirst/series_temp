import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import chardet
import io

warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="Prévision de Séries Temporelles", 
    page_icon="", 
    layout="wide"
)
st.title(" Prévision de séries annuelles")

# ==============================================
# FONCTIONS UTILITAIRES AMÉLIORÉES
# ==============================================

def test_stationarity(timeseries, window=12):
    """Test complet de stationnarité avec visualisations"""
    st.subheader("Test de Stationnarité")
    
    # 1. Test ADF
    st.markdown("### 1. Test de Dickey-Fuller Augmenté (ADF)")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                       index=['Statistique de test', 'p-value',
                             'Retards utilisés', "Nombre d'observations"])
    
    for key, value in dftest[4].items():
        dfoutput[f'Valeur critique ({key})'] = value
    
    # Affichage des résultats
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(dfoutput.to_frame().T)
    
    with col2:
        if dfoutput['p-value'] < 0.05:
            st.success("✅ La série est probablement stationnaire (p-value < 0.05)")
        else:
            st.warning("❌ La série est probablement non stationnaire (p-value ≥ 0.05)")
    
    # 2. Statistiques mobiles
    st.markdown("### 2. Analyse des Statistiques Mobiles")
    rolling_mean = timeseries.rolling(window=window).mean()
    rolling_std = timeseries.rolling(window=window).std()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timeseries, label='Série originale', color='blue')
    ax.plot(rolling_mean, label=f'Moyenne mobile ({window}périodes)', color='red')
    ax.plot(rolling_std, label=f'Écart-type mobile ({window}périodes)', color='green')
    ax.legend(loc='best')
    ax.set_title('Évolution des Statistiques Mobiles')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    # 3. Test de différenciation
    st.markdown("### 3. Test de Différenciation")
    timeseries_diff = timeseries.diff().dropna()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(timeseries)
    ax1.set_title('Série Originale')
    ax1.grid(True)
    
    ax2.plot(timeseries_diff)
    ax2.set_title('Première Différence (d=1)')
    ax2.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Test ADF sur série différenciée si nécessaire
    if dfoutput['p-value'] >= 0.05:
        st.markdown("**Test ADF sur la série différenciée:**")
        dftest_diff = adfuller(timeseries_diff, autolag='AIC')
        
        dfoutput_diff = pd.Series(dftest_diff[0:4], 
                                index=['Statistique de test', 'p-value',
                                      'Retards utilisés', "Nombre d'observations"])
        
        for key, value in dftest_diff[4].items():
            dfoutput_diff[f'Valeur critique ({key})'] = value
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(dfoutput_diff.to_frame().T)
        
        with col2:
            if dfoutput_diff['p-value'] < 0.05:
                st.success("✅ La série différenciée est stationnaire (p-value < 0.05)")
                return 1  # Ordre de différenciation nécessaire
            else:
                st.error("❌ La série reste non stationnaire après différence")
                return None
    
    return 0  # Si la série était déjà stationnaire

def evaluate_arima_model(X, arima_order):
    """Évaluation du modèle ARIMA avec validation croisée"""
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    
    for t in range(len(test)):
        try:
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        except:
            return float('inf'), float('inf')
    
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    return rmse, mae

@st.cache_data
def load_data(uploaded_file):
    """Charge les données avec gestion robuste de l'encodage"""
    try:
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        data = pd.read_csv(
            io.StringIO(raw_data.decode(encoding)), 
            sep=';',
            decimal=','
        )
        
        # Conversion et nettoyage
        data['Annee'] = pd.to_datetime(data['Annee'], format='%Y')
        data.set_index('Annee', inplace=True)
        
        # Conversion des valeurs
        if data['Valeur'].dtype == object:
            data['Valeur'] = pd.to_numeric(
                data['Valeur'].str.replace(',', '.'), 
                errors='coerce'
            ).dropna()
        
        return data
    except Exception as e:
        st.error(f"Erreur de lecture du fichier: {str(e)}")
        return None

# ==============================================
# INTERFACE UTILISATEUR
# ==============================================

# Sidebar pour les paramètres
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader(
        "Télécharger le fichier CSV", 
        type=["csv"],
        help="Fichier avec colonnes 'Annee' et 'Valeur'"
    )
    
    if uploaded_file is not None:
        st.subheader("Paramètres ARIMA")
        
        # Choix du mode
        analysis_tab = st.radio(
            "Mode d'analyse",
            ["Automatique", "Manuel"],
            index=0
        )
        
        if analysis_tab == "Manuel":
            st.markdown("**Paramètres ARIMA:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.slider("Ordre AR (p)", 0, 10, 1)
            with col2:
                d = st.slider("Ordre de différenciation (d)", 0, 2, 1)
            with col3:
                q = st.slider("Ordre MA (q)", 0, 10, 1)
        else:
            st.markdown("**Plage de recherche:**")
            col1, col2 = st.columns(2)
            with col1:
                p_range = st.slider("Pour p", 0, 10, (0, 5))
            with col2:
                q_range = st.slider("Pour q", 0, 10, (0, 5))
        
        forecast_years = st.slider(
            "Années à prévoir", 
            1, 20, 5
        )

# Contenu principal
if uploaded_file is not None:
    data = load_data(uploaded_file)
    
    if data is not None:
        series = data['Valeur']
        
        # Section 1: Exploration des données
        st.header("Analyse Exploratoire")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(series, marker='o', linestyle='-', linewidth=1)
            ax.set_title('Évolution de la Production (1961-2022)')
            ax.set_xlabel('Année')
            ax.set_ylabel('Production (tonnes)')
            ax.grid(True)
            st.pyplot(fig)
        
        with col2:
            st.metric(
                "Dernière observation", 
                f"{series.iloc[-1]:,.0f} tonnes",
                f"{(series.iloc[-1]-series.iloc[-2])/series.iloc[-2]*100:.1f}% vs année précédente"
            )
            
            st.dataframe(
                series.describe().to_frame().style.format("{:,.0f}"),
                height=250
            )
        
        # Section 2: Analyse de stationnarité
        st.header("Diagnostic de Stationnarité")
        d = test_stationarity(series)
        
        if d is not None:
            st.session_state['d_value'] = d
            st.info(f"Recommandation: utiliser d={d} pour stationnariser la série")
            
            # Afficher ACF/PACF si différenciation nécessaire
            if d > 0:
                st.subheader("Analyse ACF/PACF sur Série Stationnarisée")
                series_diff = series.diff(d).dropna()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Fonction d'Autocorrélation (ACF)**")
                    fig, ax = plt.subplots(figsize=(8, 3))
                    plot_acf(series_diff, lags=20, ax=ax)
                    st.pyplot(fig)
                
                with col2:
                    st.write("**Fonction d'Autocorrélation Partielle (PACF)**")
                    fig, ax = plt.subplots(figsize=(8, 3))
                    plot_pacf(series_diff, lags=20, ax=ax)
                    st.pyplot(fig)
        
        # Section 3: Modélisation ARIMA
        st.header("Modélisation ARIMA")
        
        if analysis_tab == "Automatique" and d is not None:
            st.info("Recherche des meilleurs paramètres ARIMA...")
            
            p_values = range(p_range[0], p_range[1]+1)
            d_values = [d]
            q_values = range(q_range[0], q_range[1]+1)
            
            best_score = float('inf')
            best_cfg = None
            results = []
            
            progress_bar = st.progress(0)
            total = len(p_values) * len(d_values) * len(q_values)
            current = 0
            
            for p in p_values:
                for d_val in d_values:
                    for q in q_values:
                        order = (p, d_val, q)
                        current += 1
                        progress_bar.progress(current/total)
                        
                        try:
                            rmse, mae = evaluate_arima_model(series, order)
                            results.append({
                                "Ordre": order,
                                "RMSE": f"{rmse:,.0f}",
                                "MAE": f"{mae:,.0f}",
                                "_RMSE": rmse
                            })
                            
                            if rmse < best_score:
                                best_score, best_cfg = rmse, order
                        except:
                            continue
            
            if best_cfg:
                st.success(f"Meilleur modèle: ARIMA{best_cfg} (RMSE: {best_score:,.0f} tonnes)")
                p, d, q = best_cfg
                
                st.dataframe(
                    pd.DataFrame(results).sort_values("_RMSE").drop("_RMSE", axis=1),
                    height=300
                )
            else:
                st.error("Aucun modèle valide trouvé. Essayez d'élargir les plages de recherche.")
        
        # Entraînement du modèle final
        if 'p' in locals() and 'd' in locals() and 'q' in locals():
            try:
                with st.spinner("Ajustement du modèle ARIMA..."):
                    model = ARIMA(series, order=(p, d, q))
                    model_fit = model.fit()
                    
                    # Affichage des résultats
                    st.subheader("Résultats du Modèle")
                    st.text(str(model_fit.summary()))
                    
                    # Analyse des résidus
                    st.subheader("Diagnostic des Résidus")
                    
                    residuals = pd.DataFrame(model_fit.resid)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Distribution des résidus**")
                        fig, ax = plt.subplots(figsize=(8, 3))
                        residuals.plot(kind='kde', ax=ax, title='Densité des résidus')
                        ax.grid(True)
                        st.pyplot(fig)
                    
                    with col2:
                        st.write("**Autocorrélation des résidus**")
                        fig, ax = plt.subplots(figsize=(8, 3))
                        plot_acf(residuals.dropna(), lags=20, ax=ax)
                        st.pyplot(fig)
                    
                    # Prévisions
                    st.subheader(f"Prévisions pour {forecast_years} ans")
                    
                    forecast = model_fit.get_forecast(steps=forecast_years)
                    forecast_mean = forecast.predicted_mean
                    conf_int = forecast.conf_int()
                    
                    # Création du graphique
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Historique
                    ax.plot(series.index, series, 
                           label='Données Historiques', 
                           marker='o')
                    
                    # Prévisions
                    future_dates = pd.date_range(
                        start=series.index[-1] + pd.DateOffset(years=1),
                        periods=forecast_years,
                        freq='YS'
                    )
                    ax.plot(future_dates, forecast_mean, 
                            label='Prévisions', 
                            color='red', 
                            marker='o')
                    ax.fill_between(future_dates,
                                   conf_int.iloc[:, 0],
                                   conf_int.iloc[:, 1],
                                   color='pink', alpha=0.3,
                                   label='Intervalle de confiance 95%')
                    
                    ax.legend(loc='upper left')
                    ax.set_title('Prévisions de Production')
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    # Tableau des prévisions
                    forecast_df = pd.DataFrame({
                        'Année': future_dates.year,
                        'Prévision (tonnes)': forecast_mean.round(),
                        'Borne inférieure': conf_int.iloc[:, 0].round(),
                        'Borne supérieure': conf_int.iloc[:, 1].round()
                    }).set_index('Année')
                    
                    st.dataframe(
                        forecast_df.style.format("{:,.0f}"),
                        height=200
                    )
                    
                    # Téléchargement
                    csv = forecast_df.to_csv().encode('utf-8')
                    st.download_button(
                        "📥 Télécharger les prévisions",
                        csv,
                        "previsions_banane.csv",
                        "text/csv"
                    )
            
            except Exception as e:
                st.error(f"Erreur dans la modélisation: {str(e)}")

else:
    st.info("Veuillez télécharger un fichier CSV pour commencer l'analyse.")
    
    # Exemple de structure
    st.markdown("""
    **Format de fichier attendu:**
    ```csv
    Annee;Valeur
    1961;500000
    1962;570000
    ```
    """)

# Pied de page
st.markdown("---")
st.caption("Application développée par la sous-direction des prévisions | © juin 2025")