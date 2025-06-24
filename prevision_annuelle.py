import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import chardet
import io
from plotly.subplots import make_subplots

# Configuration initiale
def init_page():
    st.set_page_config(
        page_title="Prévision de Séries Temporelles",
        page_icon="📈",
        layout="wide"
    )
    st.title("📊 Prévision de séries annuelles")

# Fonction pour charger les données
@st.cache_data
def load_data(uploaded_file):
    try:
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        data = pd.read_csv(
            io.StringIO(raw_data.decode(encoding)),
            sep=';',
            decimal=','
        )
        
        data['Annee'] = pd.to_datetime(data['Annee'], format='%Y')
        data.set_index('Annee', inplace=True)
        
        if data['Valeur'].dtype == object:
            data['Valeur'] = pd.to_numeric(
                data['Valeur'].str.replace(',', '.'),
                errors='coerce'
            ).dropna()
        
        return data
    except Exception as e:
        st.error(f"Erreur de lecture du fichier: {str(e)}")
        return None

# Fonction pour tracer ACF/PACF avec Plotly
def plot_plotly_acf_pacf(series, lags=20):
    fig = make_subplots(rows=2, cols=1, subplot_titles=("ACF", "PACF"))
    
    # ACF
    acf_plot = plot_acf(series, lags=lags)
    for data in acf_plot.gca().lines:
        x = data.get_xdata()
        y = data.get_ydata()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'), row=1, col=1)
    
    # PACF
    pacf_plot = plot_pacf(series, lags=lags)
    for data in pacf_plot.gca().lines:
        x = data.get_xdata()
        y = data.get_ydata()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=False)
    return fig

# Fonction de test de stationnarité
def test_stationarity(timeseries, window=12):
    st.subheader("Test de Stationnarité")
    
    # Test ADF
    st.markdown("### 1. Test de Dickey-Fuller Augmenté (ADF)")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                       index=['Statistique de test', 'p-value',
                             'Retards utilisés', "Nombre d'observations"])
    
    for key, value in dftest[4].items():
        dfoutput[f'Valeur critique ({key})'] = value
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(dfoutput.to_frame().T)
    
    with col2:
        if dfoutput['p-value'] < 0.05:
            st.success("✅ La série est probablement stationnaire (p-value < 0.05)")
        else:
            st.warning("❌ La série est probablement non stationnaire (p-value ≥ 0.05)")
    
    # Statistiques mobiles
    st.markdown("### 2. Analyse des Statistiques Mobiles")
    rolling_mean = timeseries.rolling(window=window).mean()
    rolling_std = timeseries.rolling(window=window).std()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries, name='Série originale'))
    fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, name=f'Moyenne mobile ({window}p)'))
    fig.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std, name=f'Écart-type ({window}p)'))
    fig.update_layout(title='Statistiques Mobiles')
    st.plotly_chart(fig, use_container_width=True)
    
    return dfoutput['p-value']

# Fonction d'évaluation ARIMA
def evaluate_arima_model(X, arima_order):
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

# Fonction principale
def main():
    init_page()
    warnings.filterwarnings("ignore")
    
    # Initialisation des variables
    uploaded_file = None
    p, d, q = 1, 1, 1  # Valeurs par défaut
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader(
            "Télécharger le fichier CSV",
            type=["csv"],
            help="Fichier avec colonnes 'Annee' et 'Valeur'"
        )
        
        if uploaded_file is not None:
            st.subheader("Paramètres ARIMA")
            analysis_tab = st.radio(
                "Mode d'analyse",
                ["Automatique", "Manuel"],
                index=0
            )
            
            if analysis_tab == "Manuel":
                p = st.slider("Ordre AR (p)", 0, 10, 1)
                d = st.slider("Ordre de différenciation (d)", 0, 2, 1)
                q = st.slider("Ordre MA (q)", 0, 10, 1)
            else:
                st.markdown("**Plage de recherche:**")
                p_range = st.slider("Pour p", 0, 10, (0, 5))
                q_range = st.slider("Pour q", 0, 10, (0, 5))
            
            forecast_years = st.slider("Années à prévoir", 1, 20, 5)

    # Contenu principal
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        
        if data is not None:
            series = data['Valeur']
            
            # Section Exploration
            st.header("Analyse Exploratoire")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=series,
                    mode='lines+markers'
                ))
                fig.update_layout(
                    title='Évolution Historique',
                    xaxis_title='Année',
                    yaxis_title='Valeur'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Dernière valeur", f"{series.iloc[-1]:,.0f}")
                st.dataframe(series.describe().to_frame().style.format("{:,.0f}"))

            # Section Analyse
            st.header("Analyse de Stationnarité")
            p_value = test_stationarity(series)
            
            if p_value >= 0.05:
                st.warning("La série nécessite une différenciation")
                series_diff = series.diff().dropna()
                st.plotly_chart(plot_plotly_acf_pacf(series_diff))

            # Section Modélisation
            st.header("Modélisation ARIMA")
            
            if 'analysis_tab' in locals() and analysis_tab == "Automatique":
                st.info("Recherche des meilleurs paramètres...")
                best_score, best_cfg = float("inf"), None
                for p in range(p_range[0], p_range[1]+1):
                    for q in range(q_range[0], q_range[1]+1):
                        try:
                            rmse, _ = evaluate_arima_model(series, (p,d,q))
                            if rmse < best_score:
                                best_score, best_cfg = rmse, (p,d,q)
                        except:
                            continue
                
                if best_cfg:
                    p, d, q = best_cfg
                    st.success(f"Meilleur modèle: ARIMA{best_cfg} (RMSE: {best_score:.2f})")

            try:
                model = ARIMA(series, order=(p,d,q))
                model_fit = model.fit()
                
                st.subheader("Résultats du Modèle")
                st.text(model_fit.summary())
                
                # Prévisions
                forecast = model_fit.get_forecast(steps=forecast_years)
                future_dates = pd.date_range(
                    start=series.index[-1] + pd.DateOffset(years=1),
                    periods=forecast_years,
                    freq='YS'
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series.index, y=series, name='Historique'
                ))
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast.predicted_mean,
                    name='Prévision'
                ))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([future_dates, future_dates[::-1]]),
                    y=np.concatenate([forecast.conf_int().iloc[:,0], forecast.conf_int().iloc[:,1][::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Intervalle de confiance'
                ))
                fig.update_layout(title='Prévisions')
                st.plotly_chart(fig, use_container_width=True)
                
                # Affichage tableau prévisions
                forecast_df = pd.DataFrame({
                    'Année': future_dates.year,
                    'Prévision': forecast.predicted_mean.round(2),
                    'Borne inférieure': forecast.conf_int().iloc[:,0].round(2),
                    'Borne supérieure': forecast.conf_int().iloc[:,1].round(2)
                }).set_index('Année')
                
                st.dataframe(forecast_df.style.format("{:,.2f}"))
                
                # Téléchargement
                csv = forecast_df.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Télécharger les prévisions",
                    csv,
                    "previsions.csv",
                    "text/csv"
                )
            
            except Exception as e:
                st.error(f"Erreur dans la modélisation: {str(e)}")
    else:
        st.info("Veuillez uploader un fichier CSV pour commencer")
        st.markdown("""
        **Format attendu:**
        ```
        Annee;Valeur
        2020;100
        2021;120
        ```
        """)

if __name__ == "__main__":
    main()