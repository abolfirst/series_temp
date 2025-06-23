# Application de Prévision de Séries Temporelles Annuelles
Cette application Streamlit permet d’analyser et de modéliser des séries temporelles annuelles à l’aide d’ARIMA. Elle a été conçue pour aider les économistes, analystes et planificateurs à explorer des données historiques (comme des productions agricoles), tester la stationnarité, ajuster un modèle et produire des prévisions exploitables jusqu’à 20 ans.

🔧 Fonctionnalités principales 📁 Importation de fichiers CSV (auto-détection de l’encodage)

📊 Test de stationnarité (ADF) + visualisation de tendance

⚙️ Choix entre :

Mode automatique (recherche des meilleurs paramètres ARIMA)

Mode manuel (sélection libre de p, d, q)

🔍 Affichage interactif des graphiques ACF / PACF

📉 Modélisation avec statsmodels.ARIMA

🧮 Évaluation avec RMSE, MAE

📤 Téléchargement des prévisions

🏁 Lancer l’application Pour lancer l'application , les instructions suivante sont necessaires: 
pip install -r requirements.txt 
streamlit run prevision_annuelle.py
