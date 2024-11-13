import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import randint
import time

# Dataset inladen
file_path = "Uitslagen.xlsx"
df = pd.read_excel(file_path)

# Streamlit layout instellingen
st.title("Voetbalwedstrijd Simulatie")
st.write("Voorspel de resultaten van A3MTeam tegen De Bengels.")

# Aantal simulaties instellen
num_simulaties = st.slider("Aantal simulaties", min_value=100, max_value=2000, step=100, value=500)

# Startknop om de simulatie uit te voeren
if st.button("Start Simulatie"):
    # Normaliseer de teamnamen om problemen met inconsistenties te voorkomen
    df['ThuisTeam'] = df['ThuisTeam'].str.strip().str.lower()
    df['UitTeam'] = df['UitTeam'].str.strip().str.lower()

    # Functie om de recente vorm te berekenen
    def calculate_recent_form(team, match_data, num_matches=5):
        recent_matches = match_data[(match_data['ThuisTeam'] == team) | (match_data['UitTeam'] == team)].tail(num_matches)
        points = 0
        for _, match in recent_matches.iterrows():
            if match['ThuisTeam'] == team:
                if match['ThuisWin'] == 1:
                    points += 3
                elif match['Gelijkspel'] == 1:
                    points += 1
            elif match['UitTeam'] == team:
                if match['Uitwin'] == 1:
                    points += 3
                elif match['Gelijkspel'] == 1:
                    points += 1
        return points / num_matches if recent_matches.shape[0] == num_matches else None

    # Initialiseer de nieuwe kolommen voor de features
    df['HomeTeam_Form'] = None
    df['AwayTeam_Form'] = None
    df['HomeTeam_GoalDiff'] = 0
    df['AwayTeam_GoalDiff'] = 0
    df['HomeTeam_Points'] = 0
    df['AwayTeam_Points'] = 0

    # Huidige statistieken per team bijhouden
    team_stats = {team: {'points': 0, 'goals_for': 0, 'goals_against': 0, 'goal_diff': 0} for team in pd.concat([df['ThuisTeam'], df['UitTeam']]).unique()}

    # Loop over de dataset om features per wedstrijd te berekenen
    for index, row in df.iterrows():
        thuis_team = row['ThuisTeam']
        uit_team = row['UitTeam']
        
        # Voeg de huidige vorm van elk team toe
        df.at[index, 'HomeTeam_Form'] = calculate_recent_form(thuis_team, df[:index])
        df.at[index, 'AwayTeam_Form'] = calculate_recent_form(uit_team, df[:index])
        
        # Voeg doelpunten voor en tegen en doelsaldo toe
        df.at[index, 'HomeTeam_GoalDiff'] = team_stats[thuis_team]['goal_diff']
        df.at[index, 'AwayTeam_GoalDiff'] = team_stats[uit_team]['goal_diff']
        df.at[index, 'HomeTeam_Points'] = team_stats[thuis_team]['points']
        df.at[index, 'AwayTeam_Points'] = team_stats[uit_team]['points']
        
        # Update statistieken op basis van huidige wedstrijd
        team_stats[thuis_team]['goals_for'] += row['ThuisScore']
        team_stats[thuis_team]['goals_against'] += row['UitScore']
        team_stats[uit_team]['goals_for'] += row['UitScore']
        team_stats[uit_team]['goals_against'] += row['ThuisScore']
        
        # Update doelsaldo
        team_stats[thuis_team]['goal_diff'] = team_stats[thuis_team]['goals_for'] - team_stats[thuis_team]['goals_against']
        team_stats[uit_team]['goal_diff'] = team_stats[uit_team]['goals_for'] - team_stats[uit_team]['goals_against']
        
        # Update punten
        if row['ThuisWin'] == 1:
            team_stats[thuis_team]['points'] += 3
        elif row['Uitwin'] == 1:
            team_stats[uit_team]['points'] += 3
        elif row['Gelijkspel'] == 1:
            team_stats[thuis_team]['points'] += 1
            team_stats[uit_team]['points'] += 1

    # Doelvariabele instellen
    df['Result'] = df.apply(lambda row: 1 if row['ThuisWin'] == 1 else (0 if row['Gelijkspel'] == 1 else -1), axis=1)

    # Selecteer de features en de doelvariabele
    features = ['HomeTeam_GoalDiff', 'AwayTeam_GoalDiff', 'HomeTeam_Form', 'AwayTeam_Form', 'HomeTeam_Points', 'AwayTeam_Points']
    X = df.dropna(subset=features)[features]  # Filter om ontbrekende waarden in features te verwijderen
    y = df['Result'].loc[X.index]

    # Bereken class weights om ongebalanceerde data aan te pakken
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    weights = {i: weight for i, weight in zip(np.unique(y), class_weights)}

    # Random Forest hyperparameter tuning met RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(80, 120),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'class_weight': [weights]
    }
    model = RandomForestClassifier()
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42)
    random_search.fit(X, y)
    best_model = random_search.best_estimator_

    # Simulatie uitvoeren met voortgangsbalk
    a3mteam_wins, a3mteam_draws, a3mteam_losses = 0, 0, 0
    
    progress_bar = st.progress(0)
    for sim_num in range(num_simulaties):
        stand = {team: 0 for team in pd.concat([df['ThuisTeam'], df['UitTeam']]).unique()}
        
        for _, wedstrijd in df[df['ThuisScore'].isna()].iterrows():
            thuis_team = wedstrijd['ThuisTeam']
            uit_team = wedstrijd['UitTeam']
            
            # Kenmerken van de wedstrijd met willekeurige ruis toegevoegd
            match_features = pd.DataFrame([[
                wedstrijd['HomeTeam_GoalDiff'] + np.random.normal(0, 0.1),
                wedstrijd['AwayTeam_GoalDiff'] + np.random.normal(0, 0.1),
                wedstrijd['HomeTeam_Form'] + np.random.normal(0, 0.1),
                wedstrijd['AwayTeam_Form'] + np.random.normal(0, 0.1),
                wedstrijd['HomeTeam_Points'] + np.random.normal(0, 0.1),
                wedstrijd['AwayTeam_Points'] + np.random.normal(0, 0.1)
            ]], columns=features)
            
            # Voorspelling met waarschijnlijkheden
            voorspelling_proba = best_model.predict_proba(match_features)[0]
            proba_dict = {class_label: prob for class_label, prob in zip(best_model.classes_, voorspelling_proba)}
            voorspelling_proba_reordered = [proba_dict.get(1, 0), 0.01, proba_dict.get(-1, 0)]
            total_prob = sum(voorspelling_proba_reordered)
            voorspelling_proba_reordered = [p / total_prob for p in voorspelling_proba_reordered]

            # Kies een uitkomst gebaseerd op de aangepaste waarschijnlijkheden
            voorspelling = np.random.choice([1, 0, -1], p=voorspelling_proba_reordered)

            if voorspelling == 1:  # Thuisteam wint
                if thuis_team == 'a3mteam' and uit_team == 'de bengels':
                    a3mteam_wins += 1
            elif voorspelling == -1:  # Uitteam wint
                if uit_team == 'a3mteam' and thuis_team == 'de bengels':
                    a3mteam_losses += 1
            elif voorspelling == 0:  # Gelijkspel
                if (thuis_team == 'a3mteam' and uit_team == 'de bengels') or (thuis_team == 'de bengels' and uit_team == 'a3mteam'):
                    a3mteam_draws += 1

        # Voortgangsbalk updaten
        progress_bar.progress((sim_num + 1) / num_simulaties)
        time.sleep(0.01)  # Simulatie vertragen voor visualisatie-effect in de voortgangsbalk

    # Resultaten in een histogram tonen
    st.subheader("Resultaten van A3MTeam tegen De Bengels")
    resultaten = ['Winst', 'Gelijkspel', 'Verlies']
    a3mteam_resultaten = [a3mteam_wins, a3mteam_draws, a3mteam_losses]

    # Gebruik Streamlit's ingebouwde functie voor het histogram
    st.bar_chart({
        "Resultaten": resultaten,
        "Aantal": a3mteam_resultaten
    })
    
    # Aantallen boven de balken toevoegen
    for i, resultaat in enumerate(a3mteam_resultaten):
        st.write(f"{resultaten[i]}: {resultaat} keer")

