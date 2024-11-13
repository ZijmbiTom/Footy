import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import randint
import altair as alt
import time

# Dataset inladen
file_path = "Uitslagen.xlsx"  # Zorg dat dit bestand in dezelfde map staat als je script
df = pd.read_excel(file_path)

# Streamlit titel
st.title("Voetbalwedstrijd Simulatie")
st.write("Voorspel de resultaten van A3MTeam tegen De Bengels.")

# Instellen aantal simulaties
num_simulaties = st.slider("Aantal simulaties", min_value=100, max_value=2000, step=100, value=500)

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

# Data preprocessing
df['ThuisTeam'] = df['ThuisTeam'].str.strip().str.lower()
df['UitTeam'] = df['UitTeam'].str.strip().str.lower()
df['HomeTeam_Form'] = None
df['AwayTeam_Form'] = None
df['HomeTeam_GoalDiff'] = 0
df['AwayTeam_GoalDiff'] = 0
df['HomeTeam_Points'] = 0
df['AwayTeam_Points'] = 0

team_stats = {team: {'points': 0, 'goals_for': 0, 'goals_against': 0, 'goal_diff': 0} for team in pd.concat([df['ThuisTeam'], df['UitTeam']]).unique()}

for index, row in df.iterrows():
    thuis_team = row['ThuisTeam']
    uit_team = row['UitTeam']
    df.at[index, 'HomeTeam_Form'] = calculate_recent_form(thuis_team, df[:index])
    df.at[index, 'AwayTeam_Form'] = calculate_recent_form(uit_team, df[:index])
    df.at[index, 'HomeTeam_GoalDiff'] = team_stats[thuis_team]['goal_diff']
    df.at[index, 'AwayTeam_GoalDiff'] = team_stats[uit_team]['goal_diff']
    df.at[index, 'HomeTeam_Points'] = team_stats[thuis_team]['points']
    df.at[index, 'AwayTeam_Points'] = team_stats[uit_team]['points']

    team_stats[thuis_team]['goals_for'] += row['ThuisScore']
    team_stats[thuis_team]['goals_against'] += row['UitScore']
    team_stats[uit_team]['goals_for'] += row['UitScore']
    team_stats[uit_team]['goals_against'] += row['ThuisScore']
    team_stats[thuis_team]['goal_diff'] = team_stats[thuis_team]['goals_for'] - team_stats[thuis_team]['goals_against']
    team_stats[uit_team]['goal_diff'] = team_stats[uit_team]['goals_for'] - team_stats[uit_team]['goals_against']
    if row['ThuisWin'] == 1:
        team_stats[thuis_team]['points'] += 3
    elif row['Uitwin'] == 1:
        team_stats[uit_team]['points'] += 3
    elif row['Gelijkspel'] == 1:
        team_stats[thuis_team]['points'] += 1
        team_stats[uit_team]['points'] += 1

# Random Forest Model instellen
df['Result'] = df.apply(lambda row: 1 if row['ThuisWin'] == 1 else (0 if row['Gelijkspel'] == 1 else -1), axis=1)
features = ['HomeTeam_GoalDiff', 'AwayTeam_GoalDiff', 'HomeTeam_Form', 'AwayTeam_Form', 'HomeTeam_Points', 'AwayTeam_Points']
X = df.dropna(subset=features)[features]
y = df['Result'].loc[X.index]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
weights = {i: weight for i, weight in zip(np.unique(y), class_weights)}
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

# Simulatie uitvoeren als gebruiker op de knop drukt
if st.button("Start Simulatie"):
    a3mteam_wins, a3mteam_draws, a3mteam_losses = 0, 0, 0
    
    for sim_num in range(num_simulaties):
        stand = {team: 0 for team in pd.concat([df['ThuisTeam'], df['UitTeam']]).unique()}
        
        for _, wedstrijd in df[df['ThuisScore'].isna()].iterrows():
            thuis_team = wedstrijd['ThuisTeam']
            uit_team = wedstrijd['UitTeam']
            match_features = pd.DataFrame([[
                wedstrijd['HomeTeam_GoalDiff'] + np.random.normal(0, 0.1),
                wedstrijd['AwayTeam_GoalDiff'] + np.random.normal(0, 0.1),
                wedstrijd['HomeTeam_Form'] + np.random.normal(0, 0.1),
                wedstrijd['AwayTeam_Form'] + np.random.normal(0, 0.1),
                wedstrijd['HomeTeam_Points'] + np.random.normal(0, 0.1),
                wedstrijd['AwayTeam_Points'] + np.random.normal(0, 0.1)
            ]], columns=features)
            voorspelling_proba = best_model.predict_proba(match_features)[0]
            proba_dict = {class_label: prob for class_label, prob in zip(best_model.classes_, voorspelling_proba)}
            voorspelling_proba_reordered = [proba_dict.get(1, 0), max(proba_dict.get(0, 0), 0.01), proba_dict.get(-1, 0)]
            total_prob = sum(voorspelling_proba_reordered)
            voorspelling_proba_reordered = [p / total_prob for p in voorspelling_proba_reordered]
            voorspelling = np.random.choice([1, 0, -1], p=voorspelling_proba_reordered)
            
            if voorspelling == 1 and thuis_team == 'a3mteam':
                a3mteam_wins += 1
            elif voorspelling == -1 and uit_team == 'a3mteam':
                a3mteam_losses += 1
            elif voorspelling == 0:
                a3mteam_draws += 1

    # Resultaten in een Altair-grafiek tonen
    resultaten = pd.DataFrame({
        'Resultaat': ['Winst', 'Gelijkspel', 'Verlies'],
        'Aantal': [a3mteam_wins, a3mteam_draws, a3mteam_losses]
    })
    chart = alt.Chart(resultaten).mark_bar().encode(
        y=alt.Y('Resultaat:N', sort='-x'),
        x=alt.X('Aantal:Q', title='Aantal'),
        color=alt.Color('Resultaat:N', legend=None)
    ).properties(title='Resultaten van A3MTeam tegen De Bengels')
    
    text = chart.mark_text(
        align='left',
        baseline='middle',
        dx=3
    ).encode(
        text='Aantal:Q'
    )
    
    st.altair_chart(chart + text, use_container_width=True)
