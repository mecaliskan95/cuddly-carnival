# DATA PREPROCESSING

#Import Libraries
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
random_seed = 42
np.random.seed(random_seed)

# LOAD DATA

international_matches_df = pd.read_csv(r'C:\Users\mecal\OneDrive\Documents\cuddly-carnival\ML_project\international_matches.csv')

# DATA PREPROCESSING

# Convert the date column to datetime format
international_matches_df['date'] = pd.to_datetime(international_matches_df['date'])

# Heatmap for checking the NA values in the dataset
sns.heatmap(international_matches_df.isnull(), cbar=False, cmap='viridis')

# Pre-processing for NA values
# Fill the values with average of each team
score_columns_home = ['home_team_goalkeeper_score',
                  'home_team_mean_defense_score', 'home_team_mean_offense_score',
                  'home_team_mean_midfield_score']

score_columns_away = ['away_team_goalkeeper_score',
                  'away_team_mean_defense_score','away_team_mean_offense_score',
                  'away_team_mean_midfield_score']

for team in international_matches_df['home_team'].unique():
    team_mask = (international_matches_df['home_team'] == team)
    team_means = international_matches_df.loc[team_mask, score_columns_home].mean()
    international_matches_df.loc[team_mask, score_columns_home] = international_matches_df.loc[team_mask, score_columns_home].fillna(team_means)
for team in international_matches_df['away_team'].unique():
    team_mask = (international_matches_df['away_team'] == team)
    team_means = international_matches_df.loc[team_mask, score_columns_away].mean()
    international_matches_df.loc[team_mask, score_columns_away] = international_matches_df.loc[team_mask, score_columns_away].fillna(team_means)

# There are still missing values. Let's fill them with 50, which is the average score.
international_matches_df.fillna(50,inplace=True)

# DATA ANALYSIS
plt.rc('font', size=14)
plt.rc('axes', titlesize=16, labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
# Top 10 teams based on the highest total_fifa_points
home_teams_df = international_matches_df[['home_team', 'home_team_total_fifa_points']].copy()
home_teams_df.columns = ['team', 'total_fifa_points']
away_teams_df = international_matches_df[['away_team', 'away_team_total_fifa_points']].copy()
away_teams_df.columns = ['team', 'total_fifa_points']
all_teams_df = pd.concat([home_teams_df, away_teams_df], ignore_index=True)
top_teams = all_teams_df.groupby('team')['total_fifa_points'].max().nlargest(10)
top_teams_df = top_teams.reset_index()
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(x='team', y='total_fifa_points', data=top_teams_df, palette='viridis')
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.0f'),
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.title('Top 10 Teams: Total FIFA Points')
plt.xlabel('Team')
plt.ylabel('Total FIFA Points')
plt.xticks(rotation=45)
plt.ylim(1800,2200)
plt.show()

# Top 10 teams based on the highest goalkeeper score
home_goalkeeper_df = international_matches_df[['home_team', 'home_team_goalkeeper_score']].copy()
home_goalkeeper_df.columns = ['team', 'goalkeeper_score']
away_goalkeeper_df = international_matches_df[['away_team', 'away_team_goalkeeper_score']].copy()
away_goalkeeper_df.columns = ['team', 'goalkeeper_score']
all_goalkeeper_df = pd.concat([home_goalkeeper_df, away_goalkeeper_df], ignore_index=True)
top_goalkeeper_teams = all_goalkeeper_df.groupby('team')['goalkeeper_score'].max().nlargest(10)
top_goalkeeper_teams_df = top_goalkeeper_teams.reset_index()
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(x='team', y='goalkeeper_score', data=top_goalkeeper_teams_df, palette='viridis')
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.2f'),
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.title('Top 10 Teams: Goalkeeper Score')
plt.xlabel('Team')
plt.ylabel('Goalkeeper Score')
plt.xticks(rotation=45)
plt.ylim(0,100)
plt.show()

# Top 10 teams based on the highest mean defense score
home_defense_df = international_matches_df[['home_team', 'home_team_mean_defense_score']].copy()
home_defense_df.columns = ['team', 'mean_defense_score']
away_defense_df = international_matches_df[['away_team', 'away_team_mean_defense_score']].copy()
away_defense_df.columns = ['team', 'mean_defense_score']
all_defense_df = pd.concat([home_defense_df, away_defense_df], ignore_index=True)
top_defense_teams = all_defense_df.groupby('team')['mean_defense_score'].max().nlargest(10)
top_defense_teams_df = top_defense_teams.reset_index()
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(x='team', y='mean_defense_score', data=top_defense_teams_df, palette='viridis')
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.2f'),
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.title('Top 10 Teams: Mean Defense Score')
plt.xlabel('Team')
plt.ylabel('Mean Defense Score')
plt.xticks(rotation=45)
plt.ylim(0,100)
plt.show()

# Top 10 teams based on the highest mean midfield score
home_midfield_df = international_matches_df[['home_team', 'home_team_mean_midfield_score']].copy()
home_midfield_df.columns = ['team', 'mean_midfield_score']
away_midfield_df = international_matches_df[['away_team', 'away_team_mean_midfield_score']].copy()
away_midfield_df.columns = ['team', 'mean_midfield_score']
all_midfield_df = pd.concat([home_midfield_df, away_midfield_df], ignore_index=True)
top_midfield_teams = all_midfield_df.groupby('team')['mean_midfield_score'].max().nlargest(10)
top_midfield_teams_df = top_midfield_teams.reset_index()
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(x='team', y='mean_midfield_score', data=top_midfield_teams_df, palette='viridis')
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.2f'),
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.title('Top 10 Teams: Mean Midfield Score')
plt.xlabel('Team')
plt.ylabel('Mean Midfield Score')
plt.xticks(rotation=45)
plt.ylim(0,100)
plt.show()

# Top 10 teams based on the highest mean offense score
home_offense_df = international_matches_df[['home_team', 'home_team_mean_offense_score']].copy()
home_offense_df.columns = ['team', 'mean_offense_score']
away_offense_df = international_matches_df[['away_team', 'away_team_mean_offense_score']].copy()
away_offense_df.columns = ['team', 'mean_offense_score']
all_offense_df = pd.concat([home_offense_df, away_offense_df], ignore_index=True)
top_offense_teams = all_offense_df.groupby('team')['mean_offense_score'].max().nlargest(10)
top_offense_teams_df = top_offense_teams.reset_index()
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(x='team', y='mean_offense_score', data=top_offense_teams_df, palette='viridis')
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.2f'),
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.title('Top 10 Teams: Mean Offense Score')
plt.xlabel('Team')
plt.ylabel('Mean Offense Score')
plt.xticks(rotation=45)
plt.ylim(0,100)
plt.show()

# Distribution plot for home team results
plt.figure(figsize=(12, 6))
home_team_counts = international_matches_df['home_team_result'].value_counts()
plt.pie(home_team_counts, labels=home_team_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Home Team Results')
plt.show()
international_matches_df['away_team_result'] = international_matches_df['home_team_result'].map({'Win': 'Lose', 'Draw': 'Draw', 'Lose': 'Win'})

# Top 10 teams based on the highest win rates
teams = pd.concat([international_matches_df['home_team'], international_matches_df['away_team']])
win_rate = (
    international_matches_df.loc[international_matches_df['home_team_result'] == 'Win', 'home_team'].value_counts() +
    international_matches_df.loc[international_matches_df['away_team_result'] == 'Win', 'away_team'].value_counts()
) / teams.value_counts()
top_10_teams = win_rate.sort_values(ascending=False).head(10)
top_10_teams *= 100
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(x=top_10_teams.index, y=top_10_teams.values)  # Multiply y-axis values by 100
for index, value in enumerate(top_10_teams.values):
    bar_plot.text(index, value + 0.01, f'{value:.1f}%', ha='center', va='bottom')  # Display values as percentages
plt.title('Top 10 Teams: Highest Win Rate')
plt.xlabel('Team')
plt.ylabel('Win Rate')
plt.ylim(0, 100)  # Set y-axis limits to percentage range
plt.xticks(rotation=45, ha='right')
plt.show()

# Box Plots for Scores
score_columns = ['home_team_goalkeeper_score', 'away_team_goalkeeper_score',
                  'home_team_mean_defense_score', 'away_team_mean_defense_score',
                  'home_team_mean_midfield_score', 'away_team_mean_midfield_score',
                  'home_team_mean_offense_score', 'away_team_mean_offense_score']

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.boxplot(x='home_team_result', y=score_columns[0], data=international_matches_df, palette='pastel')
plt.title(f'Box Plot of {score_columns[0]}')
plt.xlabel('Home Team Result')
plt.ylabel(score_columns[0])

plt.subplot(2, 2, 2)
sns.boxplot(x='home_team_result', y=score_columns[1], data=international_matches_df, palette='pastel')
plt.title(f'Box Plot of {score_columns[1]}')
plt.xlabel('Home Team Result')
plt.ylabel(score_columns[1])

plt.subplot(2, 2, 3)
sns.boxplot(x='home_team_result', y=score_columns[2], data=international_matches_df, palette='pastel')
plt.title(f'Box Plot of {score_columns[2]}')
plt.xlabel('Home Team Result')
plt.ylabel(score_columns[2])

plt.subplot(2, 2, 4)
sns.boxplot(x='home_team_result', y=score_columns[3], data=international_matches_df, palette='pastel')
plt.title(f'Box Plot of {score_columns[3]}')
plt.xlabel('Home Team Result')
plt.ylabel(score_columns[3])

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.boxplot(x='home_team_result', y=score_columns[4], data=international_matches_df, palette='pastel')
plt.title(f'Box Plot of {score_columns[4]}')
plt.xlabel('Home Team Result')
plt.ylabel(score_columns[4])

plt.subplot(2, 2, 2)
sns.boxplot(x='home_team_result', y=score_columns[5], data=international_matches_df, palette='pastel')
plt.title(f'Box Plot of {score_columns[5]}')
plt.xlabel('Home Team Result')
plt.ylabel(score_columns[5])

plt.subplot(2, 2, 3)
sns.boxplot(x='home_team_result', y=score_columns[6], data=international_matches_df, palette='pastel')
plt.title(f'Box Plot of {score_columns[6]}')
plt.xlabel('Home Team Result')
plt.ylabel(score_columns[6])

plt.subplot(2, 2, 4)
sns.boxplot(x='home_team_result', y=score_columns[7], data=international_matches_df, palette='pastel')
plt.title(f'Box Plot of {score_columns[7]}')
plt.xlabel('Home Team Result')
plt.ylabel(score_columns[7])

plt.tight_layout()
plt.show()

# FEATURE ENGINEERING

#international_matches_df['goal_ratio'] = international_matches_df['home_team_score'] / (international_matches_df['home_team_score']+international_matches_df['away_team_score'])
#international_matches_df['total_fifa_points_ratio'] = international_matches_df['home_team_total_fifa_points'] / international_matches_df['away_team_total_fifa_points']
#international_matches_df['fifa_rank_difference'] = international_matches_df['home_team_fifa_rank'] - international_matches_df['away_team_fifa_rank']
#international_matches_df['average_fifa_rank'] = (international_matches_df['home_team_fifa_rank'] + international_matches_df['away_team_fifa_rank'])/2
#international_matches_df['total_fifa_points_difference'] = international_matches_df['home_team_total_fifa_points'] - international_matches_df['away_team_total_fifa_points']
international_matches_df['home_advantage'] = (international_matches_df['neutral_location'] == 1).astype(int)
international_matches_df['is_win'] = (international_matches_df['home_team_result'] == 'Win').astype(int)

# Plot correlation matrix of all numeric features
numeric_columns = international_matches_df.select_dtypes(include=['number'])
correlation_matrix_numeric = numeric_columns.corr()
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix_numeric, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Matrix - Numeric Columns')
plt.show()

# Plot correlation matrix of selected features
correlation_matrix = international_matches_df[['home_team_fifa_rank','away_team_fifa_rank','home_team_total_fifa_points','away_team_total_fifa_points','home_advantage','home_team_goalkeeper_score','away_team_goalkeeper_score','home_team_mean_defense_score','away_team_mean_defense_score','home_team_mean_midfield_score','away_team_mean_midfield_score','home_team_mean_offense_score','away_team_mean_offense_score','is_win']].corr()
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Matrix - Selected Features')
plt.show()

# PREDICTION MODEL

X, y = international_matches_df.loc[:,['home_team_fifa_rank','away_team_fifa_rank','home_team_total_fifa_points','away_team_total_fifa_points','home_advantage','home_team_goalkeeper_score','away_team_goalkeeper_score','home_team_mean_defense_score','away_team_mean_defense_score','home_team_mean_midfield_score','away_team_mean_midfield_score','home_team_mean_offense_score','away_team_mean_offense_score']], international_matches_df['is_win']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

## COMPARE MODELS
models = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Bayes Classifier', GaussianNB()),
    ('Support Vector Machines', SVC(probability=True)),
    ('Neural Network', MLPClassifier(max_iter=1000)),
    ('Deep Learning', MLPClassifier(hidden_layer_sizes=(100), max_iter=1000))
]

model_results = []
for model_name, model_instance in models:
    model_instance.fit(X_train, y_train)
    predictions = model_instance.predict(X_test)
    accuracy = round(model_instance.score(X_test, y_test) * 100, 2)
    precision = round(precision_score(y_test, predictions) * 100, 2)
    recall = round(recall_score(y_test, predictions) * 100, 2)
    f1 = round(f1_score(y_test, predictions) * 100, 2)

    # Feature Importance
    if model_name == 'Decision Tree' and hasattr(model_instance, 'feature_importances_'):
        feature_importance = model_instance.feature_importances_
        feature_names = X_train.columns
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        # Annotate the bars with importance values at the middle of the columns
        for p in bar_plot.patches:
          bar_plot.annotate(format(p.get_width(), '.2f'),
                          (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2.),
                          ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('Feature Importance - Decision Tree')
        plt.show()

    metrics_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Score': [accuracy, precision, recall, f1]
    })
    metrics_summary['Score'] = metrics_summary['Score'].round(2)
    model_results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })
models_df = pd.DataFrame(model_results)
models_df.sort_values(by='Accuracy', ascending=False, inplace=True)
print("\nModel Comparison Summary:")
print(models_df.to_string(index=False))

# ROC Curve
plt.figure(figsize=(10, 6))
for model_name, model_instance in models:
    model_instance.fit(X_train, y_train)

    if hasattr(model_instance, 'predict_proba'):
        fpr, tpr, thresholds = roc_curve(y_test, model_instance.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate\n(1-specificity)')
plt.ylabel('True Positive Rate\n(sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.ylim(0,1)
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
for model_name, model_instance in models:
    predictions = model_instance.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions, labels=[1, 0])  # Specify labels for True Positive and True Negative

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Win', 'Not Win'], yticklabels=['Win', 'Not Win'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Precision-Recall Curve
plt.figure(figsize=(10, 6))
for model_name, model_instance in models:
    if hasattr(model_instance, 'predict_proba'):
        precision, recall, _ = precision_recall_curve(y_test, model_instance.predict_proba(X_test)[:, 1])
        auc_pr = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC-PR = {auc_pr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for All Models')
plt.ylim(0,1.05)
plt.legend(loc='lower left')
plt.show()

## SVM
svm = SVC(probability=True)
model = svm.fit(X_train, y_train)

## SIMULATION OF WINNER

# Assumed that the 48 teams with highest total fifa points will be qualified for World Cup 2026
sorted_df = international_matches_df.sort_values(by='home_team_total_fifa_points', ascending=False)
qualified_teams = set()  # Use a set to ensure uniqueness

for index, row in sorted_df.iterrows():
    home_team = row['home_team']
    if home_team not in qualified_teams:
        qualified_teams.add(home_team)
    if len(qualified_teams) >= 48:
        break

international_matches_df = international_matches_df.sort_values(by='date', ascending=False)

world_cup_rankings_home = international_matches_df[['home_team', 'home_team_fifa_rank', 'home_team_total_fifa_points','home_team_goalkeeper_score','home_team_mean_defense_score','home_team_mean_midfield_score','home_team_mean_offense_score']].loc[international_matches_df['home_team'].isin(qualified_teams)].drop_duplicates(subset=['home_team'])
world_cup_rankings_away = international_matches_df[['away_team', 'away_team_fifa_rank', 'away_team_total_fifa_points','away_team_goalkeeper_score','away_team_mean_defense_score','away_team_mean_midfield_score','away_team_mean_offense_score']].loc[international_matches_df['away_team'].isin(qualified_teams)].drop_duplicates(subset=['away_team'])

world_cup_rankings_home = world_cup_rankings_home.set_index(['home_team'])
world_cup_rankings_away = world_cup_rankings_away.set_index(['away_team'])

belgium_probabilities = []

def predict_winner_and_probability(team1, team2):
    home_team_data = world_cup_rankings_home.loc[team1]
    away_team_data = world_cup_rankings_away.loc[team2]

    # Calculate features
    home_team_fifa_rank = home_team_data['home_team_fifa_rank']
    away_team_fifa_rank = away_team_data['away_team_fifa_rank']
    home_team_total_fifa_points  = home_team_data['home_team_total_fifa_points']
    away_team_total_fifa_points  = away_team_data['away_team_total_fifa_points']
    home_team_continent = international_matches_df.loc[international_matches_df['home_team'] == team1, 'home_team_continent'].values[0]
    away_team_continent = international_matches_df.loc[international_matches_df['away_team'] == team2, 'away_team_continent'].values[0]
    home_advantage = 1 if (home_team_continent == 'North America' and away_team_continent != 'North America') else 0
    home_team_goalkeeper_score = home_team_data['home_team_goalkeeper_score']
    home_team_mean_defense_score = home_team_data['home_team_mean_defense_score']
    home_team_mean_midfield_score = home_team_data['home_team_mean_midfield_score']
    home_team_mean_offense_score = home_team_data['home_team_mean_offense_score']
    away_team_goalkeeper_score = away_team_data['away_team_goalkeeper_score']
    away_team_mean_defense_score = away_team_data['away_team_mean_defense_score']
    away_team_mean_midfield_score = away_team_data['away_team_mean_midfield_score']
    away_team_mean_offense_score = away_team_data['away_team_mean_offense_score']

    # Predict winner and probability using the model
    features = np.array([[home_team_fifa_rank,away_team_fifa_rank,home_team_total_fifa_points,away_team_total_fifa_points,home_advantage,home_team_goalkeeper_score,away_team_goalkeeper_score,home_team_mean_defense_score,away_team_mean_defense_score,home_team_mean_midfield_score,away_team_mean_midfield_score,home_team_mean_offense_score,away_team_mean_offense_score]])
    winner_code = model.predict(features)[0]
    winner = team1 if winner_code == 1 else team2
    probability = model.predict_proba(features)[0, 1] if winner_code == 1 else (1-model.predict_proba(features)[0, 1])

    return winner, probability

qualified_teams_list = list(qualified_teams)
winner_counts = {team: 0 for team in qualified_teams_list}

num_iterations = 10000
for iteration in range(num_iterations):
    # Shuffle the list of teams randomly
    shuffled_teams = list(qualified_teams_list)
    random.shuffle(shuffled_teams)

    # Create groups
    groups = {}
    for i in range(12):
        group_name = f'Group {chr(ord("A") + i)}'
        groups[group_name] = shuffled_teams[i * 4: (i + 1) * 4]

    # Group Stage
    group_stage_results = {}
    for group_name, teams in groups.items():
        group_stage_results[group_name] = {'teams': teams, 'points': {team: 0 for team in teams}}

        # Simulate matches within the group
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team1, team2 = teams[i], teams[j]

                sorted_teams = tuple(sorted([team1, team2]))
                winner, probability = predict_winner_and_probability(sorted_teams[0], sorted_teams[1])
                group_stage_results[group_name]['points'][winner] += 3  # Award 3 points for a win

    # Extract teams qualifying for the knockout stage based on points
    qualified_teams_iteration = set()
    for group_name, result in group_stage_results.items():
        qualified_teams_iteration.update(sorted(result['points'], key=result['points'].get, reverse=True)[:2])

    # Initialize a knockout stage bracket
    knockout_stage = {
        'Round of 32': [],
        'Round of 16': [],
        'Quarterfinals': [],
        'Semifinals': [],
        'Final': []
    }

    qualified_teams_iteration_list = list(qualified_teams_iteration)
    third_placed_teams = set()
    for group_name, result in group_stage_results.items():
        # Sort teams within each group by points and take the third-placed team
        third_placed_team = sorted(result['points'], key=result['points'].get, reverse=True)[2]
        third_placed_teams.add(third_placed_team)

    # Round of 32
    teams_in_round_of_32 = qualified_teams_iteration.union(third_placed_teams)
    teams_in_round_of_32 = list(teams_in_round_of_32)[:32]
    random.shuffle(teams_in_round_of_32)
    for i in range(0, len(teams_in_round_of_32), 2):
        team1, team2 = teams_in_round_of_32[i], teams_in_round_of_32[i + 1]
        match_winner, probability = predict_winner_and_probability(team1, team2)
        knockout_stage['Round of 32'].append((match_winner, probability))

    # Round of 16
    random.shuffle(knockout_stage['Round of 32'])
    for i in range(0, len(knockout_stage['Round of 32']), 2):
        team1, team2 = knockout_stage['Round of 32'][i][0], knockout_stage['Round of 32'][i + 1][0]
        match_winner, probability = predict_winner_and_probability(team1, team2)
        knockout_stage['Round of 16'].append((match_winner, probability))

    # Quarterfinals
    random.shuffle(knockout_stage['Round of 16'])
    for i in range(0, len(knockout_stage['Round of 16']), 2):
        team1, team2 = knockout_stage['Round of 16'][i][0], knockout_stage['Round of 16'][i + 1][0]
        match_winner, probability = predict_winner_and_probability(team1, team2)
        knockout_stage['Quarterfinals'].append((match_winner, probability))

    # Semifinals
    random.shuffle(knockout_stage['Quarterfinals'])
    for i in range(0, len(knockout_stage['Quarterfinals']), 2):
        team1, team2 = knockout_stage['Quarterfinals'][i][0], knockout_stage['Quarterfinals'][i + 1][0]
        match_winner, probability = predict_winner_and_probability(team1, team2)
        knockout_stage['Semifinals'].append((match_winner, probability))

    # Final
    random.shuffle(knockout_stage['Semifinals'])
    finalists = [knockout_stage['Semifinals'][0][0], knockout_stage['Semifinals'][1][0]]
    team1, team2 = finalists[0], finalists[1]
    winner, probability = predict_winner_and_probability(team1, team2)
    knockout_stage['Final'].append((winner, probability))

    winner = knockout_stage['Final'][0][0]
    winner_counts[winner] += 1

    # Calculate and store the probability of Belgium winning at the end of each iteration
    belgium_probability = winner_counts['Belgium'] / (iteration + 1)
    belgium_probabilities.append(belgium_probability)

for team, count in winner_counts.items():
    probability = count / num_iterations
    if probability > 0:
        print(f"Probability of {team} winning: {probability:.4f}")
        print(f"{team} wins: {count} times")

# Plot of Winners and Probabilities
plt.rc('font', size=14)
plt.rc('axes', titlesize=16, labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
teams = [team for team, count in winner_counts.items() if count > 0]
probabilities = [count / num_iterations for team, count in winner_counts.items() if count > 0]
colors = plt.cm.viridis(np.linspace(0, 1, len(teams))) # type: ignore
plt.figure(figsize=(12,6))
bars = plt.bar(teams, probabilities, color=colors)
plt.xlabel('Teams')
plt.ylabel('Probability of Winning')
plt.title('Probability of Teams Winning World Cup 2026')
plt.xticks(rotation=0, ha='right')
plt.ylim(0,1)
plt.legend(bars, teams, title='Teams', bbox_to_anchor=(1, 1), loc='upper right')
for team, probability, bar in zip(teams, probabilities, bars):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.01, f'{probability:.2%}', ha='center', color='black')
plt.tight_layout()
plt.show()

# Plot the convergence of Belgium winning probability
plt.rc('font', size=14)
plt.rc('axes', titlesize=16, labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_iterations + 1), belgium_probabilities, label='Belgium', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Probability of Belgium Winning')
plt.title('Convergence of Belgium Winning Probability')
plt.ylim(0,1)
plt.legend()
plt.grid(True)
plt.show()