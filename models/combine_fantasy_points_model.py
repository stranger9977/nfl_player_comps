import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def height_to_inches(height):
    if isinstance(height, str):
        feet, inches = height.split('-')
        return int(feet) * 12 + int(inches)
    return height

def fix_player_data(df):
    player_row = df[df['person_display_name'] == 'Elijah Moore']
    df.loc[player_row.index, 'height'] = '5-10'
    df.loc[player_row.index, 'weight'] = 184

    # Convert height values to inches
    df['height'] = df['height'].apply(height_to_inches)

    return df

def calculate_size_score(df):
    # Calculate position-wise average height and weight
    position_averages = df.groupby('position')[['height', 'weight']].mean()
    position_averages.columns = ['average_height', 'average_weight']

    # Merge position-wise averages with the main DataFrame
    df = df.merge(position_averages, on='position', how='left')

    # Calculate deviations from position average
    df['height_deviation'] = df['height'] - df['average_height']
    df['weight_deviation'] = df['weight'] - df['average_weight']

    # Normalize height and weight deviations
    scaler = MinMaxScaler()
    normalized_deviations = scaler.fit_transform(df[['height_deviation', 'weight_deviation']])
    df_normalized = pd.DataFrame(normalized_deviations, columns=['height_normalized', 'weight_normalized'])

    # Calculate size score as a weighted average of normalized height and weight deviations
    weight_height_ratio = 0.5  # Adjust this value to change the importance of height vs. weight
    df['size_score'] = weight_height_ratio * df_normalized['height_normalized'] + (1 - weight_height_ratio) * \
                       df_normalized['weight_normalized']

    return df


def prepare_position_data(df, position):
    df = df[df['position'] == position]

    X = df[['draft_grade','size_score','athleticism_score','production_score', 'draft_number']]
    y = df['Season_1_PPG']
    return X, y


def train_position_model(X, y, models, param_grids):
    position_best_models = []
    position_best_feature_importances = []

    for model, param_grid in zip(models, param_grids):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        position_best_models.append(best_model)
        position_best_feature_importances.append((best_model.feature_importances_, X.columns) if hasattr(best_model, 'feature_importances_') else None)

        print(f"Model: {best_model.__class__.__name__}")
        print(f"Best parameters: {best_model.get_params()}")
        print(f"Mean squared error: {mse:.4f}")
        print(f"R^2 score: {r2:.4f}")
        print("\n" + "=" * 50 + "\n")

    return position_best_models, position_best_feature_importances



# Load the dataset and sort by 'season'


# Load the dataset and sort by 'season' and 'entry_year'
fantasy_df = pd.read_csv('/Users/nick/sleepertoolsversion2/combine_data/2006-2022_seasonal_data.csv')
fantasy_df.sort_values(['season', 'entry_year'], inplace=True)
# Fill missing 'entry_year' values and filter the DataFrame based on 'season' and 'entry_year'
fantasy_df['entry_year'] = fantasy_df['entry_year'].fillna(method='ffill')
fantasy_df = fantasy_df[(fantasy_df['season'] >= 2011) & (fantasy_df['season'] <= 2023) & (fantasy_df['entry_year'] >= 2012)]

# Check if 2022 players are present
is_2022_players_present = fantasy_df[fantasy_df['entry_year'] == 2022]

print(is_2022_players_present)

# Calculate fantasy_points_per_game
fantasy_df['fantasy_points_per_game'] = round(fantasy_df['fantasy_points_ppr'] / fantasy_df['games'], 2)

# Calculate epa_per_play based on the player's position
def calculate_epa_per_play(row):
    if row['position'] == 'QB':
        return (row['passing_epa'] + row['rushing_epa']) / row['offense_snaps']
    elif row['position'] in ['RB', 'TE', 'WR']:
        return (row['rushing_epa'] + row['receiving_epa']) / row['offense_snaps']
    else:
        return None

fantasy_df['epa_per_play'] = fantasy_df.apply(calculate_epa_per_play, axis=1)

# Remove duplicate rows
fantasy_df = fantasy_df.drop_duplicates(subset=['season', 'player_id', 'fantasy_points_per_game'])

# Rank 'season' within each 'player_id'
fantasy_df['rn'] = fantasy_df.groupby('player_id')['season'].rank()

# Filter the DataFrame to include only the first two seasons for each player
fantasy_df = fantasy_df[fantasy_df['rn'] <= 2]
# Pivot the DataFrame
fantasy_df = pd.pivot_table(
    fantasy_df,
    values=['epa_per_play', 'fantasy_points_per_game'],
    index=['player_id', 'merge_name', 'entry_year', 'position', 'draft_number'],
    columns='rn',
    aggfunc={'epa_per_play': 'mean', 'fantasy_points_per_game': 'mean'}
)
# Rename columns
fantasy_df.columns = ['Season 1 EPA/Play', 'Season 2 EPA/Play', 'Season_1_PPG', 'Season_2_PPG']
fantasy_df = fantasy_df.reset_index()

# Calculate the mean EPA/Play and mean PPG across the first two seasons
fantasy_df['Mean EPA/Play'] = round((fantasy_df['Season 1 EPA/Play'] + fantasy_df['Season 2 EPA/Play']) / 2, 2)
fantasy_df['Mean_PPG'] = round((fantasy_df['Season_1_PPG'] + fantasy_df['Season_2_PPG']) / 2, 2)


# Load the combine dataset
combine_df = pd.read_csv('/Users/nick/sleepertoolsversion2/combine_data/apicombine.csv')

# Merge the fantasy DataFrame with the combine DataFrame
combine_fp_df = combine_df.merge(fantasy_df, on=['merge_name', 'position'])
combine_2023_df = combine_df.merge(fantasy_df, on=['merge_name', 'position'], how='left')
combine_2023_df = combine_2023_df[combine_2023_df['year'] == 2023]

combine_2023_df = calculate_size_score(combine_2023_df)
combine_2023_df.info(verbose=True)
# Fix player data
combine_fp_df = fix_player_data(combine_fp_df)

# Calculate the size score
combine_fp_df = calculate_size_score(combine_fp_df)

# Save the merged DataFrame to a CSV file
combine_fp_df.to_csv('/Users/nick/sleepertoolsversion2/combine_data/combine_and_fantasy.csv', index=False)
combine_fp_df.dropna(subset=['draft_grade'],inplace=True)
combine_fp_df.info(verbose=True)
# Train models for each position
positions = ['QB', 'RB', 'WR', 'TE']
# Define the models and their corresponding parameter grids



# Define the models and their corresponding parameter grids
models = [
    KNeighborsRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    xgb.XGBRegressor()
]

param_grids = [
    {'n_neighbors': [3, 5, 7]},  # KNN
    {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]},  # Random Forest
    {'learning_rate': [0.01, 0.1, 1], 'n_estimators': [100, 200, 300]},  # Gradient Boosting
    {'learning_rate': [0.01, 0.1, 1], 'n_estimators': [100, 200, 300]}  # XGBoost
]

best_models = []

for position in positions:
    X, y = prepare_position_data(combine_fp_df, position)
    best_model = train_position_model(X, y, models, param_grids)
    best_models.append((position, best_model))

combined_df = combine_fp_df.dropna(subset=['draft_grade'])
combine_2023_df = combine_2023_df.dropna(subset=['draft_grade'])

combined_df.info(verbose=True)
# Calculate and display feature importances for the best model of each position
# Train models for each position
positions = ['QB', 'RB', 'WR', 'TE']

for position in positions:
    # Filter the combined dataset for the specific position
    position_df = combined_df[combined_df['position'] == position].copy()
    position_2023_df = combine_2023_df[combine_2023_df['position'] == position].copy()

    # Create feature matrix X using the required features for the model
    X = position_df[['draft_grade',  'size_score', 'athleticism_score', 'production_score']]
    X_2023 = position_2023_df[['draft_grade', 'size_score', 'athleticism_score', 'production_score']]
    # Create the KNNRegressor model
    model = KNeighborsRegressor(n_neighbors=7)

    # Fit the model on the existing player data
    model.fit(X, position_df['Season_1_PPG'].dropna())

    # Predict the projected_season_1_PPG for the draft class
    position_df['projected_season_1_PPG'] = model.predict(X)
    position_2023_df['projected_season_1_PPG'] = model.predict(X_2023)

    # Assign the predicted values to the corresponding rows in the combined dataset
    combined_df.loc[combined_df['position'] == position, 'projected_season_1_PPG'] = position_df['projected_season_1_PPG']
    combine_2023_df.loc[combine_2023_df['position'] == position, 'projected_season_1_PPG'] = position_2023_df['projected_season_1_PPG']

# Concatenate the combined dataset with the 2023 dataset
df = pd.concat([combined_df, combine_2023_df], ignore_index=True)

# Display the combined dataset with projected and actual season_1_PPG along with the difference
df.to_csv('/Users/nick/sleepertoolsversion2/combine_data/projection.csv', index=False)
