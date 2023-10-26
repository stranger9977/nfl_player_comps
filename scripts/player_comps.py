import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit as st

def load_data():
    df = pd.read_csv('/Users/nick/nfl_player_comps/data/raw/apicombine.csv')
    df_projection = pd.read_csv('/Users/nick/nfl_player_comps/data/processed/projection.csv')
    df_projection = df_projection[['merge_name','Season_1_PPG','projected_season_1_PPG']]
    df = df.merge(df_projection, on='merge_name', how='left')
    df = df[(df['year'] >= 2012) & (df['year'] <= 2024)]
    df = df[(df['draft_grade'] > 70)]
    return df


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


def get_selection_options(df):
    years = sorted(df['year'].unique())
    positions = sorted(df['position'].unique())
    return years, positions


def get_filtered_df(df, selected_year, selected_position):
    df_year = df[df['year'] == selected_year]
    df_position = df_year[df_year['position'] == selected_position]
    return df_position


def get_players(df_position):
    players = sorted(df_position['person_display_name'].unique())
    return players


def create_instagram_ready_horizontal_bar_chart(output_df, features):
    if len(output_df) < 2:
        st.write("Not enough data for a comparison.")
        return


    top_player = output_df.iloc[1]

    player_names = [output_df.iloc[0]['person_display_name'], top_player['person_display_name']]
    input_percentiles = [f"{feature}_percentile" for feature in features if
                         f"{feature}_percentile" in output_df.columns]
    input_player_data = output_df.iloc[0][input_percentiles].values
    top_player_data = top_player[input_percentiles].values



    y = np.arange(len(features))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.barh(y - bar_height / 2, input_player_data, bar_height, label=player_names[0])

    if len(top_player_data) > 1:
        rects2 = ax.barh(y + bar_height / 2, top_player_data, bar_height, label=player_names[1])

        # Add season_1_ppg_projection to the bar graph
        # Check if the selected player has a null value for Season_1_PPG
        # Check if the selected player has a null value for Season_1_PPG
    if pd.isnull(output_df.iloc[0]['Season_1_PPG']):
        # Add the projected_season_1_PPG and actual Season_1_PPG as text annotations
        projection = output_df.iloc[0]['projected_season_1_PPG']
        actual = output_df.iloc[1]['Season_1_PPG']
        annotation_text = f"Projected Season 1 PPG: {projection:.2f}\nComparison Player Actual Season 1 PPG: {actual:.2f}"
        ax.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='black', ha='left',
                    va='top')

    ax.set_xlabel('Percentile Score')

    ax.set_xlabel('Percentile Score')

    ax.set_xlabel('Percentile Score')
    ax.set_title('Player Comparison')
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.legend()

    plt.tight_layout()
    st.pyplot(plt.gcf())




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


def knn_neighbors(df, player_name, n, pos):

    # Filter the DataFrame by pos
    df = calculate_size_score(df)
    pos_df = df[df['position'] == pos]
    player_season = pos_df.loc[pos_df['person_display_name'] == player_name, 'year'].values[0]
    pos_df = pos_df.loc[~((pos_df['person_display_name'] != player_name) & (pos_df['year'] == player_season))]

    player_df = pos_df[pos_df['person_display_name'] == player_name]


    # Select all float64 columns where the player is not null
    float64_cols = ['athleticism_score','production_score', 'size_score','draft_grade']
    null_cols = player_df.isnull().any()
    null_cols = null_cols[null_cols].index.tolist()
    nonnull_float64_cols = [col for col in float64_cols if col not in null_cols]

    features = nonnull_float64_cols
    pos_df = pos_df.dropna(subset=features)

    ranks_df = pos_df[['person_display_name'] + features]
    ranks_df.set_index('person_display_name', inplace=True)
    ranks_df = ranks_df[features].rank(pct=True, na_option='keep') * 100

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(ranks_df)

    knn = NearestNeighbors(n_neighbors=n)
    knn.fit(pos_df[features])

    player_features = player_df[features]
    distances, indices = knn.kneighbors(player_features)

    neighbors_df = pos_df.iloc[indices[0]].reset_index(drop=True)
    neighbors_df['similarity_score'] = 1 / (1 + distances[0])

    input_player_df = player_df.reindex(neighbors_df.columns).dropna()
    input_player_df['similarity_score'] = 1.0
    input_player_df.dropna(inplace=True)

    ranks_df.columns = [f'{col}_percentile' for col in ranks_df.columns]

    # Calculate the new projection for the selected player if their Season_1_PPG is null
    if player_df['Season_1_PPG'].isnull().values[0]:
        top_similar_player = neighbors_df.iloc[0]
        new_projection = (top_similar_player['Season_1_PPG'] + player_df['projected_season_1_PPG'].values[0]) / 2
        player_df['projected_season_1_PPG'] = new_projection

    output_df = pd.concat([input_player_df, neighbors_df])
    output_df = output_df.merge(ranks_df, right_index=True, left_on='person_display_name', how='left')

    return output_df, features
def main():
    st.title("Machine Learning Player Comps")

    df = load_data()
    df = fix_player_data(df)

    years, positions = get_selection_options(df)

    selected_year = st.selectbox("Select a Year", years, index=11)
    selected_position = st.selectbox('Select A Position', positions)

    df_position = get_filtered_df(df, selected_year, selected_position)

    players = get_players(df_position)
    selected_player = st.selectbox("Select a player", players)

    selected_position = df_position[df_position['person_display_name'] == selected_player]['position'].iloc[0]

    n_options = list(range(1, 11))
    selected_n = st.selectbox("Select the number of Comps", n_options, index=0)

    output_df, features = knn_neighbors(df, selected_player, selected_n, selected_position)

    create_instagram_ready_horizontal_bar_chart(output_df, features)


if __name__ == "__main__":
    main()







