import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


plt.rcParams['font.family'] = 'Roboto'


def create_player_card(player_data, comp_data):
    twitter_player_template = """
    <div style="background: linear-gradient(to bottom right, {}, white); border: 6px solid white; border-radius: 10px; padding: 10px; width: 506px; position: relative;">
    <h3 style=" background: linear-gradient(to right, white, {} ); font-family: 'Roboto', sans-serif; font-weight: bold; font-size: 12px; color: black; text-align: left; background-color: white; border-radius: 10px; padding: 5px; margin-bottom: 5px;">{}, {}, {}</h3>
   <div style="background: linear-gradient(to bottom right, white, {} ); text-align: center; background-color: white; border: 2px solid black; border-radius: 10px; padding: 5px; margin-bottom: 10px; position: relative;">
        <img src="{}" style="width: 100%; height: auto; object-fit: cover; border-radius: 10px;">
        <img src="{}" style="width: 130px; height: 130px; object-fit: contain; position: absolute; top: 0; left: 0;">
    </div>
        <img src="https://gridironai.com/static/gridiron/projects/public/src/assets/android-chrome-192x192.png" style="width: 80px; height: 80px; object-fit: contain; position: absolute; top: 1px; right: 1px; border-radius: 50%;">
    <div style=" background: linear-gradient(to bottom right, white, ); text-align: left; border: 2px solid black; border-radius: 10px; padding: 10px; margin-bottom: 10px; width: 100%;">
        <table style="margin: 0 auto; border-collapse: separate; border-spacing: 0;">
            <tr>
                <td style="font-family: 'Roboto', sans-serif; font-size: 14px; color: black; padding: 5px; ; font-weight: bold; border: none;"> Outlook: {} {}</td>
                <td style="font-family: 'Roboto', sans-serif; font-size: 10px; color: black; padding: 5px; border: none;"></td>
                <td style="font-family: 'Roboto', sans-serif; font-size: 14px; color: black; padding: 5px; font-weight: bold; border: none;">Draft Pick: {:.0f} </td>
                <td style="font-family: 'Roboto', sans-serif; font-size: 10px; color: black; padding: 5px; border: none;"></td>
            </tr>
            <tr>
                <td style="font-family: 'Roboto', sans-serif; font-size: 14px; color: black; padding: 5px; font-weight: bold; border: none;">AI Comp: {}</td>
                <td style="font-family: 'Roboto', sans-serif; font-size: 14px; color: black; padding: 5px; border: none;"></td>
                <td style="font-family: 'Roboto', sans-serif; font-size: 14px; color: black; padding: 5px; font-weight: bold; border: none;">Class: {}</td>
                <td style="font-family: 'Roboto', sans-serif; font-size: 14px; color: black; padding: 5px; border: none;"></td>
            </tr>
            <tr>
                 <td style="font-family: 'Roboto', sans-serif; font-size: 14px; color: black; padding: 5px; font-weight: bold; border: none;">Scout Comp: {}</td>
            <td style="font-family: 'Roboto', sans-serif; font-size: 14px; color: black; padding: 5px; border: none;"></td>
             <td style="font-family: 'Roboto', sans-serif; font-size: 14px; color: black; padding: 5px; font-weight: bold; border: none;">Team: {}</td>
            <td style="font-family: 'Roboto', sans-serif; font-size: 14px; color: black; padding: 5px; border: none;"></td>
        </tr>
                    <tr>
                <td style="font-family: 'Roboto', sans-serif; font-size: 14px; color: black; padding: 5px; ; font-weight: bold; border: none;"> Height: {}</td>
                <td style="font-family: 'Roboto', sans-serif; font-size: 10px; color: black; padding: 5px; border: none;"></td>
                <td style="font-family: 'Roboto', sans-serif; font-size: 14px; color: black; padding: 5px; font-weight: bold; border: none;">Weight: {}</td>
                <td style="font-family: 'Roboto', sans-serif; font-size: 10px; color: black; padding: 5px; border: none;"></td>
            </tr>
    </table>
    </div>

    """
    instagram_player_template = """
    <div style="background: linear-gradient(to top right, #30B5FD, white); border: 6px solid white; border-radius: 10px; padding: 10px; width: 1080px; position: relative opacity: .04;">
       <div style="text-align: center; background-color: white; border-radius: 10px; padding: 5px; margin-bottom: 10px; position: relative;">
            <img src="{}" style="width: 100%; height: auto; object-fit: cover; border-radius: 10px;">
            <img src="{}" style="width: 300px; height: 300px; object-fit: contain; position: absolute; top: 0; left: 0; opacity: 0.4;">
        </div>
        <h3 style="font-family: 'Roboto', sans-serif; font-weight: bold; font-size: 36px; color: black; text-align: center; background-color: white; border-radius: 10px; padding: 10px; margin-bottom: 10px;">{}, {}, {}</h3>
        <img src="https://gridironai.com/static/gridiron/projects/public/src/assets/android-chrome-192x192.png" style="width: 250px; height: 250px; object-fit: contain; position: absolute; top: 1px; right: 1px; border-radius: 50%;">
    <div style="text-align: left; border: 2px solid white; border-radius: 10px; padding: 20px; margin-bottom: 10px; width: 100%;">
        <table style="margin: 0 auto; background-color: white, border-collapse: separate; border-spacing: 0;">
            <tr>
                <td style="font-family: 'Roboto', sans-serif; font-size: 28px; color: black; padding: 14px; font-weight: bold; border: none;">Y1 PPG (Proj):</td>
                <td style="font-family: 'Roboto', sans-serif; font-size: 28px; color: black; padding: 14px; border: none;">{:.1f}</td>
            </tr>
            <tr>
                <td style="font-family: 'Roboto', sans-serif; font-size: 28px; color: black; padding: 14px; font-weight: bold; border: none;">Draft Pick:</td>
                <td style="font-family: 'Roboto', sans-serif; font-size: 28px; color: black; padding: 14px; border: none;">{:.0f}, {}</td>
            </tr>
            <tr>
                <td style="font-family: 'Roboto', sans-serif; font-size: 28px; color: black; padding: 14px; font-weight: bold; border: none;">AI Comp:</td>
                <td style="font-family: 'Roboto', sans-serif; font-size: 28px; color: black; padding: 14px; border: none;">{}</td>
            </tr>
                    <tr>
            <td style="font-family: 'Roboto', sans-serif; font-size: 28px; color: black; padding: 14px; font-weight: bold; border: none;">Height:</td>
            <td style="font-family: 'Roboto', sans-serif; font-size: 28px; color: black; padding: 14px; border: none;">{} lbs</td>
        </tr>
        <tr>
            <td style="font-family: 'Roboto', sans-serif; font-size: 28px; color: black; padding: 14px; font-weight: bold; border: none;">Weight:</td>
            <td style="font-family: 'Roboto', sans-serif; font-size: 28px; color: black; padding: 14px; border: none;">{} lbs</td>
        </tr>
    </table>
</div>
</div>
"""

    twitter_player_template_back = """
    <div style="background: linear-gradient(to bottom right, {}, white); border: 6px solid black; border-radius: 10px; padding: 10px; width: 506px; position: relative;">
        <h3 style=" background: linear-gradient(to right, white, {} ); font-family: 'Roboto', sans-serif; font-weight: bold; font-size: 12px; color: black; text-align: left; background-color: white; border-radius: 10px; padding: 5px; margin-bottom: 5px;">{}, {}, {}</h3>
    <div style="background: linear-gradient(to bottom right, white, {} ); text-align: center; background-color: white; border: 2px solid black; border-radius: 10px; padding: 5px; margin-bottom: 10px; position: relative;">
            <h3 style="font-family: 'Roboto', sans-serif; font-weight: bold; font-size: 14px; color: black; text-align: left; padding: 5px; margin-bottom: 5px;">Profile by NFL.com's {}</h3>
            <h4 style="font-family: 'Roboto', sans-serif; font-weight: bold; font-size: 14px; color: black; text-align: left; padding: 5px; margin-bottom: 5px;">Overview:</h3>
            <p style="font-family: 'Roboto', sans-serif; font-size: 12px; color: black; padding: 5px; margin-bottom: 10px;">{}</p>
        </div>
        <img src="https://gridironai.com/static/gridiron/projects/public/src/assets/android-chrome-192x192.png" style="width: 80px; height: 80px; object-fit: contain; position: absolute; top: 1px; right: 1px; border-radius: 50%;">
        <div style=" background: linear-gradient(to bottom right, white, ); text-align: left; border: 2px solid black; border-radius: 10px; padding: 10px; margin-bottom: 10px; width: 100%;">
            <h4 style="font-family: 'Roboto', sans-serif; font-weight: bold; font-size: 14px; color: black; text-align: left; padding: 5px; margin-bottom: 5px;">Bio:</h3>
            <p style="font-family: 'Roboto', sans-serif; font-size: 12px; color: black; padding: 5px; margin-bottom: 10px;">{}</p>
        </div>
    </div>
    """

    # Convert height from inches to feet and inches format
    height_feet = int(player_data["height"] // 12)
    height_inches = int(player_data["height"] % 12)
    height_formatted = f"{height_feet}'{height_inches}\""

    weight_formatted = f"{int(player_data['weight'])} lbs"

    comp_height_feet = int(comp_data["height"] // 12)
    comp_height_inches = int(comp_data["height"] % 12)
    comp_height_formatted = f"{comp_height_feet}'{comp_height_inches}\""

    comp_weight_formatted = f"{int(comp_data['weight'])} lbs"

    college = player_data["college"]
    college = college.split(",")[0].strip('[]"\'')

    team_colors = {
        "BAL": ["Baltimore Ravens", "#241773", "#101820", "#C9A074", "#B3060A"],
        "CIN": ["Cincinnati Bengals", "#F9461C", "#101820"],
        "CLE": ["Cleveland Browns", "#311D00", "#FF3C00"],
        "PIT": ["Pittsburgh Steelers", "#FFB81C", "#101820", "#003087", "#C60C30", "#A5A9B0"],
        "BUF": ["Buffalo Bills", "#00338D", "#C60C30"],
        "MIA": ["Miami Dolphins", "#008E97", "#F58220", "#005778"],
        "NE": ["New England Patriots", "#002244", "#C60C30", "#B0B7BC"],
        "NYJ": ["New York Jets", "#0C371D", "#000000", "#FFFFFF"],
        "HOU": ["Houston Texans",  "#C60C30", "#0C2340"],
        "IND": ["Indianapolis Colts", "#002C5F", "#A2AAAD"],
        "JAX": ["Jacksonville Jaguars", "#007198", "#FFC62F","#D7A22A", "#101820" ],
        "TEN": ["Tennessee Titans","#4B92DB", "#0C2340", "#C60C30", "#A5ACAF"],
        "DEN": ["Denver Broncos", "#FB4F14", "#002244"],
        "KC": ["Kansas City Chiefs", "#E31837", "#FFB81C"],
        "LV": ["Las Vegas Raiders", "#A5ACAF", "#000000"],
        "LAC": ["Los Angeles Chargers", "#0072C6", "#FFB81C", "#FFFFFF"],
        "CHI": ["Chicago Bears", "#0B162A", "#C83803"],
        "DET": ["Detroit Lions", "#0076B6", "#B0B7BC", "#000000", "#FFFFFF"],
        "GB": ["Green Bay Packers", "#FFB81C", "#203731"],
        "MIN": ["Minnesota Vikings", "#4F2683", "#FFC62F"],
        "DAL": ["Dallas Cowboys", "#002244", "#041E42", "#869397", "#0072C6", "#FFFFFF"],
        "NYG": ["New York Giants", "#0B2265", "#A6192E", "#A2AAAD"],
        "PHI": ["Philadelphia Eagles", "#004C54", "#A5ACAF", "#000000", "#000000", "#6C6E70"],
        "WAS": ["Washington Commanders", "#773141", "#FFB612"],
        "ATL": ["Atlanta Falcons", "#A6192E", "#101820", "#A5ACAF"],
        "CAR": ["Carolina Panthers", "#0085CA", "#101820", "#A5ACAF"],
        "NO": ["New Orleans Saints", "#D3BC8D", "#000000"],
        "TB": ["Tampa Bay Buccaneers", "#D50A0A", "#FF8200", "#000000", "#737278", "#C5B783"],
        "ARI": ["Arizona Cardinals", "#97233F", "#000000", "#FFB612"],
        "LAR": ["Los Angeles Rams", "#002244", "#85714D", "#4B2E84"],
        "SF": ["San Francisco 49ers", "#AA0000", "#B3995D"],
        "SEA": ["Seattle Seahawks","#69BE28", "#3C3B3D","#002244" ]
    }

    team_logo_color_1 = team_colors[player_data["team"]][1]

    def player_outlook(draft_score):
        if draft_score >= 93:
            outlook = "Elite"
            outlook_emoji = "🚀"
        elif draft_score >= 83:
            outlook = "Solid"
            outlook_emoji = "🪨"
        elif draft_score >= 77:
            outlook = "Boom Bust"
            outlook_emoji = "💣"
        elif draft_score < 77:
            outlook = "Low Upside"
            outlook_emoji = "🥶"
        return outlook, outlook_emoji

    draft_score = player_data["draft_grade"]
    outlook, outlook_emoji=  player_outlook(draft_score)
    player_card_html_twitter = twitter_player_template.format(
        team_logo_color_1,
        team_logo_color_1,
        player_data["name"],
        player_data["position"],
        college,
        team_logo_color_1,
        player_data["headshot_url"],
        player_data["team_logo_url"],
        outlook,
        outlook_emoji,
        player_data["draft_pick"],
        comp_data["name"],
        player_data["college_class"],
        player_data["nfl_comparison"],
        player_data["team"],
        height_formatted,
        weight_formatted,

    )

    player_card_html_twitter_back = twitter_player_template_back.format(
        team_logo_color_1,
        team_logo_color_1,
        player_data["name"],
        player_data["position"],
        college,
        team_logo_color_1,
        player_data["profile_author"],
        player_data["overview"],
        player_data["bio"],

    )

    # comp_card_html = player_template.format(
    #     comp_data["headshot_url"],
    #     comp_data["team_logo_url"],
    #     comp_data["name"],
    #     comp_data["position"],
    #     comp_data["college"],
    #     comp_data["actual_ppg"],
    #     comp_data["name"],
    #     comp_data["college_class"],
    #     comp_height_formatted,
    #     int(comp_weight_formatted) ,
    # )



    st.markdown("<h3>Front</h3>", unsafe_allow_html=True)
    st.markdown(player_card_html_twitter, unsafe_allow_html=True)

    st.markdown("<h3>Back</h3>", unsafe_allow_html=True)
    st.markdown(player_card_html_twitter_back, unsafe_allow_html=True)







def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/stranger9977/nfl_player_comps/master/data/raw/apicombine.csv')
    df_projection = pd.read_csv('https://raw.githubusercontent.com/stranger9977/nfl_player_comps/master/data/processed/projection.csv')
    print(df_projection.columns)

    df_projection = df_projection[['merge_name','Season_1_PPG','projected_season_1_PPG','team','team_logo_espn','draft_number']]

    df = df.merge(df_projection, on='merge_name', how='left')
    df = df[(df['year'] >= 2012) & (df['year'] <= 2024)]
    df = df[(df['draft_grade'] > 70)]
    df['headshot'] = df['headshot'].apply(lambda url: url.format(formatInstructions="/f_auto,q_85"))
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
def gradient_color_map(color):
    return np.linspace(color, [255, 255, 255], 256)

def create_horizontal_bar_chart(output_df):
    # Load the Roboto font
    prop = fm.FontProperties(fname='scripts/assets/Roboto/Roboto-Medium.ttf')

    labels = ['Size', 'Production', 'Athleticism', 'Overall']
    output_df['size_score'] = output_df['size_score'] * 100
    player_0_scores = [output_df.at[2, 'size_score'], output_df.at[2, 'production_score'],
                       output_df.at[2, 'athleticism_score'], output_df.at[2, 'draft_grade']]
    player_1_scores = [output_df.at[1, 'size_score'], output_df.at[1, 'production_score'],
                       output_df.at[1, 'athleticism_score'], output_df.at[1, 'draft_grade']]
    player_2_scores = [output_df.at[0, 'size_score'], output_df.at[0, 'production_score'],
                       output_df.at[0, 'athleticism_score'], output_df.at[0, 'draft_grade']]

    # Create the y positions for the bars
    y = np.arange(len(labels))

    # Define the height of each bar
    bar_height = 0.25

    # Define the colors for the bars
    colors = ['#30B5FD', '#808080', '#B19CD9']

    # Plotting the bars
    fig, ax = plt.subplots()
    rects1 = ax.barh(y + 2 * bar_height, player_0_scores, bar_height, label=output_df.at[2, 'person_display_name'], color=colors[2], edgecolor='black', linewidth=1.5)
    rects2 = ax.barh(y + bar_height, player_1_scores, bar_height, label=output_df.at[1, 'person_display_name'], color=colors[1], edgecolor='black', linewidth=1.5)
    rects3 = ax.barh(y, player_2_scores, bar_height, label=output_df.at[0, 'person_display_name'], color=colors[0], edgecolor='black', linewidth=1.5)

    # Set the x-axis labels
    for rect in rects1 + rects2 + rects3:
        width = rect.get_width()
        ax.text(width + 2, rect.get_y() + rect.get_height() / 2, int(width),
                ha='left', va='center', fontweight='bold', fontproperties=prop)

    # Set the y-axis labels
    ax.set_yticks(y + bar_height)
    ax.set_yticklabels(labels, fontproperties=prop)

    # Hide the x-axis labels and tick marks
    ax.set_xticklabels([])
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')


    # Set the subtitle text for each player
    subtitle_player_0 = f"{output_df.at[2, 'person_display_name']} - Pick {output_df.at[2, 'draft_number']:.0f} | " \
                        f"Projected Rookie PPG: {output_df.at[2, 'projected_season_1_PPG']:.1f}"
    subtitle_player_1 = f"{output_df.at[1, 'person_display_name']} - Pick {output_df.at[1, 'draft_number']:.0f} | " \
                       f"Projected Rookie PPG: {output_df.at[1, 'projected_season_1_PPG']:.1f} | " \
                       f"Actual Rookie PPG: {output_df.at[1, 'Season_1_PPG']:.1f}" if not pd.isna(
        output_df.at[1, 'Season_1_PPG']) else "???"

    subtitle_player_2 = f"{output_df.at[0, 'person_display_name']} - Pick {output_df.at[0, 'draft_number']:.0f} | " \
                        f"Projected Rookie PPG: {output_df.at[0, 'projected_season_1_PPG']:.1f} | " \
                        f"Actual Rookie PPG: {output_df.at[0, 'Season_1_PPG']:.1f}" if not pd.isna(
        output_df.at[0, 'Season_1_PPG']) else "???"

    # Set the subtitle text
    subtitle = f"{subtitle_player_0}\n{subtitle_player_1}\n{subtitle_player_2}"

    # Set the subtitle properties
    ax.set_title(output_df.at[2, 'person_display_name'] + " Top Comps", fontproperties=prop, loc='left', pad=0)

    # Add the subtitle text
    ax.text(0, -0.35, subtitle, fontsize=8, verticalalignment='top', fontproperties=prop)

    # Add the legend
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=5, frameon=True, facecolor='white', prop=prop)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Set the layout and display the plot
    plt.tight_layout()
    st.pyplot(fig)


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


def knn_neighbors(df, player_name, n, pos, draft_number=None):

    # Filter the DataFrame by pos
    df = calculate_size_score(df)
    pos_df = df[df['position'] == pos]
    player_season = pos_df.loc[pos_df['person_display_name'] == player_name, 'year'].values[0]
    pos_df = pos_df.loc[~((pos_df['person_display_name'] != player_name) & (pos_df['year'] == player_season))]

    player_df = pos_df[pos_df['person_display_name'] == player_name]

    # Update player_df with the provided draft number if available
    if draft_number:
        player_df['draft_number'] = draft_number

    # Select all float64 columns where the player is not null
    float64_cols = ['athleticism_score','production_score', 'size_score','draft_grade','draft_number', 'projected_season_1_PPG']
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
    neighbors_df['similarity_score_percentile'] = neighbors_df['similarity_score'] * 100


    input_player_df = player_df.reindex(neighbors_df.columns).dropna()
    input_player_df['similarity_score'] = 1.0
    input_player_df['similarity_score_percentile'] = input_player_df['similarity_score'] * 100
    input_player_df.dropna(inplace=True)

    ranks_df.columns = [f'{col}_percentile' for col in ranks_df.columns]

    # Calculate the new projection for the selected player if their Season_1_PPG is null
    if player_df['Season_1_PPG'].isnull().values[0]:
        top_similar_player = neighbors_df.iloc[0]
        new_projection = (top_similar_player['Season_1_PPG'] + player_df['projected_season_1_PPG'].values[0]) / 2
        player_df['projected_season_1_PPG'] = new_projection

    output_df = pd.concat([player_df, neighbors_df]).sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
    output_df = output_df.merge(ranks_df, right_index=True, left_on='person_display_name', how='left')
    return output_df, features
def main():
    st.title("Combine Data Player Comps")

    df = load_data()
    df = fix_player_data(df)

    years, positions = get_selection_options(df)

    selected_year = st.selectbox("Select a Year", years, index=11)
    selected_position = st.selectbox('Select A Position', positions)

    df_position = get_filtered_df(df, selected_year, selected_position)

    players = get_players(df_position)
    selected_player = st.selectbox("Select a player", players)

    selected_position = df_position[df_position['person_display_name'] == selected_player]['position'].iloc[0]


    # Get selected player data
    selected_player_data = df_position[df_position['person_display_name'] == selected_player].iloc[0]
    print("Selected Player_data Columns" , df_position.columns)

    # Get selected player data
    selected_player_data = df_position[df_position['person_display_name'] == selected_player].iloc[0]

    # Check if player has no team and draft number

    if pd.isna(selected_player_data['team']) and pd.isna(selected_player_data['draft_number']):
        st.write("The selected player has no team and draft number. Please provide the following information:")

        team_name_to_logo = {}

        # Iterate through the rows in the column
        for index, row in df.iterrows():
            team = row['team']
            team_logo_url = row['team_logo_espn']

            # Check if the team string is not NaN and if it is not already in the dictionary
            if pd.notna(team) and team not in team_name_to_logo:
                team_name_to_logo[team] = team_logo_url

                # Break the loop if all 32 unique teams have been found
                if len(team_name_to_logo) == 32:
                    break
            # Get user input for draft number and team
        draft_number = st.number_input("Enter the draft number", value=1, min_value=1, max_value=300)
        team = st.selectbox("Select a team", list(team_name_to_logo.keys()))

        # Update the player_data and comp_data dictionaries with the new information
        selected_player_data['draft_number'] = draft_number
        selected_player_data['team'] = team
        selected_player_data['team_logo_espn'] = team_name_to_logo[team]

        # Check if the draft number is NaN
        if pd.isna(draft_number):
            # Set the draft number to the maximum possible value
            selected_player_data['draft_number'] = 299

        # Call knn_neighbors with draft_number if it's not None
        if draft_number is not None:
            output_df, features = knn_neighbors(df, selected_player, 2, selected_position, draft_number)
        else:
            output_df, features = knn_neighbors(df, selected_player, 2, selected_position)
    else:
        # Call knn_neighbors without draft_number if it's not needed
        output_df, features = knn_neighbors(df, selected_player, 2, selected_position)
    print("output",output_df.columns)
    sorted_output_df = output_df.sort_values(by='similarity_score', ascending=False)


    print("selected",selected_player_data)
    # Prepare player data for create_player_card
    player_data = {
        # Previous fields
        "headshot_url": selected_player_data['headshot'],
        "team": selected_player_data['team'],
        "team_logo_url": selected_player_data['team_logo_espn'],
        "name": selected_player_data["person_display_name"],
        "position": selected_player_data["position"],
        "college": selected_player_data["person_college_names"],
        "draft_pick": selected_player_data["draft_number"],
        # "archetype": "Projected Archetype",  # Replace with actual archetype
        "projected_ppg": selected_player_data["projected_season_1_PPG"],
        "college_class": selected_player_data["college_class"],
        "combine_attendance": selected_player_data["combine_attendance"],
        "draft_projection": selected_player_data["draft_projection"],
        "grade": selected_player_data["grade"],
        "draft_grade": selected_player_data["draft_grade"],
        "size_score": selected_player_data["size_score"],
        "athleticism_score": selected_player_data["athleticism_score"],
        "production_score": selected_player_data["production_score"],
        "height": selected_player_data["height"],
        "weight": selected_player_data["weight"],
        "arm_length": selected_player_data["arm_length"],
        "hand_size": selected_player_data["hand_size"],
        "pro_forty_yard_dash_seconds": selected_player_data["pro_forty_yard_dash_seconds"],
        "sixty_yard_shuttle_seconds": selected_player_data["sixty_yard_shuttle_seconds"],
        "bio": selected_player_data["bio"],
        "overview": selected_player_data["overview"],
        "sources_tell_us": selected_player_data["sources_tell_us"],
        "nfl_comparison": selected_player_data["nfl_comparison"],
        "strengths": selected_player_data["strengths"],
        "weaknesses": selected_player_data["weaknesses"],
        "profile_author": selected_player_data["profile_author"],
    }

    print("output", output_df)

    comp_data_list = []
    for index in [0, 1]:  # Assuming top two comps are in rows 1 and 2
        comp_data = {
            "headshot_url": output_df.iloc[index]['headshot'],
            "team": output_df.iloc[index]['team'],
            "team_logo_url": output_df.iloc[index]['team_logo_espn'],
            "name": output_df.iloc[index]["person_display_name"],
            "position": output_df.iloc[index]["position"],
            "college": output_df.iloc[index]["person_college_names"],
            "actual_ppg": output_df.iloc[index]["Season_1_PPG"],
            "college_class": output_df.iloc[index]["college_class"],
            "combine_attendance": output_df.iloc[index]["combine_attendance"],
            "draft_projection": output_df.iloc[index]["draft_projection"],
            "grade": output_df.iloc[index]["grade"],
            "draft_grade": output_df.iloc[index]["draft_grade"],
            "size_score": output_df.iloc[index]["size_score"],
            "athleticism_score": output_df.iloc[index]["athleticism_score"],
            "production_score": output_df.iloc[index]["production_score"],
            "height": output_df.iloc[index]["height"],
            "weight": output_df.iloc[index]["weight"],
            "arm_length": output_df.iloc[index]["arm_length"],
            "hand_size": output_df.iloc[index]["hand_size"],
            "pro_forty_yard_dash_seconds": output_df.iloc[index]["pro_forty_yard_dash_seconds"],
            "sixty_yard_shuttle_seconds": output_df.iloc[index]["sixty_yard_shuttle_seconds"],
            "bio": output_df.iloc[index]["bio"],
            "overview": output_df.iloc[index]["overview"],
            "sources_tell_us": output_df.iloc[index]["sources_tell_us"],
            "nfl_comparison": output_df.iloc[index]["nfl_comparison"],
            "strengths": output_df.iloc[index]["strengths"],
            "weaknesses": output_df.iloc[index]["weaknesses"],
            "draft_pick": output_df.iloc[index]["draft_number"],
            "similarity_score": output_df.iloc[index]["similarity_score"],


        }
        comp_data_list.append(comp_data)

    # Create the player card
    create_player_card(player_data,comp_data)


    create_horizontal_bar_chart(output_df)


if __name__ == "__main__":
    main()












