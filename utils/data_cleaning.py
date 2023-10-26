import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import StrMethodFormatter


fantasy_df = pd.read_csv('/Users/nick/sleepertoolsversion2/combine_data/2006-2022_seasonal_data.csv')
fantasy_df.sort_values(['season', 'entry_year'], inplace=True)
fantasy_df['entry_year'] = fantasy_df['entry_year'].fillna(method='ffill')
fantasy_df = fantasy_df[(fantasy_df['season'] >= 2012) & (fantasy_df['season'] <= 2021)]
min_season = fantasy_df['season'].min()
max_season = fantasy_df['season'].max()
fantasy_df = fantasy_df[fantasy_df['entry_year'] >= 2012]
fantasy_df['fantasy_points_per_game'] = round(fantasy_df['fantasy_points_ppr'] / fantasy_df['games'],2)

def calculate_epa_per_play(row):
    if row['position'] == 'QB':
        return (row['passing_epa'] + row['rushing_epa']) / row['offense_snaps']
    elif row['position'] in ['RB', 'TE', 'WR']:
        return (row['rushing_epa'] + row['receiving_epa']) / row['offense_snaps']
    else:
        return None
fantasy_df['epa_per_play'] = fantasy_df.apply(calculate_epa_per_play, axis=1)

fantasy_df = fantasy_df.drop_duplicates(subset=['season', 'player_id', 'fantasy_points_per_game'])
fantasy_df['rn'] = fantasy_df.groupby('player_id')['season'].rank()
fantasy_df = fantasy_df[fantasy_df['rn'] <= 2]
fantasy_df = fantasy_df[[ 'player_id','merge_name','season','fantasy_points_per_game','rn','entry_year','position','epa_per_play']]
fantasy_df = pd.pivot_table(fantasy_df, values=['epa_per_play', 'fantasy_points_per_game'], index=['player_id','merge_name','entry_year','position'], columns='rn', aggfunc={'epa_per_play': 'mean', 'fantasy_points_per_game': 'mean'})


# rename columns to 'Season_1_PPG' and 'Season_2_PPG'
fantasy_df.columns = ['Season 1 EPA/Play',
                 'Season 2 EPA/Play','Season_1_PPG', 'Season_2_PPG']
fantasy_df = fantasy_df.reset_index()
fantasy_df['Mean EPA/Play'] = round((fantasy_df['Season 1 EPA/Play'] + fantasy_df['Season 2 EPA/Play'])/2,2)
fantasy_df['Mean_PPG'] = round((fantasy_df['Season_1_PPG'] + fantasy_df['Season_2_PPG'])/2,2)

# # compute the mean of the two PPG columns and round to 2 decimal places

#
# # reset the index of the pivot table
#
# # filter the DataFrame to include only the rows where season matches the minimum season
# # group by player and find the minimum season for each player
#
#
# # filter the DataFrame to include only the rows where entry_year matches the minimum season
#
# # forward and backward fill missing entry_year values
#
combine_df = pd.read_csv('/Users/nick/sleepertoolsversion2/combine_data/apicombine.csv')
print(combine_df.columns)



combine_fp_df = combine_df.merge(fantasy_df, on=['merge_name','position'])
# combine_fp_df = combine_fp_df.dropna(subset=['Season_1_PPG', 'Season_2_PPG'])
combine_fp_df.to_csv('/Users/nick/sleepertoolsversion2/combine_data/combine_and_fantasy.csv')
combine_fp_df.info(verbose=True)
# df = combine_fp_df
# df.rename(columns={
#     'Season_1_PPG': 'Season 1 Fantasy PPG (PPR)',
#     'Season_2_PPG': 'Season 2 Fantasy PPG (PPR)',
#     'Mean_PPG': 'Mean Fantasy PPG (PPR)',
#     'draftGrade': 'Draft Grade',
#     'productionScore':'Production Score',
#     'athleticismScore':'Athleticism Score',
#      'sizeScore':'Size Score',
#      'wt': 'Weight','ht':'Height', 'armLength':'Arm Length', 'handSize':'Hand size',
#     'shuttle':'60 Yd Shuttle', 'forty':'40 Yd Dash',
#     'tenYardSplit.seconds':'10 Yd Split', 'threeConeDrill.seconds':'3-Cone Drill', 'twentyYardShuttle.seconds':'20 Yd Shuttle',
#     'vertical':'Vertical Jump', 'bench':'225 Bench Reps', 'broad_jump':'Broad Jump'
# },inplace=True)
# positions = df['position'].unique()
# positions = np.intersect1d(positions, ['QB', 'WR', 'TE', 'RB'])
# exclude_cols = ['Season 1 Fantasy PPG (PPR)','Season 2 Fantasy PPG (PPR)','Mean Fantasy PPG (PPR)','Season 1 EPA/Play','Season 2 EPA/Play','Mean EPA/Play','Height','Weight','Hand size','year','Size Score','entry_year']
# ppg_cols = ['Mean Fantasy PPG (PPR)','Mean EPA/Play']
# df = df[df['position'] != 'P']
# # Define color palette for metrics
# metric_color_dict = {}
# # Set font properties
# font_path = '/Users/nick/sleepertoolsversion2/Exo/static/Exo-Regular.ttf'
# font_prop = font_manager.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()
#
# # Iterate over positions and metrics
#
# fig, axes = plt.subplots(nrows=len(positions), ncols=len(ppg_cols), figsize=(25, 16))
# fig_text(
#     x=0.5, y=.98,
#     s="WHICH COMBINE METRICS MATTER?",
#     va="bottom", ha="center",
#     fontsize=30, color="black", font=font_prop, weight="bold"
# )
# fig_text(
#     x=0.5, y=.92,
#     s=f"Combine Metric Correlation to <Fantasy PPG> and <EPA/Play>\nFirst Two NFL Seasons | Players Drafted in {min_season}-{max_season}\n  viz by @run_the_sims | data from NFL API and Nfl_data_py",
#     highlight_textprops=[{"weight": "bold", 'color': '#368f8b'}, {"weight": "bold", 'color': '#368f8b'}],
#     va="bottom", ha="center",
#     fontsize=15, color="black", font=font_prop, textalign='center'
# )
#
# ylim = (0, 0.65)
# for ax in axes.flatten():
#     ax.set_ylim(ylim)
#     ax.tick_params(axis='both', labelsize=14, which='both')
# for i, pos in enumerate(positions):
#     pos_df = df[df['position'] == pos]
#     for j, ppg_col in enumerate(ppg_cols):
#         corr_matrix = pos_df.corr()[ppg_col]
#         corr_series = corr_matrix.drop(exclude_cols).sort_values(ascending=False)
#         corr_above_threshold = corr_series[~corr_series.index.isin([(col, col) for col in exclude_cols])].reset_index()
#         corr_above_threshold = pd.melt(corr_above_threshold, id_vars=['index'], var_name='Feature 2',
#                                        value_name='Correlation')
#
#         # Get top 5 correlations
#         top_5_corr = corr_above_threshold.sort_values('Correlation', ascending=False).head(5)
#         num_colors = 10  # set the number of colors to 5
#         for index, row in top_5_corr.iterrows():
#             metric = row['index']
#             if metric not in metric_color_dict:
#                 # If this metric hasn't been seen before, assign a new color
#                 metric_color_dict[metric] = sns.color_palette("Set3", n_colors=num_colors)[
#                     len(metric_color_dict) % num_colors]
#
#         # Get colors for each bar based on the assigned color for its metric
#         colors = [metric_color_dict[row['index']] for index, row in top_5_corr.iterrows()]
#         axes[i][j].spines['top'].set_visible(False)
#         axes[i][j].spines['right'].set_visible(False)
#         # Plot the bar chart with the assigned colors
#         sns.barplot(x='Feature 2', y='Correlation', hue='index', data=top_5_corr, palette=colors, ax=axes[i][j], width=3, edgecolor='black', linewidth=0.5, alpha=0.7,
#                     errwidth=2, errcolor='black',)
#         axes[i][j].yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
#
#         # Set y-axis label for the four charts on the left
#         if j == 0:
#             axes[i][j].set_ylabel('Correlation Coefficient', fontproperties=font_prop, fontsize=14)
#         else:
#             axes[i][j].set_ylabel('')
#             axes[i][j].tick_params(axis='y', labelsize=14, which='both')
#         axes[i][j].set_xlabel('')
#         axes[i][j].set_xticklabels([])
#         axes[i][j].set_title(f'{pos}: {ppg_col}', fontproperties=font_prop, fontsize=20)
#         axes[i][j].legend(prop={'family': font_prop.get_family(), 'size': 16}, frameon=False, bbox_to_anchor=(1, 1), markerscale=10)
#
#         # Set x-axis limits
#         axes[i][j].set_xlim(-2, len(top_5_corr) - 0.5)
# fig.subplots_adjust(top=0.9, wspace=0.2, hspace=0.3)
# plt.tight_layout()
#
#
#
#
# # plt.tight_layout()
# plt.savefig(
# 	"/Users/nick/sleepertoolsversion2/combine_data/correlation.png",
# 	dpi = 300,
#     bbox_inches='tight'
#
# )
# #
# #
# #
