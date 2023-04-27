import nfl_data_py as nfl
import numpy as np
import pandas as pd
import unidecode

seasons = np.arange(2012,2023).tolist()
df = nfl.import_seasonal_data(seasons)
snaps = nfl.import_snap_counts(seasons)



df = df[['player_id','season','fantasy_points_ppr','games','attempts','passing_epa','carries','rushing_epa','receiving_epa']]
roster_data = pd.DataFrame()
# Loop over the past 5 years (including the current year)
for year in seasons:
    # Import the roster data for the current year
    roster_df = nfl.import_rosters([year])

    # Append the roster data to the new DataFrame
    roster_data = roster_data.append(roster_df, ignore_index=True)
roster_df = roster_data


#get team desc
team_data = nfl.import_team_desc()
team_df = pd.DataFrame(team_data)
team_df = team_df[['team_abbr', 'team_logo_espn','team_league_logo','team_color','team_color2']]

#merge rosters to
roster_df = roster_df.merge(team_df, left_on='team', right_on='team_abbr')
roster_df = roster_df[['player_id','player_name','team','position','headshot_url','entry_year','team_abbr', 'team_logo_espn','team_league_logo','team_color','team_color2','pfr_id', 'draft_number']]
df = df.merge(roster_df, on='player_id')

snaps = snaps.groupby(['pfr_player_id','season'])['offense_snaps'].sum().reset_index()
df = df.merge(snaps, left_on=['pfr_id','season'],right_on=['pfr_player_id','season'], how='left')

df = df[['player_id','player_name','team','position','headshot_url','entry_year','team_abbr','draft_number', 'team_logo_espn','team_league_logo','team_color','team_color2','offense_snaps','season','fantasy_points_ppr','games','attempts','passing_epa','carries','rushing_epa','receiving_epa']]


df.drop_duplicates(subset=['player_id','season','offense_snaps'],inplace=True)


df["first_name"] = df.player_name.apply(lambda x: x.split(" ")[0])
df["last_name"] = df.player_name.apply(lambda x: " ".join(x.split(" ")[1::]))


# Remove non-alpha numeric characters from first/last names.
df["first_name"] = df.first_name.apply(
    lambda x: "".join(c for c in x if c.isalnum())
)
df["last_name"] = df.last_name.apply(
    lambda x: "".join(c for c in x if c.isalnum())
)

# Recreate full_name to fit format "Firstname Lastname" with no accents
df["merge_name"] = df.apply(
    lambda x: x.first_name + " " + x.last_name, axis=1
)
df["merge_name"] = df.merge_name.apply(lambda x: x.lower())
df.drop(["first_name", "last_name"], axis=1, inplace=True)

df["merge_name"] = df.merge_name.apply(lambda x: unidecode.unidecode(x))

# Create Column to match with RotoGrinders
df["merge_name"] = df.merge_name.apply(
    lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:6]
)

df.to_csv('/Users/nick/sleepertoolsversion2/combine_data/2006-2022_seasonal_data.csv', index=False)


