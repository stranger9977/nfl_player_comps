import re

import requests
import pandas as pd

# send a POST request to get access token
import seaborn as sns
import unidecode
from matplotlib import pyplot as plt

nflcom_authres = requests.post(
    url="https://api.nfl.com/v1/reroute",
    data={
        'grant_type': 'client_credentials'
    },
    headers={
        'x-domain-id': '100',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
)
nflcom_headers = {
    'Referer': 'https://www.nfl.com/',
    'origin': 'https://www.nfl.com',
    'Authorization': f"Bearer {nflcom_authres.json()['access_token']}"
}

# fetch data for all years from 2006 to 2023
year_range = range(2012, 2024)
data_frames = []
for year in year_range:
    url = f"https://api.nfl.com/football/v2/combine/profiles?year={year}&limit=3000"
    response = requests.get(url, headers=nflcom_headers)
    json_data = response.json()['combineProfiles']
    data_frames.append(pd.json_normalize(json_data))

# combine all data frames and filter columns
df = pd.concat(data_frames)
df = df[['person.displayName', 'person.firstName', 'person.lastName', 'id', 'person.id', 'person.esbId', 'headshot',
         'person.hometown', 'position', 'positionGroup', 'year', 'person.collegeNames', 'collegeClass', 'combineAttendance',
         'draftProjection', 'grade', 'draftGrade', 'sizeScore', 'athleticismScore', 'productionScore', 'height', 'weight',
         'armLength', 'handSize', 'proFortyYardDash.designation', 'proFortyYardDash.seconds', 'sixtyYardShuttle.designation',
         'sixtyYardShuttle.seconds', 'fortyYardDash.designation', 'fortyYardDash.seconds', 'tenYardSplit.designation',
         'tenYardSplit.seconds', 'threeConeDrill.designation', 'threeConeDrill.seconds', 'twentyYardShuttle.designation',
         'twentyYardShuttle.seconds', 'verticalJump.designation', 'verticalJump.inches', 'benchPress.designation',
         'benchPress.repetitions', 'broadJump.designation', 'broadJump.inches', 'bio', 'overview', 'sourcesTellUs',
         'nflComparison', 'strengths', 'weaknesses', 'profileAuthor']]


df = df[df['positionGroup'].isin(['QB', 'RB', 'WR', 'TE'])]
df = df.sort_values(['year', 'draftGrade'], ascending=[False, False])

# Remove non-alpha numeric characters from first/last names.
df["first_name"] = df['person.firstName'].apply(
    lambda x: "".join(c for c in x if c.isalnum())
)
df["last_name"] = df['person.lastName'].apply(
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
df = df[df['position'].isin(['QB','WR','TE','RB'])]

def camel_to_snake(camel_str):
    snake_str = re.sub('([a-z])([A-Z])', r'\1_\2', camel_str)
    return snake_str.lower()

# Replace periods with underscores
df.columns = df.columns.str.replace('.', '_')

# Update column names to snake case
df.columns = [camel_to_snake(col) for col in df.columns]
# Remove 'designation' columns
df = df.loc[:, ~df.columns.str.contains('designation')]

# Update column names to title case with spaces


# Select the columns of interest

print(df.info(verbose=True))
df.to_csv('/Users/nick/sleepertoolsversion2/combine_data/apicombine.csv',index=False)