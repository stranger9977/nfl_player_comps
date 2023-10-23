# Player Comps Tool Powered by Next Gen Stats


Streamlit dashboard that allows users to choose players dating back to 2006. From there, a KNN model compares rookies to other players in the NFL. Features include NFL NextGenStats and combine metrics. Data from the NFL API and nflfastR database. 

## How It's Made:

**Tech used:** Python (Streamlit, Sci-Kit Learn, NumPy, pandas)

The app extracts data from nflfastR and the Nfl API. The data is cleaned and merged creating a dataset with every player entering the rookie draft since 2006. From there, the data set was used to train a KNN model using a variety of features including, NGS scores for college production, athleticism, and size. The model comparisons are displayed in a simple bar chart format displaying the Next Gen Stat Scores displayed during the draft. 

I would like to improve the player cards so they don't have giant pictures of the player's face. Additionally, the model is just for fun. It would be interesting to try different models to see if they actually have any predictive power for fantasy football or real-life football outcomes. This is mostly a fun way to follow along with the NFL draft as players are selected, but with some work could provide value to coaches and managers. I desperately need to improve the visual aspect of this app and is the next thing on the roadmap. Below are examples of outputs I have created and plan to incorporate into my app. 
## Lessons Learned:

I learned a lot about Streamlit while building this project. I got hung up on the best way to show the player comps, but ultimately landed on a simple, effective way to communicate why the model may suggest certain players as the top player comp. I am excited to produce something people will want to display on social media. 
## Examples:
