# Player Comps Tool Powered by Next Gen Stats

This project allows users to draw comparisons between NFL rookies and veterans, leveraging a KNN model trained on NFL NextGenStats and combine metrics. The dashboard presents a fun, interactive way to engage with NFL draft selections, providing insight into how rookies might perform based on historical player data. Check out the live demo and explore how your favorite rookies compare!

![Project Gif](your-gif-link-here.gif)

Link to project: [Player Comps Tool Powered by Next Gen Stats](http://recruiters-love-seeing-live-demos.com/)

## How It's Made:

**Tech used:** Python (Streamlit, Sci-Kit Learn, NumPy, pandas), HTML, CSS

### Data Extraction, Transformation, and Loading (ETL):

1. **Extraction:** 
   - Pulled data from NFL API and the nflfastR database.

2. **Transformation:** 
   - Cleaned the data, and handled missing values and duplicates.
   - Merged datasets by crafting a unique identifier, overcoming challenges like different naming conventions (e.g., 'Gabe Davis' vs 'Gabriel Davis').
  
3. **Loading:** 
   - Stored the transformed data in a project directory. Future plans include utilizing an SQL database to scale the app further.

### Model Training:

- Trained a KNN model on Next Gen Stat Scores, leveraging the similarities in features to draw comparisons between rookies and veterans.
- [Explanation of KNN model](your-youtube-link-here)
- [Explanation of Next Gen Stat Scores](your-next-gen-stat-scores-link-here)

### Project Functionality - ELI5 Version:

- The model gauges similarities based on the 4 Next Gen Stat Scores, draft positions, and other features, presenting a comparable veteran player for each rookie.
- Displays rookies and their comparable veterans in an intuitive dashboard with player cards and a grouped bar chart showcasing the Next Gen Stat Scores.

![Bar Chart Example](your-bar-chart-link-here.png)

### Rookie Year Projections:

- Utilized a KNeighbors Regressor to project rookie year performance based on the 7 most comparable veterans, optimizing the neighbor count through cross-validation.

## Optimizations:

- [ ] Modularize code for better maintainability.
- [ ] Enhance dashboard presentation, including a feature to display comparable veterans and scouting reports.
- [ ] Incorporate an explainer to elucidate the underlying mechanics.
- [ ] Simplify the draft pick and team input process for rookies.
- [ ] Craft a notebook assessing the model's mid-season performance in 2023.

## Bug Fixes:

- [ ] Ensure veterans are not compared with players drafted in subsequent years.
- [ ] Rectify instances where veterans are compared to themselves.
- [ ] Address missing rookie draft pick data for some veterans.

## Lessons Learned:

I learned a lot about Streamlit while building this project. I got hung up on the best way to show the player comps, but ultimately landed on a simple, effective way to communicate why the model may suggest certain players as the top player comp. I am excited to produce something people will want to display on social media for the 2024 draft after applying my planned optimizations. If you have any suggestions let me know! 
