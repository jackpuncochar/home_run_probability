# Predicting Pre-Pitch Home Run Probability

## Overview
The introduction of the Statcast tracking system increased the prevalence of analytics in baseball. Research emerged claiming that Statcast metrics, exit velocity and launch angle, are strong predictors of home run hitting after contact. However, there is no research exploring the pre-contact probability of hitting a home run. Understanding the probability that a hitter goes yard before the pitch can provide smarter TV audiences with another piece of valuable content. A home run probability model exists at one media company, but the model suffers from overfitting. Also, typical binary classification problems in the sports realm explore problems with balanced classes (i.e., win or lose). This study dealt with extreme class imbalance (< 5% of batted balls are home runs). Logistic regression and Naïve Bayes were trained on over 70,000 batted balls from the 2021 season using 2019 pitcher and hitter statistics and game state (count, outs) as model features. Both models performed poorly using precision and recall for model assessment. The poor performance was attributed to uncertainty in data that is only known pre-pitch and it was found that home run classification may not be meaningful when the desired output is a probability. Instead, log loss was used for model selection, and logistic regression was selected to estimate predicted probabilities on new data. An R Shiny application made it possible to display the results of the HR probability model on live pitches. The application was in the early stages and needed bug fixes and optimized code before being sent to the client. Another limitation with the R Shiny app was a lack of automation in updating as Statcast updated in real-time. The HR probability system was not sent to the client; however, the framework to efficiently process Statcast data and deploy an accurate HR probability model will help them emerge as leaders in the industry, so it is necessary to follow up with the client.

## Project Structure

## Technologies Used
- R (main project code, call statcast API, data processing, feature engineering, model selection, and final model evaluation)
- R Shiny (display HR probability as a scoreboard with live game state and hitter/pitcher data)

## Future Work
