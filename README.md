# Predicting Steam Game Review Sentiment Using Pre-Release Metadata
Author: Ramley Hirneisen
Machine Learning Final Project Report

## 1. Introduction
### 1.1 Problem Statement

Steam reviews have become one of the most important indicators of how players feel about a game. Players use these reviews to decide whether a game is worth purchasing, and developers often rely on them to understand how their game is being received.
This project explores whether it is possible to predict if a game will be reviewed positively using only pre-release metadata such as price, genres, categories, achievements, and platform support.

### 1.2 Motivation

As someone going into game development, I find the idea of using machine learning to understand or even predict how players might respond to a game before it launches extremely interesting. Developers typically have no way of knowing how their game will be received until after release, but metadata is something they already have long before launch.
If metadata alone can give even a rough estimate of review sentiment, it could help with pricing, scope planning, or deciding how to position the game on the Steam store.

### 1.3 Approach

I built a machine learning classifier that predicts whether a game will receive mostly positive reviews on Steam.
Using the Steam Games Dataset 2025 from Kaggle, I:
Cleaned and preprocessed the dataset
Removed all review-related features from the inputs
Created 81 metadata-based features
Trained Logistic Regression and Random Forest models
This allowed me to evaluate how far metadata alone can go when predicting player satisfaction.

## 2. Data
### 2.1 Introducing the Data
The dataset contains 70,000+ Steam entries with fields such as:
-developers
-publishers
-genres
-categories
-platforms
-release date
-price
-achievements
-user review statistics

After filtering unreleased games and games with zero reviews, I further restricted the dataset to games with ≥10 reviews to avoid unstable review percentages.
Final cleaned dataset:
-24,000 games
-81 metadata-based features
### 2.2 Visual Analysis

A key pattern appears when comparing positive review percentage to total review count.
Games with very few reviews can look extremely positive or extremely negative simply because a tiny number of people rated them. As review count increases, ratings stabilize.

Figure 1. Review Positivity vs Review Count (Log Scale)
<img width="2952" height="2090" alt="image" src="https://github.com/user-attachments/assets/1c0e1fcb-e38d-4eb2-a7fe-4776f00516d7" />
This justified filtering out games with very low review counts.

### 2.3 Data Preprocessing

Main preprocessing steps:
-Parsed string-encoded lists (genres, categories, platforms)
-Multi-hot encoded categorical metadata
-Created numerical features: num_developers, num_publishers
-Filled missing metacritic scores with the median
-Converted release dates to a clean release_year
-Removed review-related columns entirely (to avoid cheating)

Final output:
24,000 samples × 81 metadata features

## 3. Machine Learning Methods
### 3.1 Label Creation

Labels were created from the positive_percentual column:
-≥ 75% positive reviews → Label = 1 (positive)
-< 75% positive reviews → Label = 0 (not positive)

This matches Steam’s “Very Positive” threshold.

### 3.2 Logistic Regression

Standardized inputs using StandardScaler
Serves as a linear baseline model
Helps understand how linear metadata relationships affect prediction

### 3.3 Random Forest Classifier

Ensemble of decision trees
Captures nonlinear interactions
Handles mixed feature types
Produces feature importance scores

## 4. Results
### 4.1 Experimental Setup

80/20 train–test split

Stratified sampling

No review-based features were used as input

### 4.2 Algorithm Performance

Logistic Regression Accuracy:
0.6733

Random Forest Accuracy:
0.6672

Metadata alone predicted review sentiment with ~67% accuracy, which is surprisingly strong considering that no gameplay information or player-written text was used.

Positive games were recognized much more reliably than negative games, likely because negative games fail for a very wide variety of reasons (bugs, bad controls, performance issues) that metadata cannot capture.

### 4.3 Feature Importance

Random Forest revealed several strong metadata predictors:

price_initial

n_achievements

genre_indie

genre_action, genre_casual, genre_adventure, genre_rpg

Steam Cloud, Trading Cards, Controller Support

Mac/Linux support

Figure 2. Top 30 Most Important Metadata Features

(Insert feature importance bar chart here)

Key insights:

Indie ranked unexpectedly high as a predictive genre

Multiplayer / co-op tags ranked surprisingly low, likely because multiplayer success depends heavily on factors metadata cannot show (netcode, matchmaking, concurrency, server quality)

Steam ecosystem integration features correlated with better reception

Higher platform support (Mac/Linux) may indicate better engineering pipelines

## 5. Conclusion
### 5.1 Closure

This project showed that pre-release metadata can predict Steam review sentiment much better than expected. Achieving around 67 percent accuracy without any gameplay information suggests that how a game is presented and structured holds meaningful clues about how players will respond.

### 5.2 Challenges

Handling messy real-world data

Parsing list-encoded strings

Creating reliable labels

Ensuring no review-based features leaked into the model

### 5.3 Future Work

To improve accuracy, future work could include:

XGBoost or other advanced models

Using the game description text via TF-IDF or embeddings

Sentiment analysis on user reviews

Cluster analysis of metadata or genres

## 6. References

Steam Games Dataset 2025. Kaggle, 2025.
Pedregosa, Fabian, et al. “Scikit Learn: Machine Learning in Python.” Journal of Machine Learning Research, vol. 12, 2011.
Lundberg, Scott M., and Su-In Lee. “A Unified Approach to Interpreting Model Predictions.” NeurIPS, 2017.

## 7. Acknowledgement

I used ChatGPT to help debug parts of the preprocessing code and to help explain certain results as I went through the project. ChatGPT also helped me structure parts of this written report.
