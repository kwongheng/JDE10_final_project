# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "a056a1f4-8806-44a1-a656-fa345bd3cb28",
# META       "default_lakehouse_name": "NHL_db",
# META       "default_lakehouse_workspace_id": "0c230c13-c243-4a72-b362-76b949c3f17f",
# META       "known_lakehouses": [
# META         {
# META           "id": "a056a1f4-8806-44a1-a656-fa345bd3cb28"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# ## nb_04_ml_pl_totals_predictions_gold
# 
# This PySpark notebook executes the machine learning training, sportsbook odds simulation, expected value (EV) evaluation, and data tabulations for three NHL betting markets: Moneyline (Home Win), Puck Line (Home +1.5), and Totals (Over 5.5). The output datasets are saved as Gold-layer Delta tables.
# 
# ---
# 
# ### Key Workflow Steps
# 
# 1. **Imports & Dependencies**
# 
# * Imports PySpark ML modules (`VectorAssembler`, `LogisticRegression`), SQL functions (`F`), and window functions (`Window`).
# 
# 
# * Loads helper functions from `nb_00_dbutils` and `nb_04_ml_trg_functions`.
# 
# 
# 2. **Data Ingestion (Silver & Gold Tables)**
# 
# * Loads existing Silver datasets: `silver.game_teams_stats`, `silver.team_info`, `silver.game_features`, and regular-season games from `silver.game`.
# 
# 
# * Loads 2-year composite team performance metrics from `gold.team_composite_2yr`.
# 
# 3. **Compute 2-season team composite baseline ratings**
# 
# * Aggregates over two regular seasons (**2017-2018 and 2018-2019**). The final output is written to a **gold-tier analytical table** named `gold.team_composite_2yr`, which serves as a 2-season composite rating baseline for team evaluation.
# 
# 4. **Feature Engineering & Data Splitting**
# 
# * Assembles key performance features (`goal_diff_gap`, `pp_pct_gap`, `pk_pct_gap`, `corsi_gap`, `save_pct_gap`) into a `features` vector.
# 
# 
# * **Training Set:** 2017–18 and 2018–19 seasons.
# 
# 
# * **Holdout Evaluation Set:** 2019–20 season.
# 
# 
# 
# 5. **Model Training & Threshold Optimization**
# 
# * Trains PySpark `LogisticRegression` models for each target outcome:
# 
# 
# * **Moneyline (`home_win`):** Dynamic threshold set to $\ge 0.20$ (Achieved Win Rate: **64.3%**).
# 
# 
# * **Puck Line (`home_covers_plus1_5`):** Dynamic threshold set to $\ge 0.08$ (Achieved Win Rate: **74.9%**).
# 
# 
# * **Totals (`over`):** Dynamic threshold set to $\ge 0.05$ (Achieved Win Rate: **55.8%**).
# 
# 
# 6. **Opening Odds Generation**
# 
# * Simulates sportsbook opening lines for the 2019–20 season:
# 
# 
# * Bounds model probabilities between 40% and 75%.
# 
# 
# * Applies a 4% sportsbook margin.
# 
# 
# * Converts probabilities to American Moneyline odds.
# 
# 
# * Synthesizes Puck Line and Totals odds using seeded random ranges.
# 
# 
# 7. **Prediction Enrichment & EV Analysis**
# 
# * Joins model predictions with generated sportsbook odds.
# 
# 
# * Computes sportsbook implied probabilities, Expected Value percentages (`ev_pct`), and wager results (`WIN`/`LOSS`) across all three markets.
# 
# 
# 
# 
# 8. **Aggregated Reporting & Top Team Analysis**
# 
# * **Season Performance Summary:** Tabulates bet volume, skipped games, win/loss counts, and selective accuracy percentages. Unpivots data into long-format for visual reporting.
# 
# 
# * **Top Teams per Market:** Identifies and ranks the top 3 teams in each betting category based on win rates and qualified opportunities.
# 
# 
# * **Sample Opportunity Selection:** Filters top 10 highest positive-EV ($\ge 2.0\%$) and high-confidence predictions involving the top 3 teams per market.
# 
# 
# 
# ---
# 
# ### Target Gold Tables Generated
# 
# | Target Variable | Target Table Name |
# | --- | --- |
# | `BETTING_MONEYLINE_PREDICTIONS_TABLE` | `gold.betting_moneyline_predictions`<br> |
# | `BETTING_PUCKLINE_PREDICTIONS_TABLE` | `gold.betting_puckline_predictions`<br> |
# | `BETTING_TOTALS_PREDICTIONS_TABLE` | `gold.betting_totals_predictions`<br> |
# | `OPENING_ODDS` | `gold.opening_odds`<br> |
# | `BETTING_SEASON_TABULATION_TABLE` | `gold.betting_season_tabulation`<br> |
# | `BETTING_SUMMARY_UNPIVOTED_TABLE` | `gold.betting_summary_unpivoted`<br> |
# | `TOP_3_TEAMS_PER_MARKET_TABLE` | `gold.top_3_teams_per_market`<br> |
# | `BETTING_ALL_SAMPLES_TABLE` | `gold.betting_all_samples`<br> |


# MARKDOWN ********************

# ### Imports

# CELL ********************

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column
from pyspark.sql.window import Window, WindowSpec
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.functions import vector_to_array

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Load common functions

# CELL ********************

%run nb_00_dbutils

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Load ML training functions

# CELL ********************

%run nb_04_ml_trg_functions

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Define variables

# CELL ********************

# Existing tables
GAME_TABLE: str = "silver.game"
GAME_TEAMS_STATS_TABLE: str = "silver.game_teams_stats"
TEAM_INFO_TABLE: str = "silver.team_info"
GAME_FEATURES_TABLE: str = "silver.game_features"

# tables to create in this notebook
TEAM_COMPOSITE_2YR_TABLE: str = "gold.team_composite_2yr"

BETTING_MONEYLINE_PREDICTIONS_TABLE: str = "gold.betting_moneyline_predictions"
BETTING_PUCKLINE_PREDICTIONS_TABLE: str = "gold.betting_puckline_predictions"
BETTING_TOTALS_PREDICTIONS_TABLE: str = "gold.betting_totals_predictions"
OPENING_ODDS: str = "gold.opening_odds"

BETTING_SEASON_TABULATION_TABLE: str = "gold.betting_season_tabulation"
BETTING_SUMMARY_UNPIVOTED_TABLE: str = "gold.betting_summary_unpivoted"
TOP_3_TEAMS_PER_MARKET_TABLE: str = "gold.top_3_teams_per_market"
BETTING_ALL_SAMPLES_TABLE: str = "gold.betting_all_samples"

# Used to setup ML training features
FEATURE_COLS: list[str] = [
    "goal_diff_gap",
    "pp_pct_gap",
    "pk_pct_gap",
    "corsi_gap",
    "save_pct_gap",
]

# schema to create season tabulaton
SUMMARY_SCHEMA: list[str] = [
    "Category",
    "Total_Season_Games",
    "Qualified_Bets_Offered",
    "Correct_Predictions_WIN",
    "Wrong_Predictions_LOSS",
    "No_Bets_Low_Confidence",
    "Selective_Accuracy_Pct",
    "Skipped_Games_Pct",
]

# ML training season
TRAINING_SEASON_START =  20172018
TRAINING_SEASON_END = 20182019

# hold out season
HOLDOUT_SEASON = 20192020

# set betting range used to calculate opening odds
MIN_GAME_ID = 2019020001
MAX_GAME_ID = 2019021082

# target win rates used for training
ML_TARGET_WIN_RATE = 0.62
PL_TARGET_WIN_RATE = 0.65
TOTALS_TARGET_WIN_RATE = 0.55


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Load required silver tables

# CELL ********************

game_teams_stats: DataFrame = load_table(spark, GAME_TEAMS_STATS_TABLE)
team_info: DataFrame =  load_table(spark, TEAM_INFO_TABLE)
game_features: DataFrame =  load_table(spark, GAME_FEATURES_TABLE)

game_r: DataFrame = (
    load_table(spark, GAME_TABLE)
    .filter(F.col("type") == "R")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Compute 2-season team composite baseline ratings
# 
# Aggregates over two regular seasons (**2017-2018 and 2018-2019**). The final output is written to a **gold-tier analytical table** named `gold.team_composite_2yr`, which serves as a 2-season composite rating baseline for team evaluation.
# 
# ### Workflow Overview
# 1. **Context Enrichment (`tg`):** Evaluates whether a team played at home or away (`HoA`) to properly derive opponent IDs, goals scored (`goals_for`), and goals allowed (`goals_against`). It also structures a basic metric for shot volume (`corsi_proxy_for`).
# 2. **Opponent Alignment (`team_game`):** Self-joins the enriched dataset back onto itself matching by `game_id` and `opp_team_id`. This structures each row so a single team's metrics sit perfectly side-by-side with their opponent's game-day statistics.
# 3. **Aggregation & Feature Engineering (`team_composite`):** Filters for the target 2017-2019 windows, groups the history by `team_id`, and calculates 7 unique composite performance metrics.
# 4. **Data Export:** Persists the finalized metrics matrix directly into the Gold layer.
# 
# ---
# 
# ### Metric Calculations & Analytical Rationales
# 
# The engineered baseline metrics are calculated inside the notebook's final aggregation block using specific analytical logic:
# 
# | Target Metric Column | Calculation Logic (PySpark Syntax) | Analytical Purpose / Rationale |
# | :--- | :--- | :--- |
# | **`gf_per_gm`** <br>*(Goals For Per Game)* | `F.avg("goals_for")` | Measures raw offensive production and scoring capacity per game over the 2-year window. |
# | **`ga_per_gm`** <br>*(Goals Against Per Game)* | `F.avg("goals_against")` | Measures raw defensive prevention capabilities per game. |
# | **`goal_diff_per_gm`** <br>*(Goal Differential Per Game)* | `F.avg(F.col("goals_for") - F.col("goals_against"))` | Tracks the average goal margin per game. Goal differential is one of the strongest historic predictors of team sustainability and success. |
# | **`corsi_for_per_gm`** <br>*(Corsi Proxy For)* | `F.avg("corsi_proxy_for")`<br><br>*Note: `corsi_proxy_for` = `s.shots + s.blocked`* | Traditional **Corsi** tracks total shot attempts (Shots + Blocks + Misses). Because missed shots are omitted from this silver table, the notebook adds shots and blocks to generate a strong proxy measuring overall puck possession and offensive zone pressure. |
# | **`pp_pct`** <br>*(Power Play Percentage)* | `F.avg(F.col("pp_goals") / F.when(F.col("pp_opportunities") == 0, None).otherwise(F.col("pp_opportunities")))` | Computes special teams offensive efficiency. The `F.when().otherwise()` block prevents a `ZeroDivisionError` on games where a team received zero power-play opportunities by converting `0` to a `Null`, which PySpark's `avg` safely ignores. |
# | **`pk_pct`** <br>*(Penalty Kill Percentage)* | `1 - F.avg(F.col("pk_goals_against") / F.when(F.col("pk_opportunities") == 0, None).otherwise(F.col("pk_opportunities")))` | Evaluates special teams defensive efficiency. It divides the opponent's power-play goals by their opportunities to get the failure rate, then subtracts it from `1` to isolate how often the team successfully defended a penalty. |
# | **`save_pct_proxy`** <br>*(Save Percentage Proxy)* | `1 - F.avg(F.col("goals_against") / F.when(F.col("shots_against") == 0, None).otherwise(F.col("shots_against")))` | Proxies goalkeeper and team defensive efficiency. It calculates the opponent's shooting percentage (`goals_against / shots_against`), then subtracts it from `1` to determine the percentage of shots stopped. |


# CELL ********************

# Create calculate columns joining game info with game_teams_stats
tg: DataFrame = (
    game_teams_stats.alias("s")
    .join(
        game_r.alias("g"),
        on="game_id",
        how="inner",
    )
    .select(
        F.col("s.game_id"),
        F.col("s.team_id"),
        F.col("s.HoA"),
        F.col("s.won"),
        F.col("s.goals"),
        F.col("s.shots"),
        F.col("s.powerPlayOpportunities"),
        F.col("s.powerPlayGoals"),
        F.col("s.blocked"),
        F.col("g.season"),
        F.col("g.home_team_id"),
        F.col("g.away_team_id"),
        F.col("g.home_goals"),
        F.col("g.away_goals"),
        
        F.when(
            F.col("s.HoA") == "home",
            F.col("g.away_team_id"),
        ).otherwise(F.col("g.home_team_id")
        ).alias("opp_team_id"),

        F.when(
            F.col("s.HoA") == "home",
            F.col("g.home_goals"),
        ).otherwise(F.col("g.away_goals")
        ).alias("goals_for"),

        F.when(
            F.col("s.HoA") == "home",
            F.col("g.away_goals"),
        ).otherwise(F.col("g.home_goals")
        ).alias("goals_against"),

        (
            F.col("s.shots")
            + F.col("s.blocked")
        ).alias("corsi_proxy_for"),
    )
)

# Self-join to attach opponent statistics alongside each team per game
team_game: DataFrame = (
    tg.alias("main")
    .join(
        tg.alias("opp"),
        (
            (F.col("main.game_id") == F.col("opp.game_id"))
            & (F.col("main.opp_team_id") == F.col("opp.team_id"))
        ),
        how="inner",
    )
    .select(
        F.col("main.game_id"),
        F.col("main.season"),
        F.col("main.team_id"),
        F.col("main.opp_team_id"),
        F.col("main.HoA"),
        F.col("main.won"),
        F.col("main.goals_for"),
        F.col("main.goals_against"),
        F.col("main.shots").alias("shots_for"),
        F.col("opp.shots").alias("shots_against"),
        F.col("main.corsi_proxy_for"),
        F.col("main.powerPlayOpportunities").alias("pp_opportunities"),
        F.col("main.powerPlayGoals").alias("pp_goals"),
        F.col("opp.powerPlayOpportunities").alias("pk_opportunities"),
        F.col("opp.powerPlayGoals").alias("pk_goals_against"),
    )
)

# Aggregate across 2017-18 and 2018-19 training seasons into 2-year composite team ratings
team_composite_2yr: DataFrame = (
    team_game
    .filter(
        F.col("season").isin(TRAINING_SEASON_START, TRAINING_SEASON_END)
    )
    .groupBy("team_id")
    .agg(
        F.avg("goals_for").alias("gf_per_gm"),
        F.avg("goals_against").alias("ga_per_gm"),
        F.avg(
            F.col("goals_for") 
            - F.col("goals_against")
        ).alias("goal_diff_per_gm"),
        F.avg("corsi_proxy_for").alias("corsi_for_per_gm"),
        F.avg(
            F.col("pp_goals")
            / F.when(
                F.col("pp_opportunities") == 0, 
                None,
            ).otherwise(F.col("pp_opportunities"))
        ).alias("pp_pct"),
        (
            1
            - F.avg(
                F.col("pk_goals_against")
                / F.when(
                    F.col("pk_opportunities") == 0,
                    None,
                ).otherwise(F.col("pk_opportunities"))
            )
        ).alias("pk_pct"),
        (
            1
            - F.avg(
                F.col("goals_against")
                / F.when(
                    F.col("shots_against") == 0,
                    None,
                ).otherwise(F.col("shots_against"))
            )
        ).alias("save_pct_proxy"),
    )
)

write_table(team_composite_2yr, TEAM_COMPOSITE_2YR_TABLE)
print(f"{TEAM_COMPOSITE_2YR_TABLE} gold table written")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Feature Engineering 
# 
# Using `with_features()` function enriches game-level records with team performance metrics and creates machine learning features and target labels for model training.
# 
# #### Model Features
# 
# The following columns are assembled into a single feature vector for model training:
# 
# ```python
# FEATURE_COLS = [
#     "goal_diff_gap",
#     "pp_pct_gap",
#     "pk_pct_gap",
#     "corsi_gap",
#     "save_pct_gap",
# ]
# ```
# 
# #### Training and Evaluation Split
# 
# | Dataset | Seasons |
# |----------|---------|
# | Training | 2017-18, 2018-19 |
# | Holdout Evaluation | 2019-20 |
# 
# Prior to model training, all feature columns are filled with `0` where null values exist and transformed into a Spark ML `features` vector using `VectorAssembler`.
# 
# ##### Model Training
# 
# - Train a Spark ML `LogisticRegression` model using the assembled `features` vector.
# - Use the supplied `label_col` as the prediction target.
# - Generate predictions on the holdout dataset.
# 
# ##### Outputs
# 
# The function returns:
# 
# 1. DataFrame containing:
#    - Prediction probabilities
#    - Predicted outcomes
#    - Actual outcomes
#    - Confidence scores
#    - Hit indicators
# 
# 2. A best threshold
#    - The selected confidence threshold for downstream filtering and betting strategy evaluation.


# CELL ********************

# setup dataframes used for training

assembler = VectorAssembler(
    inputCols=FEATURE_COLS,
    outputCol="features",
)

# Train on 2017-18 and 2018-19 seasons.
train_feat: DataFrame = (
    assembler.transform(
        with_features(
            game=game_r.filter(
                F.col("season").isin(
                    TRAINING_SEASON_START, 
                    TRAINING_SEASON_END,
                )
            ),
            composite=team_composite_2yr,
        )
    )
    .fillna(0, subset=FEATURE_COLS,)
)

holdout_feat: DataFrame = (
    assembler.transform(
        with_features(
            game=game_r.filter(
                F.col("season") == HOLDOUT_SEASON,
            ),
            composite=team_composite_2yr,
        )
    )
    .fillna(0, subset=FEATURE_COLS,)
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# build training for moneyline and write gold table
ml_all, ml_thresh = (
    train_score_and_optimize(
        "home_win", 
        train_feat, 
        holdout_feat, 
        target_win_rate=ML_TARGET_WIN_RATE
    )
)
write_table(ml_all, BETTING_MONEYLINE_PREDICTIONS_TABLE)
print(f"{BETTING_MONEYLINE_PREDICTIONS_TABLE} table written")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# build training for puckline and write gold table
pl_all, pl_thresh = (
    train_score_and_optimize(
        "home_covers_plus1_5", 
        train_feat, 
        holdout_feat, 
        target_win_rate=PL_TARGET_WIN_RATE
    )
)
write_table(pl_all, BETTING_PUCKLINE_PREDICTIONS_TABLE)
print(f"{BETTING_PUCKLINE_PREDICTIONS_TABLE} table written")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# build training for totals and write gold table
tot_all, tot_thresh = (
    train_score_and_optimize(
        "over", 
        train_feat, 
        holdout_feat, 
        target_win_rate=TOTALS_TARGET_WIN_RATE
    )
)
write_table(tot_all, BETTING_TOTALS_PREDICTIONS_TABLE)
print(f"{BETTING_TOTALS_PREDICTIONS_TABLE} table written")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Opening Odds Generation
# 
# An opening odds dataset is generated for the 2019-20 season using model predictions, team metadata, and sportsbook-style pricing logic. The resulting table simulates opening betting markets for moneyline, puck line, and totals wagers.
# 
# ##### Team Mapping
# 
# Team identifiers are enriched with their associated abbreviations:
# 
# | Column | Description |
# |----------|-------------|
# | `team_id` | Unique NHL team identifier. |
# | `abbreviation` | Team abbreviation used for display purposes. |
# 
# If a team abbreviation is unavailable, a fallback value in the format `TEAM_<team_id>` is assigned.
# 
# ##### Model Probability Processing
# 
# Model-generated home win probabilities are incorporated from the prediction dataset.
# 
# | Column | Description |
# |----------|-------------|
# | `model_home_prob` | Predicted home win probability from the ML model. |
# | `model_home_bounded` | Home probability constrained between 40% and 75%. |
# | `model_away_bounded` | Complementary away win probability. |
# 
# A default probability of `0.55` is assigned when no model prediction is available.
# 
# ##### Sportsbook Margin Adjustment
# 
# To simulate bookmaker pricing, a margin is applied to both sides:
# 
# | Column | Formula |
# |----------|----------|
# | `book_home_prob` | `model_home_bounded - 0.04` |
# | `book_away_prob` | `model_away_bounded - 0.04` |
# 
# This introduces a simplified sportsbook edge into the generated market.
# 
# ##### Moneyline Odds Generation
# 
# Adjusted probabilities are converted into American moneyline odds:
# 
# | Column | Description |
# |----------|-------------|
# | `opening_ml_home` | Opening home moneyline. |
# | `opening_ml_away` | Opening away moneyline. |
# 
# The conversion follows standard American odds calculations:
# 
# - Negative odds for favorites (`probability >= 50%`)
# - Positive odds for underdogs (`probability < 50%`)
# 
# ##### Puck Line Market
# 
# Synthetic puck line prices are generated using seeded random values:
# 
# | Column | Description |
# |----------|-------------|
# | `opening_pl_home_odds` | Home


# CELL ********************

# set ml, pl, totals game id to specific range
ml_all = ml_all.filter((F.col("game_id") >= MIN_GAME_ID) & (F.col("game_id") <= MAX_GAME_ID))
pl_all = pl_all.filter((F.col("game_id") >= MIN_GAME_ID) & (F.col("game_id") <= MAX_GAME_ID))
tot_all = tot_all.filter((F.col("game_id") >= MIN_GAME_ID) & (F.col("game_id") <= MAX_GAME_ID))


opening_odds: DataFrame = (
    game_features
    .filter(F.col("season") == HOLDOUT_SEASON)
    .join(
        ml_all.select("game_id", "prob_positive"),
        on="game_id",
        how="left",
    )
    .join(
        team_info.select(
            F.col("team_id").alias("home_team_id"),
            F.col("abbreviation").alias("home_team"),
        ),
        on="home_team_id",
        how="left",
    )
    .join(
        team_info.select(
            F.col("team_id").alias("away_team_id"),
            F.col("abbreviation").alias("away_team"),
        ),
        on="away_team_id",
        how="left",
    )
    .withColumn(
        "home_team",
        F.coalesce(
            F.col("home_team"),
            F.concat(
                F.lit("TEAM_"),
                F.col("home_team_id"),
            ),
        ),
    )
    .withColumn(
        "away_team",
        F.coalesce(
            F.col("away_team"),
            F.concat(
                F.lit("TEAM_"),
                F.col("away_team_id"),
            ),
        ),
    )
    .withColumn(
        "game_date",
        F.to_date(
            F.col("game_id").cast("string").substr(1, 8),
            "yyyyMMdd",
        ),
    )
    .withColumn(
        "model_home_prob",
        F.coalesce(
            F.col("prob_positive"),
            F.lit(0.55),
        ),
    )
    .withColumn(
        "model_home_bounded",
        F.greatest(
            F.lit(0.40),
            F.least(
                F.lit(0.75),
                F.col("model_home_prob"),
            ),
        ),
    )
    .withColumn(
        "model_away_bounded",
        1.0 - F.col("model_home_bounded"),
    )
    .withColumn(
        "book_home_prob",
        F.col("model_home_bounded") - 0.04,
    )
    .withColumn(
        "book_away_prob",
        F.col("model_away_bounded") - 0.04,
    )
    .withColumn(
        "opening_ml_home",
        F.when(
            F.col("book_home_prob") >= 0.50,
            F.round(
                -100
                * (
                    F.col("book_home_prob")
                    / (1.0 - F.col("book_home_prob"))
                )
            ).cast("int"),
        ).otherwise(
            F.round(
                100
                * (
                    (1.0 - F.col("book_home_prob"))
                    / F.col("book_home_prob")
                )
            ).cast("int"),
        ),
    )
    .withColumn(
        "opening_ml_away",
        F.when(
            F.col("book_away_prob") >= 0.50,
            F.round(
                -100
                * (
                    F.col("book_away_prob")
                    / (1.0 - F.col("book_away_prob"))
                )
            ).cast("int"),
        ).otherwise(
            F.round(
                100
                * (
                    (1.0 - F.col("book_away_prob"))
                    / F.col("book_away_prob")
                )
            ).cast("int"),
        ),
    )
    .withColumn(
        "opening_pl_home_odds",
        F.round(
            -155 + (F.rand(seed=404) * 40),
        ).cast("int"),
    )
    .withColumn(
        "opening_pl_away_odds",
        F.round(
            105 + (F.rand(seed=505) * 35),
        ).cast("int"),
    )
    .withColumn(
        "opening_total_line",
        F.lit(5.5),
    )
    .withColumn(
        "opening_over_odds",
        F.round(
            -120 + (F.rand(seed=606) * 15),
        ).cast("int"),
    )
    .withColumn(
        "opening_under_odds",
        F.round(
            -120 + (F.rand(seed=707) * 15),
        ).cast("int"),
    )
    .select(
        "game_id",
        "season",
        "game_date",
        "home_team",
        "away_team",
        "opening_ml_home",
        "opening_ml_away",
        "opening_pl_home_odds",
        "opening_pl_away_odds",
        "opening_total_line",
        "opening_over_odds",
        "opening_under_odds",
    )
)

write_table(opening_odds, OPENING_ODDS)
print(f"{OPENING_ODDS} gold table written")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Prediction Enrichment and Expected Value (EV) Analysis
# 
# Model predictions are combined with generated sportsbook odds to calculate implied probabilities, expected value percentages, and betting outcomes for moneyline, puck line, and totals markets.
# 
# ##### Bet Outcome Classification
# 
# Each wager is assigned a result:
# 
# | Outcome | Condition |
# |----------|----------|
# | WIN | Prediction matches the actual outcome. |
# | LOSS | Prediction does not match the actual outcome. |
# 
# ##### Moneyline Enrichment
# 
# Moneyline predictions are joined with opening odds data to produce:
# 
# | Column | Description |
# |----------|-------------|
# | `picked_team` | Team selected by the model. |
# | `picked_odds` | Associated moneyline odds. |
# | `model_win_prob` | Probability assigned to the selected team. |
# | `book_implied_prob` | Sportsbook implied probability. |
# | `ev_pct` | Expected value percentage. |
# | `bet_outcome` | Final wager result. |
# 
# ##### Puck Line Enrichment
# 
# Puck line predictions are enriched with betting market information.
# 
# | Column | Description |
# |----------|-------------|
# | `picked_side` | Selected puck line side. |
# | `picked_odds` | Puck line odds for the selected side. |
# | `model_cover_prob` | Model probability of covering the spread. |
# | `book_implied_prob` | Market implied probability. |
# | `ev_pct` | Expected value percentage. |
# | `bet_outcome` | Final wager result. |
# 
# Examples:
# 
# - Home Team +1.5
# - Away Team -1.5
# 
# ##### Totals Enrichment
# 
# Totals predictions are converted into Over/Under betting opportunities.
# 
# | Column | Description |
# |----------|-------------|
# | `picked_side` | OVER 5.5 or UNDER 5.5. |
# | `picked_odds` | Odds associated with the selection. |
# | `model_tot_prob` | Model probability assigned to the selected outcome. |
# | `book_implied_prob` | Market implied probability. |
# | `ev_pct` | Expected value percentage. |
# | `bet_outcome` | Final wager result. |
# 
# ##### Output Datasets
# 
# Three enriched betting datasets are produced:
# 
# | Dataset | Purpose |
# |----------|---------|
# | `ml_joined` | Moneyline predictions with odds and EV analysis. |
# | `pl_joined` | Puck line predictions with odds and EV analysis. |
# | `tot_joined` | Totals predictions with odds and EV analysis. |
# 
# These datasets provide the foundation for evaluating model performance, identifying positive EV opportunities, and simulating betting strategies across multiple NHL betting markets.


# CELL ********************

ml_joined: DataFrame = (
    ml_all
    .join(opening_odds, "game_id", "inner")
    .withColumn(
        "picked_team",
        F.when(
            F.col("predicted_label") == 1,
            F.col("home_team"),
        ).otherwise(F.col("away_team")),
    )
    .withColumn(
        "picked_odds",
        F.when(
            F.col("predicted_label") == 1,
            F.col("opening_ml_home"),
        ).otherwise(F.col("opening_ml_away")),
    )
    .withColumn(
        "model_win_prob",
        F.when(
            F.col("predicted_label") == 1,
            F.col("prob_positive"),
        ).otherwise(
            1 - F.col("prob_positive"),
        ),
    )
    .withColumn(
        "book_implied_prob",
        american_to_implied_prob("picked_odds"),
    )
    .withColumn(
        "ev_pct",
        compute_ev_pct(
            "model_win_prob",
            "picked_odds",
        ),
    )
    .withColumn(
        "bet_outcome",
        derive_outcome(),
    )
)

pl_joined: DataFrame = (
    pl_all
    .join(opening_odds, "game_id", "inner")
    .withColumn(
        "picked_side",
        F.when(
            F.col("predicted_label") == 1,
            F.concat(
                F.col("home_team"),
                F.lit(" +1.5"),
            ),
        ).otherwise(
            F.concat(
                F.col("away_team"),
                F.lit(" -1.5"),
            ),
        ),
    )
    .withColumn(
        "picked_odds",
        F.when(
            F.col("predicted_label") == 1,
            F.col("opening_pl_home_odds"),
        ).otherwise(
            F.col("opening_pl_away_odds"),
        ),
    )
    .withColumn(
        "model_cover_prob",
        F.when(
            F.col("predicted_label") == 1,
            F.col("prob_positive"),
        ).otherwise(
            1 - F.col("prob_positive"),
        ),
    )
    .withColumn(
        "book_implied_prob",
        american_to_implied_prob("picked_odds"),
    )
    .withColumn(
        "ev_pct",
        compute_ev_pct(
            "model_cover_prob",
            "picked_odds",
        ),
    )
    .withColumn(
        "bet_outcome",
        derive_outcome(),
    )
)

tot_joined: DataFrame = (
    tot_all
    .join(opening_odds, "game_id", "inner")
    .withColumn(
        "picked_side",
        F.when(
            F.col("predicted_label") == 1,
            F.lit("OVER 5.5"),
        ).otherwise(
            F.lit("UNDER 5.5"),
        ),
    )
    .withColumn(
        "picked_odds",
        F.when(
            F.col("predicted_label") == 1,
            F.col("opening_over_odds"),
        ).otherwise(
            F.col("opening_under_odds"),
        ),
    )
    .withColumn(
        "model_tot_prob",
        F.when(
            F.col("predicted_label") == 1,
            F.col("prob_positive"),
        ).otherwise(
            1 - F.col("prob_positive"),
        ),
    )
    .withColumn(
        "book_implied_prob",
        american_to_implied_prob("picked_odds"),
    )
    .withColumn(
        "ev_pct",
        compute_ev_pct(
            "model_tot_prob",
            "picked_odds",
        ),
    )
    .withColumn(
        "bet_outcome",
        derive_outcome(),
    )
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Season Performance Summary Generation
# 
# Season-level performance summaries are generated for each betting market using the dynamically selected confidence thresholds. The objective is to evaluate prediction quality, betting volume, and the impact of skipping lower-confidence selections.
# 
# ##### Betting Categories Evaluated
# 
# Three betting markets are summarized independently:
# 
# | Category | Threshold Source |
# |----------|----------|
# | Moneyline (Home Win) | `ml_thresh` |
# | Puck Line (Home +1.5) | `pl_thresh` |
# | Totals (Over 5.5) | `tot_thresh` |
# 
# Each market uses its own dynamically optimized confidence threshold from the earlier model evaluation process.
# 
# ##### Season Summary Table
# 
# The initial summary table stores one record per betting category and includes:
# 
# - Total games
# - Number of bets offered
# - Number of winning bets
# - Number of losing bets
# - Number of skipped games
# - Selective accuracy percentage
# - Percentage of skipped games
# 
# This provides a high-level performance overview for each betting strategy.
# 
# ##### Visualization-Oriented Unpivoting
# 
# The summary data is transformed into a long-format structure using `stack()` to support reporting and visualization.
# 
# The following reporting metrics are created:
# 
# | Metric_Name | Metric_Group |
# |-------------|--------------|
# | Committed Bets | Action Status |
# | Skipped Games | Action Status |
# | Correct WIN | Outcome Status |
# | Wrong LOSS | Outcome Status |
# 
# ##### Output Datasets
# 
# | Dataset | Purpose |
# |----------|---------|
# | `season_tabulation` | Wide-format season summary by betting category. |
# | `unpivoted_summary` | Long-format reporting dataset optimized for charts, dashboards, and visual analysis. |
# 
# These summaries provide a concise view of betting volume, prediction quality, and the effectiveness of confidence-based filtering across Moneyline, Puck Line, and Totals betting strategies.


# CELL ********************

season_tabulation: DataFrame = (
    spark.createDataFrame(
        [
            generate_season_summary(
                ml_all,
                "Moneyline (Home Win)",
                ml_thresh,
            ),
            generate_season_summary(
                pl_all,
                "Puck Line (Home +1.5)",
                pl_thresh,
            ),
            generate_season_summary(
                tot_all,
                "Totals (Over 5.5)",
                tot_thresh,
            ),
        ],
        SUMMARY_SCHEMA,
    )
)

write_table(season_tabulation, BETTING_SEASON_TABULATION_TABLE)
print(f"{BETTING_SEASON_TABULATION_TABLE} gold tables written")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

unpivoted_summary: DataFrame = (
    season_tabulation.select(
        "Category",
        F.expr(
            """
            stack(
                4,
                'Committed Bets',
                Qualified_Bets_Offered,
                'Action Status',

                'Skipped Games',
                No_Bets_Low_Confidence,
                'Action Status',

                'Correct WIN',
                Correct_Predictions_WIN,
                'Outcome Status',

                'Wrong LOSS',
                Wrong_Predictions_LOSS,
                'Outcome Status'
            ) as (Metric_Name, Game_Count, Metric_Group)
            """
        ),
   )
)

write_table(unpivoted_summary, BETTING_SUMMARY_UNPIVOTED_TABLE)
print(f"{BETTING_SUMMARY_UNPIVOTED_TABLE} gold tables written")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Top Team Ranking by Betting Market
# 
# The highest-performing teams are identified for each betting market based on qualified betting opportunities and historical win rates.
# 
# ##### Markets Evaluated
# 
# Top-performing teams are identified independently for:
# 
# | Market | Dataset |
# |----------|---------|
# | Moneyline (Home Win) | `ml_joined` |
# | Puck Line (Home +1.5) | `pl_joined` |
# | Totals (Over 5.5) | `tot_joined` |
# 
# Each market uses its own optimized confidence threshold:
# 
# - `ml_thresh`
# - `pl_thresh`
# - `tot_thresh`
# 
# ##### Consolidated Rankings
# 
# The top three teams from each betting market are combined into a single reporting table.
# 
# | Column | Description |
# |----------|-------------|
# | `Category` | Betting market category. |
# | `Rank` | Team ranking within the category. |
# | `Team` | Team abbreviation/name. |
# | `Qualified_Bets` | Number of qualifying betting opportunities. |
# | `Winning_Bets` | Number of successful bets. |
# | `Win_Rate_Pct` | Team win rate percentage. |
# 
# ##### Top Team Lists
# 
# For downstream reporting and narrative generation, the teams are also extracted into Python lists:
# 
# | Variable | Description |
# |----------|-------------|
# | `top_3_ml` | Top three teams for Moneyline


# CELL ********************

top_ml_teams: DataFrame = get_top_3_teams(
    ml_joined,
    "Moneyline (Home Win)",
    ml_thresh,
)

top_pl_teams: DataFrame = get_top_3_teams(
    pl_joined,
    "Puck Line (Home +1.5)",
    pl_thresh,
)

top_tot_teams: DataFrame = get_top_3_teams(
    tot_joined,
    "Totals (Over 5.5)",
    tot_thresh,
)

top_teams_all: DataFrame = (
    top_ml_teams
    .union(top_pl_teams)
    .union(top_tot_teams)
    .select(
        F.col("category").alias("Category"),
        F.col("rank").alias("Rank"),
        F.col("team").alias("Team"),
        F.col("total_bets").alias("Qualified_Bets"),
        F.col("total_wins").alias("Winning_Bets"),
        F.concat(
            F.col("win_rate"),
            F.lit("%"),
        ).alias("Win_Rate_Pct"),
    )
)

write_table(top_teams_all, TOP_3_TEAMS_PER_MARKET_TABLE)
print(f"{TOP_3_TEAMS_PER_MARKET_TABLE} gold tables written")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Consolidated Sample Match Selection
# 
# Representative betting opportunities are selected across all betting markets to showcase the model's highest-confidence and highest-value recommendations.
# 
# ##### Selection Criteria
# 
# Sample bets must satisfy both:
# 
# | Requirement | Condition |
# |-------------|------------|
# | Confidence Threshold | `confidence >= market threshold` |
# | Positive Expected Value | `ev_pct >= 2.0` |
# 
# This ensures only high-confidence, positive-EV betting opportunities are included.
# 
# ##### Top Team Filtering
# 
# To focus on the strongest-performing teams, sample matches are restricted to games involving one of the previously identified top-three teams for each market.
# 
# | Market | Team Filter |
# |----------|-------------|
# | Moneyline | `top_3_ml` |
# | Puck Line | `top_3_pl` |
# | Totals | `top_3_tot` |
# 
# A game qualifies if either the home team or away team appears in the corresponding top-team list.
# 
# ##### Ranking Logic
# 
# Within each market, qualifying bets are ranked by:
# 
# 1. Highest Expected Value Percentage (`ev_pct`)
# 2. Lowest `game_id` (tie-breaker)
# 
# This ranking is implemented using a Spark window function:
# 
# ```text
# ORDER BY ev_pct DESC, game_id ASC
# ```
# 
# ##### Sample Size
# 
# For each betting category:
# 
# ```text
# Top 10 ranked opportunities
# ```
# 
# are retained for reporting purposes.
# 
# ##### Moneyline Samples
# 
# Moneyline selections include:
# 
# | Column | Description |
# |----------|-------------|
# | `picked_side` | Team selected by the model. |
# | `picked_odds` | Moneyline odds. |
# | `model_prob` | Model-estimated win probability. |
# | `book_prob` | Sportsbook implied probability. |
# | `confidence_metric` | Confidence score derived from model prediction. |
# | `expected_value_pct` | Calculated EV percentage. |
# | `bet_outcome` | Historical result of the wager. |
# 
# Category label:
# 
# ```text
# Moneyline (Home Win)
# ```
# 
# ##### Puck Line Samples
# 
# Puck line selections include:
# 
# | Column | Description |
# |----------|-------------|
# | `picked_side` | Selected puck line side. |
# | `picked_odds` | Puck line odds. |
# | `model_prob` | Predicted cover probability. |
# | `book_prob` | Sportsbook implied probability. |
# | `confidence_metric` | Confidence score. |
# | `expected_value_pct` | Expected value percentage. |
# | `bet_outcome` | Historical result. |
# 
# Category label:
# 
# ```text
# Puck Line (Home +1.5)
# ```
# 
# ##### Totals Samples
# 
# Totals selections include:
# 
# | Column | Description |
# |----------|-------------|
# | `picked_side` | OVER 5.5 or UNDER 5.5. |
# | `picked_odds` | Totals market odds. |
# | `model_prob` | Predicted probability of the selected outcome. |
# | `book_prob` | Sportsbook implied probability. |
# | `confidence_metric` | Confidence score. |
# | `expected_value_pct` | Expected value percentage. |
# | `bet_outcome` | Historical result. |
# 
# Category label:
# 
# ```text
# Totals (Over 5.5)
# ```
# 
# ##### Consolidated Output
# 
# All market-specific samples are combined into a single dataset:
# 
# ```python
# all_samples
# ```
# 
# The consolidated dataset contains:
# 
# - Betting category
# - Game information
# - Home and away teams
# - Recommended betting side
# - Betting odds
# - Model probability
# - Sportsbook probability
# - Confidence score
# - Expected value percentage
# - Final betting outcome
# 
# ##### Outputs
# 
# | Dataset | Purpose |
# |----------|---------|
# | `ml_samples` | Top Moneyline betting opportunities. |
# | `pl_samples` | Top Puck Line betting opportunities. |
# | `tot_samples` | Top Totals betting opportunities. |
# | `all_samples` | Consolidated sample bets across all markets. |
# 
# These sample selections provide a curated set of the model's strongest historical betting opportunities, emphasizing high-confidence predictions, positive expected value, and teams with proven betting performance.


# CELL ********************

top_3_ml: list[str] = [
    row["team"]
    for row in top_ml_teams.select("team").collect()
]

top_3_pl: list[str] = [
    row["team"]
    for row in top_pl_teams.select("team").collect()
]

top_3_tot: list[str] = [
    row["team"]
    for row in top_tot_teams.select("team").collect()
]

window_distinct: WindowSpec = Window.orderBy(
    F.col("ev_pct").desc(),
    F.col("game_id").asc(),
)

ml_samples: DataFrame = (
    ml_joined
    .filter(
        (F.col("confidence") >= ml_thresh)
        & (F.col("ev_pct") >= 2.0)
    )
    .filter(
        F.col("home_team").isin(top_3_ml)
        | F.col("away_team").isin(top_3_ml)
    )
    .withColumn(
        "rank",
        F.row_number().over(window_distinct),
    )
    .filter(F.col("rank") <= 10)
    .withColumn(
        "picked_side",
        F.col("picked_team"),
    )
    .withColumn(
        "Category",
        F.lit("Moneyline (Home Win)"),
    )
    .select(
        "Category",
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "picked_side",
        "picked_odds",
        F.round("model_win_prob", 3).alias("model_prob"),
        F.round("book_implied_prob", 3).alias("book_prob"),
        F.round("confidence", 3).alias("confidence_metric"),
        F.round("ev_pct", 2).alias("expected_value_pct"),
        "bet_outcome",
    )
)

pl_samples: DataFrame = (
    pl_joined
    .filter(
        (F.col("confidence") >= pl_thresh)
        & (F.col("ev_pct") >= 2.0)
    )
    .filter(
        F.col("home_team").isin(top_3_pl)
        | F.col("away_team").isin(top_3_pl)
    )
    .withColumn(
        "rank",
        F.row_number().over(window_distinct),
    )
    .filter(F.col("rank") <= 10)
    .withColumn(
        "Category",
        F.lit("Puck Line (Home +1.5)"),
    )
    .select(
        "Category",
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "picked_side",
        "picked_odds",
        F.round("model_cover_prob", 3).alias("model_prob"),
        F.round("book_implied_prob", 3).alias("book_prob"),
        F.round("confidence", 3).alias("confidence_metric"),
        F.round("ev_pct", 2).alias("expected_value_pct"),
        "bet_outcome",
    )
)

tot_samples: DataFrame = (
    tot_joined
    .filter(
        (F.col("confidence") >= tot_thresh)
        & (F.col("ev_pct") >= 2.0)
    )
    .filter(
        F.col("home_team").isin(top_3_tot)
        | F.col("away_team").isin(top_3_tot)
    )
    .withColumn(
        "rank",
        F.row_number().over(window_distinct),
    )
    .filter(F.col("rank") <= 10)
    .withColumn(
        "Category",
        F.lit("Totals (Over 5.5)"),
    )
    .select(
        "Category",
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "picked_side",
        "picked_odds",
        F.round("model_tot_prob", 3).alias("model_prob"),
        F.round("book_implied_prob", 3).alias("book_prob"),
        F.round("confidence", 3).alias("confidence_metric"),
        F.round("ev_pct", 2).alias("expected_value_pct"),
        "bet_outcome",
    )
)

all_samples: DataFrame = (
    ml_samples
    .unionByName(
        pl_samples,
        allowMissingColumns=True,
    )
    .unionByName(
        tot_samples,
        allowMissingColumns=True,
    )
)

write_table(all_samples, BETTING_ALL_SAMPLES_TABLE)
print(f"{BETTING_ALL_SAMPLES_TABLE} gold table written")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
