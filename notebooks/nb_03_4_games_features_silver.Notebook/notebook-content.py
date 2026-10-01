# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "13efe94d-6710-492a-87f5-be3b963bbad9",
# META       "default_lakehouse_name": "NHL_data",
# META       "default_lakehouse_workspace_id": "158c85ce-73a3-49f4-a3a4-a281a75c88f6",
# META       "known_lakehouses": [
# META         {
# META           "id": "13efe94d-6710-492a-87f5-be3b963bbad9"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# ## nb_03_4_game_features_silver
# 
# This notebook processes data from **silver-tier hockey tables** (`silver.game` and `silver.game_teams_stats`) to build rolling matchup features and betting prediction targets. The notebook engineers time-based team performance metrics and pivots them into a single matchup-level row containing both home and away team variables. The final output is written to a **silver feature-store table** named `silver.game_features`.
# 
# ### Workflow Overview
# 1. **Data Loading:** Imports Spark functions and loads individual team stats alongside general game metadata, filtering specifically for regular-season games (`type == "R"`).
# 2. **Base Stats Consolidation (`team_game_stats`):** Joins game attributes (date, season, game goals) with specific team box scores to build a unified chronological baseline per game per team.
# 3. **Window Definition & Data Leakage Prevention:** Creates a chronological window partitioned by `team_id`. It defines a strict sliding window covering the **previous 10 games** (`rowsBetween(-10, -1)`) to avoid data leakage from the current game's result.
# 4. **Feature Engineering (`team_features`):** Computes scheduling fatigue indicators (days of rest, back-to-back flags) and calculates 10-game moving averages for goals, shots, and win rates.
# 5. **Perspective Reshaping & Alignment:** Splits the dataset into distinct `home_team` and `away_team` views, renames columns to prevent overlaps, and joins them back together using `game_id` to establish a comparative matchup structure.
# 6. **Target & Difference Generation (`game_features`):** Calculates comparative delta features (shot differentials, rest differentials) and structures binary classification columns for Over/Under and Puckline betting targets.
# 7. **Data Export:** Persists the finalized feature matrix directly into the Silver layer table.
# 
# ---
# 
# ### Feature Engineering & Target Calculations
# 
# The engineered features and machine learning targets are derived in the main computing block using specific analytical logic:
# 
# | Target/Feature Column | Calculation Logic (PySpark Syntax) | Analytical Purpose / Rationale |
# | :--- | :--- | :--- |
# | **`days_rest`** | `F.datediff(F.to_date(F.col("date_time")), F.to_date(F.col("prev_game_date")))` | Measures the calendar days between a team's current game and their previous game to track physical recovery. |
# | **`is_back_to_back`** | `F.when(F.col("days_rest") == 1, 1.0).otherwise(0.0)` | Flags instances where a team has to play two games in two nights, a critical factor for predicting performance exhaustion. |
# | **`roll_10_avg_goals`** | `F.avg("goals").over(rolling_10_window)` | Captures recent offensive form by averaging goals scored across the team's last 10 completed matches. |
# | **`roll_10_avg_shots`** | `F.avg("shots").over(rolling_10_window)` | Evaluates sustained offensive zone presence and shooting volume trends. |
# | **`roll_10_win_rate`** | `F.avg("won").over(rolling_10_window)` | Quantifies a team's recent momentum and success rate entering the matchup. |
# | **`target_over_5_5`** | `F.when(F.col("total_goals") > 5.5, 1.0).otherwise(0.0)` | A binary target classification column indicating if the combined game goals exceeded a standard Vegas **5.5 line**. |
# | **`target_over_6_5`** | `F.when(F.col("total_goals") > 6.5, 1.0).otherwise(0.0)` | A binary target classification column indicating if the combined game goals exceeded a high-scoring Vegas **6.5 line**. |
# | **`target_puckline_home_plus1_5`** | `F.when((F.col("home_goals") + 1.5) > F.col("away_goals"), 1.0).otherwise(0.0)` | Determines whether the home team covered a **+1.5 puckline spread** (i.e., they won the game outright or lost by exactly 1 goal). |
# | **`target_moneyline`** | `F.col("home_won")` | Simply maps the home team's victory outcome as the baseline prediction goal for standard **moneyline betting** models. |
# | **`shots_diff`** | `F.col("home_roll_shots") - F.col("away_roll_shots")` | Computes the matchup delta for shot volume. Positive values indicate a home-team possession/shooting advantage. |
# | **`win_rate_diff`** | `F.col("home_roll_win_rate") - F.col("away_roll_win_rate")` | Establishes the relative form variance between the two competitors entering the ice. |
# | **`rest_diff`** | `F.col("away_is_b2b") - F.col("home_is_b2b")` | isolates fatigue imbalances. Returns `1.0` if only the away team is on a back-to-back, `-1.0` if only the home team is, and `0.0` if rest conditions match. |


# MARKDOWN ********************

# ### Imports

# CELL ********************

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window, WindowSpec

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

# ### Define variables

# CELL ********************

GAME_TABLE: str = "silver.game"
GAME_TEAMS_STATS_TABLE: str = "silver.game_teams_stats"
TEAM_INFO_TABLE: str = "silver.team_info"

GAME_FEATURES_TABLE: str = "silver.game_features"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Load required silver tables

# CELL ********************

game_teams_stats: DataFrame = load_table(spark, GAME_TEAMS_STATS_TABLE)
team_info: DataFrame = load_table(spark, TEAM_INFO_TABLE)

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

# ### Compute rolling 10-game matchup features

# CELL ********************

team_game_stats: DataFrame = (
    game_teams_stats.alias("st")
    .join(
        game_r.alias("g").select(
            "game_id",
            "date_time_GMT",
            "season",
            "away_goals",
            "home_goals",
        ),
        on="game_id",
        how="inner",
    )
    .select(
        F.col("game_id"),
        F.col("season"),
        F.col("g.date_time_GMT").alias("date_time"),
        F.col("st.team_id"),
        F.col("st.HoA"),
        F.col("st.won").cast("double").alias("won"),
        F.col("st.goals"),
        F.col("st.shots"),
        F.col("st.pim"),
        F.col("g.away_goals"),
        F.col("g.home_goals"),
    )
)

# Window partition ordered chronologically per team
team_time_window: WindowSpec = (
    Window
    .partitionBy("team_id")
    .orderBy("date_time")
)

# Strict rolling window of previous 10 games
# (-10 to -1 excludes the current game and prevents data leakage)
rolling_10_window: WindowSpec = (
    team_time_window
    .rowsBetween(-10, -1)
)

team_features: DataFrame = (
    team_game_stats
    .withColumn(
        "prev_game_date",
        F.lag("date_time", 1).over(team_time_window),
    )
    .withColumn(
        "days_rest",
        F.datediff(
            F.to_date(F.col("date_time")),
            F.to_date(F.col("prev_game_date")),
        ),
    )
    .withColumn(
        "is_back_to_back",
        F.when(F.col("days_rest") == 1, 1.0).otherwise(0.0),
    )
    .withColumn(
        "roll_10_avg_goals",
        F.avg("goals").over(rolling_10_window),
    )
    .withColumn(
        "roll_10_avg_shots",
        F.avg("shots").over(rolling_10_window),
    )
    .withColumn(
        "roll_10_win_rate",
        F.avg("won").over(rolling_10_window),
    )
    .fillna(0.0)
)

# Home-team perspective
home_team: DataFrame = (
    team_features
    .filter(F.col("HoA") == "home")
    .select(
        F.col("game_id"),
        F.col("season"),
        F.col("date_time"),
        F.col("team_id").alias("home_team_id"),
        F.col("home_goals"),
        F.col("away_goals"),
        F.col("won").alias("home_won"),
        F.col("roll_10_avg_goals").alias("home_roll_goals"),
        F.col("roll_10_avg_shots").alias("home_roll_shots"),
        F.col("roll_10_win_rate").alias("home_roll_win_rate"),
        F.col("is_back_to_back").alias("home_is_b2b"),
    )
)

# Away-team perspective
away_team: DataFrame = (
    team_features
    .filter(F.col("HoA") == "away")
    .select(
        F.col("game_id"),
        F.col("team_id").alias("away_team_id"),
        F.col("roll_10_avg_goals").alias("away_roll_goals"),
        F.col("roll_10_avg_shots").alias("away_roll_shots"),
        F.col("roll_10_win_rate").alias("away_roll_win_rate"),
        F.col("is_back_to_back").alias("away_is_b2b"),
    )
)

# Matchup-level feature set
game_features: DataFrame = (
    home_team
    .join(
        away_team,
        on="game_id",
        how="inner",
    )
    .withColumn(
        "total_goals",
        F.col("home_goals") + F.col("away_goals"),
    )
    .withColumn(
        "target_over_5_5",
        F.when(F.col("total_goals") > 5.5, 1.0).otherwise(0.0),
    )
    .withColumn(
        "target_over_6_5",
        F.when(F.col("total_goals") > 6.5, 1.0).otherwise(0.0),
    )
    .withColumn(
        "target_puckline_home_plus1_5",
        F.when(
            (F.col("home_goals") + 1.5) > F.col("away_goals"),
            1.0,
        ).otherwise(0.0),
    )
    .withColumn(
        "target_moneyline",
        F.col("home_won"),
    )
    .withColumn(
        "shots_diff",
        F.col("home_roll_shots")
        - F.col("away_roll_shots"),
    )
    .withColumn(
        "win_rate_diff",
        F.col("home_roll_win_rate")
        - F.col("away_roll_win_rate"),
    )
    .withColumn(
        "rest_diff",
        F.col("away_is_b2b")
        - F.col("home_is_b2b"),
    )
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Write silver table

# CELL ********************

write_table(game_features, GAME_FEATURES_TABLE)
print("Wrote silver table ", GAME_FEATURES_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
