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

# ## nb_04_ml_trg_functions
# 
# This notebook contains functions used to generate dataframes used for ML training
# for betting predictions
# 
# - with_features
# - train_score_and_optimize
# - american_to_decimal
# - american_to_implied_prob
# - compute_ev_pct
# - derive_outcome
# - generate_season_summary
# - get_top_3_teams

# MARKDOWN ********************

# ### Imports

# CELL ********************

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column
from pyspark.sql.window import Window, WindowSpec
from pyspark.ml.classification import LogisticRegression

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: Feature Engineering 
# 
# The `with_features()` function enriches game-level records with team performance metrics and creates machine learning features and target labels for model training.
# 
# #### Data Preparation
# 
# 1. Create home-team statistics by prefixing all columns from `composite` with `home_`.
# 2. Create away-team statistics by prefixing all columns from `composite` with `away_`.
# 3. Join the home and away statistics to each game using `home_team_id` and `away_team_id`.
# 
# #### Engineered Features
# 
# | Feature | Description |
# |----------|-------------|
# | `goal_diff_gap` | Difference in average goal differential between home and away teams, including a small home-ice advantage adjustment (+0.15). |
# | `pp_pct_gap` | Home power-play percentage minus away penalty-kill percentage. |
# | `pk_pct_gap` | Home penalty-kill percentage minus away power-play percentage. |
# | `corsi_gap` | Difference in average Corsi For between home and away teams. |
# | `save_pct_gap` | Difference in save percentage proxy between home and away teams. |
# 
# #### Derived Metrics
# 
# | Column | Description |
# |----------|-------------|
# | `home_margin` | Goal differential for the home team (`home_goals - away_goals`). |
# | `total_goals` | Combined goals scored by both teams. |
# 
# #### Target Labels
# 
# | Target | Definition |
# |----------|-------------|
# | `home_win` | `1` if the home team wins, otherwise `0`. |
# | `home_covers_plus1_5` | `1` if the home team covers a +1.5 puck line (`margin >= -1`), otherwise `0`. |
# | `over` | `1` if total goals exceed 5.5, otherwise `0`. |


# CELL ********************

def with_features(game: DataFrame, composite: DataFrame) -> DataFrame:
    """
    Enrich game records with home and away team aggregate statistics and
    generate machine learning features and target variables.

    Features compare home and away team performance metrics, while targets
    represent common betting and game outcome predictions.

    Args:
        game: Game-level dataset containing team identifiers and final scores.
        composite: Team-level aggregate statistics keyed by team_id.

    Returns:
        DataFrame containing engineered feature columns and target labels.
    """
    home_stats = composite.select(
        [F.col(c).alias(f"home_{c}") for c in composite.columns]
    )

    away_stats = composite.select(
        [F.col(c).alias(f"away_{c}") for c in composite.columns]
    )

    return (
        game
        .join(
            home_stats,
            game.home_team_id == home_stats.home_team_id,
            "inner",
        )
        .drop(home_stats["home_team_id"])
        .join(
            away_stats,
            game.away_team_id == away_stats.away_team_id,
            "inner",
        )
        .drop(away_stats["away_team_id"])
        .withColumn(
            "goal_diff_gap",
            (F.col("home_goal_diff_per_gm") + F.lit(0.15))
            - F.col("away_goal_diff_per_gm"),
        )
        .withColumn(
            "pp_pct_gap",
            F.col("home_pp_pct") - F.col("away_pk_pct"),
        )
        .withColumn(
            "pk_pct_gap",
            F.col("home_pk_pct") - F.col("away_pp_pct"),
        )
        .withColumn(
            "corsi_gap",
            F.col("home_corsi_for_per_gm")
            - F.col("away_corsi_for_per_gm"),
        )
        .withColumn(
            "save_pct_gap",
            F.col("home_save_pct_proxy")
            - F.col("away_save_pct_proxy"),
        )
        .withColumn(
            "home_margin",
            F.col("home_goals") - F.col("away_goals"),
        )
        .withColumn(
            "total_goals",
            F.col("home_goals") + F.col("away_goals"),
        )
        .withColumn(
            "home_covers_plus1_5",
            (F.col("home_margin") >= -1).cast("int"),
        )
        .withColumn(
            "home_win",
            (F.col("home_margin") > 0).cast("int"),
        )
        .withColumn(
            "over",
            (F.col("total_goals") > 5.5).cast("int"),
        )
    )


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: Model Training and Dynamic Threshold Optimization
# 
# A logistic regression model is trained for a specified target outcome and evaluated against a holdout dataset. Predictions are scored using model probabilities, and a confidence-based threshold search is performed to identify high-quality betting opportunities.
# 
# ##### Model Training
# 
# - Train a Spark ML `LogisticRegression` model using the assembled `features` vector.
# - Use the supplied `label_col` as the prediction target.
# - Generate predictions on the holdout dataset.
# 
# ##### Prediction Scoring
# 
# For each prediction, the following evaluation fields are calculated:
# 
# | Column | Description |
# |----------|-------------|
# | `prob_positive` | Predicted probability of the positive class. |
# | `predicted_label` | Model prediction converted to an integer. |
# | `actual_label` | Actual game outcome for the target label. |
# | `confidence` | Absolute distance between prediction probability and 0.5. |
# | `hit` | Indicates whether the prediction matches the actual outcome. |
# 
# ##### Dynamic Confidence Threshold Search
# 
# A series of confidence thresholds are evaluated:
# 
# ```python
# [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
# ```
# 
# For each threshold:
# 
# - Predictions below the confidence threshold are excluded.
# - At least 30 qualifying predictions are required.
# - Hit rate is calculated on the remaining selections.
# - The selected threshold is the lowest threshold that achieves or exceeds the target win rate while preserving betting volume.
# 
# ##### Fallback Logic
# 
# If no threshold achieves the target win rate:
# 
# - A default confidence threshold of `0.08` is used.
# - Hit rate is recalculated using the fallback threshold.
# 
# ##### Outputs
# 
# The function returns:
# 
# 1. `scored_df` containing:
#    - Prediction probabilities
#    - Predicted outcomes
#    - Actual outcomes
#    - Confidence scores
#    - Hit indicators
# 
# 2. `best_thresh`
#    - The selected confidence threshold for downstream filtering and betting strategy evaluation.


# CELL ********************

def train_score_and_optimize(
    label_col: str,
    train_feat: DataFrame,
    holdout_feat: DataFrame,
    target_win_rate: float = 0.65,
) -> tuple[DataFrame, float]:
    """
    Train a logistic regression model, score holdout games, and determine
    an optimal confidence threshold that meets a target win rate while
    maintaining a reasonable prediction volume.

    Args:
        label_col: Name of the target label column.
        train_feat: Training dataset containing assembled feature vectors.
        holdout_feat: Holdout dataset used for model evaluation.
        target_win_rate: Desired minimum hit rate for confidence-based
            selections.

    Returns:
        A tuple containing:
            - Scored DataFrame with predictions and confidence metrics.
            - Selected confidence threshold.
    """
    lr = LogisticRegression(
        featuresCol="features",
        labelCol=label_col,
        maxIter=100,
    )

    model = lr.fit(train_feat)
    preds = model.transform(holdout_feat)

    scored_df = (
        preds
        .withColumn(
            "prob_positive",
            vector_to_array("probability")[1],
        )
        .withColumn(
            "predicted_label",
            F.col("prediction").cast("int"),
        )
        .withColumn(
            "actual_label",
            F.col(label_col).cast("int"),
        )
        .withColumn(
            "confidence",
            F.abs(F.col("prob_positive") - 0.5),
        )
        .withColumn(
            "hit",
            F.when(
                F.col("predicted_label") == F.col("actual_label"),
                1,
            ).otherwise(0),
        )
        .select(
            "game_id",
            "season",
            "home_team_id",
            "away_team_id",
            "predicted_label",
            "actual_label",
            "prob_positive",
            "confidence",
            "hit",
        )
    )

    # Dynamic Search: Search thresholds to land closest to target accuracy
    # while preserving volume.
    best_thresh = 0.10
    best_hit_rate = 0.0

    for thresh in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        sel = scored_df.filter(F.col("confidence") >= thresh)
        n = sel.count()

        if n < 30:
            continue

        hit_rate = (sel.filter(F.col("hit") == 1).count() / n)

        if (
            hit_rate >= target_win_rate
            and (
                best_hit_rate == 0.0
                or hit_rate < best_hit_rate
            )
        ):
            best_thresh = thresh
            best_hit_rate = hit_rate

    # Fallback to reasonable threshold if model probability variance is tight.
    if best_hit_rate == 0.0:
        best_thresh = 0.08

        sel = scored_df.filter(F.col("confidence") >= best_thresh,)

        best_hit_rate = (
            sel.filter(F.col("hit") == 1).count()
            / max(sel.count(), 1)
        )

    print(
        f"[{label_col}] Selected Confidence Threshold: "
        f">= {best_thresh:.2f} "
        f"(Achieved Win Rate: {best_hit_rate:.1%})"
    )

    return scored_df, best_thresh

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: convert american odds to decimal odds

# CELL ********************

def american_to_decimal(col_name: str) -> Column:
    """
    Convert American odds to decimal odds.

    Args:
        col_name: Name of the column containing American odds.

    Returns:
        Spark Column containing decimal odds.
    """
    return (
        F.when(
            F.col(col_name) < 0,
            1 + (100 / F.abs(F.col(col_name))),
        )
        .otherwise(
            1 + (F.col(col_name) / 100),
        )
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: convert American odds to implied win probability

# CELL ********************

def american_to_implied_prob(col_name: str) -> Column:
    """
    Convert American odds to implied win probability.

    Args:
        col_name: Name of the column containing American odds.

    Returns:
        Spark Column containing implied probability values.
    """
    return (
        F.when(
            F.col(col_name) < 0,
            F.abs(F.col(col_name))
            / (F.abs(F.col(col_name)) + 100),
        )
        .otherwise(
            100 / (F.col(col_name) + 100),
        )
    )


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: calculate EV percentage

# CELL ********************

def compute_ev_pct(prob_col: str, odds_col: str) -> Column:
    """
    Calculate expected value (EV) as a percentage using model probability
    and bookmaker odds.

    Args:
        prob_col: Name of the column containing model win probability.
        odds_col: Name of the column containing American odds.

    Returns:
        Spark Column containing EV percentages rounded to two decimals.
    """
    net_payout = american_to_decimal(odds_col) - 1
    raw_prob = F.col(prob_col)

    win_prob = (
        F.when(raw_prob > 1.0, raw_prob / 100.0)
        .otherwise(raw_prob)
    )

    return F.round(
        (
            (win_prob * net_payout)
            - (1.0 - win_prob)
        )
        * 100,
        2,
    )


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: derive betting out from pedicition accuracy

# CELL ********************

def derive_outcome() -> Column:
    """
    Derive the betting outcome from prediction accuracy.

    Returns:
        Spark Column containing either 'WIN' or 'LOSS'.
    """
    return (
        F.when(F.col("hit") == 1, "WIN")
        .when(F.col("hit") == 0, "LOSS")
        .when(
            F.col("predicted_label")
            == F.col("actual_label"),
            "WIN",
        )
        .otherwise("LOSS")
    )


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: Season Performance Summary Generation
# 
# Season-level performance summaries are generated for each betting market using the dynamically selected confidence thresholds. The objective is to evaluate prediction quality, betting volume, and the impact of skipping lower-confidence selections.
# 
# ##### Summary Metrics
# 
# For each betting category, predictions are separated into:
# 
# - **Qualified Bets**: Predictions meeting or exceeding the confidence threshold.
# - **Skipped Games**: Predictions below the confidence threshold.
# - **Correct Predictions**: Qualified bets that produced winning outcomes.
# - **Wrong Predictions**: Qualified bets that produced losing outcomes.
# 
# ##### Confidence-Based Filtering
# 
# A prediction is considered actionable when:
# 
# ```python
# confidence >= threshold
# ```
# 
# Predictions below the selected threshold are treated as:
# 
# ```text
# No Bet
# ```
# 
# This allows the model to sacrifice volume in exchange for higher prediction accuracy.
# 
# ##### Calculated Statistics
# 
# The summary includes the following metrics:
# 
# | Metric | Description |
# |----------|-------------|
# | `Total_Season_Games` | Total number of games evaluated. |
# | `Qualified_Bets_Offered` | Predictions meeting the confidence threshold. |
# | `Correct_Predictions_WIN` | Correct high-confidence predictions. |
# | `Wrong_Predictions_LOSS` | Incorrect high-confidence predictions. |
# | `No_Bets_Low_Confidence` | Games excluded due to insufficient confidence. |
# | `Selective_Accuracy_Pct` | Accuracy of qualified bets only. |
# | `Skipped_Games_Pct` | Percentage of games excluded from betting. |
# 


# CELL ********************

def generate_season_summary(
    df: DataFrame,
    category_name: str,
    thresh_val: float,
) -> tuple[str, int, int, int, int, int, str, str]:
    """
    Generate a season-level summary of prediction performance using a
    confidence threshold.

    Calculates the number of eligible bets, prediction accuracy, skipped
    games, and outcome counts for a given betting category.

    Args:
        df: Prediction dataset containing confidence scores and outcomes.
        category_name: Name of the betting category being summarized.
        thresh_val: Confidence threshold used to qualify predictions.

    Returns:
        Tuple containing season summary metrics:
            - Category name
            - Total games
            - Qualified bets offered
            - Correct predictions
            - Wrong predictions
            - Skipped low-confidence games
            - Selective accuracy percentage
            - Skipped games percentage
    """
    total_games = df.count()

    high_conf_df = df.filter(
        F.col("confidence") >= thresh_val,
    )

    low_conf_count = (
        df.filter(F.col("confidence") < thresh_val)
        .count()
    )

    correct_picks = (
        high_conf_df.filter(
            (F.col("hit") == 1)
            | (
                F.col("predicted_label")
                == F.col("actual_label")
            )
        )
        .count()
    )

    selective_total = high_conf_df.count()

    wrong_picks = selective_total - correct_picks

    selective_acc = (
        correct_picks / selective_total * 100
        if selective_total > 0
        else 0.0
    )

    no_bet_pct = (
        low_conf_count / total_games * 100
        if total_games > 0
        else 0.0
    )

    return (
        category_name,
        total_games,
        selective_total,
        correct_picks,
        wrong_picks,
        low_conf_count,
        f"{selective_acc:.2f}%",
        f"{no_bet_pct:.2f}%",
    )



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: Top Team Ranking by Betting Market
# 
# The highest-performing teams are identified for each betting market based on qualified betting opportunities and historical win rates.
# 
# ##### Qualification Criteria
# 
# Only bets meeting both of the following conditions are included:
# 
# | Requirement | Condition |
# |-------------|------------|
# | Confidence Threshold | `confidence >= market threshold` |
# | Positive Expected Value | `ev_pct >= 2.0` |
# 
# This ensures rankings are based on higher-conviction, positive-EV betting opportunities.
# 
# ##### Team Performance Aggregation
# 
# Qualified bets are analyzed separately for:
# 
# - Home team appearances
# - Away team appearances
# 
# For each team, the following metrics are calculated:
# 
# | Metric | Description |
# |----------|-------------|
# | `bets` | Number of qualified betting opportunities. |
# | `wins` | Number of winning bets. |
# 
# Home and away statistics are then combined into a single team-level summary.
# 
# ##### Team Eligibility Filter
# 
# To eliminate small sample sizes, teams must have:
# 
# ```text
# Minimum 5 qualified bets
# ```
# 
# Teams with fewer than five qualifying opportunities are excluded from ranking.
# 
# ##### Win Rate Calculation
# 
# Each team's betting performance is measured using:
# 
# ```text
# Win Rate (%) = Total Wins ÷ Total Bets × 100
# ```
# 
# This metric reflects the historical success rate of betting opportunities involving that team.
# 
# ##### Team Ranking Logic
# 
# Teams are ranked using:
# 
# 1. Highest win rate.
# 2. Highest total wins (used as a tiebreaker).
# 
# Ranking is implemented using a Spark window function and only the top three teams are retained.


# CELL ********************

def get_top_3_teams(
    df: DataFrame,
    market_name: str,
    thresh_val: float,
) -> DataFrame:
    """
    Rank the top three teams for a betting market based on win rate from
    qualified betting opportunities.

    Only bets meeting the confidence threshold and minimum EV requirement
    are considered. Team performance is aggregated across both home and
    away appearances, and teams with fewer than five qualified bets are
    excluded.

    Args:
        df: Enriched betting dataset containing confidence scores,
            expected value percentages, and betting outcomes.
        market_name: Name of the betting market category.
        thresh_val: Confidence threshold used to qualify bets.

    Returns:
        DataFrame containing the top three ranked teams for the specified
        betting market, including total bets, wins, win rate, and rank.
    """
    q_df = df.filter(
        (F.col("confidence") >= thresh_val)
        & (F.col("ev_pct") >= 2.0)
    )

    home_stats = (
        q_df
        .groupBy("home_team")
        .agg(
            F.count("game_id").alias("bets"),
            F.sum(
                F.when(
                    F.col("bet_outcome") == "WIN",
                    1,
                ).otherwise(0)
            ).alias("wins"),
        )
        .withColumnRenamed(
            "home_team",
            "team",
        )
    )

    away_stats = (
        q_df
        .groupBy("away_team")
        .agg(
            F.count("game_id").alias("bets"),
            F.sum(
                F.when(
                    F.col("bet_outcome") == "WIN",
                    1,
                ).otherwise(0)
            ).alias("wins"),
        )
        .withColumnRenamed(
            "away_team",
            "team",
        )
    )

    combined = (
        home_stats
        .union(away_stats)
        .groupBy("team")
        .agg(
            F.sum("bets").alias("total_bets"),
            F.sum("wins").alias("total_wins"),
        )
        .filter(F.col("total_bets") >= 5)
        .withColumn(
            "win_rate",
            F.round(
                (
                    F.col("total_wins")
                    / F.col("total_bets")
                )
                * 100,
                2,
            ),
        )
    )

    team_window = Window.orderBy(
        F.col("win_rate").desc(),
        F.col("total_wins").desc(),
    )

    return (
        combined
        .withColumn(
            "rank",
            F.row_number().over(team_window),
        )
        .filter(F.col("rank") <= 3)
        .withColumn(
            "category",
            F.lit(market_name),
        )
    )


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
