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

# ## nb_04_teams_players_forecast_gold
# 
# This notebook serves as the **analytical orchestration engine** that aggregates granular hockey historical profiles into season-level summaries and engineers recency-weighted projection models. It evaluates team and skater trajectories using customizable decay weights, executes an iterative multi-year backtesting validation sequence, and processes out-of-sample forward projections using mathematical normalization. The finalized data assets are written as **production-ready tables in the Gold analytical layer** for semantic integration with Power BI models.
# 
# ### Workflow Overview
# 1. **Core Data Loading & Verification:** Imports structural PySpark SQL primitives and loads silver-tier transaction tables across dimensions (`dim_team`), factual box scores (`fact_team_game`, `fact_skater_game`, `fact_goalie_game`), and profile registries (`player_info`).
# 2. **Season-Level Aggregation Rollups:** Summarizes per-game events into single-row regular season metrics (`type == "R"`) for teams, skaters, and goalies. It explicitly resolves multi-team roster changes (player trades) by isolating the primary franchise via highest-game window partitions (`F.row_number()`).
# 3. **Recency-Weighted Predictive Engineering:** Encapsulates forecasting workflows into modular functions (`project_champion_asof` and `project_top_scorer_asof`). The execution logic subsets historical timelines up to an index cutoff year and calculates an asymmetric weight distribution (`3/2/1`) across the three most recent active seasons.
# 4. **Historical Validation Backtest Loop:** Executes a systematic historical validation pass across a decade-long simulation loop (seasons 2009–2018). It unions successive matrix iterations using functools reductions to establish model calibration metrics and calculate prediction top-5 hit rates.
# 5. **Softmax Betting Odds Generation (2020-21):** Computes definitive forward projections across the unobserved 2020-21 season. To structure raw point percentages into standard betting probability fields, it applies a calibrated **Softmax exponent scaling factor (\(k=15\))** across all competing franchises.
# 6. **Production Delta Provisioning & Catalog Alignment:** Persists output frames into specific schema-qualified Delta boundaries within the Lakehouse. This migration from legacy un-scoped nomenclature requires data connection updates inside the companion semantic model file (`Betting Final Report.pbix`).
# 
# ---
# 
# ### Aggregation Rollups & Target Formulations
# 
# The analytical features and forecasting metrics are generated across distinct steps within the pipeline execution:
# 
# | Target Table / Metric | Calculation Logic (PySpark Syntax) | Analytical Purpose / Rationale |
# | :--- | :--- | :--- |
# | **`team_season`** <br>*(Points Percentage)* | `safe_div(F.col("points"), 2.0 * F.col("games_played"))` | Normalizes team standings performance regardless of localized regular-season schedule disruptions or games played count. |
# | **`team_season`** <br>*(Special Teams %)* | **PP %:** `safe_div(pp_goals, pp_opportunities) * 100`<br>**PK %:** `100 - (safe_div(opp_pp_goals, opp_pp_opportunities) * 100)` | Calculates special teams conversion performance while leveraging zero-denominator validation arrays to eliminate division faults. |
# | **`primary_team_id`** <br>*(Skater/Goalie Trades)* | `F.row_number().over(Window.partitionBy("player_id", "season").orderBy(F.col("g").F.desc())) == 1` | Dedupes skater profiles down to exactly one record per season. It attributes a traded player to the specific team they registered the most games (`g`) with that year. |
# | **`skater_season`** <br>*(Points Per 60)* | `safe_div(F.col("points") * 3600.0, F.col("total_toi_seconds"))` | Standardizes skater scoring efficiency across a uniform 60 minutes of playing time, evening out variances caused by differing line deployments or ice time constraints. |
# | **`proj_points_pct`** <br>*(Recency Forecasting)* | `F.sum(F.col("points_pct") * F.col("wt")) / F.sum("wt")` <br><br>*where `wt` is defined as `4 - row_number`* | Establishes a weighted baseline projection. It weights the most recent season at \(3\times\), the prior at \(2\times\), and the earliest qualifying record at \(1\times\), requiring at least 2 active seasons. |
# | **`championship_probability`** <br>*(Softmax Transformed Odds)* | `F.exp(F.col("proj_points_pct") * 15.0) / F.lit(total_score)` | Converts linear projection margins into normalized probability vectors. The tuning parameter (\(k=15\)) accentuates top-tier outliers to emulate realistic betting markets. |
# | **`actual_2020_21_results`** <br>*(Ground Truth Audit)* | `spark.createDataFrame([...], ["category", "winner_name", "team", "detail"])` | Hardcodes authoritative historical real-world data points (such as McDavid's 105-point MVP run) to grade mathematical model validation error out-of-sample. |
# 


# MARKDOWN ********************

# ### Imports
# - importing sum as _sum to avoid using python internal sum function

# CELL ********************

from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql.window import Window, WindowSpec
from functools import reduce

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Load common db functions

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

# Define silver tables to use
DIM_TEAM_TABLE: str = "silver.dim_team"
FACT_TEAM_GAME_TABLE: str = "silver.fact_team_game"
FACT_TEAM_CHAMPION_TABLE: str = "silver.fact_team_champion"
FACT_SKATER_GAME_TABLE: str = "silver.fact_skater_game"
FACT_GOALIE_GAME_TABLE: str = "silver.fact_goalie_game"
PLAYER_INFO_TABLE: str = "silver.player_info"

# Everything this notebook produces goes in the gold schema. 
TEAM_SEASON_TABLE: str = "gold.team_season"
SKATER_SEASON_TABLE: str = "gold.skater_season"
GOALIE_SEASON_TABLE: str = "gold.goalie_season"
CHAMPION_VALIDATION_HISTORY_TABLE: str = "gold.champion_validation_history"
SKATER_VALIDATION_HISTORY_TABLE: str = "gold.skater_validation_history"
TEAM_PROJECTION_2021_RAW_TABLE: str = "gold.team_projection_2021_raw"
CHAMPIONSHIP_ODDS_2021_TABLE: str = "gold.championship_odds_2021"
BEST_PLAYER_FORECAST_2021_TABLE: str = "gold.best_player_forecast_2021"
ACTUAL_2020_21_RESULTS_TABLE: str = "gold.actual_2020_21_results"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Define custom functions

# CELL ********************

def safe_div(numer, denom):
    """Division that returns null instead of erroring/inf when denom is 0."""
    return F.when(denom == 0, F.lit(None)).otherwise(numer / denom)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Load required silver tables

# CELL ********************

dim_team: DataFrame = load_table(spark, DIM_TEAM_TABLE)
fact_team_game: DataFrame = load_table(spark, FACT_TEAM_GAME_TABLE)
fact_team_champion: DataFrame = load_table(spark, FACT_TEAM_CHAMPION_TABLE)
fact_skater_game: DataFrame = load_table(spark, FACT_SKATER_GAME_TABLE)
fact_goalie_game: DataFrame = load_table(spark, FACT_GOALIE_GAME_TABLE)
player_info: DataFrame = load_table(spark, PLAYER_INFO_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Build and write gold_team_season
# Regular season only. Rolls fact_team_game up to one row per team per season.

# CELL ********************

#fact_team_game_regular: DataFrame = fact_team_game.filter(F.col("type") == "R")

team_season_agg: DataFrame = (
    fact_team_game
    .filter(F.col("type") == "R")
    .groupBy("team_id", "season", "season_start_year")
    .agg(
        F.count("*").alias("games_played"),
        F.sum("won").alias("wins"),
        F.sum(
            F.when((F.col("won") == 0) & (F.col("settled_in") == "REG"), 1)
            .otherwise(0)
        ).alias("reg_losses"),
        F.sum(
            F.when((F.col("won") == 0) & (F.col("settled_in") == "OT"), 1)
            .otherwise(0)
        ).alias("ot_so_losses_or_ties"),
        F.sum("points").alias("points"),
        F.sum("goals").alias("goals_for"),
        F.sum("opp_goals").alias("goals_against"),
        F.sum("shots").alias("shots_for"),
        F.sum("opp_shots").alias("shots_against"),
        F.sum("powerPlayGoals").alias("pp_goals"),
        F.sum("powerPlayOpportunities").alias("pp_opportunities"),
        F.sum("opp_powerPlayGoals").alias("opp_pp_goals"),
        F.sum("opp_powerPlayOpportunities").alias("opp_pp_opportunities"),
        F.avg("faceOffWinPercentage").alias("avg_faceoff_pct"),
        F.sum(
            F.col("giveaways") - F.col("takeaways")
        ).alias("give_take_diff_total"),
    )
)

team_season_agg: DataFrame = (
    team_season_agg
    .withColumn("points_pct", safe_div(F.col("points"), 2.0 * F.col("games_played")))
    .withColumn("goal_diff", F.col("goals_for") - F.col("goals_against"))
    .withColumn("goals_for_per_game", safe_div(F.col("goals_for"), F.col("games_played")))
    .withColumn("goals_against_per_game", safe_div(F.col("goals_against"), F.col("games_played")))
    .withColumn("shots_for_per_game", safe_div(F.col("shots_for"), F.col("games_played")))
    .withColumn("shots_against_per_game", safe_div(F.col("shots_against"), F.col("games_played")))
    .withColumn("pp_pct", safe_div(F.col("pp_goals"), F.col("pp_opportunities")) * 100)
    .withColumn("pk_pct", F.lit(100) - safe_div(F.col("opp_pp_goals"), F.col("opp_pp_opportunities")) * 100)
    .withColumn("giveaway_takeaway_diff_per_game", safe_div(F.col("give_take_diff_total"), F.col("games_played")))
)

team_season: DataFrame = (
    team_season_agg
    .join(dim_team, on="team_id", how="left")
    .join(fact_team_champion.select("season", "champion_team_id"), on="season", how="left")
    .withColumn("is_champion", F.when(F.col("champion_team_id") == F.col("team_id"), 1).otherwise(0))
)

write_table(team_season, TEAM_SEASON_TABLE)
display(team_season.orderBy(F.desc("season_start_year"), F.desc("points_pct")).limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Build and write gold_skater_season
# primary_team_id = whichever team a player suited up for most that season (handles
# trades without duplicate/ambiguous rows) via a rank window instead of a correlated
# subquery.

# CELL ********************

team_game_counts: DataFrame = (
    fact_skater_game
    .filter(F.col("type") == "R")
    .groupBy("player_id", "season", "team_id")
    .agg(F.count("*").alias("g"))
)

w_primary: WindowSpec = Window.partitionBy("player_id", "season").orderBy(F.col("g").desc())
primary_team: DataFrame = (
    team_game_counts
    .withColumn("rn", F.row_number().over(w_primary))
    .filter(F.col("rn") == 1)
    .select(
        "player_id", 
        "season", 
        F.col("team_id").alias("primary_team_id")
    )
)

skater_agg: DataFrame = (
    fact_skater_game
    .filter(F.col("type") == "R")
    .groupBy("player_id", "season", "season_start_year")
    .agg(
        F.count("*").alias("games_played"),
        F.sum("goals").alias("goals"),
        F.sum("assists").alias("assists"),
        F.sum("points").alias("points"),
        F.sum("shots").alias("shots"),
        F.sum("hits").alias("hits"),
        F.sum("penaltyMinutes").alias("pim"),
        F.sum("plusMinus").alias("plus_minus"),
        F.sum("powerPlayGoals").alias("pp_goals"),
        F.sum("powerPlayAssists").alias("pp_assists"),
        F.sum("timeOnIce").alias("total_toi_seconds"),
    )
)

skater_season: DataFrame = (
    skater_agg
    .join(primary_team, on=["player_id", "season"], how="left")
    .join(player_info, on="player_id", how="left")
    .withColumn("approx_age", F.col("season_start_year") - F.year("birthDate"))
    .withColumn("points_per_game", safe_div(F.col("points"), F.col("games_played")))
    .withColumn("avg_toi_minutes_per_game", safe_div(F.col("total_toi_seconds"), F.col("games_played")) / 60.0)
    .withColumn("points_per_60", safe_div(F.col("points") * 3600.0, F.col("total_toi_seconds")))
)

write_table(skater_season, SKATER_SEASON_TABLE)
display(skater_season.orderBy(F.desc("season_start_year"), F.desc("points")).limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Build gold_goalie_season
# Not used in the "best player" forecast below, which follows the original scope and
# looks at skaters only — but it's here if the storyline extends to a Vezina-style
# goalie projection later.

# CELL ********************

goalie_team_counts: DataFrame = (
    fact_goalie_game
    .filter(F.col("type") == "R")
    .groupBy("player_id", "season", "team_id")
    .agg(F.count("*").alias("g"))
)

w_primary_g: WindowSpec = Window.partitionBy("player_id", "season").orderBy(F.col("g").desc())
primary_team_g: DataFrame = (
    goalie_team_counts
    .withColumn("rn", F.row_number().over(w_primary_g))
    .filter(F.col("rn") == 1)
    .select(
        "player_id", 
        "season", 
        F.col("team_id").alias("primary_team_id")
    )
)

goalie_agg: DataFrame = (
    fact_goalie_game
    .filter(F.col("type") == "R")
    .groupBy("player_id", "season", "season_start_year")
    .agg(
        F.count("*").alias("games_played"),
        F.sum(F.when(F.col("decision") == "W", 1).otherwise(0)).alias("wins"),
        F.sum(F.when(F.col("decision") == "L", 1).otherwise(0)).alias("losses"),
        F.sum("shots").alias("shots_against"),
        F.sum("saves").alias("saves"),
        F.sum("goals_against").alias("goals_against"),
        F.sum("timeOnIce").alias("total_toi_seconds"),
    )
)

goalie_season: DataFrame = (
    goalie_agg
    .join(primary_team_g, on=["player_id", "season"], how="left")
    .join(player_info, on="player_id", how="left")
    .withColumn("approx_age", F.col("season_start_year") - F.year("birthDate"))
    .withColumn("save_pct", safe_div(F.col("saves"), F.col("shots_against")))
    .withColumn("goals_against_avg_per60", safe_div(F.col("goals_against") * 3600.0, F.col("total_toi_seconds")))
    .withColumn("total_toi_hours", F.col("total_toi_seconds") / 3600.0)
)

write_table(goalie_season, GOALIE_SEASON_TABLE)
display(goalie_season.orderBy(F.desc("season_start_year"), F.desc("wins")).limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Forecasting functions
# Both functions take `as_of_year` and answer: *using only seasons up to and including
# this one, who do we project for the season right after it?* That's what lets the same
# code both backtest against seasons we already know the answer to, and produce the
# real, unknown 2020-21 forecast, just by passing a different year.
# 
# Method: recency-weighted average (weights 3/2/1, most recent first) over each
# team's/player's last up-to-3 qualifying seasons, then ranked. Simple and fully
# transparent on purpose — swap in a real regression model later without touching the
# ETL underneath it. They read `spark.table(...)` directly (rather than `load_table`)
# since each is called many times across the backtest loop and the real forecast.

# CELL ********************

def project_champion_asof(
    as_of_year: int,
    *,
    dim_team: DataFrame, 
    fact_team_champion: DataFrame, 
    team_season: DataFrame, 
) -> DataFrame:

    """
    Project the most likely league champion for the season immediately
    following the specified season start year.

    The projection is based on a weighted average of each team's most
    recent three seasons, where newer seasons receive greater weight
    (3, 2, 1). Teams must have at least two qualifying seasons to be
    included in the projection.

    The resulting dataset includes:
        - Projected team performance metrics
        - Projected league ranking
        - Actual champion of the following season (if available)
        - Indicator showing whether the team actually won the following season

    Args:
        as_of_year: Historical season start year used as the projection cutoff.
            Only seasons with season_start_year <= as_of_year are considered.
        dim_team: DataFrame from team dimension table
        fact_team_champion: DataFrame from team champion fact table
        team_season: Dataframe from team season gold table

    Returns:
        DataFrame containing team-level championship projections and
        actual next-season championship outcomes.
    """
    team_id_w: WindowSpec = Window.partitionBy("team_id").orderBy(F.col("season_start_year").desc())
    
    scored: DataFrame = (
        team_season
        .filter(F.col("season_start_year") <= as_of_year)
        .withColumn("season_rn", F.row_number().over(team_id_w)).filter(F.col("season_rn") <= 3)
        .withColumn("wt", 4 - F.col("season_rn"))
        .groupBy("team_id")
        .agg(
            (F.sum(F.col("points_pct") * F.col("wt")) / F.sum("wt")).alias("proj_points_pct"),
            (F.sum(F.col("goal_diff") * F.col("wt")) / F.sum("wt")).alias("proj_goal_diff"),
            F.count("*").alias("seasons_used"),
        )
        .filter(F.col("seasons_used") >= 2)   # require at least 2 of the last 3 seasons on record
    )

    rank_w: WindowSpec = Window.orderBy(F.col("proj_points_pct").desc())
    scored: DataFrame = (
        scored
        .withColumn("projected_rank", F.rank().over(rank_w))
    )
    
    champ_row: Row = (
        fact_team_champion
        .filter(F.col("season_start_year") == as_of_year + 1)
        .select("champion_team_id")
        .first()
    )
    
    champ_id: int | None = (
        champ_row["champion_team_id"]
        if champ_row is not None
        else None
    )

    return (
        scored
        .join(dim_team, on="team_id", how="left")
        .withColumn("as_of_season_start_year", F.lit(as_of_year))
        .withColumn("champion_team_id", F.lit(champ_id))
        .withColumn("actually_won_next_season", F.when(F.col("team_id") == F.lit(champ_id), 1).otherwise(0))
        .select(
            "as_of_season_start_year", 
            "team_id", 
            "teamName", 
            "abbreviation",
            "proj_points_pct", 
            "proj_goal_diff", 
            "seasons_used", 
            "projected_rank",
            "champion_team_id", 
            "actually_won_next_season",
        )
    )


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def project_top_scorer_asof(
    as_of_year: int,
    *,
    dim_team: DataFrame, 
    fact_team_champion: DataFrame, 
    skater_season: DataFrame, 
) -> DataFrame:

    """
    Project the league's top scorers for the season immediately following
    the specified season start year.

    The projection uses a weighted average of each player's most recent
    three qualifying seasons. More recent seasons are weighted more heavily
    (3, 2, 1). Players must have appeared in at least 20 games per season
    and have at least two qualifying seasons available.

    Projected scoring is expressed as:
        - Points per game (PPG)
        - Equivalent points over an 82-game season
        - Projected league-wide scoring rank

    Actual point totals from the subsequent season are included when
    available to support model evaluation.

    Args:
        as_of_year: Historical season start year used as the projection cutoff.
            Only seasons with season_start_year <= as_of_year are considered.
        dim_team: DataFrame from team dimension table
        fact_team_champion: DataFrame from team champion facts table
        skater_season: DataFrame from skater season gold table

    Returns:
        DataFrame containing player scoring projections, projected rank,
        and actual next-season point totals when available.
    """

    player_id_w: WindowSpec = Window.partitionBy("player_id").orderBy(F.col("season_start_year").desc())

    scored: DataFrame = (
        skater_season
        .filter((F.col("season_start_year") <= as_of_year) & (F.col("games_played") >= 20))
        .withColumn("season_rn", F.row_number().over(player_id_w)).filter(F.col("season_rn") <= 3)
        .withColumn("wt", 4 - F.col("season_rn"))
        .groupBy("player_id")
        .agg(
            F.max("firstName").alias("firstName"),
            F.max("lastName").alias("lastName"),
            F.max("primaryPosition").alias("primaryPosition"),
            F.max("primary_team_id").alias("primary_team_id"),
            (F.sum(F.col("points_per_game") * F.col("wt")) / F.sum("wt")).alias("proj_points_per_game"),
            F.count("*").alias("seasons_used"),
        )
        .filter(F.col("seasons_used") >= 2)
        .withColumn("proj_points_82gp", F.col("proj_points_per_game") * 82.0)
    )

    rank_w: WindowSpec = Window.orderBy(F.col("proj_points_82gp").desc())
    scored: DataFrame = (
        scored
        .withColumn("projected_rank", F.rank().over(rank_w))
    )
    
    actual_next: DataFrame = (
        skater_season
        .filter(F.col("season_start_year") == as_of_year + 1)
        .select("player_id", F.col("points").alias("actual_next_season_points"))
    )

    return (
        scored
        .join(dim_team.withColumnRenamed("team_id", "primary_team_id"), on="primary_team_id", how="left")
        .join(actual_next, on="player_id", how="left")
        .withColumn("as_of_season_start_year", F.lit(as_of_year))
        .select(
            "as_of_season_start_year", 
            "player_id", 
            "firstName",
            "lastName", 
            "primaryPosition",
            "teamName", 
            "proj_points_per_game", 
            "proj_points_82gp", 
            "seasons_used",
            "projected_rank", 
            "actual_next_season_points",
        )
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Backtest the forecast (2009–2018)
# `game.csv` has zero playoff games for any season before 2010-11, so `fact_team_champion`
# has no rows (and therefore no real outcome to check against) for season_start_year
# 2004–2009. Starting the loop at 2009 keeps `champion_validation_history` /
# `skater_validation_history` limited to as-of years that actually have a real result to
# score. The 2004-05 season is entirely absent from the source data (NHL lockout) and
# never comes up in this range, so no special-casing is needed.

# CELL ********************

champion_history: DataFrame = reduce(
    lambda a, b: a.unionByName(b),
    [project_champion_asof(
        y, 
        dim_team=dim_team, 
        fact_team_champion=fact_team_champion, 
        team_season=team_season
        ) for y in range(2009, 2019)
    ],   # backtests through the real 2018-19 -> 2019-20 pair
)
write_table(champion_history, CHAMPION_VALIDATION_HISTORY_TABLE)

scorer_history: DataFrame = reduce(
    lambda a, b: a.unionByName(b),
    [project_top_scorer_asof(
        y,
        dim_team=dim_team,
        fact_team_champion=fact_team_champion,
        skater_season=skater_season
        ) for y in range(2009, 2019)
    ],
)
write_table(scorer_history, SKATER_VALIDATION_HISTORY_TABLE)

# Quick calibration read: how often did our top-5 projected champions include the real one?
hit_rate: DataFrame = (
    champion_history
    .filter(F.col("projected_rank") <= 5)
    .agg(F.avg("actually_won_next_season").alias("top5_hit_rate"))
)
display(hit_rate)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Run the real 2020-21 forecast
# as_of = 2019 means "everything through 2019-20", predicting the season the dataset
# never saw. `points_pct` is converted into normalized "championship odds" via softmax
# so the numbers read like betting probabilities — k=15 is a tuning knob (higher k =
# more confident/top-heavy odds, lower k flattens them across more teams).

# CELL ********************

team_projection_2021_raw: DataFrame = (
    project_champion_asof(
        2019, 
        dim_team=dim_team, 
        fact_team_champion=fact_team_champion, 
        team_season=team_season
    )
)

write_table(team_projection_2021_raw, TEAM_PROJECTION_2021_RAW_TABLE)

k = 15.0
scored: DataFrame = (
    team_projection_2021_raw
    .withColumn("raw_score", F.exp(F.col("proj_points_pct") * k))
)

total_score: DataFrame = (
    scored
    .agg(F.sum("raw_score").alias("t"))
    .collect()[0]["t"]
)

championship_odds_2021: DataFrame = (
    scored
    .withColumn("championship_probability", F.col("raw_score") / F.lit(total_score))
    .select(
        "team_id", 
        "teamName", 
        "abbreviation", 
        "proj_points_pct", 
        "projected_rank", 
        "championship_probability"
    )
)

write_table(championship_odds_2021, CHAMPIONSHIP_ODDS_2021_TABLE)
display(championship_odds_2021.orderBy("projected_rank"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

best_player_forecast_2021: DataFrame = (
    project_top_scorer_asof(
        2019,
        dim_team=dim_team,
        fact_team_champion=fact_team_champion,
        skater_season=skater_season
    )
)

write_table(best_player_forecast_2021, BEST_PLAYER_FORECAST_2021_TABLE)
display(best_player_forecast_2021.orderBy("projected_rank").limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### The reveal
# 2020-21 already happened in real history even though it isn't in this dataset.
# Verified against Hockey-Reference / NHL.com coverage.

# CELL ********************

actual_2020_21_results: DataFrame = spark.createDataFrame(
    [
        ("Hart Trophy (MVP)", "Connor McDavid", "Edmonton Oilers", "105 points (33G, 72A) in a 56-game season"),
        ("Art Ross Trophy (points leader)", "Connor McDavid", "Edmonton Oilers", "105 points, led the NHL in scoring"),
        ("Stanley Cup Champion", None, "Tampa Bay Lightning", "Beat the Montreal Canadiens in the Final \u2014 back-to-back champions"),
    ],
    ["category", "winner_name", "team", "detail"],
)

write_table(actual_2020_21_results, ACTUAL_2020_21_RESULTS_TABLE)
display(actual_2020_21_results)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### What Power BI reads
# All the tables Power BI needs are now sitting in this Lakehouse: `silver.dim_team`,
# `gold.team_season`, `gold.skater_season`, `gold.goalie_season`,
# `gold.championship_odds_2021`, `gold.best_player_forecast_2021`,
# `gold.champion_validation_history`, `gold.skater_validation_history`,
# `gold.actual_2020_21_results`.
# 
# **These are new, schema-qualified names** (they used to be `dim_team`,
# `gold_team_season`, `championship_odds_2021`, etc. with no schema). The existing
# `Betting Final Report.pbix` semantic model still points at the old unqualified names
# — its table bindings need to be repointed to these before the report will refresh
# against this version of the pipeline.

# MARKDOWN ********************

# ### Diagnostics — playoff-data completeness check
# Supports the 2009 backtest cutoff above: confirms `fact_team_champion` has no rows for 2004–2009, and that `game` has zero `type='P'` rows before the 2010-11 season.

# CELL ********************

display(spark.table(FACT_TEAM_CHAMPION_TABLE).filter(F.col("season_start_year").between(2004, 2009)).orderBy("season_start_year"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

raw_game = spark.table("bronze.game")

display(
    raw_game
    .filter(F.col("type") == "P")
    .withColumn("season_start_year", F.substring("season", 1, 4).cast("int"))
    .groupBy("season_start_year")
    .count()
    .orderBy("season_start_year")
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print("🏁 Gold load complete.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
