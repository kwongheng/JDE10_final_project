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

# ## nb_03_4_teams_players_star_silver
# 
# This notebook models cleaned base silver tables into a formal Star Schema structure containing dedicated dimensions and granular factual tables for teams and players. It covers:
# * **Dimension Construction:** Builds `dim_team` containing full descriptive properties and clean naming conventions.
# * **Team Game Facts:** Models `fact_team_game` by mirroring opponent records side-by-side and generating standardized NHL standings point allocations.
# * **Playoff Champions Layer:** Inferentially isolates seasonal champions in `fact_team_champion` by tracking the concluding playoff matchup per window.
# * **Skater Granularity:** Derives per-game skater box scores (`fact_skater_game`) with unified aggregate metrics like absolute player points.
# * **Goalie Performance:** Models individual goalie records (`fact_goalie_game`) capturing total saves, decisions, and derived metrics like goals against.


# MARKDOWN ********************

# ### Imports

# CELL ********************

from pyspark.sql import functions as F
from pyspark.sql.window import Window, WindowSpec

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Load common DB functions

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
GAME_SKATER_STATS_TABLE: str = "silver.game_skater_stats"
GAME_GOALIE_STATS_TABLE: str = "silver.game_goalie_stats"
TEAM_INFO_TABLE: str = "silver.team_info"

# Raw ("game", "team_info", etc.) stay unprefixed — they're bronze, not ours to
# schema-qualify. Everything this notebook itself produces goes in the silver schema,
# matching Kelvin's silver.game_features convention.
DIM_TEAM_TABLE: str = "silver.dim_team"
FACT_TEAM_GAME_TABLE: str = "silver.fact_team_game"
FACT_TEAM_CHAMPION_TABLE: str = "silver.fact_team_champion"
FACT_SKATER_GAME_TABLE: str = "silver.fact_skater_game"
FACT_GOALIE_GAME_TABLE: str = "silver.fact_goalie_game"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Load required raw tables

# CELL ********************

game_teams_stats: DataFrame = load_table(spark, GAME_TEAMS_STATS_TABLE)
game_skater_stats: DataFrame = load_table(spark, GAME_SKATER_STATS_TABLE)
game_goalie_stats: DataFrame = load_table(spark, GAME_GOALIE_STATS_TABLE)
team_info: DataFrame = load_table(spark, TEAM_INFO_TABLE)

game: DataFrame = (
    load_table(spark, GAME_TABLE)
    .withColumn("season_start_year", F.substring("season", 1, 4).cast("int"))
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Create and write team dimension table

# CELL ********************

dim_team: DataFrame = (
    team_info
    .select(
        "team_id",
        "franchiseId",
        "shortName",
        "teamName",
        "abbreviation",
        F.concat_ws(" ", "shortName", "teamName").alias("full_name"),
    )
)

write_table(dim_team, DIM_TEAM_TABLE)
display(dim_team.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Build and write team_game fact table
# One row per team per game, opponent stats joined alongside. Points follow the standard
# NHL system: win = 2, OT/SO loss = 1, regulation loss = 0. Pre-2005-06 ties show up as
# won=FALSE, settled_in='OT' on BOTH sides in this dataset (there's no separate 'TIE'
# value) — that happens to net 1 point each side too, so no special-casing is needed. A
# small number of settled_in='tbc' rows (~0.1% of games, a data-quality artifact) default
# to 0 points here.

# CELL ********************

team_won_stats: DataFrame = (
    game_teams_stats
    .withColumn(
        "won_flag",
        F.when(
            F.col("won").cast("string").isin("TRUE", "true", "1"), 
            F.lit(1)
        ).otherwise(F.lit(0)),
    )
)

fact_team_game = (
    game.alias("g")
    .join(
        team_won_stats.alias("a"), 
        F.col("g.game_id") == F.col("a.game_id")
    )
    .join(
        team_won_stats.alias("b"), 
        (F.col("g.game_id") == F.col("b.game_id")) & 
        (F.col("a.team_id") != F.col("b.team_id"))
    )
    .select(
        F.col("g.game_id"),
        F.col("g.season"),
        F.col("g.season_start_year"),
        F.col("g.type"),
        F.col("a.team_id"),
        F.col("a.HoA"),
        F.col("a.won_flag").alias("won"),
        F.col("a.settled_in"),
        F.col("a.goals"),
        F.col("a.shots"),
        F.col("a.hits"),
        F.col("a.pim"),
        F.col("a.powerPlayOpportunities"),
        F.col("a.powerPlayGoals"),
        F.col("a.faceOffWinPercentage"),
        F.col("a.giveaways"),
        F.col("a.takeaways"),
        F.col("a.blocked"),
        F.col("b.team_id").alias("opp_team_id"),
        F.col("b.goals").alias("opp_goals"),
        F.col("b.shots").alias("opp_shots"),
        F.col("b.powerPlayOpportunities").alias("opp_powerPlayOpportunities"),
        F.col("b.powerPlayGoals").alias("opp_powerPlayGoals"),

        F.when(F.col("a.won_flag") == 1, F.lit(2)
        ).when(F.col("a.settled_in") == "OT", F.lit(1)
        ).otherwise(F.lit(0)
        ).alias("points"),
    )
)

write_table(fact_team_game, FACT_TEAM_GAME_TABLE)
display(fact_team_game.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Build fact_team_champion
# Champion = winner of each season's LAST playoff game (playoffs always end with the
# Final's deciding game, so this is a safe inference). Tied-score rows are filtered out
# first so a game that went to OT isn't picked up as the season's final state.

# CELL ********************

playoff_games: DataFrame = (
    game
    .filter(
        (F.col("type") == "P") & 
        (F.col("home_goals") != F.col("away_goals"))
    )
    .select(
        "season",
        "game_id",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
        "date_time_GMT",
        "season_start_year",
    )
)

w_last: WindowSpec = Window.partitionBy("season").orderBy(F.col("date_time_GMT").desc())

fact_team_champion: DataFrame = (
    playoff_games
    .withColumn("rn", F.row_number().over(w_last))
    .filter(F.col("rn") == 1)
    .select(
        "season",
        "season_start_year",
        F.when(
            F.col("home_goals") > F.col("away_goals"), 
            F.col("home_team_id")
        ).otherwise(F.col("away_team_id")
        ).alias("champion_team_id"),
    )
)

write_table(fact_team_champion, FACT_TEAM_CHAMPION_TABLE)
display(fact_team_champion.orderBy("season_start_year"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Build and write skater_game / goalie_game fact tables
# Per-game player rows joined to season/type.

# CELL ********************

fact_skater_game: DataFrame = (
    game_skater_stats.alias("s")
    .join(
        game.alias("g"), 
        F.col("s.game_id") == F.col("g.game_id")
    )
    .select(
        "s.game_id",
        "s.player_id",
        "s.team_id",
        "g.season",
        "g.season_start_year",
        "g.type",
        "s.timeOnIce",
        "s.goals",
        "s.assists",
        (F.col("s.goals") + F.col("s.assists")).alias("points"),
        "s.shots",
        "s.hits",
        "s.penaltyMinutes",
        "s.plusMinus",
        "s.powerPlayGoals",
        "s.powerPlayAssists",
        "s.faceOffWins",
        "s.faceoffTaken",
        "s.takeaways",
        "s.giveaways",
    )
)

write_table(fact_skater_game, FACT_SKATER_GAME_TABLE)
display(fact_skater_game.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

fact_goalie_game: DataFrame = (
    game_goalie_stats.alias("gs")
    .join(
        game.alias("g"), 
        F.col("gs.game_id") == F.col("g.game_id")
    )
    .select(
        "gs.game_id",
        "gs.player_id",
        "gs.team_id",
        "g.season",
        "g.season_start_year",
        "g.type",
        "gs.timeOnIce",
        "gs.shots",
        "gs.saves",
        (F.col("gs.shots") - F.col("gs.saves")).alias("goals_against"),
        "gs.decision",
    )
)

write_table(fact_goalie_game, FACT_GOALIE_GAME_TABLE)
display(fact_goalie_game.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
