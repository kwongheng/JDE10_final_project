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

# ## nb_03_3_game_skater_stats_silver
# 
# Cleans and validates bronze table and writes the result to a silver table
# Every cleaning/validation rule lives in its own function (defined once,
# below) and is then applied one step at a time in its own cell, so each
# intermediate result can be inspected before moving to the next step.

# MARKDOWN ********************

# ### Imports

# CELL ********************

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

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

# ### Parameters
# 
# `RUN_PIPELINE` controls whether the "Run the pipeline" steps below actually execute.
# It defaults to `True` for normal, standalone runs of this notebook.
# 
# When this notebook is loaded from another notebook via `%run` (e.g. from a test
# notebook), pass `RUN_PIPELINE = False` as a run parameter so only the function/config
# definitions are loaded and the pipeline against the real `bronze.tablename` / `silver.tablename`
# tables is skipped:
# 
# ```
# %run <notebook name> { "RUN_PIPELINE": false }
# ```

# PARAMETERS CELL ********************

# This cell is tagged "parameters" so Fabric/Synapse can override it when the
# notebook is invoked with %run nb_02_game_silver { "RUN_PIPELINE": false }
RUN_PIPELINE: bool = True

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Config

# CELL ********************

BRONZE_TABLE = "bronze.game_skater_stats"
SILVER_TABLE = "silver.game_skater_stats"

# Only these columns make it into the silver table
required_cols: list[str] = [
    "game_id",
    "player_id",
    "team_id",
    "timeOnIce",
    "goals",
    "assists",
    "shots",
    "hits",
    "penaltyMinutes",
    "plusMinus",
    "powerPlayGoals",
    "powerPlayAssists",
    "faceOffWins",
    "faceoffTaken",
    "takeaways",
    "giveaways"
]

no_null_cols: list[str] = [
    "game_id",
    "player_id",
    "team_id",
    "timeOnIce",
    "goals",
    "assists",
    "shots",
    "penaltyMinutes",
    "plusMinus",
    "powerPlayGoals",
    "powerPlayAssists",
    "faceOffWins",
    "faceoffTaken",
]


# Natural key used to de-duplicate rows
DEDUPE_KEYS: list[str] = ["game_id", "player_id", "team_id"]
PRIMARY_KEYS: list[str] = ["game_id", "player_id", "team_id"]

# Valid HoA values
VALID_HOA: tuple[str] = ("home", "away")
VALID_SETTLED_IN: tuple[str] = ("REG", "tbc", "OT")

# Prevents table from loading when called from tests
if RUN_PIPELINE:

    GAME_FOREIGN_KEYS: list[str] = ["game_id"]
    game: DataFrame = load_table(spark, "silver.game", GAME_FOREIGN_KEYS)

    TEAM_INFO_FOREIGN_KEYS: list[str] = ["team_id"]
    team_info: DataFrame = load_table(spark, "silver.team_info", TEAM_INFO_FOREIGN_KEYS)

    PLAYER_INFO_FOREIGN_KEYS: list[str] = ["player_id"]
    player_info: DataFrame = load_table(spark, "silver.player_info", PLAYER_INFO_FOREIGN_KEYS)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Run the pipeline
# Each step runs in its own cell so the result can be inspected before moving on.

# CELL ********************

if RUN_PIPELINE:

    # only load latest dataset to existing silver table
    df: DataFrame = (
        read_new_bronze_rows(
            spark, 
            BRONZE_TABLE, 
            SILVER_TABLE, 
            columns=required_cols
        )
    )

    if not df.isEmpty():    
        # ensure latest data is kept 
        # no duplicates based on primary keys
        df = keep_latest_per_key(df, PRIMARY_KEYS)

        # cleaning and transforms
        df = df.dropDuplicates(DEDUPE_KEYS)

        df = drop_foreign_key_violations(df, ftable=game, keys=GAME_FOREIGN_KEYS)
        df = drop_foreign_key_violations(df, ftable=team_info, keys=TEAM_INFO_FOREIGN_KEYS)
        df = drop_foreign_key_violations(df, ftable=player_info, keys=PLAYER_INFO_FOREIGN_KEYS)

        # Data quality checks
        df = validate_no_nulls(df, columns=no_null_cols)

        df = validate_foreign_keys(df, ftable=game, keys=GAME_FOREIGN_KEYS)
        df = validate_foreign_keys(df, ftable=team_info, keys=TEAM_INFO_FOREIGN_KEYS)
        df = validate_foreign_keys(df, ftable=player_info, keys=PLAYER_INFO_FOREIGN_KEYS)

        df = validate_primary_keys(df, keys=PRIMARY_KEYS)

        # write to SCD Type 1 silver table with latest records only
        write_to_silver(spark, df, SILVER_TABLE, PRIMARY_KEYS) 
        print(f"🏁 {SILVER_TABLE} load complete.")
    
    else:
        print(f"No new data for {SILVER_TABLE}")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
