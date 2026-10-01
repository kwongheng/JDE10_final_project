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

# ## nb_03_2_game_silver
# 
# Cleans and validates `bronze.game` and writes the result to `silver.game`.
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
# definitions are loaded and the pipeline against the real `bronze.game` / `silver.game`
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

BRONZE_TABLE = "bronze.game"
SILVER_TABLE = "silver.game"

# Only these columns make it into the silver table
required_cols: list[str] = [
    "game_id",
    "season",
    "type",
    "date_time_GMT",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
    "outcome"
]

# Natural key used to de-duplicate rows
DEDUPE_KEYS: list[str] = ["game_id"]
PRIMARY_KEYS: list[str] = ["game_id"]

# Valid game "type" values: A = All-Star, R = Regular season, P = Playoffs
VALID_TYPES: tuple[str] = ("A", "R", "P")


# this prevents table from loading when called from test script
if RUN_PIPELINE:
    TEAM_INFO_HOME_FOREIGN_KEYS = ["home_team_id", "team_id"]
    TEAM_INFO_AWAY_FOREIGN_KEYS = ["away_team_id", "team_id"]
    team_info = load_table(spark, "silver.team_info", columns=["team_id"])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: validate date_time_GMT to season
# 1. Confirm it falls within its own season's date range
#    (a season runs roughly Sept 15 of the start year through Sept 30 of the end year).
# 2. Convert date_time_GMT to a date-only column.
# Rows failing either check are dropped and reported.

# CELL ********************

def validate_game_date_to_season(
    df: DataFrame,
) -> DataFrame:
    """
    Validate that game timestamps are parseable and fall within the
    corresponding season date range.

    The season is assumed to run from September 15 of the starting year
    through September 30 of the ending year.

    Args:
        df: Input DataFrame containing ``date_time_GMT`` and ``season``
            columns.

    Returns:
        The original DataFrame if all rows pass validation.

    Raises:
        RuntimeError: If one or more rows contain a null or unparseable
            ``date_time_GMT`` value, or if a game date falls outside its
            corresponding season date range.
    """
    if (invalid_timestamp_count := df.filter(F.col("date_time_GMT").isNull()).count()) > 0:
        raise RuntimeError(
            f"Found {invalid_timestamp_count} row(s) with a null "
            f"or unparseable date_time_GMT value."
        )

    season_bounds = (
        df
        .withColumn(
            "season_start",
            F.substring(F.col("season"), 1, 4).cast("int"),
        )
        .withColumn(
            "season_end",
            F.substring(F.col("season"), 5, 4).cast("int"),
        )
        .withColumn(
            "season_start_date",
            F.to_timestamp(
                F.concat(F.col("season_start"), F.lit("-09-15 00:00:00"))
            ),
        )
        .withColumn(
            "season_end_date",
            F.to_timestamp(
                F.concat(F.col("season_end"), F.lit("-09-30 23:59:59"))
            ),
        )
    )

    invalid_season_count = (
        season_bounds
        .filter(
            (F.col("date_time_GMT") < F.col("season_start_date"))
            | (F.col("date_time_GMT") > F.col("season_end_date"))
        )
        .count()
    )

    if invalid_season_count > 0:
        raise RuntimeError(
            f"Found {invalid_season_count} row(s) with "
            f"date_time_GMT outside the corresponding season range."
        )

    return df

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
        df = df.withColumn(
            "date_time_GMT",
            F.to_timestamp(F.col("date_time_GMT"))
        )

        df = drop_foreign_key_violations(df, ftable=team_info, keys=TEAM_INFO_HOME_FOREIGN_KEYS)
        df = drop_foreign_key_violations(df, ftable=team_info, keys=TEAM_INFO_AWAY_FOREIGN_KEYS)

        # Data quality checks
        df = validate_game_date_to_season(df)

        df = validate_foreign_keys(df, ftable=team_info, keys=TEAM_INFO_HOME_FOREIGN_KEYS)
        df = validate_foreign_keys(df, ftable=team_info, keys=TEAM_INFO_AWAY_FOREIGN_KEYS)    

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
