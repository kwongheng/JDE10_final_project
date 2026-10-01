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

# ## nb_03_1_player_info_silver
# 
# Cleans and validates table in bronze schema and and writes the result to sliver schema
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
# definitions are loaded:
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

BRONZE_TABLE = "bronze.team_info"
SILVER_TABLE = "silver.team_info"

# Only these columns make it into the silver table
required_cols: list[str] = [
    "team_id",
    "franchiseId",
    "shortName",
    "teamName",
    "abbreviation",
]

# Natural key used to de-duplicate rows
PRIMARY_KEYS: list[str] = ["team_id"]
DEDUPE_KEYS: list[str] = ["team_id"]


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

        # Data quality checks
        df = validate_no_nulls(df, columns=required_cols)
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
