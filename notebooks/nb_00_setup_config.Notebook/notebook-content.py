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

# ## nb_00_setup_config
# Provides a more flexible configuration for landing and bronze
# 
# Table can be recreated anytime with new changes and must be manually executed before the pipeline

# CELL ********************

from pyspark.sql import Row
from pyspark.sql.types import *

# Ensure schemas exist (skip if already created in the Lakehouse UI)
spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")

schema = StructType([
    StructField("bronze_table_name", StringType(), False),
    StructField("kaggle_dataset", StringType(), False),
    StructField("source_file", StringType(), False),
    StructField("hash_columns", ArrayType(StringType()), False),
    StructField("csv_option", MapType(StringType(), StringType()), True),
    StructField("is_active", BooleanType(), False),
])

config_data = [
    Row(
        bronze_table_name="bronze.game",
        kaggle_dataset="martinellis/nhl-game-data",
        source_file="game.csv",
        hash_columns=["game_id", "season", "type"],
        csv_option=None,
        is_active=True
    ),
    Row(
        bronze_table_name="bronze.team_info",
        kaggle_dataset="martinellis/nhl-game-data",
        source_file="team_info.csv",
        hash_columns=[
            "team_id", 
            "franchiseId", 
            "shortName", 
            "teamName", 
            "abbreviation"
        ],
        csv_option=None,
        is_active=True
    ),
        Row(
        bronze_table_name="bronze.player_info",
        kaggle_dataset="martinellis/nhl-game-data",
        source_file="player_info.csv",
        hash_columns=[
            "player_id",
            "firstName",
            "lastName",
            "birthDate",
            "primaryPosition",
            "nationality",            
        ],
        csv_option=None,        
        is_active=True
    ),
        Row(
        bronze_table_name="bronze.game_goalie_stats",
        kaggle_dataset="martinellis/nhl-game-data",
        source_file="game_goalie_stats.csv",
        hash_columns=[
            "game_id", 
            "player_id", 
            "team_id", 
            "timeOnIce", 
        ],
        csv_option=None,
        is_active=True
    ),
        Row(
        bronze_table_name="bronze.game_skater_stats",
        kaggle_dataset="martinellis/nhl-game-data",
        source_file="game_skater_stats.csv",        
        hash_columns=[
            "game_id", 
            "player_id", 
            "team_id", 
            "timeOnIce", 
        ],
        csv_option=None,
        is_active=True
    ),
        Row(
        bronze_table_name="bronze.game_teams_stats",
        kaggle_dataset="martinellis/nhl-game-data",
        source_file="game_teams_stats.csv",        
        hash_columns=["game_id", "team_id", "HoA"],
        csv_option=None,
        is_active=True
    ),
]

(
    spark
    .createDataFrame(config_data, schema)
    .write 
    .format("delta") 
    .mode("overwrite") 
    .option("overwriteSchema", "true") 
    .saveAsTable("dbo.config_datasets")
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
