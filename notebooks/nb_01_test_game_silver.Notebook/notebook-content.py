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

# ## nb_01_test_game_silver
# 
# Unit tests for the cleaning/validation functions defined in notebook,
# using Python's built-in `unittest` framework and a local Spark session.
# 
# Passing `RUN_PIPELINE = False` as a run parameter means only the imports, config,
# and function *definitions* from that notebook execute — the `## Run the pipeline`
# cells (which read `bronze.game` and write `silver.game`) are skipped entirely, so
# this test notebook never touches real bronze/silver tables.

# MARKDOWN ********************

# ### Load functions from notebook under test (pipeline execution skipped)

# CELL ********************

%run nb_03_2_game_silver { "RUN_PIPELINE": false }

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Imports

# CELL ********************

import unittest
from datetime import date
from unittest.mock import MagicMock

from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Test data helpers
# 
# Small helper to build a `bronze.game`-shaped DataFrame from plain Python rows,
# using the current `spark` session provided by the Fabric/Synapse runtime.

# CELL ********************

RAW_SCHEMA = StructType([
    StructField("game_id", StringType(), True),
    StructField("season", StringType(), True),
    StructField("type", StringType(), True),
    StructField("date_time_GMT", StringType(), True),
])


def make_df(rows: list[tuple]):
    """Build a bronze.game-shaped DataFrame from (game_id, season, type, date_time_GMT) tuples."""
    return spark.createDataFrame(rows, schema=RAW_SCHEMA)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### TestValidateGameDateToSeason

# CELL ********************

class TestValidateGameDateToSeason(unittest.TestCase):
    """validate_game_date_to_season should:
    1. raise if date_time_GMT is NULL/unparseable
    2. raise if date_time_GMT falls outside the season window
    3. cast valid date_time_GMT to a date-only column
    """

    def test_keeps_valid_in_season_dates(self):
        df = make_df([
            ("1", "20162017", "R", "2016-10-14T23:00:00Z"),  # valid
        ])

        result = validate_game_date_to_season(df)
        self.assertEqual(result.count(), 1)

    def test_raises_on_unparseable_timestamp(self):
        df = make_df([
            ("1", "20162017", "R", None),  # NULL timestamp
        ])
        with self.assertRaises(RuntimeError) as cm:
            validate_game_date_to_season(df)
        self.assertIn("unparseable", str(cm.exception))

    def test_raises_on_date_before_season_start(self):
        df = make_df([
            ("1", "20162017", "R", "2016-08-01T00:00:00Z"),  # before start
        ])
        with self.assertRaises(RuntimeError) as cm:
            validate_game_date_to_season(df)
        self.assertIn("outside", str(cm.exception))

    def test_raises_on_date_after_season_end(self):
        df = make_df([
            ("1", "20162017", "R", "2017-10-15T00:00:00Z"),  # after end
        ])
        with self.assertRaises(RuntimeError) as cm:
            validate_game_date_to_season(df)
        self.assertIn("outside", str(cm.exception))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Run all tests

# CELL ********************

loader = unittest.TestLoader()
suite = unittest.TestSuite()
for test_case in (
    TestValidateGameDateToSeason,
):
    suite.addTests(loader.loadTestsFromTestCase(test_case))

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

print("\n========== FULL FAILURE DETAILS ==========")
print(f"Failures: {len(result.failures)}")
print(f"Errors:   {len(result.errors)}")

for test, traceback in result.failures + result.errors:
    print("\n" + "="*60)
    print(test.id())
    print("="*60)
    print(traceback)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
