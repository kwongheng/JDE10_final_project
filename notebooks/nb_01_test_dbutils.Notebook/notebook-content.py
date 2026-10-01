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

# ## nb_01_test_dbutils
# 
# Unit tests for the shared, generic Spark helper functions defined in `nb_00_dbutils`
# (load, dedupe, null-handling, validation, primary/foreign key checks, and table writes),
# using Python's built-in `unittest` framework and a local Spark session.
# 
# The functions under test are loaded with `%run nb_00_dbutils`. This notebook only exercises
# the generic dbutils functions with synthetic, game-shaped test data built in-memory — it never
# touches real bronze/silver tables.
# 
# Note: date/season validation logic (`validate_and_convert_date`) lives in `nb_02_game_silver`
# itself, not in `nb_00_dbutils`, so it is out of scope for this test notebook.

# MARKDOWN ********************

# ### Load functions under test

# CELL ********************

%run nb_00_dbutils

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Imports

# CELL ********************

import unittest
from unittest.mock import MagicMock, patch

from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql.types import TimestampType
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType
)
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

from datetime import datetime
import io
from contextlib import redirect_stdout
from typing import NamedTuple

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Test data helpers
# 
# Small helpers to build game- and team-shaped DataFrames from plain Python rows,
# using the current `spark` session provided by the Fabric/Synapse runtime.

# CELL ********************

GAME_SCHEMA = StructType([
    StructField("game_id", StringType(), True),
    StructField("season", StringType(), True),
    StructField("type", StringType(), True),
    StructField("team_id", StringType(), True),
])

TEAM_SCHEMA = StructType([
    StructField("team_id", StringType(), True),
    StructField("team_name", StringType(), True),
])

TEST_COLUMNS = ["game_id", "season", "type", "team_id"]
VALID_TYPES = ("R", "A", "P")

def make_game_df(rows: list[tuple]):
    """Build a game-shaped DataFrame from (game_id, season, type, team_id) tuples."""
    return spark.createDataFrame(rows, schema=GAME_SCHEMA)


def make_team_df(rows: list[tuple]):
    """Build a team-shaped DataFrame from (team_id, team_name) tuples."""
    return spark.createDataFrame(rows, schema=TEAM_SCHEMA)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### TestLoadTable

# CELL ********************

class TestLoadTable(unittest.TestCase):
    """load_table should select exactly the requested columns from the given table,
    or all columns when none are specified."""

    def test_selects_only_requested_columns(self):
        fake_df = make_game_df([
            ("1", "20162017", "R", "1"),
        ]).withColumn("extra_col", F.col("game_id"))

        mock_spark = MagicMock()
        mock_spark.sql.return_value = fake_df

        result = load_table(mock_spark, "bronze.game", columns=TEST_COLUMNS)

        mock_spark.sql.assert_called_once()
        self.assertEqual(result.columns, TEST_COLUMNS)
        self.assertNotIn("extra_col", result.columns)

    def test_returns_all_columns_when_not_specified(self):
        fake_df = make_game_df([
            ("1", "20162017", "R", "1"),
        ])
        mock_spark = MagicMock()
        mock_spark.sql.return_value = fake_df

        result = load_table(mock_spark, "bronze.game")

        self.assertEqual(result.columns, TEST_COLUMNS)

    def test_query_references_table_name(self):
        fake_df = make_game_df([("1", "20162017", "R", "1")])
        mock_spark = MagicMock()
        mock_spark.sql.return_value = fake_df

        load_table(mock_spark, "bronze.game")

        called_query = mock_spark.sql.call_args[0][0]
        self.assertIn("bronze.game", called_query)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### TestValidateNoNulls

# CELL ********************

class TestValidateNoNulls(unittest.TestCase):
    """validate_no_nulls should raise RuntimeError if any required column has a null,
    and otherwise return the DataFrame unchanged (hard guardrail, not a silent drop)."""

    def test_raises_on_null_in_required_column(self):
        df = make_game_df([
            ("1", "20162017", "R", "1"),
            ("2", None, "R", "2"),  # null season
        ])

        with self.assertRaises(RuntimeError):
            validate_no_nulls(df, TEST_COLUMNS)

    def test_no_nulls_returns_df_unchanged(self):
        df = make_game_df([
            ("1", "20162017", "R", "1"),
            ("2", "20172018", "R", "2"),
        ])

        result = validate_no_nulls(df, TEST_COLUMNS)

        self.assertEqual(result.count(), 2)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### TestValidateColumnValues

# CELL ********************

class TestValidateColumnValues(unittest.TestCase):
    """validate_column_values should raise RuntimeError if any row's value in `column`
    is outside the given `values`, and otherwise return the DataFrame unchanged."""

    def test_raises_on_invalid_values(self):
        df = make_game_df([
            ("1", "20162017", "R", "1"),
            ("2", "20162017", "A", "2"),
            ("3", "20162017", "X", "3"),  # invalid
        ])

        with self.assertRaises(RuntimeError):
            validate_column_values(df, "type", VALID_TYPES)

    def test_raises_on_null_value(self):
        df = make_game_df([
            ("1", "20162017", "R", "1"),
            ("2", "20162017", None, "2"),  # invalid (null)
        ])

        with self.assertRaises(RuntimeError):
            validate_column_values(df, "type", VALID_TYPES)

    def test_all_valid_returns_df_unchanged(self):
        df = make_game_df([
            ("1", "20162017", "R", "1"),
            ("2", "20162017", "A", "2"),
            ("3", "20162017", "P", "3"),
        ])

        result = validate_column_values(df, "type", VALID_TYPES)

        self.assertEqual(result.count(), 3)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### TestValidatePrimaryKeys

# CELL ********************

class TestValidatePrimaryKeys(unittest.TestCase):
    """validate_primary_keys should raise ValueError if any key combination repeats,
    and otherwise return the DataFrame unchanged."""

    def test_raises_on_duplicate_primary_key(self):
        df = make_game_df([
            ("1", "20162017", "R", "1"),
            ("1", "20172018", "R", "2"),  # duplicate game_id
        ])

        with self.assertRaises(RuntimeError):
            validate_primary_keys(df, keys=["game_id"])

    def test_unique_keys_returns_df_unchanged(self):
        df = make_game_df([
            ("1", "20162017", "R", "1"),
            ("2", "20162017", "R", "2"),
        ])

        result = validate_primary_keys(df, keys=["game_id"])

        self.assertEqual(result.count(), 2)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### TestForeignKeyFunctions

# CELL ********************

class TestForeignKeyFunctions(unittest.TestCase):
    """validate_foreign_keys / drop_foreign_key_violations should treat rows in `df`
    whose foreign key has no match in `ftable` as violations."""

    def setUp(self):
        self.teams = make_team_df([
            ("1", "Team One"),
            ("2", "Team Two"),
        ])

    def test_validate_foreign_keys_passes_when_all_match(self):
        df = make_game_df([
            ("1", "20162017", "R", "1"),
            ("2", "20162017", "R", "2"),
        ])

        result = validate_foreign_keys(df, self.teams, keys=["team_id"])

        self.assertEqual(result.count(), 2)

    def test_validate_foreign_keys_raises_on_violation(self):
        df = make_game_df([
            ("1", "20162017", "R", "1"),
            ("2", "20162017", "R", "99"),  # no matching team
        ])

        with self.assertRaises(RuntimeError):
            validate_foreign_keys(df, self.teams, keys=["team_id"])

    def test_drop_foreign_key_violations_drops_unmatched_rows(self):
        df = make_game_df([
            ("1", "20162017", "R", "1"),
            ("2", "20162017", "R", "99"),  # no matching team
        ])

        result = drop_foreign_key_violations(df, self.teams, keys=["team_id"])

        self.assertEqual(result.count(), 1)
        self.assertEqual(result.collect()[0].game_id, "1")

    def test_drop_foreign_key_violations_is_a_no_op_when_all_match(self):
        df = make_game_df([
            ("1", "20162017", "R", "1"),
            ("2", "20162017", "R", "2"),
        ])

        result = drop_foreign_key_violations(df, self.teams, keys=["team_id"])

        self.assertEqual(result.count(), 2)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### test data helpers for keep_latest_per_key

# CELL ********************

# (player_id, season, score, _ingestion_timestamp, _row_hash)
VersionedRow = tuple[str | None, str | None, int | None, datetime | None, str | None]

VERSIONED_SCHEMA: StructType = StructType([
    StructField("player_id", StringType(), True),
    StructField("season", StringType(), True),
    StructField("score", IntegerType(), True),
    StructField("_ingestion_timestamp", TimestampType(), True),
    StructField("_row_hash", StringType(), True),
])


def make_versioned_df(
    rows: list[VersionedRow],
    include_hash: bool = True,
) -> DataFrame:
    """Build a bronze-derived DataFrame with multiple versions per business key.

    Rows are (player_id, season, score, _ingestion_timestamp, _row_hash) tuples.
    When include_hash is False the `_row_hash` column is dropped entirely, to
    exercise the branch where no deterministic tiebreak is available."""
    df: DataFrame = spark.createDataFrame(rows, schema=VERSIONED_SCHEMA)
    return df if include_hash else df.drop("_row_hash")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### TestKeepLatestPerKey

# CELL ********************

class TestKeepLatestPerKey(unittest.TestCase):
    """keep_latest_per_key should keep exactly one row per business key (the one with
    the newest _ingestion_timestamp, ties broken by smallest _row_hash), rename
    _ingestion_timestamp to _bronze_ingested_at, and not leak the helper _rn column."""

    # ---- row selection ----

    def test_keeps_newest_row_per_key(self) -> None:
        df: DataFrame = make_versioned_df([
            ("p1", "20162017", 10, T1, "h1"),
            ("p1", "20162017", 20, T3, "h3"),  # newest
            ("p1", "20162017", 15, T2, "h2"),
        ])

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        self.assertEqual(result.count(), 1)
        self.assertEqual(result.collect()[0].score, 20)

    def test_keeps_one_row_per_key_across_multiple_keys(self) -> None:
        df: DataFrame = make_versioned_df([
            ("p1", "20162017", 10, T1, "h1"),
            ("p1", "20162017", 11, T2, "h2"),  # newest for p1
            ("p2", "20162017", 20, T2, "h3"),
            ("p2", "20162017", 21, T3, "h4"),  # newest for p2
            ("p3", "20162017", 30, T1, "h5"),  # only version for p3
        ])

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        by_player: dict[str, Row] = {r.player_id: r for r in result.collect()}
        self.assertEqual(result.count(), 3)
        self.assertEqual(by_player["p1"].score, 11)
        self.assertEqual(by_player["p2"].score, 21)
        self.assertEqual(by_player["p3"].score, 30)

    def test_composite_primary_key_treats_each_combination_separately(self) -> None:
        df: DataFrame = make_versioned_df([
            ("p1", "20162017", 10, T1, "h1"),
            ("p1", "20162017", 11, T2, "h2"),  # newest for (p1, 20162017)
            ("p1", "20172018", 50, T1, "h3"),  # only version for (p1, 20172018)
        ])

        result: DataFrame = keep_latest_per_key(
            df, primary_keys=["player_id", "season"]
        )

        by_season: dict[str, Row] = {r.season: r for r in result.collect()}
        self.assertEqual(result.count(), 2)
        self.assertEqual(by_season["20162017"].score, 11)
        self.assertEqual(by_season["20172018"].score, 50)

    def test_single_row_per_key_is_a_no_op_on_row_count(self) -> None:
        df: DataFrame = make_versioned_df([
            ("p1", "20162017", 10, T1, "h1"),
            ("p2", "20162017", 20, T2, "h2"),
        ])

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        self.assertEqual(result.count(), 2)

    def test_empty_dataframe_returns_empty(self) -> None:
        df: DataFrame = make_versioned_df([])

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        self.assertEqual(result.count(), 0)

    def test_null_timestamp_loses_to_non_null_timestamp(self) -> None:
        # Descending order puts nulls last, so a real timestamp always wins.
        df: DataFrame = make_versioned_df([
            ("p1", "20162017", 10, None, "h1"),
            ("p1", "20162017", 20, T1, "h2"),
        ])

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        self.assertEqual(result.count(), 1)
        self.assertEqual(result.collect()[0].score, 20)

    def test_null_key_values_are_grouped_together(self) -> None:
        df: DataFrame = make_versioned_df([
            (None, "20162017", 10, T1, "h1"),
            (None, "20162017", 20, T2, "h2"),  # newest among null-key rows
        ])

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        self.assertEqual(result.count(), 1)
        self.assertEqual(result.collect()[0].score, 20)

    # ---- tiebreaking ----

    def test_timestamp_tie_is_broken_by_smallest_row_hash(self) -> None:
        df: DataFrame = make_versioned_df([
            ("p1", "20162017", 10, T2, "b_hash"),
            ("p1", "20162017", 20, T2, "a_hash"),  # same timestamp, smaller hash wins
        ])

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        self.assertEqual(result.count(), 1)
        self.assertEqual(result.collect()[0].score, 20)

    def test_tiebreak_is_independent_of_input_order(self) -> None:
        rows: list[VersionedRow] = [
            ("p1", "20162017", 10, T2, "b_hash"),
            ("p1", "20162017", 20, T2, "a_hash"),
        ]

        forward: DataFrame = keep_latest_per_key(
            make_versioned_df(rows), primary_keys=["player_id"]
        )
        reverse: DataFrame = keep_latest_per_key(
            make_versioned_df(list(reversed(rows))), primary_keys=["player_id"]
        )

        self.assertEqual(forward.collect()[0].score, reverse.collect()[0].score)

    def test_newer_timestamp_beats_smaller_row_hash(self) -> None:
        # Timestamp takes priority; the hash only matters on a tie.
        df: DataFrame = make_versioned_df([
            ("p1", "20162017", 10, T1, "a_hash"),
            ("p1", "20162017", 20, T2, "z_hash"),
        ])

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        self.assertEqual(result.collect()[0].score, 20)

    def test_works_without_row_hash_column(self) -> None:
        df: DataFrame = make_versioned_df(
            [
                ("p1", "20162017", 10, T1, "h1"),
                ("p1", "20162017", 20, T2, "h2"),
                ("p2", "20162017", 30, T1, "h3"),
            ],
            include_hash=False,
        )

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        by_player: dict[str, Row] = {r.player_id: r for r in result.collect()}
        self.assertEqual(result.count(), 2)
        self.assertEqual(by_player["p1"].score, 20)
        self.assertEqual(by_player["p2"].score, 30)

    # ---- output shape ----

    def test_renames_ingestion_timestamp_to_bronze_ingested_at(self) -> None:
        df: DataFrame = make_versioned_df([("p1", "20162017", 10, T1, "h1")])

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        self.assertIn("_bronze_ingested_at", result.columns)
        self.assertNotIn("_ingestion_timestamp", result.columns)

    def test_renamed_column_holds_the_winning_rows_timestamp(self) -> None:
        df: DataFrame = make_versioned_df([
            ("p1", "20162017", 10, T1, "h1"),
            ("p1", "20162017", 20, T3, "h3"),
        ])

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        self.assertEqual(result.collect()[0]["_bronze_ingested_at"], T3)

    def test_does_not_leak_helper_row_number_column(self) -> None:
        df: DataFrame = make_versioned_df([("p1", "20162017", 10, T1, "h1")])

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        self.assertNotIn("_rn", result.columns)

    def test_preserves_all_other_columns(self) -> None:
        df: DataFrame = make_versioned_df([("p1", "20162017", 10, T1, "h1")])

        result: DataFrame = keep_latest_per_key(df, primary_keys=["player_id"])

        for col in ("player_id", "season", "score", "_row_hash"):
            self.assertIn(col, result.columns)

    def test_does_not_mutate_input_dataframe(self) -> None:
        df: DataFrame = make_versioned_df([("p1", "20162017", 10, T1, "h1")])
        original_columns: list[str] = list(df.columns)

        keep_latest_per_key(df, primary_keys=["player_id"])

        self.assertEqual(df.columns, original_columns)

    # ---- error handling ----

    def test_raises_when_primary_key_column_missing(self) -> None:
        df: DataFrame = make_versioned_df([("p1", "20162017", 10, T1, "h1")])

        with self.assertRaises(AnalysisException):
            keep_latest_per_key(df, primary_keys=["nonexistent_col"]).collect()

    def test_raises_when_ingestion_timestamp_column_missing(self) -> None:
        df: DataFrame = (
            make_versioned_df([("p1", "20162017", 10, T1, "h1")])
            .drop("_ingestion_timestamp")
        )

        with self.assertRaises(AnalysisException):
            keep_latest_per_key(df, primary_keys=["player_id"]).collect()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### test data helpers for read_new_bronze_rows

# CELL ********************

# (game_id, season, type, team_id, _ingestion_timestamp, _row_hash)
BronzeRow = tuple[
    str | None, str | None, str | None, str | None, datetime | None, str | None
]
# (game_id, watermark)
SilverRow = tuple[str | None, datetime | None]

BRONZE_SCHEMA: StructType = StructType([
    StructField("game_id", StringType(), True),
    StructField("season", StringType(), True),
    StructField("type", StringType(), True),
    StructField("team_id", StringType(), True),
    StructField("_ingestion_timestamp", TimestampType(), True),
    StructField("_row_hash", StringType(), True),
])

T1: datetime = datetime(2024, 1, 1, 0, 0, 0)
T2: datetime = datetime(2024, 1, 2, 0, 0, 0)
T3: datetime = datetime(2024, 1, 3, 0, 0, 0)

BRONZE_TABLE: str = "bronze.game"
SILVER_TABLE: str = "silver.game"

def make_bronze_df(rows: list[BronzeRow]) -> DataFrame:
    """Build a bronze-shaped DataFrame from
    (game_id, season, type, team_id, _ingestion_timestamp, _row_hash) tuples."""
    return spark.createDataFrame(rows, schema=BRONZE_SCHEMA)


def make_silver_df(
    rows: list[SilverRow],
    watermark_col: str = "_bronze_ingested_at",
) -> DataFrame:
    """Build a silver-shaped DataFrame from (game_id, watermark) tuples."""
    schema: StructType = StructType([
        StructField("game_id", StringType(), True),
        StructField(watermark_col, TimestampType(), True),
    ])
    return spark.createDataFrame(rows, schema=schema)


def make_mock_spark(
    bronze_df: DataFrame,
    silver_df: DataFrame | None = None,
    silver_exists: bool = True,
) -> MagicMock:
    """Mock spark whose table() returns the given frames and whose
    catalog.tableExists() returns `silver_exists`. If silver_df is None, any
    attempt to read the silver table raises KeyError, so tests can prove
    silver was never touched."""
    tables: dict[str, DataFrame] = {BRONZE_TABLE: bronze_df}
    if silver_df is not None:
        tables[SILVER_TABLE] = silver_df

    mock_spark: MagicMock = MagicMock()
    mock_spark.table.side_effect = lambda name: tables[name]
    mock_spark.catalog.tableExists.return_value = silver_exists
    return mock_spark

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### TestReadNewBronzeRows

# CELL ********************

class TestReadNewBronzeRows(unittest.TestCase):
    """read_new_bronze_rows should return only bronze rows whose _ingestion_timestamp
    is strictly greater than the max watermark already in silver, and fall back to
    the full bronze table when there is no usable watermark."""

    bronze: DataFrame

    def setUp(self) -> None:
        self.bronze = make_bronze_df([
            ("1", "20162017", "R", "1", T1, "h1"),
            ("2", "20162017", "R", "2", T2, "h2"),
            ("3", "20162017", "R", "3", T3, "h3"),
        ])

    # ---- full-load paths (no watermark) ----

    def test_returns_all_rows_when_silver_table_does_not_exist(self) -> None:
        mock_spark: MagicMock = make_mock_spark(
            self.bronze, silver_df=None, silver_exists=False
        )

        result: DataFrame = read_new_bronze_rows(mock_spark, BRONZE_TABLE, SILVER_TABLE)

        self.assertEqual(result.count(), 3)

    def test_does_not_read_silver_when_it_does_not_exist(self) -> None:
        mock_spark: MagicMock = make_mock_spark(
            self.bronze, silver_df=None, silver_exists=False
        )

        read_new_bronze_rows(mock_spark, BRONZE_TABLE, SILVER_TABLE)

        mock_spark.catalog.tableExists.assert_called_once_with(SILVER_TABLE)
        mock_spark.table.assert_called_once_with(BRONZE_TABLE)

    def test_returns_all_rows_when_silver_is_empty(self) -> None:
        empty_silver: DataFrame = make_silver_df([])
        mock_spark: MagicMock = make_mock_spark(self.bronze, empty_silver)

        result: DataFrame = read_new_bronze_rows(mock_spark, BRONZE_TABLE, SILVER_TABLE)

        self.assertEqual(result.count(), 3)

    def test_returns_all_rows_when_silver_watermark_is_all_null(self) -> None:
        silver: DataFrame = make_silver_df([("1", None), ("2", None)])
        mock_spark: MagicMock = make_mock_spark(self.bronze, silver)

        result: DataFrame = read_new_bronze_rows(mock_spark, BRONZE_TABLE, SILVER_TABLE)

        self.assertEqual(result.count(), 3)

    # ---- incremental paths ----

    def test_returns_only_rows_newer_than_watermark(self) -> None:
        silver: DataFrame = make_silver_df([("1", T1), ("2", T2)])  # max watermark = T2
        mock_spark: MagicMock = make_mock_spark(self.bronze, silver)

        result: DataFrame = read_new_bronze_rows(mock_spark, BRONZE_TABLE, SILVER_TABLE)

        self.assertEqual(result.count(), 1)
        self.assertEqual(result.collect()[0].game_id, "3")

    def test_uses_max_of_silver_watermark_not_first_or_last_row(self) -> None:
        # Silver rows deliberately out of order; max is T2.
        silver: DataFrame = make_silver_df([("2", T2), ("1", T1)])
        mock_spark: MagicMock = make_mock_spark(self.bronze, silver)

        result: DataFrame = read_new_bronze_rows(mock_spark, BRONZE_TABLE, SILVER_TABLE)

        self.assertEqual({r.game_id for r in result.collect()}, {"3"})

    def test_row_equal_to_watermark_is_excluded(self) -> None:
        # Comparison is strictly greater-than, so a row at exactly T2 is not re-read.
        silver: DataFrame = make_silver_df([("2", T2)])
        mock_spark: MagicMock = make_mock_spark(self.bronze, silver)

        result: DataFrame = read_new_bronze_rows(mock_spark, BRONZE_TABLE, SILVER_TABLE)

        ids: set[str] = {r.game_id for r in result.collect()}
        self.assertNotIn("2", ids)
        self.assertEqual(ids, {"3"})

    def test_returns_empty_when_silver_is_caught_up(self) -> None:
        silver: DataFrame = make_silver_df([("3", T3)])  # already has newest bronze row
        mock_spark: MagicMock = make_mock_spark(self.bronze, silver)

        result: DataFrame = read_new_bronze_rows(mock_spark, BRONZE_TABLE, SILVER_TABLE)

        self.assertEqual(result.count(), 0)

    def test_returns_all_rows_when_watermark_older_than_all_bronze(self) -> None:
        silver: DataFrame = make_silver_df([("0", datetime(2023, 12, 31))])
        mock_spark: MagicMock = make_mock_spark(self.bronze, silver)

        result: DataFrame = read_new_bronze_rows(mock_spark, BRONZE_TABLE, SILVER_TABLE)

        self.assertEqual(result.count(), 3)

    def test_custom_watermark_col_is_read_from_silver(self) -> None:
        silver: DataFrame = make_silver_df([("2", T2)], watermark_col="my_watermark")
        mock_spark: MagicMock = make_mock_spark(self.bronze, silver)

        result: DataFrame = read_new_bronze_rows(
            mock_spark, BRONZE_TABLE, SILVER_TABLE, watermark_col="my_watermark"
        )

        self.assertEqual({r.game_id for r in result.collect()}, {"3"})

    # ---- column selection ----

    def test_returns_all_bronze_columns_when_columns_not_specified(self) -> None:
        mock_spark: MagicMock = make_mock_spark(
            self.bronze, silver_df=None, silver_exists=False
        )

        result: DataFrame = read_new_bronze_rows(mock_spark, BRONZE_TABLE, SILVER_TABLE)

        self.assertEqual(result.columns, BRONZE_SCHEMA.fieldNames())

    def test_columns_always_include_metadata_columns(self) -> None:
        mock_spark: MagicMock = make_mock_spark(
            self.bronze, silver_df=None, silver_exists=False
        )

        result: DataFrame = read_new_bronze_rows(
            mock_spark, BRONZE_TABLE, SILVER_TABLE, columns=["game_id", "season"]
        )

        self.assertEqual(
            result.columns,
            ["game_id", "season", "_ingestion_timestamp", "_row_hash"],
        )
        self.assertNotIn("team_id", result.columns)

    def test_columns_do_not_duplicate_metadata_when_already_requested(self) -> None:
        mock_spark: MagicMock = make_mock_spark(
            self.bronze, silver_df=None, silver_exists=False
        )

        result: DataFrame = read_new_bronze_rows(
            mock_spark, BRONZE_TABLE, SILVER_TABLE,
            columns=["game_id", "_ingestion_timestamp", "_row_hash"],
        )

        self.assertEqual(
            result.columns, ["game_id", "_ingestion_timestamp", "_row_hash"]
        )

    def test_watermark_filter_works_when_timestamp_not_in_requested_columns(self) -> None:
        # _ingestion_timestamp is auto-added to the select, so filtering by it
        # must still work even though the caller only asked for game_id.
        silver: DataFrame = make_silver_df([("2", T2)])
        mock_spark: MagicMock = make_mock_spark(self.bronze, silver)

        result: DataFrame = read_new_bronze_rows(
            mock_spark, BRONZE_TABLE, SILVER_TABLE, columns=["game_id"]
        )

        self.assertEqual(result.count(), 1)
        self.assertEqual(result.collect()[0].game_id, "3")

    # ---- logging ----

    def test_logs_full_table_read_when_no_watermark(self) -> None:
        mock_spark: MagicMock = make_mock_spark(
            self.bronze, silver_df=None, silver_exists=False
        )

        buf: io.StringIO = io.StringIO()
        with redirect_stdout(buf):
            read_new_bronze_rows(mock_spark, BRONZE_TABLE, SILVER_TABLE)

        self.assertIn("No watermark found", buf.getvalue())
        self.assertIn(BRONZE_TABLE, buf.getvalue())

    def test_logs_watermark_when_reading_incrementally(self) -> None:
        silver: DataFrame = make_silver_df([("2", T2)])
        mock_spark: MagicMock = make_mock_spark(self.bronze, silver)

        buf: io.StringIO = io.StringIO()
        with redirect_stdout(buf):
            read_new_bronze_rows(mock_spark, BRONZE_TABLE, SILVER_TABLE)

        self.assertIn("Reading rows after watermark", buf.getvalue())
        self.assertIn(str(T2), buf.getvalue())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### test helpers for write_to_silver

# CELL ********************

class WriteMocks(NamedTuple):
    """Bundle of mocks for one write_to_silver call.

    delta_table_cls: stands in for the DeltaTable class.
    merger: the object returned by .merge(...), for asserting on the merge chain.
    merge_builder_matched: object returned by .whenMatchedUpdateAll(...).
    """
    spark: MagicMock
    df: MagicMock
    delta_table_cls: MagicMock
    merger: MagicMock
    merge_builder_matched: MagicMock


def make_write_mocks(
    df_empty: bool = False,
    table_exists: bool = True,
) -> WriteMocks:
    """Build mocked spark / df / DeltaTable wired up for write_to_silver.

    The DeltaTable chain mirrors the function:
      DeltaTable.forName(...).alias("t").merge(...)
          .whenMatchedUpdateAll(...).whenNotMatchedInsertAll().execute()
    """
    mock_spark: MagicMock = MagicMock()
    mock_spark.catalog.tableExists.return_value = table_exists

    mock_df: MagicMock = MagicMock()
    mock_df.isEmpty.return_value = df_empty

    mock_delta_table_cls: MagicMock = MagicMock()
    target: MagicMock = mock_delta_table_cls.forName.return_value.alias.return_value
    merger: MagicMock = target.merge.return_value
    matched: MagicMock = merger.whenMatchedUpdateAll.return_value

    return WriteMocks(
        spark=mock_spark,
        df=mock_df,
        delta_table_cls=mock_delta_table_cls,
        merger=merger,
        merge_builder_matched=matched,
    )


def run_write_to_silver(
    mocks: WriteMocks,
    primary_keys: list[str] | None = None,
) -> str:
    """Call write_to_silver with DeltaTable patched, returning captured stdout.

    Assumes `%run nb_00_dbutils` shares this notebook's global namespace, so
    patching `DeltaTable` in globals() is visible to write_to_silver."""
    keys: list[str] = primary_keys if primary_keys is not None else ["game_id"]
    buf: io.StringIO = io.StringIO()
    with patch.dict(globals(), {"DeltaTable": mocks.delta_table_cls}):
        with redirect_stdout(buf):
            write_to_silver(mocks.spark, mocks.df, SILVER_TABLE, keys)
    return buf.getvalue()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### TestWriteToSilver

# CELL ********************

class TestWriteToSilver(unittest.TestCase):
    """write_to_silver should skip empty input, create the silver table on first run,
    otherwise SCD1-merge on the business keys (updating only when the incoming
    _bronze_ingested_at is newer), and log then re-raise any failure."""

    # ---- empty input ----

    def test_empty_df_skips_all_writes(self) -> None:
        mocks: WriteMocks = make_write_mocks(df_empty=True)

        run_write_to_silver(mocks)

        mocks.df.write.format.assert_not_called()
        mocks.delta_table_cls.forName.assert_not_called()

    def test_empty_df_does_not_touch_spark_conf_or_catalog(self) -> None:
        mocks: WriteMocks = make_write_mocks(df_empty=True)

        run_write_to_silver(mocks)

        mocks.spark.conf.set.assert_not_called()
        mocks.spark.catalog.tableExists.assert_not_called()

    def test_empty_df_logs_skip_message(self) -> None:
        mocks: WriteMocks = make_write_mocks(df_empty=True)

        out: str = run_write_to_silver(mocks)

        self.assertIn("No rows to write", out)
        self.assertIn(SILVER_TABLE, out)

    # ---- first run: table does not exist ----

    def test_creates_delta_table_when_silver_does_not_exist(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=False)

        run_write_to_silver(mocks)

        mocks.df.write.format.assert_called_once_with("delta")
        mocks.df.write.format.return_value.saveAsTable.assert_called_once_with(SILVER_TABLE)

    def test_create_path_does_not_merge(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=False)

        run_write_to_silver(mocks)

        mocks.delta_table_cls.forName.assert_not_called()

    def test_create_path_logs_created_message(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=False)

        out: str = run_write_to_silver(mocks)

        self.assertIn("Created table", out)
        self.assertNotIn("Merge complete", out)

    def test_checks_existence_of_the_silver_table(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=False)

        run_write_to_silver(mocks)

        mocks.spark.catalog.tableExists.assert_called_once_with(SILVER_TABLE)

    # ---- schema auto merge ----

    def test_enables_schema_auto_merge_on_create_path(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=False)

        run_write_to_silver(mocks)

        mocks.spark.conf.set.assert_called_once_with(
            "spark.databricks.delta.schema.autoMerge.enabled", "true"
        )

    def test_enables_schema_auto_merge_on_merge_path(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=True)

        run_write_to_silver(mocks)

        mocks.spark.conf.set.assert_called_once_with(
            "spark.databricks.delta.schema.autoMerge.enabled", "true"
        )

    # ---- merge path ----

    def test_merges_into_existing_silver_table(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=True)

        run_write_to_silver(mocks)

        mocks.delta_table_cls.forName.assert_called_once_with(mocks.spark, SILVER_TABLE)
        mocks.delta_table_cls.forName.return_value.alias.assert_called_once_with("t")
        mocks.df.alias.assert_called_once_with("s")

    def test_merge_path_does_not_overwrite_via_saveastable(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=True)

        run_write_to_silver(mocks)

        mocks.df.write.format.assert_not_called()

    def test_merge_condition_single_primary_key(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=True)
        target: MagicMock = mocks.delta_table_cls.forName.return_value.alias.return_value

        run_write_to_silver(mocks, primary_keys=["game_id"])

        target.merge.assert_called_once_with(
            mocks.df.alias.return_value, "t.game_id = s.game_id"
        )

    def test_merge_condition_composite_primary_key(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=True)
        target: MagicMock = mocks.delta_table_cls.forName.return_value.alias.return_value

        run_write_to_silver(mocks, primary_keys=["player_id", "season"])

        target.merge.assert_called_once_with(
            mocks.df.alias.return_value,
            "t.player_id = s.player_id AND t.season = s.season",
        )

    def test_update_only_when_source_is_newer_or_target_is_null(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=True)

        run_write_to_silver(mocks)

        mocks.merger.whenMatchedUpdateAll.assert_called_once()
        condition: str = mocks.merger.whenMatchedUpdateAll.call_args.kwargs["condition"]
        self.assertIn("t._bronze_ingested_at IS NULL", condition)
        self.assertIn("s._bronze_ingested_at > t._bronze_ingested_at", condition)

    def test_unmatched_rows_are_inserted(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=True)

        run_write_to_silver(mocks)

        mocks.merge_builder_matched.whenNotMatchedInsertAll.assert_called_once_with()

    def test_merge_is_executed(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=True)
        insert_builder: MagicMock = (
            mocks.merge_builder_matched.whenNotMatchedInsertAll.return_value
        )

        run_write_to_silver(mocks)

        insert_builder.execute.assert_called_once_with()

    def test_merge_path_logs_merge_complete(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=True)

        out: str = run_write_to_silver(mocks)

        self.assertIn("Merge complete", out)
        self.assertNotIn("Created table", out)

    def test_returns_none(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=True)

        with patch.dict(globals(), {"DeltaTable": mocks.delta_table_cls}):
            with redirect_stdout(io.StringIO()):
                result: None = write_to_silver(
                    mocks.spark, mocks.df, SILVER_TABLE, ["game_id"]
                )

        self.assertIsNone(result)

    # ---- error handling ----

    def test_merge_failure_is_reraised(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=True)
        insert_builder: MagicMock = (
            mocks.merge_builder_matched.whenNotMatchedInsertAll.return_value
        )
        insert_builder.execute.side_effect = RuntimeError("merge exploded")

        with self.assertRaises(RuntimeError):
            run_write_to_silver(mocks)

    def test_merge_failure_is_logged_with_table_and_error_type(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=True)
        insert_builder: MagicMock = (
            mocks.merge_builder_matched.whenNotMatchedInsertAll.return_value
        )
        insert_builder.execute.side_effect = RuntimeError("merge exploded")

        buf: io.StringIO = io.StringIO()
        with patch.dict(globals(), {"DeltaTable": mocks.delta_table_cls}):
            with redirect_stdout(buf), self.assertRaises(RuntimeError):
                write_to_silver(mocks.spark, mocks.df, SILVER_TABLE, ["game_id"])

        out: str = buf.getvalue()
        self.assertIn("[FAILED]", out)
        self.assertIn(SILVER_TABLE, out)
        self.assertIn("RuntimeError", out)
        self.assertIn("merge exploded", out)

    def test_create_failure_is_reraised(self) -> None:
        mocks: WriteMocks = make_write_mocks(table_exists=False)
        mocks.df.write.format.return_value.saveAsTable.side_effect = ValueError("bad write")

        with self.assertRaises(ValueError):
            run_write_to_silver(mocks)

    def test_is_empty_failure_is_reraised_and_logged(self) -> None:
        mocks: WriteMocks = make_write_mocks()
        mocks.df.isEmpty.side_effect = RuntimeError("cannot evaluate")

        buf: io.StringIO = io.StringIO()
        with patch.dict(globals(), {"DeltaTable": mocks.delta_table_cls}):
            with redirect_stdout(buf), self.assertRaises(RuntimeError):
                write_to_silver(mocks.spark, mocks.df, SILVER_TABLE, ["game_id"])

        self.assertIn("[FAILED]", buf.getvalue())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### TestWriteTable

# CELL ********************

class TestWriteTable(unittest.TestCase):
    """write_table should write via the delta format in overwrite mode
    with schema overwrite enabled, to the given table name."""

    def test_writes_with_expected_options(self):
        mock_df = MagicMock()
        mock_writer = mock_df.write
        mock_writer.format.return_value = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.option.return_value = mock_writer
        mock_df.count.return_value = 42
        mock_df.columns = ["game_id", "season", "type", "team_id"]

        write_table(mock_df, table_name="silver.game")

        mock_writer.format.assert_called_once_with("delta")
        mock_writer.mode.assert_called_once_with("overwrite")
        mock_writer.option.assert_called_once_with("overwriteSchema", "true")
        mock_writer.saveAsTable.assert_called_once_with("silver.game")

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
    TestLoadTable,
    TestValidateNoNulls,
    TestValidateColumnValues,    
    TestValidatePrimaryKeys,
    TestForeignKeyFunctions,
    TestKeepLatestPerKey,
    TestReadNewBronzeRows,
    TestWriteToSilver,
    TestWriteTable,
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
