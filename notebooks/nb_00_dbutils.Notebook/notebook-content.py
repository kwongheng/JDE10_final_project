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

# ## nb_00_dbutils
# 
# Contains common modules used for transformation and 
# 
# including ingesting/creating delta tables using latest correct updated timestamp

# MARKDOWN ********************

# ### Imports

# CELL ********************

from typing import Any
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window, WindowSpec
from delta.tables import DeltaTable

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: Basic table loading function
# This is a generic function that can used to load any type of table, bronze, silver, gold, etc

# CELL ********************

def load_table(
    spark,
    table_name: str,
    columns: list[str] | None = None,
) -> DataFrame:
    """
    Load a Fabric/Spark table into a PySpark DataFrame.

    Optionally selects a subset of columns to reduce the amount of data
    carried through downstream transformations.

    Args:
        spark: Active SparkSession used to execute the query.
        table_name: Fully qualified table name to load.
        columns: Optional list of column names to select. If None,
            all columns are returned.

    Returns:
        A PySpark DataFrame containing the requested table data. If
        ``columns`` is provided, only the specified columns are included
        in the returned DataFrame.

    Raises:
        AnalysisException: If the table does not exist or one or more
            specified columns are invalid.
    """

    query = f"""
        SELECT *
        FROM {table_name}
    """
    df = spark.sql(query)
    print(f"✅ Loaded {table_name}")

    if columns is None:
       return df
    
    return df.select(*columns)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: Read latest records from bronze for ingestion to silver

# CELL ********************

def read_new_bronze_rows(
    spark,
    bronze_table: str,
    silver_table: str,
    columns: list[str] | None = None,
    watermark_col: str = "_bronze_ingested_at",
) -> DataFrame:
    """
    Reads bronze rows newer than the latest bronze timestamp already
    processed into silver. On the first run (no silver table, or no
    watermark yet), returns all of bronze.
    """
    df: DataFrame = spark.table(bronze_table)
    if columns:
        # using dict.fromkeys removed duplicated columns 
        # just in case metadata is included
        select_cols = list(dict.fromkeys(
            [*columns, "_ingestion_timestamp", "_row_hash"]
        ))
        df = df.select(*select_cols)

    if spark.catalog.tableExists(silver_table):
        last_seen: DataFrame = (spark.table(silver_table)
                     .agg(F.max(watermark_col))
                     .first()[0])
        if last_seen is not None:
            df = df.filter(F.col("_ingestion_timestamp") > last_seen)
            print(f"[{bronze_table}] Reading rows after watermark {last_seen}")
            return df

    print(f"[{bronze_table}] No watermark found, reading full table")
    return df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: Keep only the latest row based on latest timestamp

# CELL ********************

def keep_latest_per_key(
    df: DataFrame,
    primary_keys: list[str],
) -> DataFrame:
    """
    Keeps one row per business key (the one with the newest
    `_ingestion_timestamp`), drops bronze-only columns, and renames
    `_ingestion_timestamp` to `_bronze_ingested_at` for silver.

    Args:
        df: Bronze-derived dataframe containing the business key columns
            and `_ingestion_timestamp`.
        primary_keys: Columns identifying the entity (e.g. ["player_id"]).
    """

    order: list[Column] = [F.col("_ingestion_timestamp").desc()]
    if "_row_hash" in df.columns:
        order.append(F.col("_row_hash").asc())  # deterministic tiebreak when timestamps tie

    w: WindowSpec = Window.partitionBy(*primary_keys).orderBy(*order)

    return (
        df.withColumn("_rn", F.row_number().over(w))
          .filter(F.col("_rn") == 1)
          .drop("_rn")
          .withColumnRenamed("_ingestion_timestamp", "_bronze_ingested_at")
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: validate no nulls in required columns

# CELL ********************

def validate_no_nulls(
    df: DataFrame,
    columns: list[str],
) -> DataFrame:
    """
    Validate that specified columns contain no null values.

    Args:
        df: Input DataFrame.
        columns: Column names that must not contain null values.

    Returns:
        The original DataFrame if all specified columns pass validation.

    Raises:
        RuntimeError: If one or more required columns contain null values.
    """
    null_counts = (
        df.select([
            F.count(F.when(F.col(column_name).isNull(), 1)).alias(column_name)
            for column_name in columns
        ])
        .first()
    )

    for column_name in columns:
        if (null_count := null_counts[column_name]) > 0:
            raise RuntimeError(
                f"Column '{column_name}' contains "
                f"{null_count} null value(s)."
            )

    return df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: validate column against valid values

# CELL ********************

def validate_column_values(
    df: DataFrame,
    column: str,
    values: tuple[str],
) -> DataFrame:
    """
    Validate that a column contains only allowed non-null values.

    Args:
        df: Input DataFrame.
        column: Column to validate.
        values: Tuple of permitted values for the column.

    Returns:
        The original DataFrame if all values pass validation.

    Raises:
        RuntimeError: If the column contains null values or values not
            included in the allowed set.
    """
    invalid_count = (
        df.filter(
            F.col(column).isNull() |
            (~F.col(column).isin(*values))
        )
        .count()
    )

    if invalid_count > 0:
        raise RuntimeError(
            f"Column '{column}' contains {invalid_count} "
            f"invalid or null value(s)."
        )

    return df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: validate Primary key(s)

# CELL ********************

def validate_primary_keys(
    df: DataFrame,
    keys: list[str],
) -> DataFrame:
    """
    Validate that the specified primary key columns uniquely identify each row.

    This validation is intended to run before writing data to downstream
    tables. A failure indicates that the assumed primary key is not unique
    and that the pipeline should stop rather than persist invalid data.

    Args:
        df: Input DataFrame.
        keys: Column names that together define the primary key.

    Returns:
        The original DataFrame if the primary key constraint is satisfied.

    Raises:
        RuntimeError: If duplicate primary key values are found.
    """
    duplicate_count = (
        df.groupBy(*keys)
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    if duplicate_count > 0:
        raise RuntimeError(
            f"Primary key violation: found {duplicate_count} "
            f"duplicate key value(s) for {keys}."
        )

    return df


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: (internal) find foreign key violations

# CELL ********************

def _find_fk_violations(
    df: DataFrame,
    ftable: DataFrame,
    keys: list[str],
) -> DataFrame:
    """
    Identify rows whose foreign key values do not exist in the referenced table.

    Args:
        df: DataFrame containing the foreign key column(s) to validate.
        ftable: Referenced DataFrame containing the valid key values.
        keys: Join key definition. Provide either:
            - One column name for matching columns with the same name.
            - Two column names where keys[0] is the foreign key column
              in ``df`` and keys[1] is the referenced key column in
              ``ftable``.

    Returns:
        DataFrame containing only rows that violate the foreign key
        constraint.

    Raises:
        RuntimeError: If more than two key values are provided.
    """
    if len(keys) == 1:
        return df.join(
            ftable,
            on=keys[0],
            how="left_anti",
        )

    if len(keys) == 2:
        return df.join(
            ftable,
            df[keys[0]] == ftable[keys[1]],
            how="left_anti",
        )

    raise RuntimeError(
        f"Expected 1 or 2 key values, received {len(keys)}: {keys}"
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: drop rows with foreign key violations

# CELL ********************

def drop_foreign_key_violations(
    df: DataFrame,
    ftable: DataFrame,
    keys: list[str],
) -> DataFrame:
    """
    Remove rows whose foreign key values do not exist in the referenced table.

    Args:
        df: DataFrame containing the foreign key column(s) to validate.
        ftable: Referenced DataFrame containing valid key values.
        keys: Join key definition. Provide either:
            - One column name for matching columns with the same name.
            - Two column names where keys[0] is the foreign key column
              in ``df`` and keys[1] is the referenced key column in
              ``ftable``.

    Returns:
        DataFrame containing only rows that satisfy the foreign key
        constraint.
    """
    if len(keys) == 1:
        return df.join(
            ftable,
            on=keys[0],
            how="left_semi",
        )

    return df.join(
        ftable,
        df[keys[0]] == ftable[keys[1]],
        how="left_semi",
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: validate Foreign key(s)

# CELL ********************

def validate_foreign_keys(
    df: DataFrame,
    ftable: DataFrame,
    keys: list[str],
) -> DataFrame:
    """
    Validate that all foreign key values exist in the referenced table.

    Args:
        df: DataFrame containing the foreign key column(s) to validate.
        ftable: Referenced DataFrame containing valid key values.
        keys: Join key definition. Provide either:
            - One column name for matching columns with the same name.
            - Two column names where keys[0] is the foreign key column
              in ``df`` and keys[1] is the referenced key column in
              ``ftable``.

    Returns:
        The original DataFrame if all foreign key values are valid.

    Raises:
        RuntimeError: If foreign key violations are found.
    """
    violations = _find_fk_violations(df, ftable, keys)

    if (violation_count := violations.count()) > 0:
        sample_rows = violations.limit(5).collect()

        raise RuntimeError(
            f"Foreign key violation: found {violation_count} row(s) "
            f"without a matching reference for keys {keys}. "
            f"Example violations: {sample_rows}"
        )

    return df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: Write to SCD 1 silver table

# CELL ********************

def write_to_silver(spark, df: DataFrame, silver_table: str, primary_keys: list[str]) -> None:
    """
    Upserts a cleaned df into silver (SCD1): one row per business key.
    df must contain the business key columns and `_ingestion_timestamp`.
    """
    try:
        if df.isEmpty():
            print(f"[{silver_table}] No rows to write, skipping.")
            return

        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

        if not spark.catalog.tableExists(silver_table):
            df.write.format("delta").saveAsTable(silver_table)
            print(f"[{silver_table}] Created table.")
            return

        merge_cond = " AND ".join(f"t.{k} = s.{k}" for k in primary_keys)

        (
            DeltaTable.forName(spark, silver_table).alias("t")
            .merge(df.alias("s"), merge_cond)
            .whenMatchedUpdateAll(
                condition="t._bronze_ingested_at IS NULL "
                          "OR s._bronze_ingested_at > t._bronze_ingested_at"
            )
            .whenNotMatchedInsertAll()
            .execute()
        )
        print(f"[{silver_table}] Merge complete.")

    except Exception as e:
        print(f"[{silver_table}][FAILED] {type(e).__name__}: {e}")
        raise

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Function: write to table

# CELL ********************

def write_table(
    df: DataFrame,
    table_name: str,
) -> None:
    """
    Write a DataFrame to a Delta table, replacing any existing table
    data and schema.

    Args:
        df: DataFrame to write.
        table_name: Target table name.

    Returns:
        None.
    """
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
