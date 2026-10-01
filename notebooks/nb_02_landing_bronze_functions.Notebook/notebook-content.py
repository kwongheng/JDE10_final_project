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

# ## nb_02_landing_bronze_functions
# Download files from kaggle to landing zone and ingest them into bronze schema
# Since we are just downloading and loading into bronze schema a single notebook will do
# as they handle the same datasets.

# CELL ********************

# Install kagglehub module\
!pip install kagglehub --quiet

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Imports

# CELL ********************

from typing import Dict, List, Optional

import os
import kagglehub
import pandas as pd
from pyspark.sql import functions as F, DataFrame, Row, SparkSession
from pyspark.sql.utils import AnalysisException
import hashlib
from pyspark.sql.types import StringType
from delta.tables import DeltaTable
from datetime import datetime

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Define functions to download from kaggle to landing

# CELL ********************

def get_active_configs(spark) -> List[Row]:
    """Fetch active dataset configs from dbo.config_datasets.

    Raises ValueError if no active configs are found — there's nothing
    meaningful to do downstream if this table is empty or misconfigured.
    """
    query = """
        SELECT *
        FROM dbo.config_datasets
        WHERE is_active = true;
    """
    configs = spark.sql(query).collect()
    if not configs:
        raise ValueError("No active rows found in dbo.config_datasets — nothing to download.")
    return configs

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def __get_active_configs(spark) -> List[Row]:
    """Fetch active dataset configs from dbo.config_datasets.

    Raises ValueError if no active configs are found — there's nothing
    meaningful to do downstream if this table is empty or misconfigured.
    """
    query = """
        SELECT table_name, source_file, kaggle_dataset
        FROM dbo.config_datasets
        WHERE is_active = true;
    """
    configs = spark.sql(query).collect()
    if not configs:
        raise ValueError("No active rows found in dbo.config_datasets — nothing to download.")
    return configs

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def group_by_kaggle_dataset(configs: List[Row]) -> Dict[str, List[Row]]:
    """Group config rows by kaggle_dataset so each dataset is downloaded only once,
    even if multiple source_files come from the same dataset."""
    datasets_needed: Dict[str, List[Row]] = {}
    for row in configs:
        datasets_needed.setdefault(row.kaggle_dataset, []).append(row)
    return datasets_needed

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def download_kaggle_dataset(kaggle_dataset: str) -> str:
    """Download a Kaggle dataset via kagglehub.

    Raises RuntimeError with a clear message on auth/network failure,
    rather than letting kagglehub's raw exception surface.
    """
    try:
        download_path = kagglehub.dataset_download(kaggle_dataset)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Kaggle dataset '{kaggle_dataset}'. "
            f"Check that Kaggle credentials are configured for this environment. "
            f"Original error: {e}"
        ) from e

    if not os.path.isdir(download_path):
        raise RuntimeError(
            f"kagglehub reported success but download path does not exist: {download_path}"
        )
    return download_path

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def build_file_lookup(download_path: str) -> Dict[str, str]:
    """Build a filename -> local path lookup for every file in the downloaded dataset."""
    available_files: Dict[str, str] = {}
    for root, _dirs, files in os.walk(download_path):
        for file_name in files:
            available_files[file_name] = os.path.join(root, file_name)

    if not available_files:
        print(f"⚠️ No files found under {download_path}")
    return available_files


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def save_file_to_landing(
    local_file: str,
    source_file: str,
    table_name: str,
    landing_path: str,
) -> bool:
    """Read a local CSV and write it to the landing zone.

    Returns True on success, False on any handled failure (bad CSV,
    empty file, write error) so the caller can track and report it
    without crashing the whole run over one bad file.
    """
    try:
        df = pd.read_csv(local_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
        print(f"⚠️ Could not read '{local_file}' for table '{table_name}': {e} — skipping.")
        return False

    if df.empty:
        print(f"⚠️ '{source_file}' is empty (0 rows) for table '{table_name}' — skipping.")
        return False

    lakehouse_file_path = f"{landing_path}/{source_file}"
    try:
        df.to_csv(lakehouse_file_path, index=False)
    except OSError as e:
        print(f"⚠️ Failed to write '{source_file}' to landing zone: {e} — skipping.")
        return False

    print(f"✅ Saved {source_file} → landing/{source_file} (for table '{table_name}')")
    return True

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Define functions for landing into bronze

# CELL ********************

def has_file_content_changed(
    spark,
    landing_path: str,
    manifest_table: str = "bronze._file_manifest",
    max_bytes: int = 1024 * 1024 * 200,  # 200 MB cap for the head-read; adjust as needed
) -> dict:
    """
    Checks whether the file at `landing_path` has different content than the
    last time this function recorded it, using a full-content SHA-256 hash.
    Safe to call for any landing path -- each path/hash pair is tracked
    independently in the shared manifest table.

    Does NOT skip based on filename or LastModified -- only on actual byte
    content, so a re-upload of identical data is correctly treated as
    "no change" even though its timestamp differs.

    Parameters
    ----------
    spark : SparkSession
    landing_path : str
        Path to the file in the lakehouse (e.g. "Files/landing/mydata.csv").
    manifest_table : str
        Delta table used to track previously-seen (path, hash) pairs.
        Shared across all landing paths -- rows are distinguished by file_path.
    max_bytes : int
        Cap on how many bytes are read for hashing (protects against
        accidentally hashing a huge file in memory). Increase if your
        files legitimately exceed this and you need full-content hashing.

    Returns
    -------
    dict with keys:
        changed        : bool  -- True if content is new/different, False if identical to last run
        file_hash      : str   -- the computed SHA-256 hash
        landing_path   : str   -- echoed back for convenience
        reason         : str   -- human-readable explanation, for logging
    """
    # Read file bytes and hash them
    raw = notebookutils.fs.head(landing_path, max_bytes)
    file_bytes = raw.encode() if isinstance(raw, str) else raw
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    manifest_exists = spark.catalog.tableExists(manifest_table)

    already_seen = False
    if manifest_exists:
        already_seen = (
            spark.table(manifest_table)
            .filter((F.col("file_path") == landing_path) & (F.col("file_hash") == file_hash))
            .count() > 0
        )

    if already_seen:
        return {
            "changed": False,
            "file_hash": file_hash,
            "landing_path": landing_path,
            "reason": f"Content unchanged for '{landing_path}' since last recorded ingestion.",
        }

    # New or changed content -- record it
    manifest_row = spark.createDataFrame(
        [(landing_path, file_hash)],
        schema="file_path string, file_hash string",
    ).withColumn("detected_at", F.current_timestamp())

    if manifest_exists:
        manifest_row.write.format("delta").mode("append").saveAsTable(manifest_table)
    else:
        manifest_row.write.format("delta").saveAsTable(manifest_table)

    return {
        "changed": True,
        "file_hash": file_hash,
        "landing_path": landing_path,
        "reason": f"New or changed content detected for '{landing_path}'.",
    }

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def _log_schema_drift(
    spark: SparkSession,
    bronze_table: str,
    schema_log_table: str,
    incoming_cols: list[str],
    metadata_cols: set[str],
) -> None:
    """
    Detect and log schema drift between an incoming dataset and an existing
    bronze table.

    Compares incoming columns with the current bronze table schema,
    excluding metadata columns. Any newly added or missing columns are
    recorded in the schema log table and reported in the notebook output.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    bronze_table : str
        Target bronze Delta table being ingested.
    schema_log_table : str
        Delta table used to store schema drift events.
    incoming_cols : list[str]
        Columns present in the incoming dataset.
    metadata_cols : set[str]
        Metadata columns excluded from drift detection.
    """
    if not spark.catalog.tableExists(bronze_table):
        return  # first-ever load for this table, nothing to compare against

    existing_cols = set(spark.table(bronze_table).columns) - metadata_cols
    incoming_cols = set(incoming_cols)

    new_cols = incoming_cols - existing_cols
    dropped_cols = existing_cols - incoming_cols

    if not new_cols and not dropped_cols:
        return  # no drift

    drift_row = spark.createDataFrame(
        [
            (
                datetime.utcnow().isoformat(),
                bronze_table,
                ",".join(sorted(new_cols)) if new_cols else None,
                ",".join(sorted(dropped_cols)) if dropped_cols else None,
            )
        ],
        schema="""
            detected_at string,
            table_name string,
            new_columns string,
            dropped_columns string
        """,
    )

    if spark.catalog.tableExists(schema_log_table):
        drift_row.write.format("delta").mode("append").saveAsTable(
            schema_log_table
        )
    else:
        drift_row.write.format("delta").saveAsTable(schema_log_table)

    if new_cols:
        print(
            f"[{bronze_table}][SCHEMA DRIFT] "
            f"New columns detected: {new_cols}"
        )

    if dropped_cols:
        print(
            f"[{bronze_table}][SCHEMA DRIFT][WARNING] "
            f"Columns missing from this load: {dropped_cols}"
        )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Internal helper functions used by bronze ingestion logic

# MARKDOWN ********************

# ### Full ingestion logic for bronze tables
# Ensure idempotent when writing bronze tables
# * check for schema drifts
# * adds metadata to row updates
# * deduplicate rows based on column hash
# * default schema log in `bronze._bronze_schema_log`


# CELL ********************

def ingest_to_bronze(
    spark: SparkSession,
    landing_path: str,
    bronze_table: str,
    hash_columns: list[str],
    schema_log_table: str = "bronze._bronze_schema_log",
    csv_options: dict | None = None,
) -> dict:
    """
    Ingest a CSV file into a bronze Delta table using an idempotent,
    insert-only pattern.

    The ingestion process is metadata-driven and applies the same logic
    across source files, including schema drift detection, type
    reconciliation, row-level hashing, batch deduplication, and Delta
    merge operations.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    landing_path : str
        Path to the source CSV file.
    bronze_table : str
        Target bronze Delta table name.
    hash_columns : list[str]
        Columns used to generate a deterministic row hash that defines
        row identity for the source dataset.
    schema_log_table : str, default="bronze._bronze_schema_log"
        Delta table used to record schema drift events.
    csv_options : dict | None, optional
        Additional CSV reader options. If not provided, defaults to
        {"header": True, "inferSchema": True}.

    Returns
    -------
    dict
        Dictionary containing ingestion metrics:

        - bronze_table : str
        - rows_read : int
        - rows_deduped : int
        - rows_inserted : int
        - total_rows_now : int

    Raises
    ------
    ValueError
        If required hash columns are missing from the source data or if
        no configured hash columns are available for hash generation.
    Exception
        Re-raises any ingestion failure after logging contextual details.
    """
    metadata_cols = {"_source_file", "_ingestion_timestamp", "_row_hash",}
    csv_options = csv_options or {"header": True, "inferSchema": True,}

    try:
        spark.conf.set(
            "spark.databricks.delta.schema.autoMerge.enabled",
            "true",
        )

        # ---- Read raw CSV ---------------------------------------------------
        reader = spark.read

        for k, v in csv_options.items():
            reader = reader.option(k, v)

        raw_df = reader.csv(landing_path)
        rows_read = raw_df.count()

        print(
            f"[{bronze_table}] Rows read from "
            f"'{landing_path}': {rows_read}"
        )

        # ---- Schema drift detection + logging ------------------------------
        _log_schema_drift(
            spark,
            bronze_table,
            schema_log_table,
            raw_df.columns,
            metadata_cols,
        )

        # ---- Metadata columns + explicit-column row hash -------------------
        available_hash_cols = [c for c in hash_columns if c in raw_df.columns]
        missing_hash_cols = [c for c in hash_columns if c not in raw_df.columns]

        if missing_hash_cols:
            raise ValueError(
                f"[{bronze_table}][WARNING] Expected hash columns "
                f"not present in this load: {missing_hash_cols}"
            )

        if not available_hash_cols:
            raise ValueError(
                f"[{bronze_table}] None of the configured "
                f"hash_columns {hash_columns} are present in the "
                f"source data -- cannot compute a meaningful row "
                f"hash. Check hash_columns config."
            )

        df = (
            raw_df
            .withColumn("_source_file", F.input_file_name())
            .withColumn("_ingestion_timestamp", F.current_timestamp())
            .withColumn(
                "_row_hash",
                F.sha2(
                    F.concat_ws("||", *[F.col(c).cast(StringType()) for c in available_hash_cols]),
                    256,
                ),
            )
        )

        # ---- Collapse exact duplicates within this batch -------------------
        pre_dedup_count = df.count()

        df = df.dropDuplicates(["_row_hash"])

        post_dedup_count = df.count()
        rows_deduped = pre_dedup_count - post_dedup_count

        if rows_deduped > 0:
            print(
                f"[{bronze_table}][DEDUP] Collapsed "
                f"{rows_deduped} exact duplicate row(s) "
                f"within this batch "
                f"({pre_dedup_count} -> {post_dedup_count})."
            )

        # ---- Idempotent write ----------------------------------------------
        if not spark.catalog.tableExists(bronze_table):
            df.write.format("delta").saveAsTable(bronze_table)

            rows_inserted = df.count()
            total_rows_now = rows_inserted

            print(
                f"[{bronze_table}] Created table with "
                f"{rows_inserted} row(s)."
            )

        else:
            bronze_dt = DeltaTable.forName(
                spark,
                bronze_table,
            )

            before_count = spark.table(bronze_table).count()

            (
                bronze_dt.alias("t")
                .merge(df.alias("s"), "t._row_hash = s._row_hash")
                .whenNotMatchedInsertAll()
                .execute()
            )

            total_rows_now = spark.table(bronze_table).count()
            rows_inserted = total_rows_now - before_count

            print(
                f"[{bronze_table}] Merge complete. "
                f"Inserted {rows_inserted} new row(s). "
                f"Total rows now: {total_rows_now}."
            )

        return {
            "bronze_table": bronze_table,
            "rows_read": rows_read,
            "rows_deduped": rows_deduped,
            "rows_inserted": rows_inserted,
            "total_rows_now": total_rows_now,
        }

    except Exception as e:
        print(
            f"[{bronze_table}][FAILED] Ingestion from "
            f"'{landing_path}' failed: "
            f"{type(e).__name__}: {e}"
        )
        raise

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
