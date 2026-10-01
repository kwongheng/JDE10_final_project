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

# ## nb_02_landing_to_bronze
# Download files from kaggle to landing zone and ingest them into bronze schema
# Since we are just downloading and loading into bronze schema a single notebook will do
# as they handle the same datasets.

# MARKDOWN ********************

# ### Load functions for landing and bronze activities

# CELL ********************

%run nb_02_landing_bronze_functions

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
from pyspark.sql import Row, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Define variables

# CELL ********************

WORKSPACE: str = "DeltaTables"
LAKEHOUSE: str = "NHL_data"

ABFSS_BASE_PATH: str = f"abfss://{WORKSPACE}@onelake.dfs.fabric.microsoft.com"
LANDING_PATH: str = f"{ABFSS_BASE_PATH}/{LAKEHOUSE}.Lakehouse/Files/landing"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Run: Kaggle → landing

# CELL ********************

configs = get_active_configs(spark)
datasets_needed = group_by_kaggle_dataset(configs)

failed_landings: List[str] = []

for kaggle_dataset, rows in datasets_needed.items():
    print(f"⬇️ Downloading dataset: {kaggle_dataset}")
    try:
        download_path = download_kaggle_dataset(kaggle_dataset)
    except RuntimeError as e:
        print(f"❌ {e}")
        failed_landings.extend(row.bronze_table_name for row in rows)
        continue
    print(f"Dataset downloaded to: {download_path}")

    available_files = build_file_lookup(download_path)

    for row in rows:
        source_file = row.source_file
        table_name = row.bronze_table_name

        if source_file not in available_files:
            print(f"⚠️ {source_file} not found for table '{table_name}' — skipping.")
            failed_landings.append(table_name)
            continue

        local_file = available_files[source_file]
        if not save_file_to_landing(local_file, source_file, table_name, LANDING_PATH):
            failed_landings.append(table_name)

if failed_landings:
    print(f"\n⚠️ {len(failed_landings)} table(s) failed to land: {failed_landings}")
else:
    print("\n🏁 Landing zone load complete.")

# Fail loudly only if EVERYTHING failed — a partial landing still lets
# nb_01's own bronze step (and downstream silver notebooks) proceed for
# whatever did land. Tighten this to `if failed_landings:` if you'd rather
# treat any single failed table as fatal.
if len(failed_landings) == len(configs):
    raise RuntimeError("All configured datasets failed to land — aborting before bronze load.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Run: landing → bronze

# CELL ********************

# Re-query in case config_datasets changed since the landing step above.
# LANDING_PATH is reused from the earlier cell in this notebook's session.
configs = get_active_configs(spark)

for row in configs:
    table_name = row.bronze_table_name
    hash_columns = row.hash_columns
    csv_option = row.csv_option or None
    source_file = row.source_file

    file_path = f"{LANDING_PATH}/{source_file}"
    print(f"\n📥 Loading {source_file} → {table_name}")

    # preflight checks for content change
    result = has_file_content_changed(spark, file_path)
    if not result["changed"]:
        print(f"Skipping (no change): {file_path}")

    else: 
        print(f"Ingesting: {file_path}")

        result = (
            ingest_to_bronze(
                spark,
                file_path,
                table_name,
                hash_columns=hash_columns,
                csv_options=csv_option,
            )
        )        

print("\n🏁 Bronze load complete.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
