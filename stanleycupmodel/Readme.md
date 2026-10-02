# Stanley Cup Prediction: Semantic Model & Power BI Report

A Microsoft Fabric / Power BI project that presents the output of an **NHL Stanley Cup prediction model**. The model was trained only on 2000–2020 data and forecasts the **2020-21 season**: championship odds, the top projected scorers, and a backtest of how well past forecasts ranked the eventual champion. The report ends with a "reveal" comparing the predictions to what actually happened.

| Item | Folder | Purpose |
|------|--------|---------|
| **Semantic model** | `Betting Semantic Model.SemanticModel/` | Direct Lake model over the Fabric lakehouse (`NHL_db`): 9 tables, no measures or relationships |
| **Report** | `Betting Final Report.Report/` | 3-page Power BI report (1920×1080) connected live to the semantic model |

Both items are stored in **Fabric Git integration format** (TMDL for the model, PBIR for the report), so they diff cleanly and can be synced with a Fabric workspace.

---

## Repository structure

```
stanleycupmodel/
├── Readme.md
├── Betting Semantic Model.SemanticModel/
│   ├── .platform                     # Fabric item metadata
│   ├── definition.pbism              # Semantic model settings (v4.2)
│   └── definition/
│       ├── database.tmdl             # Compatibility level 1606
│       ├── model.tmdl                # Culture (en-US), table refs, query order
│       ├── expressions.tmdl          # Direct Lake source expression (OneLake)
│       └── tables/
│           ├── actual_2020_21_results.tmdl
│           ├── best_player_forecast_2021.tmdl
│           ├── champion_validation_history.tmdl
│           ├── championship_odds_2021.tmdl
│           ├── dim_team.tmdl
│           ├── gold_goalie_season.tmdl
│           ├── gold_skater_season.tmdl
│           ├── gold_team_season.tmdl
│           └── skater_validation_history.tmdl
└── Betting Final Report.Report/
    ├── .platform
    ├── definition.pbir               # Live connection to the semantic model
    ├── definition/
    │   ├── report.json, version.json, pages/pages.json
    │   └── pages/<page-id>/page.json + visuals/<visual-id>/visual.json
    └── StaticResources/SharedResources/BaseThemes/Fluent2-CY26SU09.json
```

---

## Semantic model

### Data source

- **Storage mode:** Direct Lake (no import refresh needed)
- **Source expression:** `DirectLake - NHL_db`, which points at a Fabric lakehouse via OneLake (`onelake.dfs.fabric.microsoft.com`)
- **Medallion layers:** the model reads eight `gold` tables and one `silver` table (`dim_team`). The lakehouse also holds bronze and silver tables (games, team/player/goalie/skater stats, facts and dimensions) and betting-prediction tables from the sibling SeasonalOdds project. These were removed from this model and are not part of it.

### Tables

**Prediction outputs** (what the model forecast)

| Table | Source | Description |
|-------|--------|-------------|
| `championship_odds_2021` | `gold.championship_odds_2021` | 2020-21 championship forecast per team: `proj_points_pct`, `projected_rank`, `championship_probability`. |
| `best_player_forecast_2021` | `gold.best_player_forecast_2021` | Projected top scorers for 2020-21: `proj_points_per_game`, `proj_points_82gp`, `projected_rank`, `seasons_used`, plus `actual_next_season_points`. |

**Backtesting / validation**

| Table | Source | Description |
|-------|--------|-------------|
| `champion_validation_history` | `gold.champion_validation_history` | Historical forecasts per team and "as-of" season (`proj_points_pct`, `proj_goal_diff`, `projected_rank`), with `actually_won_next_season` (1 = champion) to score the model. |
| `skater_validation_history` | `gold.skater_validation_history` | Historical skater forecasts (`projected_rank`, `proj_points_82gp`) alongside `actual_next_season_points`. |

**Actual results**

| Table | Source | Description |
|-------|--------|-------------|
| `actual_2020_21_results` | `gold.actual_2020_21_results` | Real 2020-21 outcomes as `category` / `winner_name` / `team` / `detail` rows (e.g. Stanley Cup Champion, Hart Trophy). |

**Season aggregates** (model inputs and context)

| Table | Source | Description |
|-------|--------|-------------|
| `gold_team_season` | `gold.team_season` | Team-season stats: W/L/OTL, points, GF/GA, shots, power play and penalty kill %, faceoff %, give/take diff, per-game rates, and `is_champion`. Includes denormalised team name fields. |
| `gold_skater_season` | `gold.skater_season` | Skater-season stats: GP, goals, assists, points, shots, hits, PIM, +/-, PP goals/assists, TOI, age, points per game, points per 60. |
| `gold_goalie_season` | `gold.goalie_season` | Goalie-season stats: GP, W/L, shots against, saves, GA, TOI, age, save %, GAA per 60. |
| `dim_team` | `silver.dim_team` | Team dimension: `team_id`, `franchiseId`, `shortName`, `teamName`, `abbreviation`, `full_name`. |

### Relationships and measures

- **Relationships:** none defined. Every visual uses columns from a single table.
- **Measures / calculated columns:** none defined. The report relies on implicit aggregations (sum, average) of raw columns.

---

## Power BI report

**Connection:** live connection (`byConnection`) to the `Betting Semantic Model` in the `JDE10` workspace.
**Theme:** Fluent 2 (CY26SU09). **Page size:** 1920×1080, fit to page.

| # | Page | Visuals | What it shows |
|---|------|---------|---------------|
| 1 | **Predicted Winners** *(opens first)* | Clustered bar chart, table | **Projected Championship Odds, 2020-21 Season**: championship probability by team (`championship_odds_2021`). **Top 10 Projected Scorers, 2020-21 Season**: rank, name, team and projected 82-game points (`best_player_forecast_2021`, filtered to `projected_rank <= 10`). |
| 2 | **How good is the model?** | Line chart, KPI card, 2 text boxes | **Model Calibration: Rank Given to the Eventual Champion, 2009–2018**: for each season cutoff year, the model's projected rank of the team that actually won (filtered to `actually_won_next_season = 1`; 1 = best). The card shows the **average rank of the actual eventual champion**. Text boxes note that *lower is better* and that playoff records in the source `games.csv` are only complete from 2010-11 onwards, so backtesting starts in 2009 (predicting that season). |
| 3 | **The Reveal** | 4 callout cards, 5 text boxes | **"Trained only on 2000–2020 data — did the bet pay off?"** Side-by-side comparison. *Our Prediction*: top-ranked team in `championship_odds_2021`. *Reality*: the actual Stanley Cup Champion from `actual_2020_21_results`. *Best Player*: top-ranked player in `best_player_forecast_2021`. *Reality*: the actual Hart Trophy (MVP) winner. |

> Page order and the default page are set in `definition/pages/pages.json`.

---

## Getting started

### Prerequisites

- Microsoft Fabric capacity with a workspace and a lakehouse (`NHL_db`) containing the `gold` and `silver` tables above
- Permission to deploy semantic models and reports in the target workspace
- Optional for local editing: Power BI Desktop (with the *TMDL view* / *PBIR* preview features enabled) or VS Code with a TMDL extension

### Deploy via Fabric Git integration

1. Push this folder to a Git repo (Azure DevOps or GitHub).
2. In your Fabric workspace, open **Workspace settings → Git integration** and connect the repo and branch.
3. Sync. Fabric creates the `Betting Semantic Model` and `Betting Final Report` items.
4. **Update the connections** (see below).

### Things to update after cloning

- **`expressions.tmdl`**: the OneLake URL contains the original workspace and lakehouse GUIDs. Replace them with your own.
- **`definition.pbir`**: the `byConnection` string references the original workspace (`JDE10`) and semantic model ID. Rebind the report to your semantic model (*Edit connection*), or switch to a `byPath` reference.
- **`.platform` logical IDs**: Fabric manages these. Leave them unchanged when syncing to the same workspace.

---

## Known issues & suggestions

These came up while reviewing the project:

1. **No relationships.** `gold_team_season`, `championship_odds_2021`, `champion_validation_history` and `dim_team` all share `team_id`, and the player tables share `player_id`. Adding relationships (e.g. `dim_team` → the team tables) would allow cross-filtering and slicers across tables.
2. **No explicit measures.** Visuals use implicit aggregations. Defining measures such as *Avg Champion Rank*, *Top-N Hit Rate* (champion in the top 3 or top 5) or *Brier score* would make the model's performance easier to audit and reuse.
3. **Implicit sum on a rank.** The calibration line chart sums `projected_rank` per year. This only gives the right answer because the champion filter leaves one row per season. Switching to average (or min) is safer if the filter changes.
4. **Hard-coded 2020-21 scope.** Table and column names (`*_2021`, `actual_2020_21_results`) and the report titles are tied to a single season, so rolling the model forward means renaming tables and editing visuals.
5. **Card labels may be static.** The *Reveal* cards use filters on `projected_rank = 1` and on `category` values such as `'Stanley Cup Champion'` and `'Hart Trophy (MVP)'`. A reference label inside one card also contains a literal player name, so check that it still updates if the data changes.
6. **Model size.** The `gold_*_season` tables include many columns the report never uses. They may be intended for future pages or can be trimmed.
7. **Compatibility level.** This model is at 1606, while the sibling SeasonalOdds model is at 1705. Fine on its own, but worth aligning if you merge them.

---

## Related project

This model uses the same lakehouse (`NHL_db`) as the **SeasonalOdds NHL betting analytics** project (moneyline, puck line and totals predictions). That project's README is `README.md`.

---

## Disclaimer

This project is for **analytics and educational purposes only**. Model probabilities and backtest results do not guarantee future results and are not betting or financial advice. Please gamble responsibly and follow the laws in your jurisdiction.
