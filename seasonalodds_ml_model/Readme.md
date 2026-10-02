# NHL Betting Analytics: Semantic Model & Power BI Report

A Microsoft Fabric / Power BI project that visualises the output of the **SeasonalOdds NHL betting ML model**. It has two parts:

| Item | Folder | Purpose |
|------|--------|---------|
| **Semantic model** | `NHL_Betting_Analytics.SemanticModel/` | Direct Lake model over the Fabric lakehouse (`NHL_db`) with DAX measures and calculated columns |
| **Report** | `NHL_Betting_Analytics.Report/` | 5-page Power BI report (1920×1080) connected live to the semantic model |

Both items are stored in **Fabric Git integration format** (TMDL for the model, PBIR for the report), so they diff cleanly and can be synced with a Fabric workspace.

---

## Repository structure

```
seasonalodds_ml_model/
├── Readme.md
├── NHL_Betting_Analytics.SemanticModel/
│   ├── .platform                     # Fabric item metadata
│   ├── definition.pbism              # Semantic model settings (v4.2)
│   └── definition/
│       ├── database.tmdl             # Compatibility level 1705
│       ├── model.tmdl                # Culture (en-US), table refs, query order
│       ├── expressions.tmdl          # Direct Lake source expression (OneLake)
│       ├── relationships.tmdl        # 2 relationships
│       └── tables/
│           ├── betting_all_samples.tmdl
│           ├── betting_season_tabulation.tmdl
│           ├── betting_summary_unpivoted.tmdl
│           ├── top_3_teams_per_market.tmdl
│           ├── team_composite_2yr.tmdl
│           └── game_features.tmdl
└── NHL_Betting_Analytics.Report/
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
- **Medallion layers:** the model reads from the `gold` schema, plus one `silver` table (`game_features`). The lakehouse also holds bronze and silver tables (games, team/player/goalie/skater stats, dim/fact tables) and other gold tables (predictions, season rollups, championship odds). These were removed from this model and are not part of it.

### Tables

| Table | Source | Description |
|-------|--------|-------------|
| `betting_all_samples` | `gold.betting_all_samples` | **Core fact table.** One row per game-level bet candidate: game, teams, picked side and odds, model probability vs. bookmaker probability, confidence, expected value %, and outcome. |
| `betting_season_tabulation` | `gold.betting_season_tabulation` | Season summary per market: total games, qualified bets, wins and losses, no-bets (low confidence), selective accuracy % and skipped-games %. |
| `betting_summary_unpivoted` | `gold.betting_summary_unpivoted` | Unpivoted version of the season summary (`Category`, `Metric_Name`, `Metric_Group`, `Game_Count`) for stacked charts. |
| `top_3_teams_per_market` | `gold.top_3_teams_per_market` | Top 3 teams per market by qualified-bet win rate (`Rank`, `Team`, `Qualified_Bets`, `Winning_Bets`, `Win_Rate_Pct`). |
| `team_composite_2yr` | `gold.team_composite_2yr` | Two-year team composite metrics: GF/GA/goal diff per game, Corsi-for per game, PP%, PK%, save % proxy. |
| `game_features` | `silver.game_features` | Model feature set per game: rolling goals/shots/win-rate for home and away, back-to-back flags, differentials (shots, win rate, rest), and targets (`target_moneyline`, `target_over_5_5`, `target_over_6_5`, `target_puckline_home_plus1_5`). |

### Markets covered

- **Moneyline** (Home Win)
- **Puck Line** (Home +1.5)
- **Totals** (Over 5.5)

### Relationships

| From | To | Notes |
|------|----|-------|
| `betting_all_samples[picked_side]` | `top_3_teams_per_market[Team]` | Links picks to team rankings |
| `betting_summary_unpivoted[Category]` | `betting_season_tabulation[Category]` | **Bi-directional** cross-filtering |

`team_composite_2yr` and `game_features` are not related to other tables.

### Measures (`betting_all_samples`)

| Measure | Definition | Notes |
|---------|-----------|-------|
| `1. Total Qualified Bets` | `SUM(betting_season_tabulation[Qualified_Bets_Offered])` | Count of bets that passed the confidence filter |
| `2. Selective Accuracy %` | `AVERAGE(betting_season_tabulation[Selective_Accuracy_Pct])` | Accuracy on committed bets only |
| `3. Avg EV %` | `AVERAGE(betting_all_samples[expected_value_pct])` | Mean expected value |
| `4. High EV Bets Count` | `COUNTROWS` where `expected_value_pct >= 2.0` | Bets with at least a 2% edge |
| `Total Qualified Bets` | `SUM('gold.betting_season_tabulation'[Qualified_Bets_Offered])` | Legacy duplicate, see [Known issues](#known-issues) |

### Calculated columns

| Table | Column | Logic |
|-------|--------|-------|
| `betting_all_samples` | `EV_Tier` | 4 tiers: Massive Value (≥15%), High Value (5–15%), Moderate Value (2–5%), Low/Negative Edge (<2%) |
| `betting_all_samples` | `EV_Category` | 3 tiers: 🟢 Massive Edge (≥15%), 🟡 Good Edge (2–15%), 🔴 Low/No Edge (<2%) |
| `betting_all_samples` | `Bet_Category_Label` | Normalises `Category` (or falls back to `picked_side` text) into Moneyline / Puckline / Totals labels |
| `betting_all_samples` | `EV_Decimal` | `expected_value_pct / 100` |
| `top_3_teams_per_market` | `Win_Rate_Num` | Numeric version of `Win_Rate_Pct` (strips the `%`) |

---

## Power BI report

**Connection:** live connection (`byConnection`) to the `NHL_Betting_Analytics` semantic model in the `JDE10` workspace.
**Theme:** Fluent 2 (CY26SU09). **Page size:** 1920×1080, fit to page.

| # | Page | Visuals | What it shows |
|---|------|---------|---------------|
| 1 | **The Confidence Filter** | Two 100% stacked column charts | *Win or Loss*: correct vs. wrong predictions per market. *Committed Bets*: committed vs. skipped games. This shows how the confidence threshold trades volume for accuracy. |
| 2 | **Expected Value (EV%)** | Clustered column, clustered bar, donut | Model vs. book probability and EV by bet category, average EV by picked side, and the share of bets in each EV category. Page filters: `expected_value_pct` (advanced filter) and `picked_side` limited to `OVER 5.5`, `TBL +1.5`, `TBL`. |
| 3 | **Team Selection Insights** | Clustered bar | Average win rate of the top 3 teams per market (winning vs. qualified bets in the tooltip). |
| 4 | **KIV - Expected Value (EV%)** | Line + clustered column combo | Model probability vs. book probability by `bet_outcome`, with EV% in the tooltip. *KIV = "keep in view"*, a working or parked page. |
| 5 | **Expected Value EV%** | Line + clustered column combo, 3 text boxes | Average EV% by picked side, with model and book probability gap lines on the secondary axis. The text boxes are empty placeholders. |

> The report opens on page 5 (`activePageName`). Page order is set in `pages/pages.json`.

---

## Getting started

### Prerequisites

- Microsoft Fabric capacity with a workspace and a lakehouse (`NHL_db`) containing the `gold` and `silver` tables above
- Permission to deploy semantic models and reports in the target workspace
- Optional for local editing: Power BI Desktop (with the *TMDL view* / *PBIR* preview features enabled) or VS Code with a TMDL extension

### Deploy via Fabric Git integration

1. Push this folder to a Git repo (Azure DevOps or GitHub).
2. In your Fabric workspace, open **Workspace settings → Git integration** and connect the repo and branch.
3. Sync. Fabric creates the `NHL_Betting_Analytics` semantic model and report.
4. **Update the connections** (see below).

### Things to update after cloning

- **`expressions.tmdl`**: the OneLake URL contains the original workspace and lakehouse GUIDs. Replace them with your own.
- **`definition.pbir`**: the `byConnection` string references the original workspace (`JDE10`) and semantic model ID. Rebind the report to your semantic model (*File → Settings* or *Edit connection*), or switch to a `byPath` reference.
- **`.platform` logical IDs**: Fabric manages these. Leave them unchanged when syncing to the same workspace.

---

## Known issues

These were found while reviewing the project and are worth fixing:

1. **Broken measure reference:** `Total Qualified Bets` and the `EV_Tier` column reference `'gold.betting_season_tabulation'` / `'gold.betting_all_samples'`, but the tables are named `betting_season_tabulation` / `betting_all_samples`. These will error until the references are corrected. `Total Qualified Bets` also duplicates `1. Total Qualified Bets`, so it can simply be deleted.
2. **Text-typed percentages:** `Selective_Accuracy_Pct`, `Skipped_Games_Pct` and `Win_Rate_Pct` are strings (e.g. `"62.5%"`). `AVERAGE` over a string column in `2. Selective Accuracy %` will not work as intended. Convert them to numeric upstream, or add numeric columns like `Win_Rate_Num`.
3. **Measures sit on an unrelated table:** `1. Total Qualified Bets` and `2. Selective Accuracy %` aggregate `betting_season_tabulation` but live on `betting_all_samples`, which has no relationship to it. Consider moving them to their home table or a dedicated `_Measures` table.
4. **Hard-coded market labels:** `Bet_Category_Label` only recognises "Over 5.5" totals and "Home +1.5" puck lines. Other lines (e.g. Over 6.5, which `game_features` has a target for) will fall through to Moneyline.
5. **Hard-coded EV thresholds:** the 2% / 5% / 15% cutoffs are repeated in measures and columns. A what-if parameter or a single threshold table would keep them in sync.
6. **Page-level filter on page 2** only shows `OVER 5.5`, `TBL +1.5` and `TBL` picks, so Tampa Bay (TBL) is hard-coded. Confirm this is intended.
7. **Cosmetic:** page 5 has three empty text boxes, and page 4 is labelled `KIV` (parked).

---

## Disclaimer

This project is for **analytics and educational purposes only**. Model probabilities, expected value and historical accuracy do not guarantee future results and are not betting or financial advice. Please gamble responsibly and follow the laws in your jurisdiction.