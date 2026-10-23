# Data

No ABCD data are, or may ever be, stored in this repository. ABCD data require an approved Data Use Certification (see `docs/data.md`).

| File | Contents |
|---|---|
| `simulated/CLEAN_ABCD_5_1_panel_simulated.csv` | Fully simulated panel (300 synthetic participants, waves T0 to T4) with the same columns as the analysis file. Values are drawn at random from column-level summaries, so it holds no participant data and no real associations. Used by the demo, the notebooks and the tests. |
| `variable_map.csv` | Every variable in the analysis file with its source ABCD table, original variable name and figure label. Derived composites, change scores and thresholds are marked `derived`. |

To analyze ABCD data, build the merged panel with the extraction code in [abcd-visualizations-and-data](https://github.com/ra2yamamoto/abcd-visualizations-and-data), store it outside this repository, and set `setup.DATA_PATH` in your configuration.
