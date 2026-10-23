# ABCD Biopsychosocial Prediction

High-dimensional ensemble stacking machine learning pipeline for hierarchical, interactive, and dynamic prediction of psychiatric, cognitive, and social outcomes in the ABCD Study.

Which of the wide range of currently available biopsychosocial measurements best predict hallmark psychiatric outcomes in adolescents and adults, and where do they converge or diverge in their predictive architecture? This is the code behind Roberts et al., *Psychiatry and Its Hierarchical & Dynamic Discontents: Biopsychosocial Prediction in Depression, Anxiety, ADHD, Social Relationships, and Cognition* (under review at *Nature*).

![Fig 5 of the article: within-category performance for adolescent anxiety and low mood/depression](assets/fig5_within_category.png)

*Fig 5 of the article: within-category ensemble meta-model performance (R²) for adolescent anxiety and hallmark low-mood/depression features in each biopsychosocial measurement category, at every available wave from age 9 to 13. Feature sets from age 9 were also used to predict age 13 outcomes.*

![Fig 7 of the article: across-category Sankey diagrams for five adult targets](assets/fig7_across_category_sankey.png)

*Fig 7 of the article: across-category ensemble performance and the top 15 consensus-weighted features for five adult targets at baseline, 2-year and 4-year follow-up. Ribbon width and opacity reflect weighted ensemble importance, and shared features are annotated (nT).*

## Start here

### Checking the article's code

The exact code used for the article is the notebook in `paper/` (release v1.0.0, [Zenodo](https://doi.org/10.5281/zenodo.23540757)). [Methods in brief](methods.md) describes what it does, [Reproducing the paper](reproducing.md) maps each figure and supplementary analysis to its stages and settings, and [Equivalence](equivalence.md) shows that this package gives the same results as the notebook at every stage on simulated panels.

### Running it on ABCD data

Read [Data access and governance](data.md), then [Installation](installation.md), [Reproducing the paper](reproducing.md) and [Configuration](configuration.md).

### Trying it without ABCD data

Follow the [Quickstart](quickstart.md). It runs in a minute or two on a laptop and checks that the pipeline recovers a planted signal.

```bash
pip install -e .
bpsp example-panel outputs/example_panel.csv
bpsp run configs/demo.yaml
```

### Everything else

The rest of the site covers the [concepts](concepts.md) behind the analyses, the [stages](stages.md), how the package is [organized](architecture.md) and [tested](testing.md), its [known issues](known_issues.md) and the [model card](model_card.md).
