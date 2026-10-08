# Data

## Source

**Orange Belgium churn-uplift dataset**

- Paper: T. Verhelst, D. Mercier, J. Shrestha, G. Bontempi, *A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling* (2023), [arXiv:2312.07206](https://arxiv.org/abs/2312.07206). Also published in the ECML PKDD 2023 workshop proceedings (Springer).
- Distribution: OpenML dataset **45580** (`churn-uplift-mlg`, version 1), <https://www.openml.org/search?type=data&id=45580>
- Benchmark code from the authors: <https://github.com/TheoVerhelst/Churn-Uplift-Dataset-Paper>
- **Licence: CC BY-NC-ND 4.0.** Because of the no-derivatives / non-commercial terms, the raw data is **not committed** to this repository.

## How to obtain it

You don't need to download anything by hand. The first call to `src.functions.load_data()` downloads the data from OpenML with `sklearn.datasets.fetch_openml(data_id=45580)` and caches it to `data/raw/churn_uplift.parquet`. That folder is git-ignored. `data/processed/` (also git-ignored) holds intermediate model outputs passed between notebooks.

## Contents

| Item | Value |
|---|---|
| Observations | 11,896 customers (one row per customer) |
| Columns | 180: `PC1`–`PC160`, `FACTOR1`–`FACTOR18`, `t`, `y` |
| Treatment `t` | 1 = targeted by a retention campaign (call-centre contact proposing a better-suited offer/tariff); 0 = held-out control. 9,010 treated (75.7%), 2,886 control (24.3%) |
| Outcome `y` | 1 = **churned** within the two months after the campaign; 0 = retained. 408 churners (3.4%) |
| Treatment arms | One treatment arm and one control arm |
| Observation period | Campaigns run September–December 2020; outcome window of two months after each campaign |
| Missing values | None |
| Duplicates | None |

### Covariates

- **`PC1`–`PC160` (continuous).** The provider projected the original numeric features (usage, e.g. data consumption and number of calls; billing/tariff price; demographics such as age) onto principal components to anonymise them. The components have no business meaning. `PC159` and `PC160` have (near-)zero variance and are dropped.
- **`FACTOR1`–`FACTOR18` (categorical, 1–20 levels).** Anonymised categorical features such as region or tariff plan, with generic level names `V1`, `V2`, …. `FACTOR3` is constant and `FACTOR8` is near-constant (4 customers outside the modal level), so both are dropped.

### Outcome convention

Because `y = 1` means churn, an effective retention intervention has a **negative** effect on `y`. When customers are ranked for targeting, the project scores them by the estimated **reduction** in churn probability (positive = good) and labels this explicitly. The outcome itself is never recoded.

## Known limitations

1. **Anonymisation.** Segments can be found, but they cannot be described in business terms.
2. **Pooled campaigns without a campaign identifier.** The treated share varies from about 63% to 86% across levels of `FACTOR4`/`FACTOR18`/`FACTOR6` (see notebooks 01–03). Treatment therefore looks randomised *within* campaigns or segments rather than unconditionally. The analysis adjusts for covariates to account for this.
3. **Few events.** There are only 408 churners in total and 105 in the control arm, which severely limits statistical power for both the average effect and heterogeneity.
4. **Telecom, not banking.** The intervention, customer base and churn mechanism differ from a bank's. The methodology transfers, but the numerical findings do not.
5. **Timing of features.** The paper describes the features as measured before each campaign, but this cannot be verified independently.
