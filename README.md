# Medicare GLP-1 Bridge: NHANES analysis

One-script analysis for the revised manuscript using NHANES 2017–March 2020.

## Run

```bash
python -m pip install -r requirements.txt
python analysis.py --download
```

NHANES files are downloaded from the CDC into `data/`; CSV results are written to `outputs/`. To use existing files or change locations:

```bash
python analysis.py --data-dir data --out-dir outputs
```

## Outputs

- `main_results.csv`: candidate population, diagnosis-detectable proxy, and potentially missed group
- `restriction_estimates.csv`: diabetes, current GLP-1 use, and exploratory MASH-proxy restrictions
- `sensitivity_estimates.csv`: alternative clinical and screening definitions
- `subgroup_estimates.csv`: demographic and socioeconomic subgroups
- `awareness_estimates.csv`: self-reported prediabetes and hypertension awareness
- `missing_data_bounds.csv`: deterministic missing-data bounds
- `missingness.csv`: component missingness

Potential candidates are not validated 2026 program eligibility. The potentially missed group is a diagnosis-based screening proxy, not observed claims invisibility; NHANES was not linked to Medicare claims. The MASH measure is an exploratory elastography proxy, and obstructive sleep apnea is not quantified.
