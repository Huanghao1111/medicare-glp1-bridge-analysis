#!/usr/bin/env python3
"""NHANES 2017–March 2020 analysis of potential Medicare GLP-1 Bridge candidates."""

import argparse
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import beta, t as student_t

CDC = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
M = 1_000_000
GLP1 = ["SEMAGLUTIDE", "LIRAGLUTIDE", "DULAGLUTIDE", "EXENATIDE", "ALBIGLUTIDE", "LIXISENATIDE", "TIRZEPATIDE"]
ANTIHYPERTENSIVE = {
    "ACEi": ["LISINOPRIL", "ENALAPRIL", "BENAZEPRIL", "RAMIPRIL", "QUINAPRIL", "FOSINOPRIL", "PERINDOPRIL", "CAPTOPRIL", "MOEXIPRIL", "TRANDOLAPRIL"],
    "ARB": ["LOSARTAN", "VALSARTAN", "IRBESARTAN", "OLMESARTAN", "CANDESARTAN", "TELMISARTAN", "AZILSARTAN", "EPROSARTAN"],
    "CCB": ["AMLODIPINE", "NIFEDIPINE", "DILTIAZEM", "VERAPAMIL", "FELODIPINE", "ISRADIPINE", "NICARDIPINE", "NISOLDIPINE"],
    "THZ": ["HYDROCHLOROTHIAZIDE", "CHLORTHALIDONE", "INDAPAMIDE", "METOLAZONE", "CHLOROTHIAZIDE"],
    "BB": ["METOPROLOL", "ATENOLOL", "CARVEDILOL", "PROPRANOLOL", "BISOPROLOL", "LABETALOL", "NADOLOL", "NEBIVOLOL", "ACEBUTOLOL", "BETAXOLOL", "PINDOLOL"],
    "MRA": ["SPIRONOLACTONE", "EPLERENONE"],
    "A1B": ["DOXAZOSIN", "PRAZOSIN", "TERAZOSIN"],
    "CEN": ["CLONIDINE", "METHYLDOPA", "GUANFACINE"],
    "VAS": ["HYDRALAZINE", "MINOXIDIL"],
    "REN": ["ALISKIREN"],
}
TABLES = {
    "P_DEMO": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR", "WTMECPRP", "SDMVPSU", "SDMVSTRA"],
    "P_BMX": ["SEQN", "BMXBMI"],
    "P_BPXO": ["SEQN", "BPXOSY1", "BPXODI1", "BPXOSY2", "BPXODI2", "BPXOSY3", "BPXODI3"],
    "P_MCQ": ["SEQN", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
    "P_DIQ": ["SEQN", "DIQ010", "DIQ050", "DIQ070", "DIQ160"],
    "P_BPQ": ["SEQN", "BPQ020", "BPQ040A"],
    "P_GHB": ["SEQN", "LBXGH"],
    "P_BIOPRO": ["SEQN", "LBXSCR"],
    "P_HIQ": ["SEQN", "HIQ032B", "HIQ260"],
    "P_LUX": ["SEQN", "LUAXSTAT", "LUXSMED", "LUXCAPM"],
    "P_RXQ_RX": ["SEQN", "RXDUSE", "RXDCOUNT", "RXDDRUG"],
}


def read_table(data_dir, name):
    return pd.read_sas(Path(data_dir) / f"{name}.xpt", format="xport")[TABLES[name]]


def load_data(data_dir, download=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if download:
        for name in TABLES:
            destination = data_dir / f"{name}.xpt"
            if not destination.exists():
                print(f"Downloading {name}")
                request = urllib.request.Request(f"{CDC}/{name}.xpt", headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request) as response:
                    destination.write_bytes(response.read())

    names = [name for name in TABLES if name != "P_RXQ_RX"]
    full = read_table(data_dir, names[0])
    for name in names[1:]:
        full = full.merge(read_table(data_dir, name), on="SEQN", how="left")

    rx = read_table(data_dir, "P_RXQ_RX").copy()
    rx["drug"] = rx["RXDDRUG"].astype(str).str.upper()
    status = rx[["SEQN", "RXDUSE", "RXDCOUNT"]].drop_duplicates("SEQN")
    full = full.merge(status, on="SEQN", how="left")
    full["on_glp1"] = full["SEQN"].isin(rx.loc[rx["drug"].str.contains("|".join(GLP1), na=False), "SEQN"])

    def count_drugs(drugs, keywords):
        drugs = list(drugs)
        return sum(any(keyword in drug for drug in drugs) for keyword in keywords)

    grouped = rx.groupby("SEQN")["drug"]
    classes = grouped.apply(lambda x: sum(count_drugs(x, words) > 0 for words in ANTIHYPERTENSIVE.values()))
    ingredients = grouped.apply(lambda x: count_drugs(x, [word for words in ANTIHYPERTENSIVE.values() for word in words]))
    full["n_antihtn"] = full["SEQN"].map(classes).fillna(0)
    full["n_antihtn_ingredients"] = full["SEQN"].map(ingredients).fillna(0)
    full = full.loc[full["WTMECPRP"] > 0].copy().reset_index(drop=True)

    for column in ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "DIQ010", "DIQ050", "DIQ070", "DIQ160", "BPQ020", "BPQ040A"]:
        valid = [1, 2, 3] if column == "DIQ010" else [1, 2]
        full[column] = full[column].where(full[column].isin(valid))

    full["sbp"] = full[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    full["dbp"] = full[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    female = full["RIAGENDR"] == 2
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    scr = full["LBXSCR"]
    full["egfr"] = 142 * np.minimum(scr / k, 1) ** a * np.maximum(scr / k, 1) ** -1.2 * 0.9938 ** full["RIDAGEYR"] * np.where(female, 1.012, 1.0)

    full["domain"] = (full["RIDAGEYR"] >= 65) & ((full["HIQ032B"] == 2) | (full["HIQ260"] == 1)) & full["BMXBMI"].notna()
    full["chf"] = full["MCQ160B"] == 1
    full["chd"] = full["MCQ160C"] == 1
    full["angina"] = full["MCQ160D"] == 1
    full["mi"] = full["MCQ160E"] == 1
    full["stroke"] = full["MCQ160F"] == 1
    full["diabetes"] = (full["DIQ010"] == 1) | (full["LBXGH"] >= 6.5) | (full["DIQ050"] == 1) | (full["DIQ070"] == 1)
    full["prediabetes"] = ~full["diabetes"] & full["LBXGH"].between(5.7, 6.4)
    full["prediabetes_selfreport"] = ~full["diabetes"] & ((full["DIQ160"] == 1) | (full["DIQ010"] == 3))
    full["prediabetes_either"] = full["prediabetes"] | full["prediabetes_selfreport"]
    full["ckd"] = full["egfr"] < 60
    full["htn_class"] = (full["n_antihtn"] >= 2) & ((full["sbp"] > 140) | (full["dbp"] > 90))
    full["htn_ingredients"] = (full["n_antihtn_ingredients"] >= 2) & ((full["sbp"] > 140) | (full["dbp"] > 90))
    full["lux_complete"] = full["LUAXSTAT"].eq(1) & full["LUXSMED"].notna() & full["LUXCAPM"].notna()
    full["mash"] = full["lux_complete"] & (full["LUXCAPM"] >= 288) & full["LUXSMED"].between(8, 12.5, inclusive="left")
    full["core_complete"] = (
        full[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "DIQ010"]].notna().all(axis=1)
        & full["LBXGH"].notna() & full["LBXSCR"].notna() & full["sbp"].notna() & full["dbp"].notna() & full["RXDUSE"].isin([1, 2])
    )
    full["sex"] = np.where(full["RIAGENDR"] == 1, "Men", "Women")
    full["agegrp"] = np.where(full["RIDAGEYR"] < 75, "Age 65–74", "Age ≥75")
    full["race"] = full["RIDRETH3"].map({1: "Hispanic", 2: "Hispanic", 3: "NH White", 4: "NH Black", 6: "NH Asian", 7: "Other/Multi"})
    full["pir"] = pd.cut(full["INDFMPIR"], [0, 1.3, 3.5, 100], labels=["PIR <1.3", "PIR 1.3–3.5", "PIR >3.5"])
    return full


def build(full, htn=None, use_mash=False, selfreport_prediabetes=False, remove_chf=False, complete_case=False):
    htn = full["htn_class"] if htn is None else htn
    prediabetes = full["prediabetes_either"] if selfreport_prediabetes else full["prediabetes"]
    chf = pd.Series(False, index=full.index) if remove_chf else full["chf"]
    path_a = full["BMXBMI"] >= 35
    path_b = (full["BMXBMI"] >= 30) & (chf | htn | full["ckd"])
    path_c = (full["BMXBMI"] >= 27) & (prediabetes | full["mi"] | full["stroke"])
    clinical = full["domain"] & (path_a | path_b | path_c)
    allowed = ~full["diabetes"] & ~full["on_glp1"]
    if use_mash:
        allowed &= ~full["mash"]
    eligible = clinical & allowed
    if complete_case:
        eligible &= full["core_complete"]
    detectable_condition = full["chf"] | full["ckd"] | full["mi"] | full["stroke"]
    missed = eligible & ~detectable_condition
    return {
        "bmi27plus": full["domain"] & (full["BMXBMI"] >= 27),
        "path_a": path_a, "path_b": path_b, "path_c": path_c,
        "clinical": clinical, "eligible": eligible,
        "tier_a": eligible & path_a,
        "tier_b": eligible & ~path_a & path_b,
        "tier_c": eligible & ~path_a & ~path_b & path_c,
        "detectable": eligible & detectable_condition,
        "missed": missed,
        "missed_chd": eligible & ~(detectable_condition | full["chd"] | full["angina"]),
        "missed_ckd": eligible & ~(full["chf"] | full["mi"] | full["stroke"]),
        "missed_prediabetes_visible": missed & ~full["prediabetes_selfreport"],
        "missed_a": missed & path_a,
        "missed_htn": missed & ~path_a & path_b & htn,
        "missed_prediabetes": missed & ~path_a & ~path_b & path_c & prediabetes,
    }


def variance(u, full):
    design = pd.DataFrame({"u": u, "strata": full["SDMVSTRA"].to_numpy(), "psu": full["SDMVPSU"].to_numpy()})
    psu = design.groupby(["strata", "psu"])["u"].sum().reset_index()
    value = 0.0
    for _, group in psu.groupby("strata"):
        n = len(group)
        if n > 1:
            value += n / (n - 1) * ((group["u"] - group["u"].mean()) ** 2).sum()
    return value


def total(y, full, domain):
    y = np.asarray(y, dtype=float)
    domain = np.asarray(domain, dtype=bool)
    u = np.where(domain, full["WTMECPRP"].to_numpy() * np.nan_to_num(y), 0.0)
    return u.sum(), np.sqrt(variance(u, full))


def proportion(y, full, domain):
    y = np.asarray(y, dtype=float)
    domain = np.asarray(domain, dtype=bool)
    denominator, _ = total(np.ones(len(full)), full, domain)
    numerator, _ = total(y, full, domain)
    p = numerator / denominator
    u = np.where(domain, full["WTMECPRP"].to_numpy() * (np.nan_to_num(y) - p), 0.0) / denominator
    return p, np.sqrt(variance(u, full))


def kg_interval(p, se, n, df):
    if n <= 0 or df <= 0 or not np.isfinite(p) or p <= 0 or p >= 1 or se <= 0:
        return np.nan, np.nan, np.nan, np.nan
    deff = se ** 2 / (p * (1 - p) / n)
    if not np.isfinite(deff) or deff <= 0:
        return np.nan, np.nan, np.nan, np.nan
    neff = min(n, n / deff)
    neff_star = min(n, neff * (student_t.ppf(0.975, n - 1) / student_t.ppf(0.975, df)) ** 2)
    x = neff_star * p
    return beta.ppf(0.025, x, neff_star - x + 1), beta.ppf(0.975, x + 1, neff_star - x), neff, deff


def estimate(full, mask, domain):
    mask = np.asarray(mask, dtype=bool) & np.asarray(domain, dtype=bool)
    domain = np.asarray(domain, dtype=bool)
    n, se_n = total(mask, full, np.ones(len(full), dtype=bool))
    p, se_p = proportion(mask, full, domain)
    design = full.loc[domain, ["SDMVSTRA", "SDMVPSU"]].dropna()
    n_strata = design["SDMVSTRA"].nunique()
    df = int(design.drop_duplicates().shape[0] - n_strata)
    tcrit = student_t.ppf(0.975, df)
    lo, hi, neff, deff = kg_interval(p, se_p, int(domain.sum()), df)
    width = 100 * (hi - lo) if np.isfinite(lo) else np.nan
    relative_width = width / p if p > 0 and np.isfinite(lo) else np.nan
    presentable = int(domain.sum()) >= 30 and np.isfinite(neff) and neff >= 30 and int(mask.sum()) > 0 and df >= 8 and np.isfinite(width) and width < 30 and not (5 < width < 30 and relative_width > 130)
    return {
        "n_unweighted": int(mask.sum()), "domain_n_unweighted": int(domain.sum()),
        "millions": n / M, "millions_lo_t": (n - tcrit * se_n) / M, "millions_hi_t": (n + tcrit * se_n) / M,
        "pct": 100 * p, "pct_lo_kg": 100 * lo, "pct_hi_kg": 100 * hi,
        "design_df": df, "effective_n": neff, "design_effect": deff,
        "nchs_presentable_proportion": presentable,
    }


def result(full, mask, domain, variant, measure, denominator):
    return {"variant": variant, "measure": measure, "denominator": denominator, **estimate(full, mask, domain)}


def tri(index, yes, no):
    yes = np.asarray(pd.Series(yes).fillna(False), dtype=bool)
    no = np.asarray(pd.Series(no).fillna(False), dtype=bool)
    out = pd.Series(pd.NA, index=index, dtype="boolean")
    out.iloc[np.flatnonzero(no)] = False
    out.iloc[np.flatnonzero(yes)] = True
    return out


def missing_status(full):
    index = full.index
    diabetes_yes = (full["DIQ010"] == 1) | (full["LBXGH"] >= 6.5) | (full["DIQ050"] == 1) | (full["DIQ070"] == 1)
    diabetes_no = ~diabetes_yes & full["DIQ010"].isin([2, 3]) & full["LBXGH"].notna() & (full["LBXGH"] < 6.5)
    diabetes = tri(index, diabetes_yes, diabetes_no)
    rx_known = full["RXDUSE"].isin([1, 2])
    glp1 = tri(index, full["on_glp1"], rx_known & ~full["on_glp1"])
    chf = tri(index, full["MCQ160B"] == 1, full["MCQ160B"] == 2)
    ckd = tri(index, full["egfr"].notna() & (full["egfr"] < 60), full["egfr"].notna() & (full["egfr"] >= 60))
    htn_no = ~full["htn_class"] & ((rx_known & (full["n_antihtn"] < 2)) | (full["sbp"].notna() & full["dbp"].notna() & (full["sbp"] <= 140) & (full["dbp"] <= 90)))
    htn = tri(index, full["htn_class"], htn_no)
    mi_stroke = tri(index, (full["MCQ160E"] == 1) | (full["MCQ160F"] == 1), (full["MCQ160E"] == 2) & (full["MCQ160F"] == 2))
    prediabetes_yes = ~diabetes_yes & full["LBXGH"].between(5.7, 6.4)
    prediabetes_no = diabetes_yes | (full["LBXGH"].notna() & ~full["LBXGH"].between(5.7, 6.4))
    prediabetes = tri(index, prediabetes_yes, prediabetes_no)
    tier_a = tri(index, full["BMXBMI"] >= 35, full["BMXBMI"] < 35)
    b_comorbidity = chf | htn | ckd
    c_comorbidity = prediabetes | mi_stroke
    bmi_b = full["BMXBMI"] >= 30
    bmi_c = full["BMXBMI"] >= 27
    tier_b = tri(index, bmi_b & b_comorbidity.fillna(False), ~bmi_b | (bmi_b & (b_comorbidity == False).fillna(False)))
    tier_c = tri(index, bmi_c & c_comorbidity.fillna(False), ~bmi_c | (bmi_c & (c_comorbidity == False).fillna(False)))
    eligible = (tier_a | tier_b | tier_c) & ~(diabetes | glp1)
    eligible.iloc[np.flatnonzero(~full["domain"].to_numpy())] = False
    return eligible


def run(data_dir, out_dir, download=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = load_data(data_dir, download)
    domain = full["domain"]
    primary = build(full)

    rows = [result(full, domain, domain, "primary", "analytic_domain", "analytic_domain")]
    for measure in ["bmi27plus", "clinical", "eligible", "tier_a", "tier_b", "tier_c", "detectable"]:
        rows.append(result(full, primary[measure], domain, "primary", measure, "analytic_domain"))
    rows.append(result(full, primary["missed"], domain, "primary", "lab_base", "analytic_domain"))
    for measure in ["missed", "missed_chd", "missed_ckd", "missed_prediabetes_visible", "missed_a", "missed_htn", "missed_prediabetes"]:
        output_measure = {"missed": "lab", "missed_chd": "lab_chd", "missed_ckd": "lab_ckd_measurement", "missed_prediabetes_visible": "lab_prediabetes_visible", "missed_a": "invisible_a", "missed_htn": "invisible_htn", "missed_prediabetes": "invisible_prediab"}[measure]
        eligible_denominator = measure in ["missed", "missed_chd", "missed_ckd", "missed_prediabetes_visible"]
        rows.append(result(full, primary[measure], primary["eligible"] if eligible_denominator else primary["missed"], "primary", output_measure, "eligible" if eligible_denominator else "claims_invisible"))
    pd.DataFrame(rows).to_csv(out_dir / "main_results.csv", index=False)

    clinical = primary["clinical"]
    mash_variant = build(full, use_mash=True)
    restrictions = [
        ("clinical_before_exclusions", clinical),
        ("diabetes_proxy", clinical & full["diabetes"]),
        ("current_glp1_nonexclusive", clinical & full["on_glp1"]),
        ("mash_proxy_nonexclusive", clinical & full["mash"]),
        ("diabetes_or_mash_deduplicated", clinical & (full["diabetes"] | full["mash"])),
        ("diabetes_or_glp1_deduplicated", clinical & (full["diabetes"] | full["on_glp1"])),
        ("diabetes_or_glp1_or_mash_deduplicated", clinical & (full["diabetes"] | full["on_glp1"] | full["mash"])),
        ("hierarchy_1_diabetes", clinical & full["diabetes"]),
        ("hierarchy_2_current_glp1_after_diabetes", clinical & ~full["diabetes"] & full["on_glp1"]),
        ("hierarchy_3_mash_proxy_after_diabetes_and_glp1", clinical & ~full["diabetes"] & ~full["on_glp1"] & full["mash"]),
        ("hierarchy_4_remaining_after_mash_proxy", mash_variant["eligible"]),
        ("primary_after_diabetes_and_glp1", primary["eligible"]),
        ("vcte_incomplete_among_clinical", clinical & ~full["lux_complete"]),
    ]
    pd.DataFrame([result(full, mask, clinical, "CMS_restrictions", name, "clinical_before_exclusions") for name, mask in restrictions]).to_csv(out_dir / "restriction_estimates.csv", index=False)

    variants = {
        "primary": primary,
        "any_antihypertensive_class": build(full, htn=(full["n_antihtn"] >= 1) & ((full["sbp"] > 140) | (full["dbp"] > 90))),
        "two_active_ingredients": build(full, htn=full["htn_ingredients"]),
        "without_chf_proxy": build(full, remove_chf=True),
        "include_selfreported_prediabetes": build(full, selfreport_prediabetes=True),
        "exploratory_mash_proxy_excluded": mash_variant,
        "core_complete_case": build(full, complete_case=True),
    }
    sensitivity = []
    for name, variant in variants.items():
        sensitivity.append(result(full, variant["eligible"], domain, name, "eligible", "analytic_domain"))
        sensitivity.append(result(full, variant["missed"], variant["eligible"], name, "claims_invisible", "eligible"))
    pd.DataFrame(sensitivity).to_csv(out_dir / "sensitivity_estimates.csv", index=False)

    subgroups = [("Overall", domain)]
    for variable, levels in [("sex", ["Men", "Women"]), ("agegrp", ["Age 65–74", "Age ≥75"]), ("race", ["NH White", "NH Black", "Hispanic", "NH Asian", "Other/Multi"]), ("pir", ["PIR <1.3", "PIR 1.3–3.5", "PIR >3.5"])]:
        subgroups.extend((level, domain & full[variable].eq(level)) for level in levels)
    subgroup_rows = []
    for label, subgroup in subgroups:
        subgroup_rows.append({"subgroup": label, **result(full, primary["eligible"] & subgroup, subgroup, "subgroup", "eligible", "analytic_subgroup")})
        subgroup_rows.append({"subgroup": label, **result(full, primary["missed"] & subgroup, primary["eligible"] & subgroup, "subgroup", "claims_invisible", "eligible_subgroup")})
    pd.DataFrame(subgroup_rows).to_csv(out_dir / "subgroup_estimates.csv", index=False)

    missed = primary["missed"]
    measured_prediabetes = missed & full["prediabetes"]
    awareness = [
        ("told_prediabetes_among_invisible_with_measured_prediabetes", measured_prediabetes & (full["DIQ160"] == 1), measured_prediabetes),
        ("told_hypertension_among_invisible", missed & (full["BPQ020"] == 1), missed),
        ("told_neither_among_invisible", missed & (full["DIQ160"] == 2) & (full["BPQ020"] == 2), missed),
    ]
    pd.DataFrame([result(full, mask, denom, "awareness", name, "specified_subgroup") for name, mask, denom in awareness]).to_csv(out_dir / "awareness_estimates.csv", index=False)

    status = missing_status(full)
    definite = status.eq(True).fillna(False).to_numpy(dtype=bool)
    possible = definite | np.asarray(status.isna(), dtype=bool)
    bounds = [("definite_eligible_lower_bound", definite), ("observed_primary", primary["eligible"]), ("possible_eligible_upper_bound", possible), ("unknown_eligibility", np.asarray(status.isna()) & np.asarray(domain))]
    pd.DataFrame([result(full, mask, domain, "missing_data_bounds", name, "analytic_domain") for name, mask in bounds]).to_csv(out_dir / "missing_data_bounds.csv", index=False)

    missingness = [("HbA1c", full["LBXGH"].isna()), ("creatinine", full["LBXSCR"].isna()), ("blood_pressure", full["sbp"].isna() | full["dbp"].isna()), ("family_income_to_poverty_ratio", full["INDFMPIR"].isna()), ("VCTE_incomplete", ~full["lux_complete"])]
    pd.DataFrame([result(full, domain & mask, domain, "missingness", name, "analytic_domain") for name, mask in missingness]).to_csv(out_dir / "missingness.csv", index=False)

    main = pd.DataFrame(rows)
    print(main.loc[main["measure"].isin(["clinical", "eligible", "detectable", "lab"]), ["measure", "n_unweighted", "millions", "pct", "pct_lo_kg", "pct_hi_kg"]].to_string(index=False))
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, args.download)
