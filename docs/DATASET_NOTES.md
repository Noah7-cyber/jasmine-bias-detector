# Dataset notes

## ProPublica COMPAS

- Source: [ProPublica compas-analysis repository](https://github.com/propublica/compas-analysis), `compas-scores-two-years.csv`.
- Intended use: reproduces the proposal's filtered Broward County recidivism analysis.
- Preparation: filter screening-date/arrest-date differences to −30 through +30 days inclusive, remove `is_recid == -1`, ordinary traffic offences (`c_charge_degree == O`), and unlabelled risk scores. Record the count after each filter.
- Analysis fields: race, sex, age category, COMPAS decile score, two-year recidivism truth. Positive prediction is `decile_score >= 5`, a **higher-risk/adverse flag**.
- Privacy: raw bytes are processed in memory; only the selected analysis columns are written to the bundled sample. Names, dates of birth, case identifiers and other identifying fields are excluded.
- Cite ProPublica's [analysis methodology](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm).

## UCI Adult / Census Income

- Source: [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult), DOI 10.24432/C5XW20, CC BY 4.0 attribution required.
- Intended use: second group comparison with an untuned baseline prediction column because the original table supplies income truth but not model output.
- Preparation: official train/test files; strip whitespace and `.` on test labels; interpret `?` as missing; drop rows incomplete on selected features for classifier fitting/prediction. Fit encoders, imputers, scaler and untuned logistic regression only on training rows; predict only held-out test rows. Do not tune.
- Analysis fields in the saved sample: sex, race, documented age band, income truth, income prediction. Positive is income over $50,000.
- The saved analysis sample is a held-out subset, not the entire 48,842-record dataset. Performance benchmarking must label any full-size stress fixture accurately.

## Synthetic scenarios

The generator sets group confusion counts exactly, shuffles rows with a fixed NumPy seed, and has 100 reproducible run seeds per each of six proposal scenarios. Expected outcomes and counts must be frozen independently in evaluation fixtures before comparing application results. These constructed scenarios provide known answers and do not model the full complexity of real-world data.
