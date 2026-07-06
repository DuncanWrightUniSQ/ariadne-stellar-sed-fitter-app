# ARIADNE Stellar SED Fitter App

Verified version: `0.1.0`

This is the main app development line for a Streamlit interface around
[jvines/astroariadne](https://github.com/jvines/astroariadne).

The app resolves a stellar target, retrieves Gaia/2MASS/WISE photometry, displays
the SED with passband-width bars and flux uncertainties, runs an ARIADNE dynesty
fit, and reports fitted stellar parameters including radius. Gaia RUWE and Gaia
distance in parsecs/light years are shown after retrieval.

The `main` branch continues app development after the verified `v0.1.0` release.
It includes separate buttons for a faster single-grid fit and a heavier Bayesian
Model Averaging (BMA) fit, with larger plot axis labels/tick labels for
readability. Fit runs show a near-button status panel and scroll the page to it
so long Bayesian Model Averaging (BMA) jobs are visibly in progress.
The sidebar has Quick, Balanced, and Detailed convergence presets. Quick skips
the MIST age/mass HR-diagram stage by default; Balanced and Detailed enable it
with progressively stricter convergence settings.

## Run Locally

```bash
./setup_env.sh
./run_streamlit.sh
```

Then open:

```text
http://127.0.0.1:8501
```

## Notes

- The SFD dustmap can be downloaded from the app sidebar.
- The optional ARIADNE spectra cache is about 2.8 GB / 2.6 GiB and can be
  downloaded from the app sidebar or from the results message when a continuous
  fitted SED model plot is unavailable. It is only needed for some fitted SED
  model plots, not for the numerical fit, parameter tables, corner plots, or
  Bayesian Model Averaging (BMA) histograms.
- Fit uncertainty values are displayed as lower/upper offsets from the best fit:
  `- / +`.
- Use **Run single-grid fit** for a faster fit using the selected single model.
- Use **Run Bayesian Model Averaging (BMA) fit** to average across the selected
  model grids. This is slower, but can produce additional posterior/model-weight
  plots and HR/isochrone plots when ARIADNE has the required samples.
- Use the convergence preset and MIST isochrone controls to trade precision
  against run time. The MIST age/mass stage is required for the HR diagram and is
  often the longest step.
- Existing MIST isochrone cache directories are reused safely during Bayesian
  Model Averaging (BMA) age/mass estimation.
- The HR diagram is shown when ARIADNE produces the required age/isochrone
  samples for the selected fit output.
