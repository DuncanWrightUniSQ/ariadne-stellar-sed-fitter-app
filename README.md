# ARIADNE Stellar SED Fitter App

Verified version: `0.1.0`

This is the main app development line for a Streamlit interface around
[jvines/astroariadne](https://github.com/jvines/astroariadne).

The app resolves a stellar target, retrieves Gaia/2MASS/WISE photometry, displays
the SED with passband-width bars and flux uncertainties, runs an ARIADNE dynesty
fit, and reports fitted stellar parameters including radius. Gaia RUWE and Gaia
distance in parsecs/light years are shown after retrieval.

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
- The optional ARIADNE spectra cache is about 2.6 GB and can also be downloaded
  from the app sidebar. It enables the continuous fitted SED model curve.
- Fit uncertainty values are displayed as lower/upper offsets from the best fit:
  `- / +`.
- The HR diagram is shown when ARIADNE produces the required age/isochrone
  samples for the selected fit output.
