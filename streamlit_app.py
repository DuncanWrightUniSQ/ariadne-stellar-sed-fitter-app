from __future__ import annotations

import contextlib
import io
import pickle
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import astropy.units as u
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa
from astroquery.simbad import Simbad
from dustmaps.config import config as dustmaps_config

import importlib.resources._common as resources_common

resources_common.sys = sys


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "work" / "streamlit-runs"
DUSTMAP_DIR = ROOT / "work" / "dustmaps"
PC_TO_LY = 3.261563777
APP_VERSION = "0.1.1-dev"
SOLAR_RADIUS = "R\u2299"
SOLAR_MASS = "M\u2299"
SOLAR_LUMINOSITY = "L\u2299"

PARAMETER_DISPLAY = {
    "teff": ("Effective Temperature", "K", "{:.0f}"),
    "logg": ("Surface Gravity", "dex", "{:.3f}"),
    "z": ("Metallicity [Fe/H]", "dex", "{:.3f}"),
    "dist": ("Distance", "pc", "{:.3f}"),
    "rad": ("Stellar Radius", SOLAR_RADIUS, "{:.4f}"),
    "radius": ("Stellar Radius", SOLAR_RADIUS, "{:.4f}"),
    "Av": ("Visual Extinction", "mag", "{:.4f}"),
    "grav_mass": ("Gravity-derived Mass", SOLAR_MASS, "{:.4f}"),
    "iso_mass": ("Isochrone Mass", SOLAR_MASS, "{:.4f}"),
    "lum": ("Luminosity", SOLAR_LUMINOSITY, "{:.4f}"),
    "AD": ("Angular Diameter", "mas", "{:.4f}"),
    "age": ("Isochrone Age", "Gyr", "{:.3f}"),
    "eep": ("Equivalent Evolutionary Phase", "", "{:.2f}"),
    "loglike": ("Log Likelihood", "", "{:.3f}"),
}

ARIADNE_BANDS = [
    "GaiaDR2v2_BP",
    "GaiaDR2v2_G",
    "GaiaDR2v2_RP",
    "2MASS_J",
    "2MASS_H",
    "2MASS_Ks",
    "WISE_RSR_W1",
    "WISE_RSR_W2",
]

DISPLAY_NAMES = {
    "GaiaDR2v2_BP": "Gaia BP",
    "GaiaDR2v2_G": "Gaia G",
    "GaiaDR2v2_RP": "Gaia RP",
    "2MASS_J": "2MASS J",
    "2MASS_H": "2MASS H",
    "2MASS_Ks": "2MASS Ks",
    "WISE_RSR_W1": "WISE W1",
    "WISE_RSR_W2": "WISE W2",
}


st.set_page_config(
    page_title="ARIADNE Stellar SED Fitter",
    page_icon="*",
    layout="wide",
)

DUSTMAP_DIR.mkdir(parents=True, exist_ok=True)
dustmaps_config["data_dir"] = str(DUSTMAP_DIR)


def spectra_cache_path() -> Path:
    return ROOT / "astroariadne" / "astroARIADNE" / "Datafiles" / "spectra_cache.h5"


def _configure_spectra_cache() -> Path | None:
    cache_path = spectra_cache_path()
    if not cache_path.exists():
        return None

    with contextlib.suppress(Exception):
        import astroARIADNE.config as ariadne_config
        import astroARIADNE.plotter as ariadne_plotter

        ariadne_config.spectra_cache = str(cache_path)
        ariadne_plotter.spectra_cache = str(cache_path)
    return cache_path


def _import_ariadne():
    resources_common.sys = sys

    from astroARIADNE.fitter import Fitter
    from astroARIADNE.plotter import SEDPlotter
    from astroARIADNE.star import Star

    _configure_spectra_cache()
    return Star, Fitter, SEDPlotter


def fetch_spectra_cache() -> Path:
    _import_ariadne()
    from astroARIADNE.fetch import fetch_spectra_cache as fetch

    return Path(fetch())


def _colname(table, desired: str) -> str | None:
    desired_lower = desired.lower()
    for name in table.colnames:
        if name.lower() == desired_lower:
            return name
    return None


def _clean_star_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_") or "star"


def _extract_gaia_dr3_id(ids_blob: str | None) -> int | None:
    if not ids_blob:
        return None
    match = re.search(r"Gaia\s+DR3\s+(\d+)", ids_blob)
    if match:
        return int(match.group(1))
    match = re.search(r"Gaia\s+EDR3\s+(\d+)", ids_blob)
    if match:
        return int(match.group(1))
    return None


def _simbad_coord(ra_value, dec_value) -> SkyCoord:
    try:
        return SkyCoord(float(ra_value), float(dec_value), unit=(u.deg, u.deg))
    except (TypeError, ValueError):
        return SkyCoord(str(ra_value), str(dec_value), unit=(u.hourangle, u.deg))


def _first_xml_text(root: ET.Element, tag: str) -> str | None:
    item = next(root.iter(tag), None)
    if item is None or item.text is None:
        return None
    text = item.text.strip()
    return text or None


def _resolve_with_sesame(star_name: str) -> dict:
    errors = []
    quoted = quote(star_name.strip())
    urls = [
        f"https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV?{quoted}",
        f"https://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-oxp/SNV?{quoted}",
    ]

    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            ra_text = _first_xml_text(root, "jradeg")
            dec_text = _first_xml_text(root, "jdedeg")
            main_id = _first_xml_text(root, "oname") or star_name
            if not ra_text or not dec_text:
                error_text = _first_xml_text(root, "ERROR") or "no coordinates returned"
                errors.append(f"{url}: {error_text}")
                continue

            ra_deg = float(ra_text)
            dec_deg = float(dec_text)
            gaia_id = gaia_source_from_position(ra_deg, dec_deg)
            return {
                "main_id": main_id,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "gaia_dr3_id": gaia_id,
                "ids": "",
                "resolved_by": "CDS Sesame/SIMBAD fallback",
            }
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(f"CDS Sesame could not resolve '{star_name}'. " + " | ".join(errors))


@st.cache_data(show_spinner=False, ttl=3600)
def resolve_with_simbad(star_name: str) -> dict:
    simbad = Simbad()
    try:
        with contextlib.suppress(Exception):
            simbad.add_votable_fields("ids")
        result = simbad.query_object(star_name)
    except Exception:
        return _resolve_with_sesame(star_name)

    if result is None or len(result) == 0:
        return _resolve_with_sesame(star_name)

    row = result[0]
    main_id_col = _colname(result, "main_id") or _colname(result, "MAIN_ID")
    ra_col = _colname(result, "ra") or _colname(result, "RA")
    dec_col = _colname(result, "dec") or _colname(result, "DEC")
    ids_col = _colname(result, "ids") or _colname(result, "IDS")
    if not ra_col or not dec_col:
        raise RuntimeError("SIMBAD resolved the object but did not return coordinates.")

    coord = _simbad_coord(row[ra_col], row[dec_col])
    ids_blob = str(row[ids_col]) if ids_col else ""
    gaia_id = _extract_gaia_dr3_id(ids_blob)
    if gaia_id is None:
        gaia_id = gaia_source_from_position(coord.ra.deg, coord.dec.deg)

    return {
        "main_id": str(row[main_id_col]) if main_id_col else star_name,
        "ra_deg": float(coord.ra.deg),
        "dec_deg": float(coord.dec.deg),
        "gaia_dr3_id": gaia_id,
        "ids": ids_blob,
        "resolved_by": "SIMBAD",
    }


@st.cache_data(show_spinner=False, ttl=3600)
def gaia_source_from_position(ra_deg: float, dec_deg: float, radius_arcsec: float = 5.0) -> int | None:
    query = f"""
        SELECT TOP 1 source_id,
               DISTANCE(
                 POINT('ICRS', ra, dec),
                 POINT('ICRS', {ra_deg:.12f}, {dec_deg:.12f})
               ) AS ang_sep
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(
          POINT('ICRS', ra, dec),
          CIRCLE('ICRS', {ra_deg:.12f}, {dec_deg:.12f}, {radius_arcsec / 3600.0:.12f})
        )
        ORDER BY ang_sep ASC
    """
    data = run_gaia_tap_json(query, timeout=30)
    if not data:
        return None
    return int(data[0][0])


@st.cache_data(show_spinner=False, ttl=3600)
def query_gaia_ruwe(gaia_dr3_id: int | None) -> float | None:
    if gaia_dr3_id is None:
        return None
    query = f"SELECT source_id, ruwe FROM gaiadr3.gaia_source WHERE source_id = {int(gaia_dr3_id)}"
    data = run_gaia_tap_json(query, timeout=30)
    if not data or data[0][1] is None:
        return None
    return float(data[0][1])


def _gaia_mag_error(flux_over_error) -> float:
    if flux_over_error in (None, 0):
        return 0.01
    try:
        snr = float(flux_over_error)
    except (TypeError, ValueError):
        return 0.01
    return max(2.5 / np.log(10) / snr, 0.001)


def _maybe_float(row, name: str):
    if name not in row.colnames:
        return None
    value = row[name]
    if value is np.ma.masked:
        return None
    return float(value)


@st.cache_data(show_spinner=False, ttl=3600)
def download_gaia_observables(gaia_dr3_id: int | None) -> dict:
    mag_dict = {}
    plx = plx_e = ruwe = None
    gaia_ra = gaia_dec = None

    if gaia_dr3_id is not None:
        query = f"""
            SELECT source_id, ra, dec, parallax, parallax_error,
                   phot_g_mean_mag, phot_g_mean_flux_over_error,
                   phot_bp_mean_mag, phot_bp_mean_flux_over_error,
                   phot_rp_mean_mag, phot_rp_mean_flux_over_error,
                   ruwe
            FROM gaiadr3.gaia_source
            WHERE source_id = {int(gaia_dr3_id)}
        """
        rows = run_gaia_tap_json(query, timeout=30)
        if rows:
            row = rows[0]
            gaia_ra, gaia_dec = row[1], row[2]
            plx, plx_e = row[3], row[4]
            for band, mag_index, snr_index in [
                ("GaiaDR2v2_G", 5, 6),
                ("GaiaDR2v2_BP", 7, 8),
                ("GaiaDR2v2_RP", 9, 10),
            ]:
                if row[mag_index] is not None:
                    mag_dict[band] = (float(row[mag_index]), _gaia_mag_error(row[snr_index]))
            ruwe = row[11]

    dist = dist_e = None
    if plx is not None and plx > 0:
        dist = 1000.0 / float(plx)
        dist_e = abs(dist * float(plx_e or 0.0) / float(plx))

    return {
        "mag_dict": mag_dict,
        "plx": plx,
        "plx_e": plx_e,
        "dist": dist,
        "dist_e": dist_e,
        "ruwe": float(ruwe) if ruwe is not None else None,
        "ra_deg": gaia_ra,
        "dec_deg": gaia_dec,
    }


@st.cache_data(show_spinner=False, ttl=3600)
def download_2mass_observables(ra_deg: float, dec_deg: float) -> dict:
    mag_dict = {}
    coord = SkyCoord(ra_deg, dec_deg, unit="deg")
    with contextlib.suppress(Exception):
        tmass = Irsa.query_region(coord, catalog="fp_psc", spatial="Cone", radius=5 * u.arcsec)
        if len(tmass):
            row = tmass[0]
            for band, mag_col, err_col in [
                ("2MASS_J", "j_m", "j_msigcom"),
                ("2MASS_H", "h_m", "h_msigcom"),
                ("2MASS_Ks", "k_m", "k_msigcom"),
            ]:
                mag = _maybe_float(row, mag_col)
                err = _maybe_float(row, err_col)
                if mag is not None:
                    mag_dict[band] = (mag, err if err is not None and err > 0 else 0.03)
    return mag_dict


@st.cache_data(show_spinner=False, ttl=3600)
def download_wise_observables(ra_deg: float, dec_deg: float) -> dict:
    mag_dict = {}
    coord = SkyCoord(ra_deg, dec_deg, unit="deg")
    with contextlib.suppress(Exception):
        wise = Irsa.query_region(coord, catalog="allwise_p3as_psd", spatial="Cone", radius=5 * u.arcsec)
        if len(wise):
            row = wise[0]
            for band, mag_col, err_col in [
                ("WISE_RSR_W1", "w1mpro", "w1sigmpro"),
                ("WISE_RSR_W2", "w2mpro", "w2sigmpro"),
            ]:
                mag = _maybe_float(row, mag_col)
                err = _maybe_float(row, err_col)
                if mag is not None:
                    mag_dict[band] = (mag, err if err is not None and err > 0 else 0.03)
    return mag_dict


@st.cache_data(show_spinner=False, ttl=3600)
def download_observables(ra_deg: float, dec_deg: float, gaia_dr3_id: int | None) -> dict:
    gaia = download_gaia_observables(gaia_dr3_id)
    query_ra = gaia["ra_deg"] or ra_deg
    query_dec = gaia["dec_deg"] or dec_deg
    mag_dict = {}
    mag_dict.update(gaia["mag_dict"])
    mag_dict.update(download_2mass_observables(query_ra, query_dec))
    mag_dict.update(download_wise_observables(query_ra, query_dec))

    return {
        "mag_dict": mag_dict,
        "plx": gaia["plx"],
        "plx_e": gaia["plx_e"],
        "dist": gaia["dist"],
        "dist_e": gaia["dist_e"],
        "ruwe": gaia["ruwe"],
    }


def run_gaia_tap_json(query: str, timeout: int = 30) -> list:
    response = requests.post(
        "https://gea.esac.esa.int/tap-server/tap/sync",
        data={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "json",
            "QUERY": query,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", [])


def fetch_sfd_dustmap() -> None:
    from dustmaps.sfd import fetch

    fetch()


@st.cache_resource(show_spinner=False)
def build_star(
    star_name: str,
    ra_deg: float,
    dec_deg: float,
    gaia_dr3_id: int | None,
    dustmap: str,
    mag_items: tuple,
    plx: float | None,
    plx_e: float | None,
    dist: float | None,
    dist_e: float | None,
):
    Star, _, _ = _import_ariadne()
    mag_dict = dict(mag_items)
    with contextlib.redirect_stdout(io.StringIO()):
        return Star(
            star_name,
            float(ra_deg),
            float(dec_deg),
            g_id=gaia_dr3_id,
            plx=plx,
            plx_e=plx_e,
            dist=dist,
            dist_e=dist_e,
            mag_dict=mag_dict,
            offline=True,
            dustmap=dustmap,
            verbose=False,
        )


def photometry_dataframe(star) -> pd.DataFrame:
    rows = []
    for band in ARIADNE_BANDS:
        idx = np.where(star.filter_names == band)[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        used = bool(star.used_filters[i] == 1)
        rows.append(
            {
                "Band": DISPLAY_NAMES.get(band, band),
                "ARIADNE filter": band,
                "Magnitude": np.nan if not used else float(star.mags[i]),
                "Magnitude error": np.nan if not used else float(star.mag_errs[i]),
                "Wavelength (um)": np.nan if not used else float(star.wave[i]),
                "Band half-width (um)": np.nan if not used else float(star.bandpass[i]),
                "Flux": np.nan if not used else float(star.flux[i]),
                "Flux error": np.nan if not used else float(star.flux_er[i]),
                "Status": "found" if used else "missing",
            }
        )
    return pd.DataFrame(rows)


def plot_sed(df: pd.DataFrame):
    found = df[df["Status"] == "found"].copy()
    if found.empty:
        return None
    found["Flux x wavelength"] = found["Flux"] * found["Wavelength (um)"]
    found["Flux error x wavelength"] = found["Flux error"] * found["Wavelength (um)"]
    fig = px.scatter(
        found,
        x="Wavelength (um)",
        y="Flux x wavelength",
        error_x="Band half-width (um)",
        error_y="Flux error x wavelength",
        color="Band",
        hover_data={
            "ARIADNE filter": True,
            "Magnitude": ":.4f",
            "Magnitude error": ":.4f",
            "Wavelength (um)": ":.4f",
            "Band half-width (um)": ":.4f",
            "Flux": ":.4e",
            "Flux error": ":.4e",
            "Flux x wavelength": ":.4e",
            "Flux error x wavelength": ":.2e",
        },
        labels={
            "Wavelength (um)": "Wavelength (micron)",
            "Flux x wavelength": "lambda F_lambda (erg s^-1 cm^-2)",
            "Band half-width (um)": "Filter half-width (micron)",
        },
        template="plotly_white",
    )
    fig.update_traces(marker={"size": 11, "line": {"width": 1, "color": "#1f2933"}})
    fig.update_xaxes(
        type="log",
        showgrid=True,
        title_font={"size": 22},
        tickfont={"size": 17},
        ticks="outside",
    )
    fig.update_yaxes(
        type="log",
        showgrid=True,
        title_font={"size": 22},
        tickfont={"size": 17},
        ticks="outside",
    )
    fig.update_layout(
        height=540,
        margin={"l": 92, "r": 34, "t": 34, "b": 86},
        legend_title_text="",
        font={"size": 16},
    )
    return fig


def fit_output_file(out_folder: Path, bma: bool, grid: str) -> Path:
    return out_folder / ("BMA.pkl" if bma else f"{grid}_out.pkl")


def run_fit(star, resolved: dict, settings: dict) -> tuple[dict, Path, Path, bool]:
    _, Fitter, SEDPlotter = _import_ariadne()

    run_name = _clean_star_name(resolved["main_id"])
    out_folder = RUNS_DIR / run_name
    if settings["replace_output"] and out_folder.exists():
        shutil.rmtree(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    f = Fitter()
    f.star = star
    f.setup = [
        "dynesty",
        settings["nlive"],
        settings["dlogz"],
        settings["bound"],
        settings["sample"],
        settings["threads"],
        settings["dynamic"],
    ]
    f.av_law = settings["av_law"]
    f.out_folder = str(out_folder)
    f.bma = settings["bma"]
    f.n_samples = settings["n_samples"]
    f.prior_setup = {
        "teff": "default",
        "logg": "default",
        "z": "default",
        "dist": "default",
        "rad": "default",
        "Av": "default",
    }

    reused_existing = False

    if settings["bma"]:
        f.models = settings["models"]
        f.n_grid_jobs = settings["n_grid_jobs"]
        in_file = out_folder / "BMA.pkl"
        if in_file.exists():
            reused_existing = True
        else:
            f.initialize()
            f.fit_bma()
    else:
        f.grid = settings["grid"]
        in_file = fit_output_file(out_folder, False, settings["grid"])
        if in_file.exists():
            reused_existing = True
        else:
            f.initialize()
            f.fit_dynesty(out_file=str(in_file))

    plots_folder = out_folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    _configure_spectra_cache()
    with contextlib.suppress(Exception):
        artist = SEDPlotter(str(in_file), str(plots_folder), png=True)
    if "artist" in locals():
        artist.fontsize = 24
        artist.tick_labelsize = 18
        artist.corner_fontsize = 17
        artist.corner_tick_fontsize = 14
    if "artist" in locals():
        with contextlib.suppress(Exception):
            artist.plot_SED_no_model()
        with contextlib.suppress(Exception):
            artist.plot_SED()
        with contextlib.suppress(Exception):
            artist.plot_bma_HR(25)
        if settings["bma"]:
            with contextlib.suppress(Exception):
                artist.plot_bma_hist()
        with contextlib.suppress(Exception):
            artist.plot_corner()
        with contextlib.suppress(Exception):
            artist.clean()

    out = {}
    if in_file.exists():
        with open(in_file, "rb") as handle:
            out = pickle.load(handle)
    elif hasattr(f, "to_dict"):
        with contextlib.suppress(Exception):
            out = f.to_dict()
    return out, out_folder, plots_folder, reused_existing


def _parameter_metadata(key: str) -> tuple[str, str, str]:
    if key in PARAMETER_DISPLAY:
        return PARAMETER_DISPLAY[key]
    if key.endswith("_noise"):
        band = key.removesuffix("_noise")
        return (f"{band} extra noise", "mag", "{:.4f}")
    return (key.replace("_", " ").title(), "", "{:.4g}")


def _format_number(value, fmt: str) -> str:
    try:
        if value is None or value is np.ma.masked:
            return ""
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size == 0:
                return ""
            if arr.size == 1:
                return fmt.format(arr[0])
            return " / ".join(fmt.format(v) for v in arr[:2])
        return fmt.format(float(value))
    except Exception:
        return str(value)


def _format_with_unit(value, unit: str, fmt: str) -> str:
    number = _format_number(value, fmt)
    if not number:
        return ""
    return f"{number} {unit}" if unit else number


def best_fit_dataframe(out: dict) -> pd.DataFrame:
    candidates = [
        out.get("best_fit_averaged"),
        out.get("best_fit"),
        out.get("best_fit_samples"),
    ]
    best = next((item for item in candidates if isinstance(item, dict)), {})
    uncertainty = out.get("uncertainties_averaged") or out.get("uncertainties") or {}
    rows = []
    for key, value in best.items():
        if key == "loglike":
            continue
        err = uncertainty.get(key)
        label, unit, fmt = _parameter_metadata(key)
        rows.append(
            {
                "Parameter": f"{label} ({key})",
                "Ariadne parameter": key,
                "Value": _format_with_unit(value, unit, fmt),
                "Uncertainty (- / +)": _format_with_unit(err, unit, fmt),
            }
        )
    return pd.DataFrame(rows)


st.title("ARIADNE Stellar SED Fitter")
st.caption(
    f"Version {APP_VERSION}. Resolve a star, fetch Gaia/2MASS/WISE photometry with ARIADNE, "
    "inspect the SED, then launch a dynesty fit."
)

with st.sidebar:
    st.header("Fit setup")
    grid = st.selectbox("Single-grid model", ["phoenix", "bosz", "btsettl", "btnextgen", "btcond", "kurucz", "ck04", "sphinx", "tlusty"])
    models = st.multiselect(
        "BMA models",
        ["phoenix", "btsettl", "btnextgen", "btcond", "kurucz", "ck04", "bosz"],
        default=["phoenix", "btsettl", "kurucz", "ck04", "bosz"],
    )
    nlive = st.number_input("Live points", min_value=25, max_value=2000, value=150, step=25)
    dlogz = st.number_input("Evidence tolerance", min_value=0.01, max_value=5.0, value=0.5, step=0.05)
    threads = st.number_input("Threads", min_value=1, max_value=16, value=2, step=1)
    n_grid_jobs = st.number_input("BMA grid jobs", min_value=1, max_value=8, value=1, step=1)
    n_samples = st.number_input("Posterior samples saved", min_value=1000, max_value=200000, value=25000, step=1000)
    av_law = st.selectbox("Extinction law", ["fitzpatrick", "cardelli", "odonnell", "calzetti"])
    bound = st.selectbox("Bound", ["multi", "single", "balls", "cubes"], index=0)
    sample = st.selectbox("Sampler", ["rwalk", "unif", "rslice"], index=0)
    dynamic = st.checkbox("Dynamic nested sampler", value=False)

    st.divider()
    if st.button("Download SFD dustmap", use_container_width=True):
        with st.spinner("Downloading SFD dustmap data..."):
            fetch_sfd_dustmap()
        st.success("SFD dustmap data is installed for this app.")

    cache_path = spectra_cache_path()
    if cache_path.exists():
        st.success("ARIADNE spectra cache is installed.")
    else:
        st.info("The fitted continuous SED curve needs ARIADNE's optional spectra cache (~2.6 GB).")
    if st.button("Download spectra cache (~2.6 GB)", use_container_width=True):
        with st.spinner("Downloading ARIADNE spectra cache from Zenodo..."):
            downloaded = fetch_spectra_cache()
            _configure_spectra_cache()
        st.success(f"Spectra cache installed: {downloaded}")


star_name = st.text_input("Star name", value="WASP-19", placeholder="e.g. WASP-19, HD 209458, TIC 267263253")

if "resolved" not in st.session_state:
    st.session_state.resolved = None
if "star" not in st.session_state:
    st.session_state.star = None

resolve_col, fit_col, bma_col = st.columns([1.1, 1, 1])
with resolve_col:
    resolve_clicked = st.button("Resolve and fetch photometry", type="primary", use_container_width=True)
with fit_col:
    fit_clicked = st.button("Run single-grid fit", use_container_width=True)
with bma_col:
    bma_fit_clicked = st.button(
        "Run BMA fit",
        use_container_width=True,
        help="Runs Bayesian Model Averaging across the selected BMA models. This is slower, but can provide the additional ARIADNE BMA/isochrone plots when the fit output includes the required samples.",
    )

replace_output = st.checkbox(
    "Rerun fit even if an existing result file is present",
    value=False,
    help="Leave this off to reuse a completed result such as phoenix_out.pkl. Turn it on when you want to discard the previous output and run the sampler again.",
)

if resolve_clicked:
    try:
        with st.status("Resolving target and retrieving photometry...", expanded=True) as status:
            st.write("Querying SIMBAD for the target name and coordinates; if SIMBAD TAP is unavailable, using CDS Sesame as a fallback.")
            resolved = resolve_with_simbad(star_name)
            st.write(
                f"Resolved via {resolved.get('resolved_by', 'SIMBAD')} as {resolved['main_id']} "
                f"at RA {resolved['ra_deg']:.7f} deg, "
                f"Dec {resolved['dec_deg']:.7f} deg."
            )
            if resolved["gaia_dr3_id"]:
                st.write(f"Matched Gaia DR3 source `{resolved['gaia_dr3_id']}`.")
            else:
                st.write("No Gaia DR3 identifier found from SIMBAD; trying a Gaia cone search.")

            st.write("Querying Gaia DR3 for BP/G/RP photometry, parallax distance, and RUWE.")
            gaia = download_gaia_observables(resolved["gaia_dr3_id"])
            query_ra = gaia["ra_deg"] or resolved["ra_deg"]
            query_dec = gaia["dec_deg"] or resolved["dec_deg"]
            gaia_bands = [DISPLAY_NAMES[k] for k in ARIADNE_BANDS if k in gaia["mag_dict"]]
            st.write(
                "Gaia returned "
                + (", ".join(gaia_bands) if gaia_bands else "no requested photometry")
                + (
                    f"; parallax distance {gaia['dist']:.2f} pc."
                    if gaia["dist"] is not None
                    else "; no positive parallax distance."
                )
            )

            st.write("Querying IRSA 2MASS PSC for J/H/Ks photometry.")
            tmass = download_2mass_observables(query_ra, query_dec)
            tmass_bands = [DISPLAY_NAMES[k] for k in ARIADNE_BANDS if k in tmass]
            st.write("2MASS returned " + (", ".join(tmass_bands) if tmass_bands else "no requested photometry") + ".")

            st.write("Querying IRSA AllWISE for W1/W2 photometry.")
            wise = download_wise_observables(query_ra, query_dec)
            wise_bands = [DISPLAY_NAMES[k] for k in ARIADNE_BANDS if k in wise]
            st.write("AllWISE returned " + (", ".join(wise_bands) if wise_bands else "no requested photometry") + ".")

            mag_dict = {}
            mag_dict.update(gaia["mag_dict"])
            mag_dict.update(tmass)
            mag_dict.update(wise)
            missing = [DISPLAY_NAMES[k] for k in ARIADNE_BANDS if k not in mag_dict]
            if missing:
                st.write("Missing requested band(s): " + ", ".join(missing) + ".")
            else:
                st.write("All requested Gaia, 2MASS, and WISE bands were found.")

            observables = {
                "mag_dict": mag_dict,
                "plx": gaia["plx"],
                "plx_e": gaia["plx_e"],
                "dist": gaia["dist"],
                "dist_e": gaia["dist_e"],
                "ruwe": gaia["ruwe"],
            }
            resolved["ruwe"] = observables["ruwe"]

            st.write("Constructing the ARIADNE Star object and converting magnitudes to fluxes.")
            star = build_star(
                resolved["main_id"],
                resolved["ra_deg"],
                resolved["dec_deg"],
                resolved["gaia_dr3_id"],
                "SFD",
                tuple(sorted(observables["mag_dict"].items())),
                observables["plx"],
                observables["plx_e"],
                observables["dist"],
                observables["dist_e"],
            )
            status.update(label="Photometry retrieval complete.", state="complete", expanded=False)
        st.session_state.resolved = resolved
        st.session_state.star = star
        st.success(f"Resolved {resolved['main_id']}.")
    except FileNotFoundError as exc:
        st.error("The SFD dustmap data is not installed yet. Use the sidebar button to download it, then try again.")
        st.exception(exc)
    except Exception as exc:
        st.error("Could not resolve/fetch this target.")
        st.exception(exc)

resolved = st.session_state.resolved
star = st.session_state.star

if resolved and star:
    meta1, meta2, meta3, meta4, meta5 = st.columns(5)
    meta1.metric("SIMBAD name", resolved["main_id"])
    meta2.metric("Gaia DR3 source", resolved["gaia_dr3_id"] or "not found")
    meta3.metric(
        "Distance",
        f"{star.dist:.2f} pc" if star.dist not in (-1, None) else "not found",
        f"+/- {star.dist_e:.2f} pc" if star.dist_e not in (-1, None) else None,
    )
    meta4.metric(
        "Distance",
        f"{star.dist * PC_TO_LY:.2f} ly" if star.dist not in (-1, None) else "not found",
        f"+/- {star.dist_e * PC_TO_LY:.2f} ly" if star.dist_e not in (-1, None) else None,
    )
    meta5.metric(
        "Gaia RUWE",
        f"{resolved['ruwe']:.3f}" if resolved.get("ruwe") is not None else "not found",
        help="RUWE is the Gaia Renormalized Unit Weight Error. Values near 1 indicate a well-behaved single-star astrometric solution; values much above about 1.4 can flag unresolved companions, blending, or other astrometric issues.",
    )

    st.write(f"Coordinates: RA {resolved['ra_deg']:.7f} deg, Dec {resolved['dec_deg']:.7f} deg")

    df = photometry_dataframe(star)
    chart = plot_sed(df)
    if chart:
        st.plotly_chart(chart, use_container_width=True)
        st.caption(
            "Vertical bars show flux uncertainty. Horizontal bars show the approximate filter half-width in wavelength, "
            "so they represent the passband used for the flux measurement rather than uncertainty in wavelength."
        )
    st.dataframe(df, use_container_width=True, hide_index=True)

    run_requested = fit_clicked or bma_fit_clicked
    if run_requested:
        run_bma = bool(bma_fit_clicked)
        if run_bma and not models:
            st.error("Choose at least one BMA model.")
        else:
            settings = {
                "bma": run_bma,
                "grid": grid,
                "models": models,
                "nlive": int(nlive),
                "dlogz": float(dlogz),
                "threads": int(threads),
                "n_grid_jobs": int(n_grid_jobs),
                "n_samples": int(n_samples),
                "av_law": av_law,
                "bound": bound,
                "sample": sample,
                "dynamic": bool(dynamic),
                "replace_output": bool(replace_output),
            }
            try:
                fit_label = "BMA" if run_bma else f"{grid} single-grid"
                with st.spinner(f"Running ARIADNE {fit_label} fit. This can take a while, especially for BMA..."):
                    out, out_folder, plots_folder, reused_existing = run_fit(star, resolved, settings)
                if reused_existing:
                    st.info(f"Loaded existing {fit_label} fit result from {out_folder}.")
                else:
                    st.success(f"{fit_label} fit complete. Output folder: {out_folder}")
                fit_df = best_fit_dataframe(out)
                if not fit_df.empty:
                    display_df = fit_df.drop(columns=["Ariadne parameter"])
                    radius = fit_df[fit_df["Ariadne parameter"].isin(["rad", "radius"])].drop(columns=["Ariadne parameter"])
                    if not radius.empty:
                        st.subheader("Stellar radius")
                        st.dataframe(radius, hide_index=True, use_container_width=True)
                    st.subheader("Fit parameters")
                    st.dataframe(display_df, hide_index=True, use_container_width=True)
                fitted_sed = plots_folder / "SED.png"
                raw_sed = plots_folder / "SED_no_model.png"
                if fitted_sed.exists():
                    st.subheader("Fitted SED model")
                    st.image(
                        str(fitted_sed),
                        caption="ARIADNE fitted SED model, observed fluxes, synthetic model fluxes, and residuals.",
                        use_container_width=True,
                    )
                elif raw_sed.exists():
                    st.subheader("Photometry SED")
                    st.image(
                        str(raw_sed),
                        caption="Photometry-only SED. Download the ARIADNE spectra cache to enable the continuous fitted model curve.",
                        use_container_width=True,
                    )
                    st.info("The fit completed, but ARIADNE did not create the continuous SED model plot. The usual reason is that the optional spectra cache is not installed.")

                hr_diagram = plots_folder / "HR_diagram.png"
                if hr_diagram.exists():
                    st.subheader("HR diagram")
                    st.image(
                        str(hr_diagram),
                        caption="ARIADNE HR diagram / isochrone plot.",
                        use_container_width=True,
                    )
                else:
                    st.info("No HR diagram was produced for this result. ARIADNE only makes that plot when the fit output includes the age/isochrone samples, which is most likely with BMA/isochrone-enabled output.")

                bma_histograms = sorted((plots_folder / "histograms").glob("*.png"))
                if bma_histograms:
                    with st.expander("BMA posterior and model-weight plots", expanded=run_bma):
                        for image_path in bma_histograms:
                            st.image(str(image_path), caption=image_path.stem.replace("_", " "), use_container_width=True)

                corner_plot = plots_folder / "CORNER.png"
                if corner_plot.exists():
                    st.subheader("Posterior corner plot")
                    st.image(str(corner_plot), caption="ARIADNE posterior corner plot.", use_container_width=True)
            except Exception as exc:
                st.error("The fit did not complete.")
                st.exception(exc)
elif fit_clicked or bma_fit_clicked:
    st.warning("Resolve and fetch photometry before running a fit.")
