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
import streamlit.components.v1 as components
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse_single_table
from astroquery.ipac.irsa import Irsa
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from dustmaps.config import config as dustmaps_config

import importlib.resources._common as resources_common

resources_common.sys = sys


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "work" / "streamlit-runs"
DUSTMAP_DIR = ROOT / "work" / "dustmaps"
PC_TO_LY = 3.261563777
APP_VERSION = "0.2"
SPECTRA_CACHE_SIZE = "~2.8 GB / 2.6 GiB"
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
    "GALEX_FUV",
    "GALEX_NUV",
    "SkyMapper_u",
    "SDSS_u",
    "GROUND_JOHNSON_U",
    "SkyMapper_v",
    "TYCHO_B_MvB",
    "GROUND_JOHNSON_B",
    "SDSS_g",
    "PS1_g",
    "SkyMapper_g",
    "GaiaDR2v2_BP",
    "TYCHO_V_MvB",
    "GROUND_JOHNSON_V",
    "SkyMapper_r",
    "SDSS_r",
    "PS1_r",
    "GaiaDR2v2_G",
    "SDSS_i",
    "PS1_i",
    "SkyMapper_i",
    "GaiaDR2v2_RP",
    "PS1_z",
    "SDSS_z",
    "SkyMapper_z",
    "PS1_y",
    "2MASS_J",
    "2MASS_H",
    "2MASS_Ks",
    "WISE_RSR_W1",
    "WISE_RSR_W2",
    "WISE_RSR_W3",
    "WISE_RSR_W4",
]

DISPLAY_NAMES = {
    "GALEX_FUV": "GALEX FUV",
    "GALEX_NUV": "GALEX NUV",
    "SkyMapper_u": "SkyMapper u",
    "SkyMapper_v": "SkyMapper v",
    "SkyMapper_g": "SkyMapper g",
    "SkyMapper_r": "SkyMapper r",
    "SkyMapper_i": "SkyMapper i",
    "SkyMapper_z": "SkyMapper z",
    "SDSS_u": "SDSS u",
    "SDSS_g": "SDSS g",
    "SDSS_r": "SDSS r",
    "SDSS_i": "SDSS i",
    "SDSS_z": "SDSS z",
    "PS1_g": "Pan-STARRS g",
    "PS1_r": "Pan-STARRS r",
    "PS1_i": "Pan-STARRS i",
    "PS1_z": "Pan-STARRS z",
    "PS1_y": "Pan-STARRS y",
    "GaiaDR2v2_BP": "Gaia BP",
    "GaiaDR2v2_G": "Gaia G",
    "GaiaDR2v2_RP": "Gaia RP",
    "TYCHO_B_MvB": "Tycho BT",
    "TYCHO_V_MvB": "Tycho VT",
    "GROUND_JOHNSON_U": "Johnson U",
    "GROUND_JOHNSON_B": "Johnson B",
    "GROUND_JOHNSON_V": "Johnson V",
    "2MASS_J": "2MASS J",
    "2MASS_H": "2MASS H",
    "2MASS_Ks": "2MASS Ks",
    "WISE_RSR_W1": "WISE W1",
    "WISE_RSR_W2": "WISE W2",
    "WISE_RSR_W3": "WISE W3",
    "WISE_RSR_W4": "WISE W4",
}

PHOTOMETRY_SOURCE_BANDS = {
    "gaia": ["GaiaDR2v2_BP", "GaiaDR2v2_G", "GaiaDR2v2_RP"],
    "2mass": ["2MASS_J", "2MASS_H", "2MASS_Ks"],
    "wise12": ["WISE_RSR_W1", "WISE_RSR_W2"],
    "wise34": ["WISE_RSR_W3", "WISE_RSR_W4"],
    "panstarrs": ["PS1_g", "PS1_r", "PS1_i", "PS1_z", "PS1_y"],
    "sdss": ["SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z"],
    "skymapper": ["SkyMapper_u", "SkyMapper_v", "SkyMapper_g", "SkyMapper_r", "SkyMapper_i", "SkyMapper_z"],
    "apass": ["GROUND_JOHNSON_B", "GROUND_JOHNSON_V", "SDSS_g", "SDSS_r", "SDSS_i"],
    "tycho": ["TYCHO_B_MvB", "TYCHO_V_MvB"],
    "galex": ["GALEX_FUV", "GALEX_NUV"],
    "johnson": ["GROUND_JOHNSON_U", "GROUND_JOHNSON_B", "GROUND_JOHNSON_V"],
}

PHOTOMETRY_SOURCE_LABELS = {
    "gaia": "Gaia BP/G/RP",
    "2mass": "2MASS J/H/Ks",
    "wise12": "WISE W1/W2",
    "wise34": "WISE W3/W4",
    "panstarrs": "Pan-STARRS grizy",
    "sdss": "SDSS ugriz",
    "skymapper": "SkyMapper uvgriz",
    "apass": "APASS BVgri",
    "tycho": "Tycho-2 BT/VT",
    "galex": "GALEX FUV/NUV",
    "johnson": "Johnson UBV",
}

GAIA_TAP_SERVICES = [
    ("ESA Gaia Archive", "https://gea.esac.esa.int/tap-server/tap/sync", "json"),
    ("Gaia@AIP", "https://gaia.aip.de/tap/sync", "votable"),
]


st.set_page_config(
    page_title="ARIADNE Stellar SED Fitter",
    page_icon="*",
    layout="wide",
)

DUSTMAP_DIR.mkdir(parents=True, exist_ok=True)
dustmaps_config["data_dir"] = str(DUSTMAP_DIR)


def spectra_cache_path() -> Path:
    return ROOT / "astroariadne" / "astroARIADNE" / "Datafiles" / "spectra_cache.h5"


def spectra_cache_part_path() -> Path:
    return spectra_cache_path().with_name(f"{spectra_cache_path().name}.part")


def _format_download_size(size: int) -> str:
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.2f} GB"
    if size >= 1_000_000:
        return f"{size / 1_000_000:.0f} MB"
    return f"{size / 1_000:.0f} KB"


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


def _patch_isochrones_cache_creation() -> None:
    with contextlib.suppress(Exception):
        import os
        from isochrones.grid import Grid, download_file, getLogger

        if not getattr(pd.read_table, "_ariadne_app_safe", False):
            original_read_table = pd.read_table

            def safe_read_table(*args, **kwargs):
                if kwargs.pop("delim_whitespace", False):
                    kwargs.setdefault("sep", r"\s+")
                return original_read_table(*args, **kwargs)

            safe_read_table._ariadne_app_safe = True
            pd.read_table = safe_read_table

        if not getattr(pd.DataFrame.to_hdf, "_ariadne_app_safe", False):
            original_to_hdf = pd.DataFrame.to_hdf

            def safe_to_hdf(self, path_or_buf, *args, **kwargs):
                if args:
                    kwargs.setdefault("key", args[0])
                    args = args[1:]
                return original_to_hdf(self, path_or_buf, *args, **kwargs)

            safe_to_hdf._ariadne_app_safe = True
            pd.DataFrame.to_hdf = safe_to_hdf

        if getattr(Grid.download_tarball, "_ariadne_app_safe", False):
            return

        def safe_download_tarball(self, **kwargs):
            os.makedirs(self.datadir, exist_ok=True)
            tarball = self.get_tarball_file(**kwargs)
            if not os.path.exists(tarball):
                url = self.get_tarball_url(**kwargs)
                getLogger().info("Downloading {}...".format(url))
                download_file(url, tarball)

        safe_download_tarball._ariadne_app_safe = True
        Grid.download_tarball = safe_download_tarball

        import astroARIADNE.fitter as ariadne_fitter
        import astroARIADNE.isochrone as ariadne_isochrone

        if not getattr(ariadne_fitter.estimate, "_ariadne_app_safe", False):
            original_estimate = ariadne_fitter.estimate

            def safe_estimate(*args, **kwargs):
                isochrone_nlive = getattr(ariadne_fitter, "_ariadne_app_isochrone_nlive", None)
                if not isochrone_nlive:
                    return original_estimate(*args, **kwargs)

                original_nested_sampler = ariadne_isochrone.dynesty.NestedSampler

                def nested_sampler_with_app_nlive(*sampler_args, **sampler_kwargs):
                    sampler_kwargs["nlive"] = int(isochrone_nlive)
                    return original_nested_sampler(*sampler_args, **sampler_kwargs)

                ariadne_isochrone.dynesty.NestedSampler = nested_sampler_with_app_nlive
                try:
                    return original_estimate(*args, **kwargs)
                finally:
                    ariadne_isochrone.dynesty.NestedSampler = original_nested_sampler

            safe_estimate._ariadne_app_safe = True
            ariadne_fitter.estimate = safe_estimate


def _import_ariadne():
    resources_common.sys = sys

    from astroARIADNE.fitter import Fitter
    from astroARIADNE.plotter import SEDPlotter
    from astroARIADNE.star import Star

    _configure_spectra_cache()
    _patch_isochrones_cache_creation()
    return Star, Fitter, SEDPlotter


def fetch_spectra_cache() -> Path:
    _import_ariadne()
    from astroARIADNE import fetch as ariadne_fetch

    dest = spectra_cache_path()
    if dest.exists():
        _configure_spectra_cache()
        return dest

    tmp = spectra_cache_part_path()
    downloaded = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

    progress = st.progress(0, text="Preparing spectra cache download...")
    status = st.empty()

    with requests.get(ariadne_fetch._download_url(), headers=headers, stream=True, timeout=(30, 120)) as response:
        if downloaded and response.status_code == 200:
            downloaded = 0
            tmp.unlink(missing_ok=True)
        response.raise_for_status()

        total_remaining = int(response.headers.get("content-length", 0))
        total_size = downloaded + total_remaining if response.status_code == 206 else total_remaining
        mode = "ab" if downloaded else "wb"
        dest.parent.mkdir(parents=True, exist_ok=True)

        with tmp.open(mode) as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    fraction = min(downloaded / total_size, 1.0)
                    progress.progress(
                        fraction,
                        text=(
                            f"Downloading optional spectra cache: {fraction:.0%} "
                            f"({_format_download_size(downloaded)} / {_format_download_size(total_size)})"
                        ),
                    )
                else:
                    status.info(f"Downloaded {_format_download_size(downloaded)}...")

    if ariadne_fetch.ZENODO_SHA256 is not None:
        progress.progress(1.0, text="Verifying optional spectra cache checksum...")
        actual = ariadne_fetch._sha256(tmp)
        if actual != ariadne_fetch.ZENODO_SHA256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch: expected {ariadne_fetch.ZENODO_SHA256}, got {actual}"
            )

    tmp.replace(dest)
    _configure_spectra_cache()
    progress.progress(1.0, text="Optional spectra cache installed.")
    status.empty()
    return dest


def render_spectra_cache_button(key: str) -> None:
    cache_installed = spectra_cache_path().exists()
    partial_cache = spectra_cache_part_path()

    st.info(
        f"The continuous fitted SED model curve needs ARIADNE's optional spectra cache ({SPECTRA_CACHE_SIZE}). "
        "It is not required for the fit itself, parameter tables, corner plots, or Bayesian Model Averaging (BMA) histograms."
    )
    if cache_installed:
        st.success("ARIADNE spectra cache is installed.")
    elif partial_cache.exists():
        st.caption(f"Partial download found: {_format_download_size(partial_cache.stat().st_size)}. The download will resume from this file.")
    if st.button(
        f"Download optional spectra cache ({SPECTRA_CACHE_SIZE})",
        key=key,
        use_container_width=True,
        disabled=cache_installed,
    ):
        if spectra_cache_path().exists():
            _configure_spectra_cache()
            st.success("ARIADNE spectra cache is already installed.")
            return
        with st.spinner(f"Downloading ARIADNE spectra cache ({SPECTRA_CACHE_SIZE}) from Zenodo..."):
            downloaded = fetch_spectra_cache()
        st.success(f"Spectra cache installed: {downloaded}")


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
    with contextlib.suppress(Exception):
        if np.ma.is_masked(value):
            return None
    return float(value)


def _add_mag(mag_dict: dict, band: str, mag, err, default_err: float = 0.03) -> None:
    if band in mag_dict or mag is None:
        return
    with contextlib.suppress(Exception):
        mag = float(mag)
        err = float(err) if err is not None and float(err) > 0 else default_err
        if np.isfinite(mag) and np.isfinite(err):
            mag_dict[band] = (mag, err)


def _nearest_vizier_row(ra_deg: float, dec_deg: float, catalog: str, radius_arcsec: float = 5):
    coord = SkyCoord(ra_deg, dec_deg, unit="deg")
    tables = Vizier(columns=["**", "+_r"], row_limit=20).query_region(
        coord,
        catalog=catalog,
        radius=radius_arcsec * u.arcsec,
    )
    if not tables:
        return None
    table = tables[0]
    if len(table) == 0:
        return None
    if "_r" in table.colnames:
        table.sort("_r")
    return table[0]


def _download_vizier_observables(
    ra_deg: float,
    dec_deg: float,
    catalog: str,
    bands: list[tuple[str, str, str]],
    radius_arcsec: float = 5,
) -> dict:
    mag_dict = {}
    with contextlib.suppress(Exception):
        row = _nearest_vizier_row(ra_deg, dec_deg, catalog, radius_arcsec=radius_arcsec)
        if row is None:
            return mag_dict
        for band, mag_col, err_col in bands:
            _add_mag(mag_dict, band, _maybe_float(row, mag_col), _maybe_float(row, err_col))
    return mag_dict


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

    usable_plx = float(plx) if plx is not None and plx > 0 else None
    usable_plx_e = float(plx_e) if usable_plx is not None and plx_e is not None else None
    dist = dist_e = None
    if usable_plx is not None:
        dist = 1000.0 / usable_plx
        dist_e = abs(dist * float(usable_plx_e or 0.0) / usable_plx)

    return {
        "mag_dict": mag_dict,
        "plx": usable_plx,
        "plx_e": usable_plx_e,
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
                ("WISE_RSR_W3", "w3mpro", "w3sigmpro"),
                ("WISE_RSR_W4", "w4mpro", "w4sigmpro"),
            ]:
                mag = _maybe_float(row, mag_col)
                err = _maybe_float(row, err_col)
                _add_mag(mag_dict, band, mag, err)
    return mag_dict


@st.cache_data(show_spinner=False, ttl=3600)
def download_panstarrs_observables(ra_deg: float, dec_deg: float) -> dict:
    return _download_vizier_observables(
        ra_deg,
        dec_deg,
        "II/349/ps1",
        [
            ("PS1_g", "gmag", "e_gmag"),
            ("PS1_r", "rmag", "e_rmag"),
            ("PS1_i", "imag", "e_imag"),
            ("PS1_z", "zmag", "e_zmag"),
            ("PS1_y", "ymag", "e_ymag"),
        ],
    )


@st.cache_data(show_spinner=False, ttl=3600)
def download_sdss_observables(ra_deg: float, dec_deg: float) -> dict:
    return _download_vizier_observables(
        ra_deg,
        dec_deg,
        "V/147/sdss12",
        [
            ("SDSS_u", "umag", "e_umag"),
            ("SDSS_g", "gmag", "e_gmag"),
            ("SDSS_r", "rmag", "e_rmag"),
            ("SDSS_i", "imag", "e_imag"),
            ("SDSS_z", "zmag", "e_zmag"),
        ],
    )


@st.cache_data(show_spinner=False, ttl=3600)
def download_apass_observables(ra_deg: float, dec_deg: float) -> dict:
    return _download_vizier_observables(
        ra_deg,
        dec_deg,
        "II/336/apass9",
        [
            ("GROUND_JOHNSON_V", "Vmag", "e_Vmag"),
            ("GROUND_JOHNSON_B", "Bmag", "e_Bmag"),
            ("SDSS_g", "g'mag", "e_g'mag"),
            ("SDSS_r", "r'mag", "e_r'mag"),
            ("SDSS_i", "i'mag", "e_i'mag"),
        ],
    )


@st.cache_data(show_spinner=False, ttl=3600)
def download_tycho_observables(ra_deg: float, dec_deg: float) -> dict:
    return _download_vizier_observables(
        ra_deg,
        dec_deg,
        "I/259/tyc2",
        [
            ("TYCHO_B_MvB", "BTmag", "e_BTmag"),
            ("TYCHO_V_MvB", "VTmag", "e_VTmag"),
        ],
    )


@st.cache_data(show_spinner=False, ttl=3600)
def download_galex_observables(ra_deg: float, dec_deg: float) -> dict:
    return _download_vizier_observables(
        ra_deg,
        dec_deg,
        "II/312/ais",
        [
            ("GALEX_FUV", "FUV", "e_FUV"),
            ("GALEX_NUV", "NUV", "e_NUV"),
        ],
    )


@st.cache_data(show_spinner=False, ttl=3600)
def download_johnson_observables(ra_deg: float, dec_deg: float) -> dict:
    mag_dict = {}
    with contextlib.suppress(Exception):
        row = _nearest_vizier_row(ra_deg, dec_deg, "II/168/ubvmeans", radius_arcsec=180)
        if row is None:
            return mag_dict
        v = _maybe_float(row, "Vmag")
        v_e = _maybe_float(row, "e_Vmag")
        _add_mag(mag_dict, "GROUND_JOHNSON_V", v, v_e)
        bv = _maybe_float(row, "B-V")
        bv_e = _maybe_float(row, "e_B-V")
        if v is not None and bv is not None:
            b_e = float(np.sqrt((v_e or 0.0) ** 2 + (bv_e or 0.0) ** 2)) or 0.03
            b = float(v) + float(bv)
            _add_mag(mag_dict, "GROUND_JOHNSON_B", b, b_e)
            ub = _maybe_float(row, "U-B")
            ub_e = _maybe_float(row, "e_U-B")
            if ub is not None:
                u_e = float(np.sqrt(b_e**2 + (ub_e or 0.0) ** 2)) or 0.03
                _add_mag(mag_dict, "GROUND_JOHNSON_U", b + float(ub), u_e)
    return mag_dict


@st.cache_data(show_spinner=False, ttl=3600)
def download_skymapper_observables(ra_deg: float, dec_deg: float) -> dict:
    return _download_vizier_observables(
        ra_deg,
        dec_deg,
        "II/379/smssdr4",
        [
            ("SkyMapper_u", "uPSF", "e_uPSF"),
            ("SkyMapper_v", "vPSF", "e_vPSF"),
            ("SkyMapper_g", "gPSF", "e_gPSF"),
            ("SkyMapper_r", "rPSF", "e_rPSF"),
            ("SkyMapper_i", "iPSF", "e_iPSF"),
            ("SkyMapper_z", "zPSF", "e_zPSF"),
        ],
    )


def _source_subset(mag_dict: dict, bands: list[str]) -> dict:
    return {band: mag_dict[band] for band in bands if band in mag_dict}


def _merge_source(target: dict, source_by_band: dict, source_name: str, values: dict, allowed_bands: list[str]) -> None:
    for band, value in values.items():
        if band in allowed_bands and band not in target:
            target[band] = value
            source_by_band[band] = source_name


def filter_fit_magnitudes(mag_dict: dict, fit_sources: tuple[str, ...]) -> dict:
    fit_bands = set()
    for source in fit_sources:
        fit_bands.update(PHOTOMETRY_SOURCE_BANDS[source])
    return {band: value for band, value in mag_dict.items() if band in fit_bands}


@st.cache_data(show_spinner=False, ttl=3600)
def download_observables(
    ra_deg: float,
    dec_deg: float,
    gaia_dr3_id: int | None,
    retrieve_sources: tuple[str, ...],
) -> dict:
    gaia = download_gaia_observables(gaia_dr3_id)
    query_ra = gaia["ra_deg"] or ra_deg
    query_dec = gaia["dec_deg"] or dec_deg
    mag_dict = {}
    source_by_band = {}

    if "gaia" in retrieve_sources:
        _merge_source(mag_dict, source_by_band, "Gaia", gaia["mag_dict"], PHOTOMETRY_SOURCE_BANDS["gaia"])
    if "2mass" in retrieve_sources:
        _merge_source(mag_dict, source_by_band, "2MASS", download_2mass_observables(query_ra, query_dec), PHOTOMETRY_SOURCE_BANDS["2mass"])
    if "wise12" in retrieve_sources or "wise34" in retrieve_sources:
        wise = download_wise_observables(query_ra, query_dec)
        if "wise12" in retrieve_sources:
            _merge_source(mag_dict, source_by_band, "AllWISE", wise, PHOTOMETRY_SOURCE_BANDS["wise12"])
        if "wise34" in retrieve_sources:
            _merge_source(mag_dict, source_by_band, "AllWISE", wise, PHOTOMETRY_SOURCE_BANDS["wise34"])
    if "panstarrs" in retrieve_sources:
        _merge_source(mag_dict, source_by_band, "Pan-STARRS", download_panstarrs_observables(query_ra, query_dec), PHOTOMETRY_SOURCE_BANDS["panstarrs"])
    if "sdss" in retrieve_sources:
        _merge_source(mag_dict, source_by_band, "SDSS", download_sdss_observables(query_ra, query_dec), PHOTOMETRY_SOURCE_BANDS["sdss"])
    if "skymapper" in retrieve_sources:
        _merge_source(mag_dict, source_by_band, "SkyMapper", download_skymapper_observables(query_ra, query_dec), PHOTOMETRY_SOURCE_BANDS["skymapper"])
    if "apass" in retrieve_sources:
        _merge_source(mag_dict, source_by_band, "APASS", download_apass_observables(query_ra, query_dec), PHOTOMETRY_SOURCE_BANDS["apass"])
    if "tycho" in retrieve_sources:
        _merge_source(mag_dict, source_by_band, "Tycho-2", download_tycho_observables(query_ra, query_dec), PHOTOMETRY_SOURCE_BANDS["tycho"])
    if "galex" in retrieve_sources:
        _merge_source(mag_dict, source_by_band, "GALEX", download_galex_observables(query_ra, query_dec), PHOTOMETRY_SOURCE_BANDS["galex"])
    if "johnson" in retrieve_sources:
        _merge_source(mag_dict, source_by_band, "Johnson UBV", download_johnson_observables(query_ra, query_dec), PHOTOMETRY_SOURCE_BANDS["johnson"])

    return {
        "mag_dict": mag_dict,
        "source_by_band": source_by_band,
        "plx": gaia["plx"],
        "plx_e": gaia["plx_e"],
        "dist": gaia["dist"],
        "dist_e": gaia["dist_e"],
        "ruwe": gaia["ruwe"],
    }


def _response_excerpt(response: requests.Response, limit: int = 240) -> str:
    text = response.text.strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] or "<empty response>"


def _coerce_tap_value(value):
    if value is np.ma.masked:
        return None
    with contextlib.suppress(Exception):
        if np.ma.is_masked(value):
            return None
    with contextlib.suppress(Exception):
        return value.item()
    return value


def _votable_rows(content: bytes) -> list:
    table = parse_single_table(io.BytesIO(content)).to_table()
    return [[_coerce_tap_value(row[name]) for name in table.colnames] for row in table]


def run_gaia_tap_json(query: str, timeout: int = 30) -> list:
    errors = []
    for service_name, url, response_format in GAIA_TAP_SERVICES:
        try:
            response = requests.post(
                url,
                data={
                    "REQUEST": "doQuery",
                    "LANG": "ADQL",
                    "FORMAT": response_format,
                    "QUERY": query,
                },
                timeout=timeout,
            )
            response.raise_for_status()

            if response_format == "json":
                content_type = response.headers.get("content-type", "")
                if "json" not in content_type.lower() and not response.text.lstrip().startswith("{"):
                    errors.append(f"{service_name} returned non-JSON: {_response_excerpt(response)}")
                    continue
                payload = response.json()
                return payload.get("data", [])

            return _votable_rows(response.content)
        except Exception as exc:
            errors.append(f"{service_name}: {exc}")

    raise RuntimeError("Gaia TAP query failed. " + " | ".join(errors))


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
    extinction_kwargs = {}
    if dist is None:
        extinction_kwargs["Av"] = 0.0
        extinction_kwargs["Av_e"] = 0.0
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
            **extinction_kwargs,
        )


def photometry_dataframe(star, fit_bands: set[str] | None = None, source_by_band: dict | None = None) -> pd.DataFrame:
    fit_bands = fit_bands or set()
    source_by_band = source_by_band or {}
    rows = []
    for band in ARIADNE_BANDS:
        idx = np.where(star.filter_names == band)[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        used = bool(star.used_filters[i] == 1)
        fit_used = used and band in fit_bands
        rows.append(
            {
                "Band": DISPLAY_NAMES.get(band, band),
                "ARIADNE filter": band,
                "Source": source_by_band.get(band, ""),
                "Magnitude": np.nan if not used else float(star.mags[i]),
                "Magnitude error": np.nan if not used else float(star.mag_errs[i]),
                "Wavelength (um)": np.nan if not used else float(star.wave[i]),
                "Band half-width (um)": np.nan if not used else float(star.bandpass[i]),
                "Flux": np.nan if not used else float(star.flux[i]),
                "Flux error": np.nan if not used else float(star.flux_er[i]),
                "Status": "fit" if fit_used else ("display only" if used else "missing"),
            }
        )
    return pd.DataFrame(rows)


def plot_sed(df: pd.DataFrame):
    found = df[df["Status"] != "missing"].copy()
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
        color="Status",
        symbol="Band",
        hover_data={
            "Band": True,
            "ARIADNE filter": True,
            "Source": True,
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


def run_fit(star, resolved: dict, settings: dict, progress_callback=None) -> tuple[dict, Path, Path, bool, list[str], list[str]]:
    _, Fitter, SEDPlotter = _import_ariadne()

    run_name = _clean_star_name(resolved["main_id"])
    out_folder = RUNS_DIR / run_name
    if progress_callback:
        progress_callback(f"Preparing output folder `{out_folder}`.")
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
    f.estimate_age = settings["estimate_age"]
    f.isochrone_dlogz = settings["isochrone_dlogz"]
    f.isochrone_nlive = settings["isochrone_nlive"]
    f.prior_setup = {
        "teff": "default",
        "logg": "default",
        "z": "default",
        "dist": "default",
        "rad": "default",
        "Av": "default",
    }

    reused_existing = False
    plot_messages: list[str] = []
    plot_warnings: list[str] = []

    if settings["bma"]:
        _patch_isochrones_cache_creation()
        import astroARIADNE.fitter as ariadne_fitter

        ariadne_fitter._ariadne_app_isochrone_nlive = settings["isochrone_nlive"]
        f.models = settings["models"]
        f.n_grid_jobs = settings["n_grid_jobs"]
        in_file = out_folder / "BMA.pkl"
        if in_file.exists():
            reused_existing = True
        else:
            if progress_callback:
                progress_callback(
                    "Running Bayesian Model Averaging (BMA) grid fits with dynesty. ARIADNE writes detailed sampler iteration output to the terminal; the app will update when this phase finishes."
                )
                if settings["estimate_age"]:
                    progress_callback(
                        f"MIST age/mass and HR-diagram stage is enabled with {settings['isochrone_nlive']} live points and evidence tolerance {settings['isochrone_dlogz']}. This is usually the slowest part."
                    )
                else:
                    progress_callback(
                        "MIST age/mass and HR-diagram stage is disabled for this quicker exploratory run."
                    )
            f.initialize()
            f.fit_bma()
    else:
        f.grid = settings["grid"]
        in_file = fit_output_file(out_folder, False, settings["grid"])
        if in_file.exists():
            reused_existing = True
        else:
            if progress_callback:
                progress_callback("Running single-grid dynesty sampler.")
            f.initialize()
            f.fit_dynesty(out_file=str(in_file))

    plots_folder = out_folder / "plots"
    if progress_callback:
        progress_callback("Generating ARIADNE plots and loading fit output.")
    plots_folder.mkdir(exist_ok=True)
    _configure_spectra_cache()
    try:
        if settings["bma"]:
            artist = SEDPlotter(str(in_file), str(plots_folder))
        else:
            artist = SEDPlotter("raw", str(plots_folder))
        artist.fontsize = 24
        artist.tick_labelsize = 18
        artist.corner_fontsize = 17
        artist.corner_tick_fontsize = 14

        if settings["bma"]:
            plot_jobs = [
                ("Photometry-only SED", artist.plot_SED_no_model, plots_folder / "SED_no_model.png"),
                ("Fitted SED model", artist.plot_SED, plots_folder / "SED.png"),
                ("HR diagram", lambda: artist.plot_bma_HR(25), plots_folder / "HR_diagram.png"),
                (
                    "Bayesian Model Averaging (BMA) posterior and model-weight plots",
                    artist.plot_bma_hist,
                    plots_folder / "histograms",
                ),
                ("Posterior corner plot", artist.plot_corner, plots_folder / "CORNER.png"),
            ]
        else:
            plot_jobs = [
                ("Photometry-only SED", lambda: artist.plot_SED_no_model(star), plots_folder / "SED_no_model.png"),
            ]
            plot_messages.append(
                "Single-grid run: created the photometry SED. The fitted continuous SED diagnostic is only generated by this app for Bayesian Model Averaging (BMA) output with the optional spectra cache installed."
            )

        for label, plot_func, expected_path in plot_jobs:
            try:
                plot_func()
                if expected_path.is_dir():
                    if any(expected_path.glob("*.png")):
                        plot_messages.append(f"Created {label}.")
                    else:
                        plot_warnings.append(f"{label}: ARIADNE did not create any PNG files in {expected_path}.")
                elif expected_path.exists():
                    plot_messages.append(f"Created {label}.")
                else:
                    if label == "Fitted SED model" and not spectra_cache_path().exists():
                        plot_warnings.append(
                            f"{label}: ARIADNE did not create {expected_path.name} because the optional spectra cache ({SPECTRA_CACHE_SIZE}) is not installed. The cache is only needed for some fitted SED model plots, not for the numerical fit."
                        )
                    elif label == "HR diagram" and not settings["estimate_age"]:
                        plot_warnings.append(
                            f"{label}: ARIADNE did not create {expected_path.name} because the MIST age/mass HR-diagram stage was disabled for this quick run."
                        )
                    else:
                        plot_warnings.append(f"{label}: ARIADNE did not create {expected_path.name}.")
            except Exception as exc:
                plot_warnings.append(f"{label}: {type(exc).__name__}: {exc}")

        try:
            artist.clean()
        except Exception as exc:
            plot_warnings.append(f"Plot cleanup: {type(exc).__name__}: {exc}")
    except Exception as exc:
        plot_warnings.append(f"Could not initialise ARIADNE plotter: {type(exc).__name__}: {exc}")

    out = {}
    if in_file.exists():
        with open(in_file, "rb") as handle:
            out = pickle.load(handle)
    elif hasattr(f, "to_dict"):
        with contextlib.suppress(Exception):
            out = f.to_dict()
    return out, out_folder, plots_folder, reused_existing, plot_messages, plot_warnings


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


def _parameters_dataframe(best: dict | None, uncertainty: dict | None) -> pd.DataFrame:
    if not isinstance(best, dict):
        return pd.DataFrame()

    uncertainty = uncertainty or {}
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


def bma_fit_dataframe(out: dict) -> pd.DataFrame:
    return _parameters_dataframe(out.get("best_fit_averaged"), out.get("uncertainties_averaged"))


def simple_fit_dataframe(out: dict) -> pd.DataFrame:
    candidates = [
        (out.get("best_fit"), out.get("uncertainties")),
        (out.get("best_fit_samples"), out.get("uncertainties_samples")),
        (out.get("best_fit_averaged"), out.get("uncertainties_averaged")),
    ]
    best, uncertainty = next(((best, uncertainty) for best, uncertainty in candidates if isinstance(best, dict)), ({}, {}))
    return _parameters_dataframe(best, uncertainty)


def render_fit_outputs(out: dict, plots_folder: Path, run_bma: bool) -> None:
    if run_bma:
        bma_df = bma_fit_dataframe(out)
        if not bma_df.empty:
            st.subheader("Bayesian Model Averaging (BMA) fit parameters")
            st.dataframe(bma_df.drop(columns=["Ariadne parameter"]), hide_index=True, use_container_width=True)

    simple_df = simple_fit_dataframe(out)
    if not simple_df.empty:
        display_df = simple_df.drop(columns=["Ariadne parameter"])
        radius = simple_df[simple_df["Ariadne parameter"].isin(["rad", "radius"])].drop(columns=["Ariadne parameter"])
        if not radius.empty:
            st.subheader("Stellar radius")
            st.dataframe(radius, hide_index=True, use_container_width=True)
        st.subheader("Simple fit parameters")
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
            caption=f"Photometry-only SED. Download the optional ARIADNE spectra cache ({SPECTRA_CACHE_SIZE}) to enable the continuous fitted model curve where ARIADNE supports it.",
            use_container_width=True,
        )
        st.info(
            f"The fit completed, but ARIADNE did not create the continuous SED model plot. "
            f"The usual reason is that the optional spectra cache ({SPECTRA_CACHE_SIZE}) is not installed. "
            "The cache is only needed for some fitted SED model plots."
        )

    hr_diagram = plots_folder / "HR_diagram.png"
    if hr_diagram.exists():
        st.subheader("HR diagram")
        st.image(
            str(hr_diagram),
            caption="ARIADNE HR diagram / isochrone plot.",
            use_container_width=True,
        )
    else:
        st.info("No HR diagram was produced for this result. ARIADNE only makes that plot when the fit output includes the age/isochrone samples, which is most likely with Bayesian Model Averaging (BMA) and the MIST age/mass stage enabled.")

    bma_histograms = sorted((plots_folder / "histograms").glob("*.png"))
    if bma_histograms:
        with st.expander("Bayesian Model Averaging (BMA) posterior and model-weight plots", expanded=run_bma):
            for image_path in bma_histograms:
                st.image(str(image_path), caption=image_path.stem.replace("_", " "), use_container_width=True)

    corner_plot = plots_folder / "CORNER.png"
    if corner_plot.exists():
        st.subheader("Posterior corner plot")
        st.image(str(corner_plot), caption="ARIADNE posterior corner plot.", use_container_width=True)


st.title("ARIADNE Stellar SED Fitter")
st.caption(
    f"Version {APP_VERSION}. Resolve a star, fetch Gaia/2MASS/WISE photometry with ARIADNE, "
    "inspect the SED, then launch a dynesty fit."
)
render_spectra_cache_button("download_spectra_cache_top")

with st.sidebar:
    st.header("Fit setup")
    fit_preset = st.selectbox(
        "Convergence preset",
        [
            "Quick exploratory",
            "Balanced",
            "Detailed",
        ],
        help=(
            "Quick uses fewer live points and looser evidence tolerances for a faster first look. "
            "Balanced and Detailed spend more time on convergence, especially in the MIST isochrone stage."
        ),
    )
    preset_defaults = {
        "Quick exploratory": {
            "nlive": 75,
            "dlogz": 2.0,
            "n_samples": 5000,
            "estimate_age": False,
            "isochrone_dlogz": 2.0,
            "isochrone_nlive": 150,
        },
        "Balanced": {
            "nlive": 150,
            "dlogz": 0.5,
            "n_samples": 25000,
            "estimate_age": True,
            "isochrone_dlogz": 1.0,
            "isochrone_nlive": 250,
        },
        "Detailed": {
            "nlive": 500,
            "dlogz": 0.1,
            "n_samples": 50000,
            "estimate_age": True,
            "isochrone_dlogz": 0.1,
            "isochrone_nlive": 500,
        },
    }[fit_preset]
    grid = st.selectbox("Single-grid model", ["phoenix", "bosz", "btsettl", "btnextgen", "btcond", "kurucz", "ck04", "sphinx", "tlusty"])
    models = st.multiselect(
        "Bayesian Model Averaging (BMA) models",
        ["phoenix", "btsettl", "btnextgen", "btcond", "kurucz", "ck04", "bosz"],
        default=["phoenix", "btsettl", "kurucz", "ck04", "bosz"],
    )
    nlive = st.number_input("Live points", min_value=25, max_value=2000, value=preset_defaults["nlive"], step=25)
    dlogz = st.number_input("Evidence tolerance", min_value=0.01, max_value=5.0, value=preset_defaults["dlogz"], step=0.05)
    threads = st.number_input("Threads", min_value=1, max_value=16, value=2, step=1)
    n_grid_jobs = st.number_input("Bayesian Model Averaging (BMA) grid jobs", min_value=1, max_value=8, value=1, step=1)
    n_samples = st.number_input("Posterior samples saved", min_value=1000, max_value=200000, value=preset_defaults["n_samples"], step=1000)
    estimate_age = st.checkbox(
        "Run MIST age/mass and HR-diagram stage",
        value=preset_defaults["estimate_age"],
        help=(
            "This is required for ARIADNE's HR/isochrone plot, but it is usually the slowest part of Bayesian Model Averaging (BMA). "
            "Turn it off for a much faster exploratory Bayesian Model Averaging (BMA) fit."
        ),
    )
    isochrone_dlogz = st.number_input(
        "MIST isochrone evidence tolerance",
        min_value=0.01,
        max_value=5.0,
        value=preset_defaults["isochrone_dlogz"],
        step=0.05,
        help="Higher values stop the MIST age/mass nested sampler earlier; lower values are more precise but slower.",
        disabled=not estimate_age,
    )
    isochrone_nlive = st.number_input(
        "MIST isochrone live points",
        min_value=50,
        max_value=2000,
        value=preset_defaults["isochrone_nlive"],
        step=25,
        help="Fewer live points make the HR/age stage faster but rougher.",
        disabled=not estimate_age,
    )
    av_law = st.selectbox("Extinction law", ["fitzpatrick", "cardelli", "odonnell", "calzetti"])
    bound = st.selectbox("Bound", ["multi", "single", "balls", "cubes"], index=0)
    sample = st.selectbox("Sampler", ["rwalk", "unif", "rslice"], index=0)
    dynamic = st.checkbox("Dynamic nested sampler", value=False)

    st.divider()
    st.header("Photometry")
    retrieve_defaults = {
        "gaia": True,
        "2mass": True,
        "wise12": True,
        "wise34": True,
        "panstarrs": True,
        "sdss": True,
        "skymapper": True,
        "apass": True,
        "tycho": True,
        "galex": False,
        "johnson": True,
    }
    fit_defaults = {
        "gaia": True,
        "2mass": True,
        "wise12": True,
        "wise34": False,
        "panstarrs": True,
        "sdss": True,
        "skymapper": False,
        "apass": True,
        "tycho": True,
        "galex": False,
        "johnson": True,
    }
    retrieve_sources = []
    fit_sources = []
    with st.expander("Photometry sources", expanded=False):
        st.caption("Retrieve controls what is shown. Include in fit controls what ARIADNE uses.")
        for source_key, label in PHOTOMETRY_SOURCE_LABELS.items():
            retrieve = st.checkbox(
                f"Retrieve {label}",
                value=retrieve_defaults[source_key],
                key=f"retrieve_{source_key}",
            )
            if retrieve:
                retrieve_sources.append(source_key)
            fit = st.checkbox(
                f"Include {label} in fit",
                value=fit_defaults[source_key],
                key=f"fit_{source_key}",
                disabled=not retrieve,
                help="WISE W3/W4 is often useful for infrared-excess inspection but is off for fitting by default.",
            )
            if retrieve and fit:
                fit_sources.append(source_key)

    st.divider()
    if st.button("Download SFD dustmap", use_container_width=True):
        with st.spinner("Downloading SFD dustmap data..."):
            fetch_sfd_dustmap()
        st.success("SFD dustmap data is installed for this app.")


star_name = st.text_input("Star name", value="WASP-19", placeholder="e.g. WASP-19, HD 209458, TIC 267263253")

if "resolved" not in st.session_state:
    st.session_state.resolved = None
if "star" not in st.session_state:
    st.session_state.star = None
if "display_star" not in st.session_state:
    st.session_state.display_star = None
if "photometry_source_by_band" not in st.session_state:
    st.session_state.photometry_source_by_band = {}
if "fit_bands" not in st.session_state:
    st.session_state.fit_bands = set()

resolve_col, fit_col, bma_col = st.columns([1.1, 1, 1])
with resolve_col:
    resolve_clicked = st.button("Resolve and fetch photometry", type="primary", use_container_width=True)
with fit_col:
    fit_clicked = st.button("Run single-grid fit", use_container_width=True)
with bma_col:
    bma_fit_clicked = st.button(
        "Run Bayesian Model Averaging (BMA) fit",
        use_container_width=True,
        help="Runs Bayesian Model Averaging (BMA) across the selected model grids. This is slower, but can provide additional posterior/model-weight plots, and HR/isochrone plots when the MIST stage is enabled.",
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
            observables = download_observables(
                resolved["ra_deg"],
                resolved["dec_deg"],
                resolved["gaia_dr3_id"],
                tuple(retrieve_sources),
            )
            gaia = {
                "mag_dict": _source_subset(observables["mag_dict"], PHOTOMETRY_SOURCE_BANDS["gaia"]),
                "plx": observables["plx"],
                "plx_e": observables["plx_e"],
                "dist": observables["dist"],
                "dist_e": observables["dist_e"],
                "ruwe": observables["ruwe"],
                "ra_deg": None,
                "dec_deg": None,
            }
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

            st.write("Querying selected photometry catalogs around the resolved coordinates.")
            for source_key in retrieve_sources:
                bands = [DISPLAY_NAMES[band] for band in PHOTOMETRY_SOURCE_BANDS[source_key] if band in observables["mag_dict"]]
                st.write(f"{PHOTOMETRY_SOURCE_LABELS[source_key]} returned " + (", ".join(bands) if bands else "no requested photometry") + ".")

            mag_dict = observables["mag_dict"]
            fit_mag_dict = filter_fit_magnitudes(mag_dict, tuple(fit_sources))
            missing = [DISPLAY_NAMES[k] for k in ARIADNE_BANDS if k not in mag_dict]
            if missing:
                st.write("Missing requested band(s): " + ", ".join(missing) + ".")
            else:
                st.write("All requested bands were found.")
            fit_band_names = [DISPLAY_NAMES[k] for k in ARIADNE_BANDS if k in fit_mag_dict]
            st.write("Fit will use " + (", ".join(fit_band_names) if fit_band_names else "no photometry bands") + ".")
            if not fit_mag_dict:
                raise RuntimeError("No selected photometry bands were retrieved for fitting. Enable more retrieve/include sources and try again.")
            resolved["ruwe"] = observables["ruwe"]

            st.write("Constructing the ARIADNE Star object and converting magnitudes to fluxes.")
            if observables["dist"] is None:
                st.write(
                    "No usable Gaia parallax distance is available, so the app will skip distance-dependent dust-map extinction for this target and let ARIADNE use its broad default distance prior during fitting."
                )
            display_star = build_star(
                resolved["main_id"],
                resolved["ra_deg"],
                resolved["dec_deg"],
                resolved["gaia_dr3_id"],
                "SFD",
                tuple(sorted(mag_dict.items())),
                observables["plx"],
                observables["plx_e"],
                observables["dist"],
                observables["dist_e"],
            )
            star = build_star(
                resolved["main_id"],
                resolved["ra_deg"],
                resolved["dec_deg"],
                resolved["gaia_dr3_id"],
                "SFD",
                tuple(sorted(fit_mag_dict.items())),
                observables["plx"],
                observables["plx_e"],
                observables["dist"],
                observables["dist_e"],
            )
            status.update(label="Photometry retrieval complete.", state="complete", expanded=False)
        st.session_state.resolved = resolved
        st.session_state.star = star
        st.session_state.display_star = display_star
        st.session_state.photometry_source_by_band = observables["source_by_band"]
        st.session_state.fit_bands = set(fit_mag_dict)
        st.success(f"Resolved {resolved['main_id']}.")
    except FileNotFoundError as exc:
        st.error("The SFD dustmap data is not installed yet. Use the sidebar button to download it, then try again.")
        st.exception(exc)
    except Exception as exc:
        st.error("Could not resolve/fetch this target.")
        st.exception(exc)

resolved = st.session_state.resolved
star = st.session_state.star
display_star = st.session_state.display_star or star

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

    run_requested = fit_clicked or bma_fit_clicked
    if run_requested:
        run_bma = bool(bma_fit_clicked)
        st.markdown('<div id="fit-progress-anchor"></div>', unsafe_allow_html=True)
        components.html(
            """
            <script>
            const target = window.parent.document.getElementById("fit-progress-anchor");
            if (target) {
              target.scrollIntoView({behavior: "smooth", block: "start"});
            }
            </script>
            """,
            height=0,
        )
        if run_bma and not models:
            st.error("Choose at least one Bayesian Model Averaging (BMA) model.")
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
                "estimate_age": bool(estimate_age) if run_bma else False,
                "isochrone_dlogz": float(isochrone_dlogz),
                "isochrone_nlive": int(isochrone_nlive),
                "av_law": av_law,
                "bound": bound,
                "sample": sample,
                "dynamic": bool(dynamic),
                "replace_output": bool(replace_output),
            }
            try:
                fit_label = "Bayesian Model Averaging (BMA)" if run_bma else f"{grid} single-grid"
                with st.status(f"Running ARIADNE {fit_label} fit...", expanded=True) as status:
                    status.write(
                        f"Settings: live points {settings['nlive']}, evidence tolerance {settings['dlogz']}, "
                        f"threads {settings['threads']}."
                    )
                    if run_bma:
                        status.write(
                            "Bayesian Model Averaging (BMA) is deliberately slower: ARIADNE fits each selected model grid, combines the evidence, then optionally estimates age/mass with MIST isochrones."
                        )
                        if settings["estimate_age"]:
                            status.write(
                                f"MIST stage settings: {settings['isochrone_nlive']} live points, evidence tolerance {settings['isochrone_dlogz']}."
                            )

                    def report_progress(message: str) -> None:
                        status.write(message)

                    out, out_folder, plots_folder, reused_existing, plot_messages, plot_warnings = run_fit(
                        star, resolved, settings, progress_callback=report_progress
                    )
                    status.update(label=f"ARIADNE {fit_label} fit finished.", state="complete", expanded=False)
                if reused_existing:
                    st.info(f"Loaded existing {fit_label} fit result from {out_folder}.")
                else:
                    st.success(f"{fit_label} fit complete. Output folder: {out_folder}")
                for message in plot_messages:
                    st.caption(message)
                if plot_warnings:
                    with st.expander("Plot generation notes", expanded=True):
                        for warning in plot_warnings:
                            st.warning(warning)
                render_fit_outputs(out, plots_folder, run_bma)
            except Exception as exc:
                st.error("The fit did not complete.")
                st.exception(exc)

    df = photometry_dataframe(
        display_star,
        fit_bands=st.session_state.fit_bands,
        source_by_band=st.session_state.photometry_source_by_band,
    )
    chart = plot_sed(df)
    if chart:
        st.plotly_chart(chart, use_container_width=True)
        st.caption(
            "Vertical bars show flux uncertainty. Horizontal bars show the approximate filter half-width in wavelength, "
            "so they represent the passband used for the flux measurement rather than uncertainty in wavelength."
        )
    st.dataframe(df, use_container_width=True, hide_index=True)
elif fit_clicked or bma_fit_clicked:
    st.warning("Resolve and fetch photometry before running a fit.")
