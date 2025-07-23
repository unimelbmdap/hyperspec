import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
import spectral
import xarray as xr
import xmltodict

__all__ = [
    "read_cube",
    "read_png",
    "read_specim_manifest",
    "read_specim_metadata",
    "read_specim",
]


def read_specim_manifest(path: Path | str) -> dict[str, dict[str, str]]:
    path = Path(path)
    if path.suffix != ".xml":
        path = path / "manifest.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    result = {}
    for child in root:
        typ = child.get("type")
        if typ not in result:
            result[typ] = {}
        result[typ][child.get("extension")] = child.text
    return result


def read_specim(path: Path | str, item: str) -> xr.Dataset | xr.DataArray:
    path = Path(path)
    manifest = read_specim_manifest(path / "manifest.xml")
    if item not in manifest.keys():
        _err = f"Item {item} not found in manifest. Avilable items: {list(manifest.keys())}"
        raise KeyError(_err)

    result = xr.Dataset()
    item_dict = manifest[item]
    for key, value in item_dict.items():
        if key == "hdr":
            result["cube"] = read_cube(path / value)
        elif key == "png":
            result[key] = read_png(path / value)

    data_vals = list(result.data_vars)
    if len(data_vals) == 1:
        result = result[data_vals[0]]

    return result


def _reformat_metadata(d: dict | list) -> dict | list:
    if isinstance(d, list):
        for ii, v in enumerate(d):
            d[ii] = _reformat_metadata(v)
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = _reformat_metadata(v)
        if "@field" in d:
            if "#text" in d:
                d = {d["@field"]: d["#text"]}
            else:
                d = d["@field"]
    return d


def read_specim_metadata(path: Path | str) -> dict:
    path = Path(path)
    if path.suffix != ".xml":
        path = (path / "metadata").glob("*.xml").__next__()
    result = _reformat_metadata(xmltodict.parse(path.read_text(encoding="utf-8")))
    assert isinstance(result, dict), "Expected a dictionary from XML parsing"
    result = result["properties"]
    for rk, rv in result.items():
        if isinstance(rv, dict) and "key" in rv:
            if isinstance(rv["key"], list):
                result[rk] = {k: v for d in rv["key"] for k, v in d.items()}
            else:
                result[rk] = rv["key"]
    return result


def read_cube(path: Path | str) -> xr.DataArray:
    """
    Reads a BIL hypercube and optionally crops it.
    Args:
      path: The path to the BIL hypercube.
    Returns:
      xr.DataArray: The hypercube.
    """
    raw = spectral.open_image(str(path))
    if not isinstance(raw, spectral.io.bilfile.BilFile):
        _err = f"Expected BIL hypercube, got {type(raw)}"
        raise ValueError(_err)

    data = np.rot90(raw.asarray(), -1)

    cube = xr.DataArray(
        data,
        dims=("y", "x", "wavelength"),
        coords={
            "x": np.arange(raw.ncols),
            "y": np.arange(raw.nrows),
            "wavelength": raw.bands.centers,
        },
    )

    return cube


def read_png(cube_path: Path | str, *, greyscale: bool = False) -> xr.DataArray:
    """
    Reads a preview image and optionally converts it to greyscale.
    Args:
      cube_path (Path): The path to the BIL hypercube.
      greyscale (bool, optional): Whether to convert the image to greyscale (default: False).
    Returns:
      xr.DataArray: The preview image.
    """
    path = Path(cube_path)
    if not path.exists():
        _err = f"Preview image not found at {path}"
        raise FileNotFoundError(_err)
    preview = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if greyscale:
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    else:
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    return xr.DataArray(
        preview,
        dims=("y", "x", "rgb"),
        coords={
            "x": np.arange(preview.shape[1]),
            "y": np.arange(preview.shape[0]),
            "rgb": ["red", "green", "blue"],
        },
    )
