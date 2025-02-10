from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import spectral
import xarray as xr

__all__ = ["read_cube", "read_preview"]


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


def read_preview(cube_path: Path | str, *, greyscale: bool = False) -> npt.NDArray:
    """
    Reads a preview image and optionally converts it to greyscale.
    Args:
      cube_path (Path): The path to the BIL hypercube.
      greyscale (bool, optional): Whether to convert the image to greyscale (default: False).
    Returns:
      xr.DataArray: The preview image.
    """
    cube_path = Path(cube_path)
    ident = cube_path.name.removeprefix("REFLECTANCE_").removesuffix(".hdr")
    path = cube_path.parents[1] / f"{ident}.png"
    if not path.exists():
        _err = f"Preview image not found at {path}"
        raise FileNotFoundError(_err)
    preview = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if greyscale:
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    return preview
