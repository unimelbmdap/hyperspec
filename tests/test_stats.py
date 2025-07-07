import numpy as np
import xarray as xr
from typing import Tuple

from hyperspec import stats


def create_random_cube(
    shape: Tuple[int, int, int] = (50, 50, 50), random_state: int | None = None
) -> xr.DataArray:
    """Creates a random xarray DataArray cube for testing."""
    rng = np.random.default_rng(random_state)
    data = rng.random(shape)
    return xr.DataArray(
        data,
        dims=["y", "x", "wavelength"],
        coords={
            "y": range(shape[0]),
            "x": range(shape[1]),
            "wavelength": range(shape[2]),
        },
    )


def test_random_state(random_state: int = 42, n_calls: int = 5) -> None:
    cube = create_random_cube(random_state=random_state)

    result, model = stats.pca(cube, n_components=3, random_state=random_state)

    for _ in range(n_calls):
        new_result, new_model = stats.pca(
            cube, n_components=3, random_state=random_state
        )
        assert np.array_equal(
            result["components"].values, new_result["components"].values
        )
        assert np.array_equal(model.components_, new_model.components_)
        assert model.explained_variance_.shape == new_model.explained_variance_.shape
        assert (
            model.explained_variance_ratio_.shape
            == new_model.explained_variance_ratio_.shape
        )

    mnf = stats.MNF()
    result = mnf.fit_transform(cube, random_state=random_state)

    for _ in range(n_calls):
        new_result = mnf.fit_transform(cube, random_state=random_state)
        assert np.array_equal(result.values, new_result.values)
