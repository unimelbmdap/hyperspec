import numpy as np
import xarray as xr
from sklearn import decomposition as decomp
from sklearn.preprocessing import StandardScaler

__all__ = ["pca"]


def pca(cube: xr.DataArray, n_components: int | float = 3) -> tuple[xr.Dataset, decomp.PCA]:
    """
    Computes principal components of a cube.
    Args:
      cube: The cube to compute principal components of.
      n_components: The number of components to compute (or minimum contained
                    variance if 0.0 < n_components < 1.0).
    Returns:
      xr.Dataset: A dataset containing the principal components.
      decomp.PCA: The fitted PCA model.
    """

    model = decomp.PCA(
        n_components=n_components, svd_solver="full" if (n_components > 0) and (n_components <= 1) else "auto"
    )
    wavelengths = cube.wavelength.values
    X = StandardScaler().fit_transform(  # noqa: N806  <- sklearn norm
        cube.dropna("x", how="all").dropna("y", how="all").values.reshape((-1, wavelengths.size))
    )
    model.fit_transform(X)
    total_var = np.cumsum(model.explained_variance_)
    total_var /= total_var[-1]
    components = xr.Dataset(
        {
            str(ii): (("wavelength",), component)
            for ii, component in zip(np.arange(model.n_components_), model.components_)
        },
        coords={"wavelength": wavelengths},
    )
    for ii in range(model.n_components_):
        components[str(ii)].attrs["explained_variance_ratio"] = model.explained_variance_ratio_[ii]
        components[str(ii)].attrs["cumulative_variance_ratio"] = total_var[ii]

    return components, model
