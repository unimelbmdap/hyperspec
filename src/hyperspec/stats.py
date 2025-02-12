import numpy as np
import xarray as xr
from sklearn import decomposition as decomp
from sklearn.preprocessing import StandardScaler

__all__ = ["pca", "cube_to_features", "features_to_cube"]


def cube_to_features(cube: xr.DataArray) -> np.ndarray:
    """
    Prepares a cube for sklearn usage by converting to (n_samples, n_features).

    Args:
      cube: The cube to prepare.

    Returns:
      np.ndarray: The prepared data.
    """
    return cube.dropna("x", how="all").dropna("y", how="all").values.reshape((-1, cube.shape[-1]))


def features_to_cube(features: np.ndarray, cube: xr.DataArray, name: str) -> xr.DataArray:
    """
    Converts a feature array into an xarray DataArray with specified dimensions and coordinates.

    Args:
        features: The feature array to be reshaped and converted.
        cube: The reference DataArray providing shape and coordinate information.
        name: The name for the new dimension in the resulting DataArray.

    Returns:
        xr.DataArray: A DataArray with reshaped features and appropriate dimensions and coordinates.

    Raises:
        ValueError: If the number of dimensions in features is not 1 or 2.
    """
    if features.ndim == 2:
        return xr.DataArray(
            features.reshape((*cube.shape[:2], -1)),
            dims=[*cube.dims[:2], name],
            coords={"y": cube.y, "x": cube.x, name: range(features.shape[-1])},
        )
    elif features.ndim == 1:
        return xr.DataArray(
            features.reshape(cube.shape[:2]),
            dims=cube.dims[:2],
            coords={"y": cube.y, "x": cube.x},
        )
    else:
        raise ValueError(f"Invalid number of dimensions: {features.ndim}")


def pca(cube: xr.DataArray, n_components: int | float = 3) -> tuple[xr.Dataset, decomp.PCA]:
    """
    Computes principal components of a cube.
    Args:
      cube: The cube to compute principal components of.
      n_components: The number of components to compute (or minimum contained
                    variance if 0.0 < n_components < 1.0).
    Returns:
      xr.Dataset: A dataset containing the principal components and projected cube.
      decomp.PCA: The fitted PCA model.
    """

    data = cube_to_features(cube)
    X = StandardScaler().fit_transform(data)

    model = decomp.PCA(
        n_components=n_components, svd_solver="full" if (n_components > 0) and (n_components <= 1) else "auto"
    )
    model.fit_transform(X)

    total_var = np.cumsum(model.explained_variance_)
    total_var /= total_var[-1]

    result = xr.Dataset(
        {
            "components": (("component", "wavelength"), model.components_),
        },
        coords={"component": np.arange(model.components_.shape[0]), "wavelength": cube.wavelength.values},
    )

    result["explained_variance_ratios"] = xr.DataArray(model.explained_variance_ratio_, dims=("component",))
    result["cumulative_variance_ratio"] = xr.DataArray(total_var, dims=("component",))

    result["projected"] = features_to_cube(model.transform(X), cube, "component")

    return result, model
