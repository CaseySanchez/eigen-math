# eigen-geometry

## `multidimensional_scaling`

Classical multidimensional scaling projects a squared-distance matrix down to lower-dimensional coordinates via symmetric eigendecomposition.

The `feature_count` argument allows embedding a subset of the `FEATURE_DIM` points (padding the distance matrix with zeros).

## `umeyama`

Rigid point cloud alignment (rotation and translation). Generic over spatial dimension and point count. The algorithm is based on: "Least-squares estimation of transformation parameters between two point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573.