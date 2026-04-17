from datashader.compiler import compile_components
from datashader.utils import Dispatcher
from datashader.glyphs.line import LinesXarrayCommonX
from datashader.glyphs.quadmesh import (
    QuadMeshRaster, QuadMeshRectilinear, QuadMeshCurvilinear, build_scale_translate
)
from .xarray import _extract_third_dim
from datashader.utils import apply
import dask
import numpy as np
import xarray as xr
from dask.base import tokenize, compute, flatten
from dask.array.overlap import overlap
dask_glyph_dispatch = Dispatcher()


def _prepare_3d_coords_and_dims(third_dim, xr_ds, axis, glyph):
    pass


def dask_xarray_pipeline(glyph, xr_ds, schema, canvas, summary, *, antialias=False, cuda=False):
    pass


def shape_bounds_st_and_axis(xr_ds, canvas, glyph):
    if not canvas.x_range or not canvas.y_range:
        x_extents, y_extents = glyph.compute_bounds_dask(xr_ds)
    else:
        x_extents, y_extents = None, None

    x_range = canvas.x_range or x_extents
    y_range = canvas.y_range or y_extents
    x_min, x_max, y_min, y_max = bounds = compute(*(x_range + y_range))
    x_range, y_range = (x_min, x_max), (y_min, y_max)

    width = canvas.plot_width
    height = canvas.plot_height

    x_st = canvas.x_axis.compute_scale_and_translate(x_range, width)
    y_st = canvas.y_axis.compute_scale_and_translate(y_range, height)
    st = x_st + y_st
    shape = (height, width)

    x_axis = canvas.x_axis.compute_index(x_st, width)
    y_axis = canvas.y_axis.compute_index(y_st, height)
    axis = dict([(glyph.x_label, x_axis), (glyph.y_label, y_axis)])

    return shape, bounds, st, axis

def _data_info_3d(xr_ds, canvas, glyph):
    pass


def dask_rectilinear(glyph, xr_ds, schema, canvas, summary, *, antialias=False, cuda=False):
    pass


def dask_raster(glyph, xr_ds, schema, canvas, summary, *, antialias=False, cuda=False):
    pass


def dask_curvilinear(glyph, xr_ds, schema, canvas, summary, *, antialias=False, cuda=False):
    pass


def dask_xarray_lines(
    glyph: LinesXarrayCommonX, xr_ds: xr.Dataset, schema, canvas, summary,
    *, antialias=False, cuda=False,
):
    pass


dask_glyph_dispatch.register(QuadMeshRectilinear)(dask_rectilinear)
dask_glyph_dispatch.register(QuadMeshRaster)(dask_raster)
dask_glyph_dispatch.register(QuadMeshCurvilinear)(dask_curvilinear)
dask_glyph_dispatch.register(LinesXarrayCommonX)(dask_xarray_lines)
