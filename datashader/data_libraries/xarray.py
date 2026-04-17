from __future__ import annotations
from datashader.glyphs.line import LinesXarrayCommonX
from datashader.glyphs.quadmesh import _QuadMeshLike
from datashader.data_libraries.pandas import default
from datashader.core import bypixel
import xarray as xr
from datashader.utils import Dispatcher
from datashader.compiler import compile_components


try:
    import cupy
except Exception:
    cupy = None

glyph_dispatch = Dispatcher()


@bypixel.pipeline.register(xr.Dataset)
def xarray_pipeline(xr_ds, schema, canvas, glyph, summary, *, antialias=False):
    pass


def _extract_third_dim(glyph, source):
    x_dims = set(source.coords[glyph.x].dims) if glyph.x in source.coords else {glyph.x}
    y_dims = set(source.coords[glyph.y].dims) if glyph.y in source.coords else {glyph.y}
    dims = set(source.dims) - (x_dims | y_dims)
    match len(dims):
        case 0:
            return None
        case 1:
            return next(iter(dims))
        case _:
            raise ValueError("Only one additional dimension supported for QuadMesh glyphs.")


@glyph_dispatch.register(_QuadMeshLike)
def quadmesh_default(glyph, source, schema, canvas, summary, *, antialias=False, cuda=False):
    pass

# Default to default pandas implementation
glyph_dispatch.register(LinesXarrayCommonX)(default)
