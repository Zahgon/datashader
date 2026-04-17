from __future__ import annotations
from contextlib import suppress
from datashader.data_libraries.dask import dask_pipeline
from datashader.core import bypixel


def dask_cudf_pipeline(df, schema, canvas, glyph, summary, *, antialias=False):
    pass


with suppress(ImportError):
    import dask_cudf

    dask_cudf_pipeline = bypixel.pipeline.register(dask_cudf.DataFrame)(dask_cudf_pipeline)
