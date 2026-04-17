"""Bundle a graph's edges to emphasize the graph structure.

Given a large graph, the underlying structure can be obscured by edges in close
proximity. To uncover the group structure for clearer visualization, edges are
split into smaller edges and bundled with neighbors.

Ian Calvert's `Edgehammer`_ is the original implementation of the main
algorithm.

.. _Edgehammer:
   https://gitlab.com/ianjcalvert/edgehammer
"""

from __future__ import annotations

from math import ceil

from pandas import DataFrame
from collections import namedtuple

try:
    import dask
    from dask import compute, delayed
except ImportError:
    dask, compute = None, None
    def delayed(*args, **kwargs):
        def func(*args, **kwargs):
            raise ImportError("dask is required to use delayed functions")
        return func
try:
    import skimage
    from skimage.filters import gaussian, sobel_h, sobel_v
except Exception:
    skimage = None

import numpy as np
import pandas as pd
import param
import numba as nb
import itertools

from .utils import ngjit

SegmentLength  = namedtuple('SegmentLength', ['min', 'max', 'mean'])
segment_length_type = nb.types.NamedUniTuple(nb.float64, 3, SegmentLength)


@nb.jit(
    nb.float32(nb.float32[::1], nb.float32[::1]),
    nopython=True,
    nogil=True,
    fastmath=True,
    locals={"result": nb.float32, "diff": nb.float32},
)
def distance_between(a, b):
    """Find the Euclidean distance between two points."""
    pass

@nb.jit(
    nb.float32[:,::1](nb.float32[:,::1], nb.float32[:,::1], nb.uint16[::1], nb.int64),
    nopython=True,
    nogil=True,
    fastmath=True,
    locals={
        'next_point': nb.float32[::1],
        'current_point': nb.float32[::1],
        'step_vector': nb.float32[::1],
        'i': nb.uint16,
        'pos': nb.uint64,
        'index': nb.uint64,
        'distance': nb.float32
    }
)
def resample_segment(segments, new_segments, n_points_to_add, ndims):
    pass

@nb.jit(
    nb.types.Tuple((nb.boolean, nb.uint64, nb.uint16[::1]))(nb.float32[:,::1], segment_length_type),
    nopython=True,
    nogil=True,
    fastmath=True,
    locals={
        'next_point': nb.float32[::1],
        'current_point': nb.float32[::1],
        'pos': nb.uint64,
        'index': nb.uint64,
        'distance': nb.float32
    }
)
def calculate_resampling(segments, squared_segment_length):
    pass

@nb.jit(
    nb.float32[:, ::1](nb.float32[:, ::1], segment_length_type, nb.int64),
    nopython=True,
    nogil=True,
)
def resample_edge(segments, squared_segment_length, ndims):
    pass


def resample_edges(edge_segments, squared_segment_length, ndims):
    pass


@nb.jit(
    nb.void(nb.float32[:,::1], nb.float32, nb.int64, nb.int64),
    nopython=True,
    nogil=True,
    fastmath=True,
    locals={
        "i": nb.uint16,
        "segments": nb.float32[:,::1],
        "previous": nb.float32[::1],
        "current": nb.float32[::1],
        "next_point": nb.float32[::1]
    }
)
def smooth_segment(segments, tension, idx, idy):
    pass

def smooth(edge_segments, tension, idx, idy):
    pass

@nb.jit(
    nb.float32[:,::1](nb.float32[:,::1], nb.float32[:,::1], nb.float32[:,::1], nb.int64, nb.float64,
                      segment_length_type, nb.uint64, nb.uint64, nb.int64),
    nopython=True,
    nogil=True,
    fastmath=True,
    locals={'it': nb.uint8, "i": nb.uint16, "x": nb.uint16, "y": nb.uint16}
)
def advect_and_resample(vert, horiz, segments, iterations, accuracy, squared_segment_length,
                        idx, idy, ndims):
    pass

def advect_resample_all(gradients, edge_segments, iterations, accuracy, squared_segment_length,
                        idx, idy, ndims):
    pass


def batches(seq, n):
    """Yield successive n-sized batches from seq."""
    pass

def draw_to_surface(edge_segments, bandwidth, accuracy, accumulator):
    pass

@nb.jit(
    nb.void(nb.float32[:,::1], nb.float32[:,::1]),
    nopython=True,
    nogil=True,
    fastmath=True,
)
def normalize_gradients(vert, horiz):
    pass

def get_gradients(img):
    pass


class BaseSegment:
    @classmethod
    def create_delimiter(cls):
        pass


class UnweightedSegment(BaseSegment):
    ndims = 3
    idx, idy = 1, 2

    @staticmethod
    def get_columns(params):
        pass

    @staticmethod
    def get_merged_columns(params):
        pass

    @staticmethod
    @ngjit
    def create_segment(edge):
        pass

    @staticmethod
    @ngjit
    def accumulate(img, points, accuracy):
        pass


class EdgelessUnweightedSegment(BaseSegment):
    ndims = 2
    idx, idy = 0, 1

    @staticmethod
    def get_columns(params):
        pass

    @staticmethod
    def get_merged_columns(params):
        pass

    @staticmethod
    @ngjit
    def create_segment(edge):
        pass

    @staticmethod
    @ngjit
    def accumulate(img, points, accuracy):
        pass


class WeightedSegment(BaseSegment):
    ndims = 4
    idx, idy = 1, 2

    @staticmethod
    def get_columns(params):
        pass

    @staticmethod
    def get_merged_columns(params):
        pass

    @staticmethod
    @ngjit
    def create_segment(edge):
        pass

    @staticmethod
    @ngjit
    def accumulate(img, points, accuracy):
        pass


class EdgelessWeightedSegment(BaseSegment):
    ndims = 3
    idx, idy = 0, 1

    @staticmethod
    def get_columns(params):
        pass

    @staticmethod
    def get_merged_columns(params):
        pass

    @staticmethod
    @ngjit
    def create_segment(edge):
        pass

    @staticmethod
    @ngjit
    def accumulate(img, points, accuracy):
        pass


def _convert_graph_to_edge_segments(nodes, edges, params):
    """
    Merge graph dataframes into a list of edge segments.

    Given a graph defined as a pair of dataframes (nodes and edges), the
    nodes (id, coordinates) and edges (id, source, target, weight) are
    joined by node id to create a single dataframe with each source/target
    of an edge (including its optional weight) replaced with the respective
    coordinates. For both nodes and edges, each id column is assumed to be
    the index.

    We also return the dimensions of each point in the final dataframe and
    the accumulator function for drawing to an image.
    """
    pass


def _convert_edge_segments_to_dataframe(edge_segments, segment_class, params):
    """
    Convert list of edge segments into a dataframe.

    For all edge segments, we create a dataframe to represent a path
    as successive points separated by a point with NaN as the x or y
    value.
    """
    pass


class connect_edges(param.ParameterizedFunction):
    """
    Convert a graph into paths suitable for datashading.

    Base class that connects each edge using a single line segment.
    Subclasses can add more complex algorithms for connecting with
    curved or manhattan-style polylines.
    """

    x = param.String(default='x', doc="""
        Column name for each node's x coordinate.""")

    y = param.String(default='y', doc="""
        Column name for each node's y coordinate.""")

    source = param.String(default='source', doc="""
        Column name for each edge's source.""")

    target = param.String(default='target', doc="""
        Column name for each edge's target.""")

    weight = param.String(default=None, allow_None=True, doc="""
        Column name for each edge weight. If None, weights are ignored.""")

    include_edge_id = param.Boolean(default=False, doc="""
        Include edge IDs in bundled dataframe""")

    def __call__(self, nodes, edges, **params):
        """
        Convert a graph data structure into a path structure for plotting

        Given a set of nodes (as a dataframe with a unique ID for each
        node) and a set of edges (as a dataframe with with columns for the
        source and destination IDs for each edge), returns a dataframe
        with with one path for each edge suitable for use with
        Datashader. The returned dataframe has columns for x and y
        location, with paths represented as successive points separated by
        a point with NaN as the x or y value.
        """
        p = param.ParamOverrides(self, params)
        edges, segment_class = _convert_graph_to_edge_segments(nodes, edges, p)
        return _convert_edge_segments_to_dataframe(edges, segment_class, p)

directly_connect_edges = connect_edges # For backwards compatibility; deprecated


def minmax_normalize(X, lower, upper):
    pass


def minmax_denormalize(X, lower, upper):
    pass


class hammer_bundle(connect_edges):
    """
    Iteratively group edges and return as paths suitable for datashading.

    Breaks each edge into a path with multiple line segments, and
    iteratively curves this path to bundle edges into groups.
    """

    initial_bandwidth = param.Number(default=0.05,bounds=(0.0,None),doc="""
        Initial value of the bandwidth....""")

    decay = param.Number(default=0.7,bounds=(0.0,1.0),doc="""
        Rate of decay in the bandwidth value, with 1.0 indicating no decay.""")

    iterations = param.Integer(default=4,bounds=(1,None),doc="""
        Number of passes for the smoothing algorithm""")

    batch_size = param.Integer(default=20000,bounds=(1,None),doc="""
        Number of edges to process together""")

    tension = param.Number(default=0.3,bounds=(0,None),precedence=-0.5,doc="""
        Exponential smoothing factor to use when smoothing""")

    accuracy = param.Integer(default=500,bounds=(1,65535),precedence=-0.5,doc="""
        Number of entries in table for...""")

    advect_iterations = param.Integer(default=50,bounds=(0,None),precedence=-0.5,doc="""
        Number of iterations to move edges along gradients""")

    min_segment_length = param.Number(default=0.008,bounds=(0,None),precedence=-0.5,doc="""
        Minimum length (in data space?) for an edge segment""")

    max_segment_length = param.Number(default=0.016,bounds=(0,None),precedence=-0.5,doc="""
        Maximum length (in data space?) for an edge segment""")

    weight = param.String(default='weight', allow_None=True, doc="""
        Column name for each edge weight. If None, weights are ignored.""")

    use_dask = param.Boolean(default=False, doc="""
        Whether to use dask to parallelize the computation.""")

    def __call__(self, nodes, edges, **params):
        if dask is None or skimage is None:
            raise ImportError("hammer_bundle operation requires dask and scikit-image. "
                              "Ensure you install the dependency before applying "
                              "bundling.")

        p = param.ParamOverrides(self, params)

        if p.use_dask:
            resample_edges_fn = delayed(resample_edges)
            draw_to_surface_fn = delayed(draw_to_surface)
            get_gradients_fn = delayed(get_gradients)
            advect_resample_all_fn = delayed(advect_resample_all)
        else:
            resample_edges_fn = resample_edges
            draw_to_surface_fn = draw_to_surface
            get_gradients_fn = get_gradients
            advect_resample_all_fn = advect_resample_all

        # Calculate min/max for coordinates
        xmin, xmax = np.min(nodes[p.x]), np.max(nodes[p.x])
        ymin, ymax = np.min(nodes[p.y]), np.max(nodes[p.y])

        # Normalize coordinates
        nodes = nodes.copy()
        nodes[p.x] = minmax_normalize(nodes[p.x], xmin, xmax)
        nodes[p.y] = minmax_normalize(nodes[p.y], ymin, ymax)

        # Convert graph into list of edge segments
        edges, segment_class = _convert_graph_to_edge_segments(nodes, edges, p)

        # This is simply to let the work split out over multiple cores
        edge_batches = list(batches(edges, p.batch_size))

        squared_segment_length = SegmentLength(
            p.min_segment_length**2, p.max_segment_length**2,
            ((p.min_segment_length + p.max_segment_length) / 2)**2
        )

        # This gets the edges split into lots of small segments
        # Doing this inside a delayed function lowers the transmission overhead
        edge_segments = [resample_edges_fn(batch, squared_segment_length,
                                        segment_class.ndims) for batch in edge_batches]

        for i in range(p.iterations):
            # Each step, the size of the 'blur' shrinks
            bandwidth = p.initial_bandwidth * p.decay**(i + 1) * p.accuracy

            # If it's this small, there won't be a change anyway
            if bandwidth < 2:
                break

            # Draw the density maps and combine them
            images = [draw_to_surface_fn(segment, bandwidth, p.accuracy, segment_class.accumulate)
                      for segment in edge_segments]
            overall_image = sum(images)

            gradients = get_gradients_fn(overall_image)

            # Move edges along the gradients and resample when necessary
            # This could include smoothing to adjust the amount a graph can change
            edge_segments = [advect_resample_all_fn(gradients, segment, p.advect_iterations,
                                                 p.accuracy, squared_segment_length,
                                                 segment_class.idx, segment_class.idy,
                                                 segment_class.ndims)
                             for segment in edge_segments]

        # Do a final resample to a smaller size for nicer rendering
        edge_segments = [resample_edges_fn(segment, squared_segment_length,
                                        segment_class.ndims) for segment in edge_segments]

        if p.use_dask:
            # Finally things can be sent for computation
            edge_segments = compute(edge_segments)[0]

        # Flatten things
        new_segs = []
        for batch in edge_segments:
            new_segs.extend(batch)

        smooth(new_segs, p.tension, segment_class.idx, segment_class.idy)

        # Convert list of edge segments to Pandas dataframe
        df = _convert_edge_segments_to_dataframe(new_segs, segment_class, p)

        # Denormalize coordinates
        df[p.x] = minmax_denormalize(df[p.x], xmin, xmax)
        df[p.y] = minmax_denormalize(df[p.y], ymin, ymax)

        return df
