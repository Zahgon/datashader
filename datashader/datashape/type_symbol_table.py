"""
A symbol table object to hold types for the parser.
"""

import ctypes
from itertools import chain

from . import coretypes as ct

__all__ = ['TypeSymbolTable', 'sym']


_is_64bit = (ctypes.sizeof(ctypes.c_void_p) == 8)


def _complex(tp):
    """Simple temporary type constructor for complex"""
    pass


def _struct(names, dshapes):
    """Simple temporary type constructor for struct"""
    pass


def _funcproto(args, ret):
    """Simple temporary type constructor for funcproto"""
    pass


def _typevar_dim(name):
    """Simple temporary type constructor for typevar as a dim"""
    pass


def _typevar_dtype(name):
    """Simple temporary type constructor for typevar as a dtype"""
    pass


def _ellipsis(name):
    pass

# data types with no type constructor
no_constructor_types = [
    ('bool', ct.bool_),
    ('int8', ct.int8),
    ('int16', ct.int16),
    ('int32', ct.int32),
    ('int64', ct.int64),
    ('intptr', ct.int64 if _is_64bit else ct.int32),
    ('int', ct.int32),
    ('uint8', ct.uint8),
    ('uint16', ct.uint16),
    ('uint32', ct.uint32),
    ('uint64', ct.uint64),
    ('uintptr', ct.uint64 if _is_64bit else ct.uint32),
    ('float16', ct.float16),
    ('float32', ct.float32),
    ('float64', ct.float64),
    ('complex64', ct.complex64),
    ('complex128', ct.complex128),
    ('real', ct.float64),
    ('complex', ct.complex_float64),
    ('string', ct.string),
    ('json', ct.json),
    ('date', ct.date_),
    ('time', ct.time_),
    ('datetime', ct.datetime_),
    ('timedelta', ct.timedelta_),
    ('null', ct.null),
    ('void', ct.void),
    ('object', ct.object_),
]

# data types with a type constructor
constructor_types = [
    ('complex', _complex),
    ('string', ct.String),
    ('struct', _struct),
    ('tuple', ct.Tuple),
    ('funcproto', _funcproto),
    ('typevar', _typevar_dtype),
    ('option', ct.Option),
    ('map', ct.Map),
    ('time', ct.Time),
    ('datetime', ct.DateTime),
    ('timedelta', ct.TimeDelta),
    ('units', ct.Units),
    ('decimal', ct.Decimal),
    ('categorical', ct.Categorical),
]

# dim types with no type constructor
dim_no_constructor = [
    ('var', ct.Var()),
    ('ellipsis', ct.Ellipsis()),
]

# dim types with a type constructor
dim_constructor = [
    ('fixed', ct.Fixed),
    ('typevar', _typevar_dim),
    ('ellipsis', _ellipsis),
]


class TypeSymbolTable:

    """
    This is a class which holds symbols for types and type constructors,
    and is used by the datashape parser to build types during its parsing.
    A TypeSymbolTable sym has four tables, as follows:

    sym.dtype
        Data type symbols with no type constructor.
    sym.dtype_constr
        Data type symbols with a type constructor. This may contain
        symbols also in sym.dtype, e.g. for 'complex' and 'complex[float64]'.
    sym.dim
        Dimension symbols with no type constructor.
    sym.dim_constr
        Dimension symbols with a type constructor.
    """
    __slots__ = ['dtype', 'dtype_constr', 'dim', 'dim_constr']

    def __init__(self, bare=False):
        # Initialize all the symbol tables to empty dicts1
        self.dtype = {}
        self.dtype_constr = {}
        self.dim = {}
        self.dim_constr = {}
        if not bare:
            self.add_default_types()

    def add_default_types(self):
        """
        Adds all the default datashape types to the symbol table.
        """
        pass

# Create the default global type symbol table
sym = TypeSymbolTable()
