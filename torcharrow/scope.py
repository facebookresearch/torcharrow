# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import typing as ty
import warnings

import torcharrow as ta
import torcharrow.dtypes as dt

from .dispatcher import Device, Dispatcher
from .trace import Trace, trace

# ---------------------------------------------------------------------------
# Scope, pipelines global state...

# helper class


class Counter:
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        return self._value

    def next(self):
        n = self._value
        self._value += 1
        return n


# Scope will be deprecated in the future.
# Please DON'T use it in user-level code.
# pyre-fixme[13]: Attribute `default` is never initialized.
class Scope:
    """
    This class will be deprecated in the future.
    Please don't use it in user code.
    """

    default: ty.ClassVar["Scope"]

    default_config: ty.Dict = {
        "device": "cpu",
        "tracing": False,
        "types_to_trace": [],
    }

    def __init__(self, config: ty.Union[dict, str, None] = None):
        if config is None:
            self.config = type(self).default_config
        elif isinstance(config, str):
            path = config
            with open(path) as f:
                self.config = {**type(self).default_config, **json.load(f)}
        elif isinstance(config, dict):
            self.config = {**type(self).default_config, **config}

        self.ct = Counter()
        self.id = "s0"
        self._scope = self

        tracing = ty.cast(bool, self.config["tracing"])
        types = ty.cast(ty.Iterable[ty.Type], self.config["types_to_trace"])
        self.trace = Trace(tracing, tuple(types))

    # device handling --------------------------------------------------------

    @property
    def device(self):
        return self.config["device"]

    # tracing handling --------------------------------------------------------
    @property
    def tracing(self):
        return self.config["tracing"]

    # only one scope --------------------------------------------------------

    def is_same(self, other):
        return id(self) == id(other)

    def check_is_same(self, other):
        if id(self) != id(other) or self.device != other.device:
            raise TypeError("scope and device must be the same")

    def check_are_same(self, others):
        if not all(
            self.is_same(other) and self.device == other.device for other in others
        ):
            raise TypeError("scope and device must be the same")

    # TODO: refactor these static methods out of scope.py
    @staticmethod
    def _require_column_constructors_to_be_registered():
        from .idataframe import dataframe
        from .ilist_column import ListColumn
        from .imap_column import MapColumn
        from .istring_column import StringColumn
        from .velox_rt import NumericalColumnCpu

    # private column/dataframe constructors -----------------------------------
    @staticmethod
    def _EmptyColumn(dtype, device=""):
        """
        Column row builder method

        * This methods return a "column builder" that may not support all the methods of column.
        * Backend may provide more efficient _append_row implementation that doesn't need to copy the whole column per appending.
        * _finalize has to be called before returning the Column to user

        TODO: rename this method to _EmptyColumnBuilder to emphasize it returns a builder
        """
        Scope._require_column_constructors_to_be_registered()

        device = device or Scope.default.device
        call = Dispatcher.lookup((dtype.typecode + "_empty", device))

        return call(device, dtype)

    @staticmethod
    def _FullColumn(data, dtype, device="", mask=None):
        """
        Column vector builder method -- data is already in right form
        """
        Scope._require_column_constructors_to_be_registered()

        device = device or Scope.default.device
        call = Dispatcher.lookup((dtype.typecode + "_full", device))

        return call(device, data, dtype, mask)

    @staticmethod
    def _FromPySequence(data: ty.Sequence, dtype: dt.DType, device=""):
        """
        Convert from plain Python sequence (sequence of scalars or containers).
        """
        Scope._require_column_constructors_to_be_registered()

        device = device or Scope.default.device
        call = Dispatcher.lookup((dtype.typecode + "_from_pysequence", device))

        return call(device, data, dtype)

    @staticmethod
    def null_check_from_pysequence(data: ty.Sequence, dtype: dt.DType, device=""):
        result = Scope._FromPySequence(data, dtype, device)

        # since dtype is known, check the nullability
        if not dtype.nullable and result.null_count != 0:
            raise ValueError(f"None found in the list for non-nullable type: {dtype}")
        return result

    @staticmethod
    @trace
    def _Column(
        data=None,
        dtype: ty.Optional[dt.DType] = None,
        device: Device = "",
    ):
        """
        Column factory method
        """

        device = device or Scope.default.device

        if (data is None) and (dtype is None):
            raise TypeError(
                f"Column requires data and/or dtype parameter {data} {dtype}"
            )

        if isinstance(data, dt.DType) and isinstance(dtype, dt.DType):
            raise TypeError("Column can only have one dtype parameter")

        if isinstance(data, dt.DType):
            (data, dtype) = (dtype, data)

        # data is a list or a tuple; we only allow type inference from flat tuples
        # pyre-fixme[6]: For 2nd param expected `Union[Type[typing.Any],
        #  typing.Tuple[Type[typing.Any], ...]]` but got `_SpecialForm`.
        if isinstance(data, ty.List) or isinstance(data, ty.Tuple):
            # TODO: infer the type from the whole list
            dtype = dtype or dt.infer_dtype_from_prefix(data[:7])
            if dtype is None or dt.is_any(dtype):
                raise ValueError("Column cannot infer type from data")
            if dt.contains_tuple(dtype):
                raise TypeError("Cannot infer type from nested Python tuple")

            return Scope.null_check_from_pysequence(data, dtype, device)

        if Scope._is_column(data):
            dtype = dtype or data.dtype
            if data.device == device and data.dtype == dtype:
                return data
            else:
                # TODO: More efficient interop, such as leveraging arrow
                return ta.from_pysequence(data.to_pylist(), dtype=dtype, device=device)

        if data is not None:
            warnings.warn(
                "Constructing column from non Python list/Column may result in degenerated performance"
            )

        # dtype given, optional data
        if isinstance(dtype, dt.DType):
            col = Scope._EmptyColumn(dtype, device)
            if data is not None:
                for i in data:
                    col._append(i)
            return col._finalize()

        # data given, optional column
        if data is not None:
            if isinstance(data, ty.Iterable):
                # make sure that we are dealing with an actual Iterator
                data = iter(data)

                prefix = []
                for i, v in enumerate(data):
                    prefix.append(v)
                    if i > 5:
                        break
                dtype = dt.infer_dtype_from_prefix(prefix)
                if dtype is None or dt.is_any(dtype):
                    raise ValueError("Column cannot infer type from data")

                if dt.is_tuple(dtype):
                    # TODO fix me
                    raise TypeError(
                        "Column cannot be used to created structs, use Dataframe constructor instead"
                    )
                col = Scope._EmptyColumn(dtype, device=device)
                # add prefix and ...
                for p in prefix:
                    col._append(p)
                # ... continue to add the rest of the data
                for i in data:
                    col._append(i)
                return col._finalize()
            else:
                raise TypeError(
                    f"data parameter of ty.Iterable type expected (got {type(dtype).__name__})"
                )
        else:
            raise AssertionError("unexpected case")

    # public dataframe (column)) constructor
    @staticmethod
    @trace
    def _DataFrame(
        data=None,  # : DataOrDTypeOrNone = None,
        dtype=None,  # : ty.Optional[dt.DType] = None,
        columns=None,  # : ty.Optional[List[str]] = None,
        device="",
    ):
        """
        Dataframe factory method
        """

        if data is None and dtype is None:
            assert columns is None
            return Scope._EmptyColumn(dt.Struct([]), device=device)._finalize()

        if data is not None and isinstance(data, dt.DType):
            if dtype is not None and isinstance(dtype, dt.DType):
                raise TypeError("Dataframe can only have one dtype parameter")
            dtype = data
            data = None

        if Scope._is_dataframe(data):
            dtype = dtype or data.dtype
            if data.device == device and data.dtype == dtype:
                return data
            else:
                dtype_fields = {f.name for f in dtype.fields}
                data_fields = {f.name for f in data.dtype.fields}
                if dtype_fields != data_fields:
                    raise TypeError(
                        f"data fields are {data_fields} while dtype fields are {dtype_fields}"
                    )

                res = {
                    n: Scope._Column(data[n], dtype=dtype.get(n), device=device)
                    for n in data.columns
                }
                return Scope._DataFrame(res, dtype=dtype, device=device)

        # dtype given, optional data
        if dtype is not None:
            if not dt.is_struct(dtype):
                raise TypeError(
                    f"Dataframe takes a dt.Struct dtype as parameter (got {dtype})"
                )
            dtype = ty.cast(dt.Struct, dtype)
            if data is None:
                return Scope._EmptyColumn(dtype, device=device)._finalize()
            else:
                if isinstance(data, ty.Sequence):
                    return Scope._Column(data, dtype, device)
                elif isinstance(data, ty.Mapping):
                    res = {}
                    dtype_fields = {f.name: f.dtype for f in dtype.fields}

                    if len(data) != len(dtype_fields):
                        raise TypeError(
                            f"""dtype provides {len(dtype.fields)} fields: {dtype_fields.keys()}
but data only provides {len(data)} fields: {data.keys()}
"""
                        )

                    for n, c in data.items():
                        if n not in dtype_fields:
                            raise AttributeError(
                                f"Column {n} is present in the data but absent in explicitly provided dtype"
                            )
                        if Scope._is_column(c):
                            if c.dtype != dtype_fields[n]:
                                raise TypeError(
                                    f"Wrong type for column {n}: dtype specifies {dtype_fields[n]} while column of {c.dtype} is provided"
                                )
                        else:
                            c = Scope._Column(c, dtype_fields[n])
                        res[n] = c

                    return Scope._FullColumn(res, dtype)
                else:
                    raise TypeError(
                        f"Dataframe does not support constructor for data of type {type(data).__name__}"
                    )

        # data given, optional column
        if data is not None:
            if isinstance(data, ty.Sequence):
                prefix = []
                for i, v in enumerate(data):
                    prefix.append(v)
                    if i > 5:
                        break
                dtype = dt.infer_dtype_from_prefix(prefix)
                if dtype is None or not dt.is_tuple(dtype):
                    raise TypeError("Dataframe cannot infer struct type from data")
                dtype = ty.cast(dt.Tuple, dtype)
                if columns is not None:
                    if len(dtype.fields) != len(columns):
                        raise TypeError("Dataframe column length must equal row length")
                else:
                    first_tuple_columns = None
                    for v in data:
                        if v is not None:
                            if hasattr(v, "_fields"):
                                first_tuple_columns = v._fields
                                break
                    if first_tuple_columns is None:
                        raise TypeError(
                            "DataFrame construction from tuples requires dtype or columns to be given"
                        )
                    columns = list(first_tuple_columns)
                dtype = dt.Struct(
                    [dt.Field(n, t) for n, t in zip(columns, dtype.fields)]
                )

                return Scope._Column(data, dtype, device)
            elif isinstance(data, ty.Mapping):
                res = {}
                for n, c in data.items():
                    if Scope._is_column(c):
                        res[n] = c
                    elif isinstance(c, ty.Sequence):
                        res[n] = Scope._Column(c, device=device)
                    else:
                        raise TypeError(
                            f"dataframe does not support constructor for column data of type {type(c).__name__}"
                        )
                return Scope._FullColumn(
                    res,
                    dtype=dt.Struct([dt.Field(n, c.dtype) for n, c in res.items()]),
                    device=device,
                )
            else:
                raise TypeError(
                    f"dataframe does not support constructor for data of type {type(data).__name__}"
                )
        else:
            raise AssertionError("unexpected case")

    # helper ------------------------------------------------------------------
    @staticmethod
    def _is_column(c):
        # NOTE: should be isinstance(c, Column)
        # But can't do tha due to cyclic reference, so we use ...
        return c is not None and hasattr(c, "_dtype") and hasattr(c, "_device")

    @staticmethod
    def _is_dataframe(c):
        # NOTE: should be isinstance(c, DataFrame)
        # But can't do tha due to cyclic reference, so we use ...
        return (
            c is not None
            and hasattr(c, "_dtype")
            and hasattr(c, "_device")
            and hasattr(c, "columns")
        )


# ------------------------------------------------------------------------------
# registering the default scope
Scope.default = Scope()
