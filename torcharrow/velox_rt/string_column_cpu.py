# Copyright (c) Facebook, Inc. and its affiliates.
import array as ar
from dataclasses import dataclass
from typing import cast, List

import numpy as np
import numpy.ma as ma
import torcharrow._torcharrow as velox
import torcharrow.dtypes as dt
from tabulate import tabulate
from torcharrow.expression import expression
from torcharrow.functional import functional
from torcharrow.istring_column import IStringColumn, IStringMethods
from torcharrow.scope import ColumnFactory, Scope, Device

from .column import ColumnFromVelox
from .typing import get_velox_type

# ------------------------------------------------------------------------------
# StringColumnCpu


class StringColumnCpu(IStringColumn, ColumnFromVelox):

    # private constructor
    def __init__(self, scope, device, dtype, data, mask):  # REP offsets
        assert dt.is_string(dtype)
        super().__init__(scope, device, dtype)

        self._data = velox.Column(get_velox_type(dtype))
        for m, d in zip(mask.tolist(), data):
            if m:
                self._data.append_null()
            else:
                self._data.append(d)
        self._finialized = False

        self.str = StringMethodsCpu(self)
        # REP: self._offsets = offsets

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(scope, device, dtype):
        # REP  ar.array("I", [0])
        return StringColumnCpu(scope, device, dtype, [], ar.array("b"))

    @staticmethod
    def _full(scope, device, data, dtype=None, mask=None):
        assert isinstance(data, np.ndarray) and data.ndim == 1
        if dtype is None:
            dtype = dt.typeof_np_ndarray(data.dtype)
            if dtype is None:  # could be an object array
                mask = np.vectorize(_is_not_str)(data)
                dtype = dt.string
        else:
            pass
            # if dtype != typeof_np_ndarray:
            # pass
            # TODO refine this test
            # raise TypeError(f'type of data {data.dtype} and given type {dtype }must be the same')
        if not dt.is_string(dtype):
            raise TypeError(f"construction of columns of type {dtype} not supported")
        if mask is None:
            mask = np.vectorize(_is_not_str)(data)
        elif len(data) != len(mask):
            raise ValueError(f"data length {len(data)} must be mask length {len(mask)}")
        # TODO check that all non-masked items are strings
        return StringColumnCpu(scope, device, dtype, data, mask)

    @staticmethod
    def _fromlist(scope, device: str, data: List[str], dtype: dt.DType):
        velox_column = velox.Column(get_velox_type(dtype), data)
        return ColumnFromVelox.from_velox(
            scope,
            device,
            dtype,
            velox_column,
            True,
        )

    def _append_null(self):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        self._data.append_null()

    def _append_value(self, value):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        else:
            self._data.append(value)

    def _finalize(self):
        self._finialized = True
        return self

    def __len__(self):
        return len(self._data)

    def null_count(self):
        return self._data.get_null_count()

    def getmask(self, i):
        if i < 0:
            i += len(self._data)
        return self._data.is_null_at(i)

    def getdata(self, i):
        if i < 0:
            i += len(self._data)
        if self._data.is_null_at(i):
            return self.dtype.default
        else:
            return self._data[i]

    @staticmethod
    def _valid_mask(ct):
        raise np.full((ct,), False, dtype=np.bool8)

    def gets(self, indices):
        data = self._data[indices]
        mask = self._mask[indices]
        return self.scope._FullColumn(data, self.dtype, self.device, mask)

    def slice(self, start, stop, step):
        range = slice(start, stop, step)
        return self.scope._FullColumn(
            self._data[range], self.dtype, self.device, self._mask[range]
        )

    # operators ---------------------------------------------------------------
    @expression
    def __eq__(self, other):
        if isinstance(other, StringColumnCpu):
            res = self._EmptyColumn(
                dt.Boolean(self.dtype.nullable or other.dtype.nullable),
            )
            for (m, i), (n, j) in zip(self.items(), other.items()):
                if m or n:
                    res._data.append_null()
                else:
                    res._data.append(i == j)
            return res._finalize()
        else:
            res = self._EmptyColumn(dt.Boolean(self.dtype.nullable))
            for (m, i) in self.items():
                if m:
                    res._data.append_null()
                else:
                    res._data.append(i == other)
            return res._finalize()

    # printing ----------------------------------------------------------------

    def __str__(self):
        def quote(x):
            return f"'{x}'"

        return f"Column([{', '.join('None' if i is None else quote(i) for i in self)}])"

    def __repr__(self):
        tab = tabulate(
            [["None" if i is None else f"'{i}'"] for i in self],
            tablefmt="plain",
            showindex=True,
        )
        typ = f"dtype: {self.dtype}, length: {self.length()}, null_count: {self.null_count()}, device: cpu"
        return tab + dt.NL + typ


# ------------------------------------------------------------------------------
# StringMethodsCpu


class StringMethodsCpu(IStringMethods):
    """Vectorized string functions for IStringColumn"""

    def __init__(self, parent: StringColumnCpu):
        super().__init__(parent)

    def cat(self, others=None, sep: str = "", fill_value: str = None) -> IStringColumn:
        """
        Concatenate strings with given separator and n/a substitition.
        """
        me = cast(StringColumnCpu, self._parent)
        assert all(me.device == other.device for other in others)

        _all = [me] + others

        # mask
        res_mask = me._mask
        if fill_value is None:
            for one in _all:
                if res_mask is None:
                    res_mak = one.mask
                elif one.mask is not None:
                    res_mask = res_mask | one.mask

        # fill
        res_filled = []
        for one in _all:
            if fill_value is None:
                res_filled.append(one.fillna(fill_value))
            else:
                res_filled.append(one)
        # join
        has_nulls = fill_value is None and any(one.nullable for one in _all)
        res = me._EmptyColumn(dt.String(has_nulls))

        for ws in zip(res_filled):
            # will throw if join is applied on null
            res._append_value(sep.join(ws))
        return res._finalize()

    def slice(self, start: int = None, stop: int = None) -> IStringColumn:
        start = start or 0
        if stop is None:
            return functional.substr(self._parent, start + 1).with_null(
                self._parent.dtype.nullable
            )
        else:
            return functional.substr(self._parent, start + 1, stop - start).with_null(
                self._parent.dtype.nullable
            )

    def lower(self) -> IStringColumn:
        return functional.lower(self._parent).with_null(self._parent.dtype.nullable)

    def upper(self) -> IStringColumn:
        return functional.upper(self._parent).with_null(self._parent.dtype.nullable)

    def isalpha(self) -> IStringColumn:
        return functional.torcharrow_isalpha(self._parent).with_null(
            self._parent.dtype.nullable
        )

    def isalnum(self) -> IStringColumn:
        return functional.torcharrow_isalnum(self._parent).with_null(
            self._parent.dtype.nullable
        )

    def isdecimal(self) -> IStringColumn:
        return functional.isdecimal(self._parent).with_null(self._parent.dtype.nullable)

    def islower(self) -> IStringColumn:
        return functional.torcharrow_islower(self._parent).with_null(
            self._parent.dtype.nullable
        )

    def isupper(self) -> IStringColumn:
        return functional.isupper(self._parent).with_null(self._parent.dtype.nullable)

    def startswith(self, pat):
        return (
            functional.substr(self._parent, 1, len(pat)).with_null(
                self._parent.dtype.nullable
            )
            == pat
        )

    def isspace(self) -> IStringColumn:
        return functional.torcharrow_isspace(self._parent).with_null(
            self._parent.dtype.nullable
        )

    def istitle(self) -> IStringColumn:
        return functional.torcharrow_istitle(self._parent).with_null(
            self._parent.dtype.nullable
        )

    def isnumeric(self) -> IStringColumn:
        return functional.isnumeric(self._parent).with_null(self._parent.dtype.nullable)

    def match_re(self, pattern: str):
        return functional.match_re(self._parent, pattern).with_null(
            self._parent.dtype.nullable
        )

    def contains_re(
        self,
        pattern: str,
    ):
        return functional.regexp_like(self._parent, pattern).with_null(
            self._parent.dtype.nullable
        )


# ------------------------------------------------------------------------------
# registering the factory
ColumnFactory.register((dt.String.typecode + "_empty", "cpu"), StringColumnCpu._empty)
ColumnFactory.register((dt.String.typecode + "_full", "cpu"), StringColumnCpu._full)
ColumnFactory.register(
    (dt.String.typecode + "_fromlist", "cpu"), StringColumnCpu._fromlist
)


def _is_not_str(s):
    return not isinstance(s, str)
