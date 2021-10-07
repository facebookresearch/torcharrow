# Copyright (c) Facebook, Inc. and its affiliates.
import array as ar
import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import numpy as np
import torcharrow.dtypes as dt
import torcharrow.pytorch as pytorch
from tabulate import tabulate
from torcharrow.column_factory import ColumnFactory
from torcharrow.icolumn import IColumn
from torcharrow.imap_column import IMapColumn, IMapMethods


# -----------------------------------------------------------------------------
# IMapColumn


class MapColumnDemo(IMapColumn):
    def __init__(self, scope, device, dtype, key_data, item_data, mask):
        assert dt.is_map(dtype)
        super().__init__(scope, device, dtype)

        self._key_data = key_data
        self._item_data = item_data
        self._mask = mask

        self.maps = MapMethodsStd(self)

    # Lifecycle: _empty -> _append* -> _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(scope, device, dtype):
        key_data = scope._EmptyColumn(
            dt.List(dtype.key_dtype).with_null(dtype.nullable)
        )
        item_data = scope._EmptyColumn(
            dt.List(dtype.item_dtype).with_null(dtype.nullable)
        )
        return MapColumnDemo(scope, device, dtype, key_data, item_data, ar.array("b"))

    @staticmethod
    def _full(scope, device, data, dtype=None, mask=None):
        assert isinstance(data, tuple) and len(data) == 2
        key_data, item_data = data
        assert isinstance(key_data, IColumn)
        assert isinstance(item_data, IColumn)
        assert len(item_data) == len(key_data)

        if dtype is None:
            dtype = dt.Map(
                dt.typeof_np_ndarray(key_data.dtype),
                dt.typeof_np_ndarray(item_data.dtype),
            )
        # else:
        #     if dtype != dt.typeof_np_dtype(data.dtype):
        #         # TODO fix nullability
        #         # raise TypeError(f'type of data {data.dtype} and given type {dtype} must be the same')
        #         pass
        if not dt.is_map(dtype):
            raise TypeError(f"construction of columns of type {dtype} not supported")
        if mask is None:
            mask = IMapColumn._valid_mask(len(key_data))
        elif len(key_data) != len(mask):
            raise ValueError(
                f"data length {len(key_data)} must be the same as mask length {len(mask)}"
            )
        # TODO check that all non-masked items are legal numbers (i.e not nan)
        return MapColumnDemo(scope, device, dtype, key_data, item_data, mask)

    @staticmethod
    def _fromlist(scope, device, data: List, dtype):
        # default implementation
        col = MapColumnDemo._empty(scope, device, dtype)
        for i in data:
            col._append(i)
        return col._finalize()

    def _append_null(self):
        self._mask.append(True)
        self._key_data._append_null()
        self._item_data._append_null()

    def _append_value(self, value):
        self._mask.append(False)
        self._key_data._append_value(list(value.keys()))
        self._item_data._append_value(list(value.values()))

    def _append_data(self, value):
        self._key_data._append_value(list(value.keys()))
        self._item_data._append_value(list(value.values()))

    def _finalize(self, mask=None):
        self._key_data = self._key_data._finalize()
        self._item_data = self._item_data._finalize()
        if not isinstance(self._mask, np.ndarray):
            self._mask = np.array(self._mask, dtype=np.bool8, copy=False)
        return self

    def __len__(self):
        return len(self._key_data)

    def null_count(self):
        return self._mask.sum()

    def getmask(self, i):
        return self._mask[i]

    def getdata(self, i):
        return {k: v for k, v in zip(self._key_data[i], self._item_data[i])}

    @staticmethod
    def _valid_mask(ct):
        raise np.full((ct,), False, dtype=np.bool8)

    def append(self, values):
        """Returns column/dataframe with values appended."""
        tmp = self.scope.Column(values, dtype=self.dtype, device=self.device)
        return self.scope._FullColumn(
            (
                self._key_data.append(tmp._key_data),
                self._item_data.append(tmp._item_data),
            ),
            self.dtype,
            self.device,
            np.append(self._mask, tmp._mask),
        )

    # printing ----------------------------------------------------------------
    def __str__(self):
        return f"Column([{', '.join('None' if i is None else str(i) for i in self)}])"

    def __repr__(self):
        tab = tabulate(
            [["None" if i is None else str(i)] for i in self],
            tablefmt="plain",
            showindex=True,
        )
        typ = f"dtype: {self.dtype}, length: {self.length()}, null_count: {self.null_count()}"
        return tab + dt.NL + typ

    def to_torch(self):
        pytorch.ensure_available()
        import torch

        keys = self._key_data.to_torch(_propagate_py_list=False)
        vals = self._item_data.to_torch(_propagate_py_list=False)
        # TODO: should we propagate python list if both keys and vals are lists of strings?
        if isinstance(keys, pytorch.WithPresence):
            keys = keys.values
        if isinstance(vals, pytorch.WithPresence):
            vals = vals.values
        assert isinstance(keys, pytorch.PackedList), keys
        assert isinstance(vals, pytorch.PackedList), vals
        assert torch.all(keys.offsets == vals.offsets)
        res = pytorch.PackedMap(
            keys=keys.values, values=vals.values, offsets=keys.offsets
        )
        if not self._dtype.nullable:
            return res

        presence = torch.tensor(self._mask, dtype=torch.bool).bitwise_not()
        return pytorch.WithPresence(values=res, presence=presence)


# ------------------------------------------------------------------------------
# registering the factory
ColumnFactory.register((dt.Map.typecode + "_empty", "demo"), MapColumnDemo._empty)
ColumnFactory.register((dt.Map.typecode + "_full", "demo"), MapColumnDemo._full)
ColumnFactory.register((dt.Map.typecode + "_fromlist", "demo"), MapColumnDemo._fromlist)
# -----------------------------------------------------------------------------
# MapMethods


@dataclass
class MapMethodsStd(IMapMethods):
    """Vectorized list functions for IListColumn"""

    def __init__(self, parent: MapColumnDemo):
        super().__init__(parent)

    def keys(self):
        me = self._parent
        return me._key_data

    def values(self):
        me = self._parent
        return me._item_data


# ops on maps --------------------------------------------------------------
#  'get',
#  'items',
#  'keys',
#  'pop',
#  'popitem',
#  'setdefault',
#  'update',
#  'values'
