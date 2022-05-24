.. currentmodule:: torcharrow

torcharrow.Column
==========================

A :class:`torcharrow.Column` is a 1-dimension torch.Tensor like data structure containing 
elements of a single data type. It also supports non-numeric types such as string, 
list, struct.


Column class reference
------------------------------
.. class:: Column()

.. autoattribute:: Column.dtype
.. autoattribute:: Column.device
.. autoattribute:: Column.length
.. autoattribute:: Column.null_count

.. autosummary::
    :toctree: generated
    :nosignatures:

    Column.head
    Column.tail
    Column.cast
    Column.is_valid_at
    Column.append
    Column.isin

    Column.all
    Column.any

    Column.map
    Column.filter
    Column.flatmap
    Column.transform

    Column.fill_null
    Column.drop_null
    Column.drop_duplicates

    Column.to_arrow
    Column.to_tensor
    Column.to_pylist
    Column.to_pandas    


NumericalColumn class reference
-----------------------------------
.. class:: NumericalColumn()

.. autosummary::
    :toctree: generated
    :nosignatures:

    NumericalColumn.abs
    NumericalColumn.ceil
    NumericalColumn.floor
    NumericalColumn.round
    NumericalColumn.log

    NumericalColumn.describe
    NumericalColumn.min
    NumericalColumn.max
    NumericalColumn.sum
    NumericalColumn.mean
    NumericalColumn.std
    NumericalColumn.median


StringColumn class reference
-----------------------------------
.. class:: StringColumn()

.. autosummary::
    :toctree: generated
    :nosignatures:

    torcharrow.istring_column.StringMethods.length
    torcharrow.istring_column.StringMethods.slice
    torcharrow.istring_column.StringMethods.split
    torcharrow.istring_column.StringMethods.strip

    torcharrow.istring_column.StringMethods.isalpha
    torcharrow.istring_column.StringMethods.isnumeric
    torcharrow.istring_column.StringMethods.isalnum
    torcharrow.istring_column.StringMethods.isdigit
    torcharrow.istring_column.StringMethods.isdecimal
    torcharrow.istring_column.StringMethods.isspace
    torcharrow.istring_column.StringMethods.islower
    torcharrow.istring_column.StringMethods.isupper
    torcharrow.istring_column.StringMethods.istitle

    torcharrow.istring_column.StringMethods.lower
    torcharrow.istring_column.StringMethods.upper

    torcharrow.istring_column.StringMethods.startswith
    torcharrow.istring_column.StringMethods.endswith
    torcharrow.istring_column.StringMethods.count
    torcharrow.istring_column.StringMethods.find
    torcharrow.istring_column.StringMethods.replace
    torcharrow.istring_column.StringMethods.match
    torcharrow.istring_column.StringMethods.contains
    torcharrow.istring_column.StringMethods.findall
    
ListColumn class reference
-----------------------------------
.. class:: ListColumn()

.. autosummary::
    :toctree: generated
    :nosignatures:

    torcharrow.ilist_column.ListMethods.length
    torcharrow.ilist_column.ListMethods.slice
    torcharrow.ilist_column.ListMethods.vmap
