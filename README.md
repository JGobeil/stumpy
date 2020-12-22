# Stumpy

Stumpy is a Python 3 library to read, visualize and analyze the STM data files create by the [Nanonis](https://www.specs-group.com/nanonis/products/) control system.

Each opened files have two main attributes: the `header` and the `data`. In the `header` is the information contained in the header (date and time, experiments parameters, ...). The `data` contains the raw data as a `numpy.ndarray` for `.sxm` and a `pandas.DataFrame` for `.dat`.

Plotting is easy and customizable and use the powerful `matplotlib` library.

The library also contains specialized classes for bias and z spectroscopy experiments and tools to modify the scanned image (automatic drift correction, interpolating, cutting, ...).





