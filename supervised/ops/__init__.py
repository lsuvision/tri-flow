"""
Operators package
=================

This package contains low‑level operators used by the rest of the
project.  Currently only ``flow.py`` is provided, exposing basic
functions for working with optical flow fields:

* :func:`_base_grid` – create a base coordinate grid
* :func:`flow_warp` – warp one flow field by another
* :func:`flow_compose` – compose two flow fields
* :func:`in_bounds_mask` – compute a mask of valid flow destinations

These functions are imported in various places throughout the codebase
using either relative imports (e.g. ``from ..ops.flow import ...``)
or absolute imports (``from ops.flow import ...``).  Both import
styles are supported provided that the ``ops`` package resides at the
same level as the importing module.
"""

from .flow import _base_grid, flow_warp, flow_compose, in_bounds_mask

__all__ = ["_base_grid", "flow_warp", "flow_compose", "in_bounds_mask"]