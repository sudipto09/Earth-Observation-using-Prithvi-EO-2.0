"""
reporting/__init__.py

Reporting sub-package for batch summary figures. Exports build_batch_report()
and THEME. Assembles KPI cards, classification panels, ranked table, and field
thumbnails with NDVI trajectory sparklines into a single publication-ready figure.


"""

from reporting.batch_report import build_batch_report
from reporting.theme import THEME

__all__ = ['build_batch_report', 'THEME']