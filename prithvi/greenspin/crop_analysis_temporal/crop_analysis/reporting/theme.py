"""
reporting/theme.py

Frozen Theme dataclass with all colours, font sizes, and DPI for report figures.
Call apply_matplotlib_defaults(THEME) once at the start of any reporting script.
label_color(label) maps a classification string to its agronomic accent colour.


"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Theme:
    
    # agronomic semantic colors
    color_homogeneous: str  = '#2E7D32'   #  uniform / healthy
    color_intra_crop:  str  = '#F9A825'   # within-field variability
    color_multi_crop:  str  = '#C62828'   #  distinct land cover
    color_weak:    str  = '#9E9E9E'   # weakly variable

  
    color_text: str  = '#212121'
    color_text_muted:str  = '#616161'
    color_bg:   str  = '#FAFAFA'
    color_bg_panel: str  = '#FFFFFF'
    color_grid: str  = '#E0E0E0'
    color_border: str  = '#BDBDBD'

    
    color_kpi_value:   str  = '#1565C0'
    color_kpi_label:   str  = '#616161'

    
    font_family:  str  = 'DejaVu Sans'
    font_size_title:int  = 14
    font_size_axis: int  = 11
    font_size_kpi: int  = 28
    font_size_kpi_lbl: int  = 11
    font_size_small: int  = 9
    font_size_footer:int  = 8

    
    fig_dpi:int  = 140
    panel_alpha: float = 0.85

    def label_color(self, label: str) -> str:
        return {
            'homogeneous': self.color_homogeneous,
            'multi-crop': self.color_multi_crop,
            'intra-crop':  self.color_intra_crop,
            'weakly-variable': self.color_weak,
        }.get(label, self.color_weak)



THEME = Theme()


def apply_matplotlib_defaults(theme: Theme = THEME) -> None:
    
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family':  theme.font_family,
        'font.size': theme.font_size_axis,
        'axes.titlesize': theme.font_size_title,
        'axes.labelsize': theme.font_size_axis,
        'axes.edgecolor': theme.color_border,
        'axes.labelcolor':theme.color_text,
        'axes.titlecolor': theme.color_text,
        'axes.titleweight':'bold',
        'axes.spines.top': False,
        'axes.spines.right':False,
        'xtick.color':  theme.color_text_muted,
        'ytick.color':theme.color_text_muted,
        'grid.color':  theme.color_grid,
        'grid.linewidth':  0.6,
        'figure.facecolor': theme.color_bg,
        'axes.facecolor':theme.color_bg_panel,
        'savefig.facecolor':theme.color_bg,
        'savefig.dpi': theme.fig_dpi,
    })