"""Support functions for dashboards."""

from pathlib import Path
from typing import Generator

import datashader as ds
import pandas as pd
from colorcet import palette
from datashader.colors import Elevation, Sets1to3, inferno, viridis

cmap_selection = [
    'bgy',
    'bgyw',
    'bjy',
    'bkr',
    'bky',
    'blues',
    'bmw',
    'bmy',
    'bwy',
    'circle_mgbm_67_c31_s25',
    'colorwheel',
    'coolwarm',
    'cwr',
    'cyclic_bgrmb_35_70_c75',
    'cyclic_bgrmb_35_70_c75_s25',
    'cyclic_grey_15_85_c0',
    'cyclic_grey_15_85_c0_s25',
    'cyclic_isoluminant',
    'cyclic_mrybm_35_75_c68',
    'cyclic_mrybm_35_75_c68_s25',
    'cyclic_mybm_20_100_c48',
    'cyclic_mybm_20_100_c48_s25',
    'cyclic_mygbm_30_95_c78',
    'cyclic_mygbm_50_90_c46',
    'cyclic_mygbm_50_90_c46_s25',
    'cyclic_protanopic_deuteranopic_bwyk_16_96_c31',
    'cyclic_protanopic_deuteranopic_wywb_55_96_c33',
    'cyclic_rygcbmr_50_90_c64',
    'cyclic_rygcbmr_50_90_c64_s25',
    'cyclic_tritanopic_cwrk_40_100_c20',
    'cyclic_tritanopic_wrwc_70_100_c20',
    'cyclic_wrkbw_10_90_c43',
    'cyclic_wrkbw_10_90_c43_s25',
    'cyclic_wrwbw_40_90_c42',
    'cyclic_wrwbw_40_90_c42_s25',
    'cyclic_ymcgy_60_90_c67',
    'cyclic_ymcgy_60_90_c67_s25',
    'dimgray',
    'diverging_bwg_20_95_c41',
    'diverging_bwr_20_95_c54',
    'diverging_bwr_55_98_c37',
    'diverging_cwm_80_100_c22',
    'diverging_gkr_60_10_c40',
    'diverging_gwr_55_95_c38',
    'diverging_isoluminant_cjm_75_c24',
    'diverging_linear_bjr_30_55_c53',
    'diverging_linear_protanopic_deuteranopic_bjy_57_89_c34',
    'diverging_rainbow_bgymr_45_85_c67',
    'fire',
    'glasbey',
    'glasbey_bw',
    'glasbey_bw_minc_20',
    'glasbey_bw_minc_20_hue_150_280',
    'glasbey_bw_minc_20_hue_330_100',
    'glasbey_bw_minc_20_maxl_70',
    'glasbey_bw_minc_20_minl_30',
    'glasbey_category10',
    'glasbey_cool',
    'glasbey_dark',
    'glasbey_hv',
    'glasbey_light',
    'glasbey_warm',
    'gouldian',
    'gray',
    'gwv',
    'isolum',
    'isoluminant_cgo_70_c39',
    'isoluminant_cm_70_c39',
    'kb',
    'kbc',
    'kbgyw',
    'kg',
    'kgy',
    'kr',
    'linear_bgyw_15_100_c67',
    'linear_bmw_5_95_c86',
    'linear_bmy_10_95_c71',
    'linear_gow_65_90_c35',
    'linear_kbgyw_5_98_c62',
    'linear_kry_0_97_c73',
    'linear_kryw_5_100_c67',
    'linear_protanopic_deuteranopic_kbjyw_5_95_c25',
    'linear_protanopic_deuteranopic_kbw_5_95_c34',
    'linear_protanopic_deuteranopic_kbw_5_98_c40',
    'linear_protanopic_deuteranopic_kyw_5_95_c49',
    'linear_tritanopic_kcw_5_95_c22',
    'linear_tritanopic_krjcw_5_95_c24',
    'linear_tritanopic_krjcw_5_98_c46',
    'linear_tritanopic_krw_5_95_c46',
    'linear_wcmr_100_45_c42',
    'linear_worb_100_25_c53',
    'linear_wyor_100_45_c55',
    'rainbow',
    'rainbow4',
    'rainbow_bgyrm_35_85_c71',
]

colormaps = {k: v for (k, v) in sorted(palette.items()) if k in cmap_selection}
colormaps['viridis'] = viridis
colormaps['inferno'] = inferno
colormaps['Sets1to3'] = Sets1to3
colormaps['Elevation'] = Elevation


def render_attractor(
    trajectory: pd.DataFrame, plot_type: str = 'points', cmap: list | None = None, size: int = 700, **kwargs
) -> Generator:
    """Render attractor's trajectory into an image using datashader."""
    if cmap is None:
        cmap = colormaps['inferno']
    cvs = ds.Canvas(plot_width=size, plot_height=size)
    agg = getattr(cvs, plot_type)(trajectory, 'x', 'y', agg=ds.count())
    yield ds.tf.shade(agg, cmap=cmap, **kwargs)


def save_image(img: ds.tf.Image, output_path: Path | str, color: str = 'black') -> None:
    """Export image to png file."""
    output_image = ds.tf.set_background(img, color=color)
    output_image.to_pil().save(output_path, format='png')
