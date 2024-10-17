"""Panel dashboard to visualise attractors of various types.

This app is based on a panel notebook available here:
https://examples.holoviz.org/gallery/attractors/attractors.html

It can be launched with:

    > panel serve --show src/attractor_explorer/attractors_explorer.py

"""
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false

import panel as pn
import param

from attractor_explorer import attractors as at
from attractor_explorer.shared import colormaps, render_attractor

try:
    from attractor_explorer._version import __version__
except ModuleNotFoundError:
    __version__ = 'unknown'

RESOLUTIONS = {'Low': 200_000, 'Medium': 10_000_000, 'High': 50_000_000, 'Very High': 100_000_000}
PLOT_SIZE = 800
SIDEBAR_WIDTH = 360
CSS = """
:root {
--background-color: black
--design-background-color: black;
--design-background-text-color: #ff7f00;   # orange
--panel-surface-color: black;
}
#sidebar {
background-color: black;
}
"""
# Hack can be removed when https://github.com/holoviz/panel/issues/7360 has been solved
CMAP_CSS_HACK = 'div, div:hover {background: #2b3035; color: white}'

pn.extension('katex', design='material', global_css=[CSS], sizing_mode='stretch_width')  # type: ignore
pn.config.throttled = True


class AttractorsExplorer(pn.viewable.Viewer):
    """Select and render attractors."""

    param_sets = at.ParameterSets(name='Attractors')
    attractor_type = param.Selector(
        default=param_sets.attractors['Clifford'], objects=param_sets.attractors, precedence=0.5
    )

    resolution = param.Selector(
        doc='Resolution (n points)',
        objects=RESOLUTIONS.keys(),
        default='Low',
        precedence=0.6,
    )
    n_points = param.Integer(
        RESOLUTIONS['Low'],
        bounds=(1, None),
        softbounds=(1, RESOLUTIONS['Very High']),
        doc='Number of points',
        precedence=0.8,
    )
    colormap = pn.widgets.ColorMap(
        value=colormaps['fire'],
        options=colormaps,
        ncols=1,
        swatch_width=100,
        margin=(25, 0, 200, 0),
        stylesheets=[CMAP_CSS_HACK],
    )

    # Interpolation method in datashader
    interpolation = pn.widgets.RadioButtonGroup(
        value='eq_hist', options=['eq_hist', 'cbrt', 'log', 'linear'], button_type='primary'
    )

    @param.depends('attractor_type.param', 'n_points', 'colormap.value', 'interpolation.value')
    def __panel__(self):
        trajectory = self.attractor_type(n=self.n_points)  # type: ignore
        return render_attractor(trajectory, cmap=self.colormap.value, size=PLOT_SIZE, how=self.interpolation.value)

    @param.depends('attractor_type.param')
    def equations(self):
        if not self.attractor_type.equations:
            return pn.Column()
        return pn.Column(
            *[pn.pane.LaTeX(e, styles={'font-size': '15pt'}) for e in self.attractor_type.equations],
        )

    @param.depends('resolution', watch=True)
    def set_npoints(self):
        self.n_points = RESOLUTIONS[self.resolution]

    @pn.depends('param_sets.example', watch=True)
    def update_attractor(self):
        a = self.param_sets.get_attractor(*self.param_sets.example)
        if a is not self.attractor_type:
            self.param.update(attractor_type=a)
            self.colormap.value = colormaps[self.param_sets.example[1]]  # type: ignore


ats = AttractorsExplorer(name='Attractors Explorer')
ats.param_sets.current = lambda: ats.attractor_type


pn.template.FastListTemplate(
    title='Attractor Explorer',
    sidebar=[
        pn.Param(
            ats.param,
            widgets={
                'attractor_type': {
                    'widget_type': pn.widgets.RadioButtonGroup,
                    'orientation': 'vertical',
                    'button_type': 'warning',
                    'button_style': 'outline',
                    'stylesheets': [':host(.outline) .bk-btn-group .bk-btn-warning.bk-active {color:white}'],
                },
                'resolution': {'widget_type': pn.widgets.RadioButtonGroup, 'button_type': 'success'},
            },
            expand=True,
            show_name=False,
        ),
        ats.interpolation,
        ats.colormap,
        pn.layout.Card(
            pn.Param(
                ats.param_sets.param,
                widgets={
                    'input_examples_filename': {
                        'widget_type': pn.widgets.TextInput,
                        'stylesheets': ['.bk-input-group > label {background-color: black}'],
                        'name': '',
                    },
                    'example': {
                        'stylesheets': ['bk-panel-models-widgets-CustomSelect {background: #2b3035; color: black}'],
                        'name': '',
                    },
                },
                show_name=False,
                parameters=['input_examples_filename', 'load', 'example', 'remember_this_one'],
            ),
            title='Load and Save',
            collapsed=True,
            header_color='white',
            header_background='#2c71b4',
        ),
    ],
    main=[
        ats.equations,
        ats,
    ],
    main_layout=None,
    sidebar_width=SIDEBAR_WIDTH,
    sidebar_footer=__version__,
    accent_base_color='goldenrod',
    background_color='black',
    header_background='teal',
    theme='dark',
    theme_toggle=False,
).servable('Attractor Explorer')
