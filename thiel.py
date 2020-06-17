''' Create a simple stocks correlation dashboard.

Choose stocks to compare in the drop down widgets, and make selections
on the plots to update the summary and histograms accordingly.

.. note::
    Running this example requires downloading sample data. See
    the included `README`_ for more information.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve stocks

at your command prompt. Then navigate to the URL

    http://localhost:5006/stocks

.. _README: https://github.com/bokeh/bokeh/blob/master/examples/app/stocks/README.md

'''
import px4tools
from functools import lru_cache
from os.path import dirname, join
from bokeh.io import output_file, show
from bokeh.models.widgets import FileInput
from bokeh.models.widgets import Paragraph
from bokeh.models import CheckboxGroup

from bokeh.models import Div


import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PreText, Select
from bokeh.plotting import figure

DATA_DIR = join(dirname(__file__), 'datalogs')

DEFAULT_FIELDS = ['XY', 'LatLon', 'VxVy']

def nix(val, lst):
    return [x for x in lst if x != val]

@lru_cache()
def load_data(data_type):
    fname = join(DATA_DIR, '%s.csv' % data_type)  # fix this later
    data = pd.read_csv(fname)
    return pd.DataFrame(data)
#    return pd.DataFrame({data_type: data.y})

@lru_cache()
def get_data():
    dfsim = load_data('sim2')
#    print("sim",dfsim.y)
    dfreal = load_data('real2')
#    print("real",dfreal.y)
#    data = pd.concat([dfsim, dfreal], keys=['dfsim', 'dfreal'], names=['Sim', 'Real'], ignore_index = True, axis=1)
#    data = pd.concat([dfsim, dfreal], axis=1)
#    data = data.dropna()   # remove missing values
    dfsim = pd.concat([dfsim.simx,dfsim.simy], axis=1)
    dfreal = pd.concat([dfreal.realx,dfreal.realy], axis=1)
    return dfsim,dfreal

# set up widgets

# stats = PreText(text='This is some initial text', width=500)
sim = Select(value='XY', options=nix('VxVy', DEFAULT_FIELDS))
real = Select(value='VxVy', options=nix('XY', DEFAULT_FIELDS))

# set up plots

dfsim, dfreal = get_data()
simsource = ColumnDataSource(dfsim)
simsource_static = ColumnDataSource(dfsim)
realsource = ColumnDataSource(dfreal)
realsource_static = ColumnDataSource(dfreal)

tools = 'pan,wheel_zoom,xbox_select,reset'

#corr = figure(plot_width=350, plot_height=350, tools='pan,wheel_zoom,box_select,reset')
#corr.circle('Sim X', 'Real X', size=2, source=simsource, selection_color="orange", alpha=0.6, nonselection_alpha=0.1, selection_alpha=0.4)

ts1 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='linear', active_drag="xbox_select")
ts1.line('simx', 'simy', source=simsource_static)
ts1.circle('simx', 'simy', size=1, source=simsource, color=None, selection_color="orange")

ts2 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='linear', active_drag="xbox_select")
ts2.x_range = ts1.x_range
ts2.line('realx', 'realy', source=realsource_static)
ts2.circle('realx', 'realy', size=1, source=realsource, color=None, selection_color="orange")

# set up callbacks

def sim_change(attrname, old, new):
    real.options = nix(new, DEFAULT_FIELDS)
    update()

def real_change(attrname, old, new):
    sim.options = nix(new, DEFAULT_FIELDS)
    update()

def update(selected=None):
    global simsource, source_static, realsource, realsource_static
    dfsim, dfreal = get_data()
    print("sim", sim)
    print("real",real)
    data = pd.concat([dfsim, dfreal], axis=1)
    print(data)
    simsource = dfsim
    simsource_static = dfsim
    realsource = dfreal
    realsource_static = dfreal

#    update_stats(data, sim, real)

#    corr.title.text = 'Sim vs Real'
    ts1.title.text, ts2.title.text = 'Sim', 'Real'

def update_stats(data, sim, real):
    stats.text = str(data[[df1.y, df2.y, 'Sim', 'Real']].describe())

sim.on_change('value', sim_change)
real.on_change('value', real_change)

def selection_change(attrname, old, new):
#    sim, real = sim.value, real.value
    data = get_data()
    selected = simsource.selected.indices
    if selected:
        data = data.iloc[selected, :]
#    update_stats(data, sim, real)
    
simsource.selected.on_change('indices', selection_change)
    
#file_input = FileInput(accept=".ulg, .csv")
#file_input.on_change('value', get_data())
#file_input2 = FileInput(accept=".ulg, .csv")
#file_input2.on_change('value', get_data())

intro_text = Div(text="""<H2>Sim/Real Theil Coefficient Calculator</H2>""",width=500, height=100, align="center")
sim_upload_text = Paragraph(text="Upload a simulator datalog:",width=500, height=15)
real_upload_text = Paragraph(text="Upload a corresponding real-world datalog:",width=500, height=15)
#checkbox_group = CheckboxGroup(labels=["x", "y", "vx","vy","lat","lon"], active=[0, 1])

simsource.selected.on_change('indices', selection_change)

# set up layout
widgets = column(sim, real)
# widgets = column(stats, sim, real)
#main_row = row(corr, widgets)
main_row = row(widgets)
series = column(ts1, ts2)
layout = column(main_row, series)

# initialize
update()

curdoc().add_root(intro_text)
curdoc().add_root(sim_upload_text)
#curdoc().add_root(file_input)
curdoc().add_root(real_upload_text)
#curdoc().add_root(file_input2)
#curdoc().add_root(checkbox_group)
curdoc().add_root(layout)
curdoc().title = "Flight data"



