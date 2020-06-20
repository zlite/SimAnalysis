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
import numpy as np
import math
from functools import lru_cache
from os.path import dirname, join
from bokeh.io import output_file, show
from bokeh.models.widgets import FileInput
from bokeh.models.widgets import Paragraph
from bokeh.models import CheckboxGroup
import time

from bokeh.models import Div


import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PreText, Select
from bokeh.plotting import figure

DATA_DIR = join(dirname(__file__), 'datalogs')

DEFAULT_FIELDS = ['XY', 'LatLon', 'VxVy']

simname = 'sim2'
realname = 'real2'


@lru_cache()
def load_data(name):
#    fname = join(DATA_DIR, '%s.csv' % name)  # fix this later
#    data = pd.read_csv(fname)
    data = pd.read_csv(name)
    return pd.DataFrame(data)

@lru_cache()
def get_data(simname,realname):
    dfsim = load_data(simname)
    dfreal = load_data(realname)
    data = pd.concat([dfsim, dfreal], axis=1)
    data = data.dropna()   # remove missing values
    data['simy'] = data.simy
    data['simx'] = data.simx
    data['realy'] = data.realy
    data['realx'] = data.realx
    return data

# set up widgets

stats = PreText(text='Thiel Coefficient', width=500)
datatype = Select(value='XY', options=DEFAULT_FIELDS)

# set up plots

source = ColumnDataSource(data = dict(simx=[],simy=[],realx=[],realy=[]))
source_static = ColumnDataSource(data = dict(simx=[],simy=[],realx=[],realy=[]))
tools = 'pan,wheel_zoom,xbox_select,reset'

ts1 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='linear', active_drag="xbox_select")
ts1.line('simx', 'simy', source=source_static)
ts1.circle('simx', 'simy', size=1, source=source, color=None, selection_color="orange")

ts2 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='linear', active_drag="xbox_select")
ts2.x_range = ts1.x_range
ts2.line('realx', 'realy', source=source_static)
ts2.circle('realx', 'realy', size=1, source=source, color=None, selection_color="orange")

# set up callbacks

def sim_change(attrname, old, new):
    real.options = nix(new, DEFAULT_FIELDS)
    update()

def update(selected=None):
    tempdata = get_data(simname,realname)
    data = tempdata[['simx', 'simy','realx','realy']]
    source.data = data
    source_static.data = data
    ts1.title.text, ts2.title.text = 'Sim', 'Real'

def update_stats(data):
    real = np.array(data.realy)
    sim = np.array(data.simy)
    sign = -1  # if the sign of the real data has to be reversed. This is just for debugging 
    sum1 = 0
    sum2 = 0
    sum3 = 0
#    for n in np.nditer(data):
    for n in range(len(data)):
        sum1 = sum1 + (sign*real[int(n)]-sim[int(n)])**2
        sum2 = sum2 + real[int(n)]**2
        sum3 = sum3 + sim[int(n)]**2
    sum1 = 1/len(real) * sum1
    sum2 = 1/len(real) * sum2
    sum3 = 1/len(real) * sum3
    sum1 = math.sqrt(sum1)
    sum2 = math.sqrt(sum2)
    sum3 = math.sqrt(sum3)
    TIC = sum1/(sum2 + sum3)
    stats.text = 'Thiel coefficient = ' + str(round(TIC,3))

datatype.on_change('value', sim_change)

def selection_change(attrname, old, new):
#    sim, real = sim.value, real.value
    data = get_data(simname,realname)
    selected = source.selected.indices
    if selected:
        data = data.iloc[selected, :]
    update_stats(data)
    
source.selected.on_change('indices', selection_change)
    
file_input = FileInput(accept=".ulg, .csv")
file_input.on_change('value', load_data(file_input))
file_input2 = FileInput(accept=".ulg, .csv")
file_input2.on_change('value', load_data(file_input))

intro_text = Div(text="""<H2>Sim/Real Theil Coefficient Calculator</H2>""",width=500, height=100, align="center")
sim_upload_text = Paragraph(text="Upload a simulator datalog:",width=500, height=15)
real_upload_text = Paragraph(text="Upload a corresponding real-world datalog:",width=500, height=15)
#checkbox_group = CheckboxGroup(labels=["x", "y", "vx","vy","lat","lon"], active=[0, 1])

source.selected.on_change('indices', selection_change)

# set up layout
widgets = column(datatype,stats)
main_row = row(widgets)
series = column(ts1, ts2)
layout = column(main_row, series)

# initialize
update()

curdoc().add_root(intro_text)
curdoc().add_root(sim_upload_text)
curdoc().add_root(file_input)
curdoc().add_root(real_upload_text)
curdoc().add_root(file_input2)
curdoc().add_root(layout)
curdoc().title = "Flight data"



