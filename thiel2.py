''' Create a simple real/sim data correlation dashboard.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve thiel.py

at your command prompt. Then navigate to the URL

    http://localhost:5006

'''
import px4tools
import numpy as np
import math
import io
from functools import lru_cache
from os.path import dirname, join
from bokeh.io import output_file, show
from bokeh.models.widgets import FileInput
from bokeh.models.widgets import Paragraph
from bokeh.models import CheckboxGroup
from bokeh.models import RadioButtonGroup

import time
import copy
from bokeh.models import Div


import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PreText, Select
from bokeh.plotting import figure

DATA_DIR = join(dirname(__file__), 'datalogs')

DEFAULT_FIELDS = ['XY', 'LatLon', 'VxVy']

simname = 'sim2.csv'
realname = 'real2.csv'
sim_polarity = 1  # determines if we should reverse the Y data
real_polarity = 1
simx_offset = 0
realx_offset = 0


@lru_cache()
def load_data_sim(simname):
    fname = join(DATA_DIR, simname)
    data = pd.read_csv(fname)
    dfsim = pd.DataFrame(data)
    return dfsim

@lru_cache()
def load_data_real(realname):
    fname = join(DATA_DIR, realname)
    data = pd.read_csv(fname)
    dfreal = pd.DataFrame(data)
    return dfreal


@lru_cache()
def get_data(simname,realname):
    global original_data
    dfsim = load_data_sim(simname)
    dfreal = load_data_real(realname)
    data = pd.concat([dfsim, dfreal], axis=1)
    data = data.dropna()   # remove missing values
    data['simy'] = data.simy
    data['simx'] = data.simx
    data['realy'] = data.realy
    data['realx'] = data.realx
    original_data = copy.deepcopy(data)
    return data

# set up widgets

stats = PreText(text='Thiel Coefficient', width=500)
datatype = Select(value='XY', options=DEFAULT_FIELDS)

# set up plots

realsource = ColumnDataSource(data = dict(realx=[],realy=[]))
realsource_static = ColumnDataSource(data = dict(realx=[],realy=[]))
simsource = ColumnDataSource(data = dict(simx=[],simy=[]))
simsource_static = ColumnDataSource(data = dict(simx=[],simy=[]))


tools = 'pan,wheel_zoom,xbox_select,reset'


ts1 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='linear', active_drag="xbox_select")
ts1.line('simx', 'simy', source=simsource, line_width=2)
ts1.circle('simx', 'simy', size=1, source=simsource_static, color=None, selection_color="orange")

ts2 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='linear', active_drag="xbox_select")
# ts2.x_range = ts1.x_range
#ts2.line('realx', 'realy', source=source_static)
ts2.line('realx', 'realy', source=realsource, line_width=2)
ts2.circle('realx', 'realy', size=1, source=realsource_static, color=None, selection_color="orange")

# set up callbacks

def sim_change(attrname, old, new):
    real.options = nix(new, DEFAULT_FIELDS)
    update()

def update(selected=None):
    global tempdata, select_data
    tempdata = get_data(simname, realname)
    print("Sim offset", simx_offset)
    print("Real offset", realx_offset)

 #   print(simsource_static.data)
    tempdata[['simy']] = sim_polarity * original_data[['simy']]  # reverse data if neessary
    tempdata[['realy']] = real_polarity * original_data[['realy']]
    data = tempdata[['simx', 'simy','realx','realy']]
    realsource.data = data
    realsource_static.data = data
    simsource.data = data
    simsource_static.data = data
    select_data = copy.deepcopy(tempdata)
    ts1.title.text, ts2.title.text = 'Sim', 'Real'

def upload_new_data_sim(attr, old, new):
    global simname
    decoded = b64decode(new)
    simname = io.BytesIO(decoded)
    update()

def upload_new_data_real(attr, old, new):
    global realname
    decoded = b64decode(new)
    realname = io.BytesIO(decoded)
    update()

def update_stats(data):
    real = np.array(data.realy)
    sim = np.array(data.simy)
    sum1 = 0
    sum2 = 0
    sum3 = 0
#    for n in np.nditer(data):
    for n in range(len(data)):
        sum1 = sum1 + (real[int(n)]-sim[int(n)])**2
        sum2 = sum2 + real[int(n)]**2
        sum3 = sum3 + sim[int(n)]**2
    sum1 = 1/len(real) * sum1
    sum2 = 1/len(real) * sum2
    sum3 = 1/len(real) * sum3
    sum1 = math.sqrt(sum1)
    sum2 = math.sqrt(sum2)
    sum3 = math.sqrt(sum3)
    TIC = sum1/(sum2 + sum3)
    stats.text = 'Thiel coefficient: ' + str(round(TIC,3))

datatype.on_change('value', sim_change)

def simselection_change(attrname, old, new):
    a = 1
    data = select_data
 #   update()
    selected = simsource_static.selected.indices
    if selected:
        data = select_data.iloc[selected, :]
    update_stats(data)
    if (len(simsource.data['simy']) != 0):
        print(simsource.data['simy'][1])
        simsource.data['simy'][1] = 2
        print(simsource.data['simy'][1])

def realselection_change(attrname, old, new):
    data = select_data
 #   update()
    selected = realsource_static.selected.indices
    if selected:
        data = select_data.iloc[selected, :]
    update_stats(data)
    if (len(simsource.data['realy']) != 0):
        print(simsource.data['realy'][1])
        simsource.data['realy'][1] = 3
        print(simsource.data['realy'][1])

def reverse_sim():
    global sim_polarity
    if (sim_reverse_button.active == 1): sim_polarity = -1
    else: sim_polarity = 1
    update()

def reverse_real():
    global real_polarity
    if (real_reverse_button.active == 1): real_polarity = -1
    else: real_polarity = 1
    update()

def change_sim_scale(shift):
    global simx_offset
    simx_offset = shift

def change_real_scale(shift):
    global realx_offset
    realx_offset = shift
 
    
file_input = FileInput(accept=".ulg, .csv")
file_input.on_change('value', upload_new_data_sim)
file_input2 = FileInput(accept=".ulg, .csv")
file_input2.on_change('value', upload_new_data_real)

intro_text = Div(text="""<H2>Sim/Real Thiel Coefficient Calculator</H2>""",width=500, height=100, align="center")
sim_upload_text = Paragraph(text="Upload a simulator datalog:",width=500, height=15)
real_upload_text = Paragraph(text="Upload a corresponding real-world datalog:",width=500, height=15)
#checkbox_group = CheckboxGroup(labels=["x", "y", "vx","vy","lat","lon"], active=[0, 1])

sim_reverse_button = RadioButtonGroup(
        labels=["Sim Default", "Reversed"], active=0)
sim_reverse_button.on_change('active', lambda attr, old, new: reverse_sim())
real_reverse_button = RadioButtonGroup(
        labels=["Real Default", "Reversed"], active=0)
real_reverse_button.on_change('active', lambda attr, old, new: reverse_real())

simsource_static.selected.on_change('indices', simselection_change)
realsource_static.selected.on_change('indices', realselection_change)
# The below are in case you want to see the x axis range change as you pan. Poorly documented elsewhere!
#ts1.x_range.on_change('end', lambda attr, old, new: print ("TS1 X range = ", ts1.x_range.start, ts1.x_range.end))
#ts2.x_range.on_change('end', lambda attr, old, new: print ("TS2 X range = ", ts2.x_range.start, ts2.x_range.end))

ts1.x_range.on_change('end', lambda attr, old, new: change_sim_scale(ts1.x_range.start))
ts2.x_range.on_change('end', lambda attr, old, new: change_real_scale(ts2.x_range.start))



# set up layout
widgets = column(datatype,stats)
sim_button = column(sim_reverse_button)
real_button = column(real_reverse_button)
main_row = row(widgets)
series = column(ts1, sim_button, ts2, real_button)
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