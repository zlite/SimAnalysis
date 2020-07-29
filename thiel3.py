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
from bokeh.models import Range1d

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
read_file = True
reverse = False
new_data = True

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
 #   global original_data
    dfsim = load_data_sim(simname)
    dfreal = load_data_real(realname)
    data = pd.concat([dfsim, dfreal], axis=1)
    data = data.dropna()   # remove missing values
    data['simy'] = data.simy
    data['simx'] = data.simx
    data['realy'] = data.realy
    data['realx'] = data.realx
#    original_data = copy.deepcopy(data)
    return data

# set up widgets

stats = PreText(text='Thiel Coefficient', width=500)
datatype = Select(value='XY', options=DEFAULT_FIELDS)

# set up plots

source = ColumnDataSource(data = dict(realx=[],realy=[],simx=[],simy=[]))
source_static = ColumnDataSource(data = dict(realx=[],realy=[],simx=[],simy=[]))


tools = 'xpan,wheel_zoom,xbox_select,reset'


ts1 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='linear', active_drag="xbox_select")
ts1.line('simx', 'simy', source=source, line_width=2)
ts1.circle('simx', 'simy', size=1, source=source_static, color=None, selection_color="orange")

ts2 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='linear', active_drag="xbox_select")
# to adjust ranges, add something like this: x_range=Range1d(0, 1000), y_range = None,
# ts2.x_range = ts1.x_range
# ts2.line('realx', 'realy', source=source_static)
ts2.line('realx', 'realy', source=source, line_width=2)
ts2.circle('realx', 'realy', size=1, source=source_static, color=None, selection_color="orange")

# set up callbacks

def sim_change(attrname, old, new):
    real.options = nix(new, DEFAULT_FIELDS)
    update()

def update(selected=None):
    global read_file, reverse, new_data, source, source_static, original_data, data, new_data
    if (read_file):
       original_data = get_data(simname, realname)
       data = copy.deepcopy(original_data)
       read_file = False
    print("Sim offset", simx_offset)
    print("Real offset", realx_offset)
    if reverse:
        data[['simy']] = sim_polarity * original_data[['simy']]  # reverse data if necessary
        data[['realy']] = real_polarity * original_data[['realy']]
        source.data = data
        source_static.data = data
        simmax = round(max(data[['simy']].values)[0])  # reset the axis scales as appopriate (auto scaling doesn't work)
        simmin = round(min(data[['simy']].values)[0])
        realmax = round(max(data[['realy']].values)[0])
        realmin = round(min(data[['realy']].values)[0])
        ts1.y_range.start = simmin - abs((simmax-simmin)/10)
        ts1.y_range.end = simmax + abs((simmax-simmin)/10)
        ts2.y_range.start = realmin - abs((realmax-realmin)/10)
        ts2.y_range.end = realmax + abs((realmax-realmin)/10)
        reverse = False
    if new_data:
        source.data = data[['simx', 'simy','realx','realy']]
        source_static.data = data[['simx', 'simy','realx','realy']]
        new_data = False
#    select_data = copy.deepcopy(tempdata)
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
    real = np.array(data['realy'])
    sim = np.array(data['simy'])
    sum1 = 0
    sum2 = 0
    sum3 = 0
#    for n in np.nditer(data):
    for n in range(len(real)):
        sum1 = sum1 + (real[int(n)]-sim[int(n)])**2
        sum2 = sum2 + real[int(n)]**2
        sum3 = sum3 + sim[int(n)]**2
    sum1 = 1/len(real) * sum1
    sum2 = 1/len(real) * sum2
    sum3 = 1/len(real) * sum3
    print("Sum 1", sum1, "sum2", sum2, "sum3", sum3)
    sum1 = math.sqrt(sum1)
    sum2 = math.sqrt(sum2)
    sum3 = math.sqrt(sum3)
    TIC = sum1/(sum2 + sum3)
    stats.text = 'Thiel coefficient: ' + str(round(TIC,3))

datatype.on_change('value', sim_change)

def selection_change(attrname, old, new):
    selected = source_static.selected.indices
    print("selected:", selected)
    if selected:
        seldata = data.iloc[selected, :]
        update_stats(seldata)
        print("update stats")
 #   if (len(data['realy']) != 0):
 #       for x in range(len(source_static.data['realx'])):
 #           source_static.data['realx'][x] = source_static.data['realx'][x] - realx_offset
#            tempdata['realx'][x] = tempdata['realx'][x] - realx_offset
#            print(tempdata['realx'][x])
   
    update()

def reverse_sim():
    global sim_polarity, reverse
    if (sim_reverse_button.active == 1): sim_polarity = -1
    else: sim_polarity = 1
    reverse= True
    update()

def reverse_real():
    global real_polarity, reverse
    if (real_reverse_button.active == 1): real_polarity = -1
    else: real_polarity = 1
    reverse = True
    update()

def change_sim_scale(shift):
    global simx_offset, new_data
    simx_offset = shift
    new_data = True
    update()

def change_real_scale(shift):
    global realx_offset, new_data
    realx_offset = shift
    new_data = True
    update()
 
    
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

source_static.selected.on_change('indices', selection_change)

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
