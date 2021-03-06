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

DEFAULT_FIELDS = ['XY', 'Lat Lon', 'Gyro', 'Baro']

def nix(val, lst):
    return [x for x in lst if x != val]

@lru_cache()
def load_field(field):
    fname = join(DATA_DIR, '%s.csv' % field.lower())
    data = pd.read_csv(file_input.filename)
    data = data.set_index('x')
    return pd.DataFrame({field: data.y)

@lru_cache()
def get_data(t1, t2):
    df1 = load_field(t1)
    df2 = load_field(t2)
    data = pd.concat([df1, df2], axis=1)
    data = data.dropna()
    data['t1'] = data[t1]
    data['t2'] = data[t2]
    data['t1_returns'] = data[t1+'_returns']
    data['t2_returns'] = data[t2+'_returns']
    return data

# set up widgets

stats = PreText(text='This is some initial text', width=500)
data1 = Select(value='XY', options=nix('Lat Lon', DEFAULT_FIELDS))
data2 = Select(value='XY', options=nix('Lat Lon', DEFAULT_FIELDS))

# set up plots

source = ColumnDataSource(data=dict(date=[], t1=[], t2=[], t1_returns=[], t2_returns=[]))
source_static = ColumnDataSource(data=dict(date=[], t1=[], t2=[], t1_returns=[], t2_returns=[]))
tools = 'pan,wheel_zoom,xbox_select,reset'

corr = figure(plot_width=350, plot_height=350,
              tools='pan,wheel_zoom,box_select,reset')
corr.circle('t1_returns', 't2_returns', size=2, source=source,
            selection_color="orange", alpha=0.6, nonselection_alpha=0.1, selection_alpha=0.4)

ts1 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts1.line('date', 't1', source=source_static)
ts1.circle('date', 't1', size=1, source=source, color=None, selection_color="orange")

ts2 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts2.x_range = ts1.x_range
ts2.line('date', 't2', source=source_static)
ts2.circle('date', 't2', size=1, source=source, color=None, selection_color="orange")

# set up callbacks

def data1_change(attrname, old, new):
    data2.options = nix(new, DEFAULT_FIELDS)
    update()

def data2_change(attrname, old, new):
    data1.options = nix(new, DEFAULT_FIELDS)
    update()

def update(selected=None):
    t1, t2 = data1.value, data2.value

    df = get_data(t1, t2)
    data = df[['t1', 't2', 't1_returns', 't2_returns']]
    source.data = data
    source_static.data = data

    update_stats(df, t1, t2)

    corr.title.text = '%s returns vs. %s returns' % (t1, t2)
    ts1.title.text, ts2.title.text = t1, t2

def update_stats(data, t1, t2):
    stats.text = str(data[[t1, t2, t1+'_returns', t2+'_returns']].describe())

data1.on_change('value', data1_change)
data2.on_change('value', data2_change)

def selection_change(attrname, old, new):
    t1, t2 = data1.value, data2.value
    data = get_data(t1, t2)
    selected = source.selected.indices
    if selected:
        data = data.iloc[selected, :]
    update_stats(data, t1, t2)

def upload_sim_datalog(attr, old, new):
    print("sim data upload succeeded")
    df1 = pd.read_csv(file_input.filename)

def upload_real_datalog(attr, old, new):
    print("real data upload succeeded")
    df2 = pd.read_csv(file_input2.filename)

    
file_input = FileInput(accept=".ulg, .csv")
file_input.on_change('value', upload_sim_datalog)
file_input2 = FileInput(accept=".ulg, .csv")
file_input2.on_change('value', upload_real_datalog)

intro_text = Div(text="""<H2>Sim/Real Theil Coefficient Calculator</H2>""",width=500, height=100, align="center")
sim_upload_text = Paragraph(text="Upload a simulator datalog:",width=500, height=15)
real_upload_text = Paragraph(text="Upload a corresponding real-world datalog:",width=500, height=15)

source.selected.on_change('indices', selection_change)

# set up layout
widgets = column(data1, data2)
main_row = row(corr, widgets)
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
curdoc().title = "Stocks"



