from bokeh.io import curdoc
from bokeh.models.widgets import FileInput

def upload_fit_data(attr, old, new):
    print("fit data upload succeeded")
    print(file_input.value)
file_input = FileInput(accept=".csv,.json,.txt")
file_input.on_change('value', upload_fit_data)

doc=curdoc()
doc.add_root(file_input)
