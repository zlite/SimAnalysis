##
##from __future__ import print_function
##
##import argparse
##import os
##import re
##import subprocess
##
### df1 = px4tools.read_ulog('real.ulg')
###print(df1.t_wind_estimate_0)
##sdlog2_path="sdlog2_dump.py"
##
##file_path = 'real.ulg'
##file_out = re.sub('.px4log', '.csv', file_path)
##cmd = 'python {sdlog2_path:s} {file_path:s}' \
##          ' -f {file_out:s} -e'.format(**locals())
##p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
##out, err = p.communicate()

import pandas as pd
import px4tools

a = 2
df1 = pd.read_csv(r'real.csv')
df2 = pd.read_csv(r'sim.csv')
df3 = pd.read_csv(r'sim.csv')
print(df1.x)
