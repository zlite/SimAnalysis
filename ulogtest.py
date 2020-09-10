#! /usr/bin/env python

"""
Convert a ULog file into CSV file(s)
"""

from __future__ import print_function

import argparse
import os

from pyulog import ULog

#pylint: disable=too-many-locals, invalid-name, consider-using-enumerate
keylist = []


def convert_ulog2csv(ulog_file_name):
    """
    Coverts and ULog file to a CSV file.

    :param ulog_file_name: The ULog filename to open and read
    :param messages: A list of message names
    :param output: Output file path
    :param delimiter: CSV delimiter

    :return: None
    """

    msg_filter = None
    disable_str_exceptions=False
    delimiter = ','

    ulog = ULog(ulog_file_name, msg_filter, disable_str_exceptions)
    data = ulog.data_list

    for d in data:
        # use same field order as in the log, except for the timestamp
        data_keys = [f.field_name for f in d.field_data]
        data_keys.remove('timestamp')
      #  data_keys.insert(0, 'timestamp')  # we want timestamp at first position

        # we don't use np.savetxt, because we have multiple arrays with
        # potentially different data types. However the following is quite
        # slow...

        # write the header
        # print(delimiter.join(data_keys) + '\n')

        # write the data
        # last_elem = len(data_keys)-1
        # for i in range(len(d.data['timestamp'])):
        #     for k in range(len(data_keys)):
        #         csvfile.write(str(d.data[data_keys[k]][i]))
        #         if k != last_elem:
        #             csvfile.write(delimiter)
        #     csvfile.write('\n')
        print("Key =", data_keys)
        for g in data_keys:
            keylist.append(g)
#        print("Data =", d.data)
    print("Keylist =", keylist)

convert_ulog2csv("airtonomysim.ulg")
