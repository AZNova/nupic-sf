import csv
import math
from pandas import *
import numpy

def run():
    lines_list = []

    with open('./file.in') as fin:
        reader = csv.reader(fin)
        lines = list(reader)

    for idx in xrange(len(lines)):
        if not lines[idx][2]:
            lines_list.append(numpy.nan)
        else:
            lines_list.append(float(lines[idx][2]))
    import ipdb; ipdb.set_trace() # BREAKPOINT
    lines_array = numpy.array(lines_list)
#    lines_interpolated = Series(lines_array).interpolate().values
    lines_interpolated = Series(lines_array).values

    with open('f.out', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(["timestamp", "traffic"])
        writer.writerow(["datetime", "float"])
        writer.writerow(["", ""])

        for idx in xrange(len(lines)):
            writer.writerow([lines[idx][1], lines_interpolated[idx]])

if __name__ == "__main__":
    run()


