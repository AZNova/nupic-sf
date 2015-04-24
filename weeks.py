import csv
import math
import pandas as pd
import numpy
import datetime as dt
import matplotlib.pyplot as plt
import pylab
import sys
import os


def run(filename,field):
    import ipdb; ipdb.set_trace() # BREAKPOINT
    df = pd.read_csv(filename, parse_dates=['timestamp'],
            header=0, skiprows=[1,2], index_col='timestamp')
    rm = pd.rolling_mean(df.resample("60Min", fill_method="ffill"), 
            window=3, min_periods=1)
    #    df[field].plot()
    #    rm[field].plot()
    #    pylab.ion()
    #    pylab.show()

    kw = lambda x: ((x.isocalendar()[0] * 100) + x.isocalendar()[1] + 100) - 100
    by_week = rm.groupby([rm.index.map(kw)], sort=False)

    for each_year in xrange(2013, 2016):
        for each_week in xrange(1, 53):
            try:
                filename = 'out-{0}{1:02d}-{2}.csv'.format(each_year, each_week,field)
                if len(by_week.get_group((each_year * 100) + each_week).head()) > 0:
                    if os.path.exists(filename):
                        os.rename(filename, filename + '.bak')
                    by_week.get_group((each_year * 100) + each_week).to_csv(filename, header=True, index=True)
                    print('Created {0}'.format(filename))
            except KeyError, e:
                pass
            except:
                raise

        
#    rm.to_csv('rec-center-hourly_out.csv', header=True, index=True)

    sys.exit(0)


    for idx in xrange(len(lines)):
        if not lines[idx][2]:
            lines_list.append(numpy.nan)
        else:
            lines_list.append(float(lines[idx][2]))
    import ipdb; ipdb.set_trace() # BREAKPOINT
    lines_array = numpy.array(lines_list)
    lines_interpolated = Series(lines_array).interpolate().values

    import ipdb; ipdb.set_trace() # BREAKPOINT
    rm = rolling_mean(lines_interpolated, window=3, min_periods=1)

    with open('f.out', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(["timestamp", field])
        writer.writerow(["datetime", "float"])
        writer.writerow(["", ""])

        for idx in xrange(len(lines)):
            writer.writerow([lines[idx][1], lines_interpolated[idx]])

if __name__ == "__main__":
    if sys.argv[1] is None:
        print("I need a filename!")
        sys.exit(1)
    if sys.argv[2] is None:
        print("I need a field!")
        sys.exit(1)

    run(sys.argv[1], sys.argv[2])


