#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import importlib
import sys, os
import csv
import datetime
import pprint
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import datetime as dt

from nupic.swarming import permutations_runner
from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.model import Model
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager


SWARM_CONFIG = {
    "includedFields": [
        {
            "fieldName": "timestamp",
            "fieldType": "datetime"
        },
        {
            "fieldName": "traffic",
            "fieldType": "float",
            "maxValue": 1000.0,
            "minValue": 0.0
        }
    ],
    "streamDef": {
        "info": "traffic",
        "version": 1,
        "streams": [
            {
                "info": "HTTP Traffic",
                "source": "file://rolling-out.csv",
                "columns": [ "*" ]
            }
        ]
    },
    "inferenceType": "TemporalAnomaly",
    #"inferenceType": "TemporalMultiStep",
    "inferenceArgs": {
        "predictionSteps": [ 1 ],
        "predictedField": "traffic"
    },
    "iterationCount": -1,
    "swarmSize": "small"
}

SWARM_TEMP_FOLDER = "./swarmTemp"
MODEL_PARAMS = "modelParams"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_METRIC_SPECS = (
    MetricSpec(
        field='traffic',
        metric='multiStep',
        inferenceElement='multiStepBestPredictions',
        params={'errorMetric': 'aae', 'window': 1000, 'steps': 1}),
    MetricSpec(
        field='traffic',
        metric='trivial',
        inferenceElement='prediction',
        params={'errorMetric': 'aae', 'window': 1000, 'steps': 1}),
    MetricSpec(
        field='traffic',
        metric='multiStep',
        inferenceElement='multiStepBestPredictions',
        params={'errorMetric': 'altMAPE', 'window': 1000, 'steps': 1}),
    MetricSpec(
        field='traffic',
        metric='trivial',
        inferenceElement='prediction',
        params={'errorMetric': 'altMAPE', 'window': 1000, 'steps': 1}),
    )



def swarm():
    # swarm using the config and get the result params
    params = permutations_runner.runWithConfig(
        SWARM_CONFIG,
        { 'maxWorkers': 4, 'overwrite': True },
        outDir=SWARM_TEMP_FOLDER,
        permWorkDir=SWARM_TEMP_FOLDER,
        verbosity=0
    )

    # save these so they can be loaded by the learn / model create
    pp = pprint.PrettyPrinter( indent=2 )
    formatted = pp.pformat( params )
    with open( MODEL_PARAMS + ".py", 'wb' ) as paramsFile:
        paramsFile.write( 'MODEL_PARAMS = \\\n%s' % formatted )


def train():
    # load the params module
    try:
        modelModule = importlib.import_module( MODEL_PARAMS ).MODEL_PARAMS
    except ImportError:
        raise Exception( "FU buddy, no model params found" )

    # create the model
    model = ModelFactory.create(modelModule)
    model.enableInference({"predictedField": "traffic"})

    # get the data into a csv reader
    inputFile = open("rolling-out.csv", "rb")
    csvReader = csv.reader(inputFile)

    # create the output csv writer, the results
    outputFile = open("rolling-out_train.csv", "wb" )
    csvWriter = csv.writer( outputFile )
    csvWriter.writerow( ['timestamp','traffic','predictedTraffic'] )

    counter = 0
    for trainCounter in range(0):
        # reset the file to position zero again
        inputFile.seek(0)
        csvReader.next()  # skip the header rows
        csvReader.next()
        csvReader.next()

        for row in csvReader:
            counter += 1

            timestamp = datetime.datetime.strptime( row[0], DATE_FORMAT )
            traffic = float(row[1])

            result = model.run({ "timestamp": timestamp, "traffic": traffic })
            p1 = result.inferences["multiStepBestPredictions"][1]
            a = result.inferences["anomalyScore"]
            csvWriter.writerow([timestamp, traffic, ' NP:'+str(p1), ' A:'+str(a)])

            if counter % 100 == 0:
                print("pass {0}, {1} records loaded".format(trainCounter, counter))

    inputFile.close()
    outputFile.close()

    model.save( os.path.abspath( 'modelSave' ) )

def test(args):

    inputName = ''
    if 'good' in args:
        inputName = 'rolling-outGOOD.csv'
    elif 'bad' in args:
        inputName = 'rolling-outBAD.csv'
    else:
        print 'Yer killin me smalls, specify a test (good, bad)'
        return

    shifter = InferenceShifter()

    # load the previously trained model from disk
    model = Model.load( os.path.abspath( 'modelSave' ))

    # setup the metrics handling
    metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(), 
            model.getInferenceType())

    in_df = pd.read_csv(inputName, parse_dates=['timestamp'],
            header=0, skiprows=[1,2], index_col='timestamp')
    in_rm = pd.rolling_mean(in_df.resample("60Min", fill_method="ffill"), 
            window=3, min_periods=1)

    ts_list = []
    tr_list = []
    pr_list = []
    an_list = []

    for timestamp, traffic in in_rm.itertuples():
        # run the data
        result = model.run({ "timestamp": timestamp,
            "traffic": traffic })

        # get the metrics data, over-time avarages and such
        # not sure how to interpret these yet... tbd
        result.metrics = metricsManager.update( result )

        result = shifter.shift(result)

        # get the raw inference data
        p1 = result.inferences["multiStepBestPredictions"][1]
        a = result.inferences["anomalyScore"]

        ts_list.append(timestamp)
        tr_list.append(traffic)
        pr_list.append(p1)
        an_list.append(a)

        # print result.metrics
        #print result.metrics[
        #    "multiStepBestPredictions:multiStep:"
        #    "errorMetric='altMAPE':steps=1:window=1000:"
        #    "field=kw_energy_consumption"]

    a_dict = {'timestamp':ts_list, 'traffic':tr_list,
            'nextPredictedTraffic':pr_list, 'currentAnomaly':an_list}
    a_dict1 = {'traffic':tr_list,
            'nextPredictedTraffic':pr_list, 'currentAnomaly':an_list}

    row_df = pd.DataFrame(a_dict, columns=['traffic',
        'nextPredictedTraffic', 'currentAnomaly'], index=ts_list)
    row_df.index.name = 'timestamp'

    row_df.to_csv('rolling-out_test.csv', header=True, index=True)

    # save it again, if the recently learned data needs to be updated
    # model.save( os.path.abspath( 'modelSave' ) )

    fig, ax1 = plt.subplots()
    ax1.set_title('Click on thelegend to toggles lines on/off')
    line1 = ax1.plot(ts_list, row_df['traffic'], 'b-', 
            label='Traffic')
    line2 = ax1.plot(ts_list, row_df['nextPredictedTraffic'], 'g-', 
            label='Prediction')
    ax1.set_xlabel('Dates')
    ax1.set_ylabel('HTTP Connections', color='b')
    for t1 in ax1.get_yticklabels():
        t1.set_color('b')


    ax2 = ax1.twinx()
#    line3 = ax2.plot(ts_list, row_df['currentAnomaly'], 'r-', 
#            label='Anomaly')
    ax2.set_ylabel('currentAnomaly', color='r')
    for t1 in ax2.get_yticklabels():
        t1.set_color('r')

    # hande combined legend for axes
    lined = {}
#    lines = line1 + line2 + line3
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, loc='upper center', 
            fancybox=True, shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    frame.set_alpha(0.4)
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
    for legline, origline in zip(legend.get_lines(), lines):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline

    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()


if __name__ == "__main__":

  args = sys.argv[1:]
  if "swarm" in args:
      swarm()

  if "train" in args:
      train()

  if "test" in args:
      test( args )
