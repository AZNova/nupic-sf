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
from collections import deque
import datetime
import random
import pprint
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab
import datetime as dt
import glob

from nupic.swarming import permutations_runner
from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.model import Model
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager
from nupic_output import NuPICFileOutput, NuPICPlotOutput


WINDOW = 360

SWARM_CONFIG = {
    "includedFields": [
        {
            "fieldName": "timestamp",
            "fieldType": "datetime"
        },
        {
            "fieldName": "traffic",
            "fieldType": "float",
            "maxValue": 350.0,
            "minValue": 0.0
        }
    ],
    "streamDef": {
        "info": "traffic",
        "version": 1,
        "streams": [
            {
                "info": "traffic",
                "source": "file://out-week7-rolling.csv",
                "columns": [ "*" ]
            }
        ]
    },
    #"inferenceType": "TemporalAnomaly",
    "inferenceType": "TemporalMultiStep",
    "inferenceArgs": {
        "predictionSteps": [ 1 ],
        "predictedField": "traffic"
    },
    "iterationCount": -1,
    "swarmSize": "medium"
}

SWARM_TEMP_FOLDER = "./swarmTemp"
MODEL_PARAMS = "modelParams"

#DATE_FORMAT = "%m/%d/%y %H:%M"
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

    counter = 0
    for trainCounter in xrange(1,53):
        #        file_picker = random.randint(6,12)
        filename = 'out-2014{0:02d}'.format(trainCounter)
        inputName = filename + '.csv'
        outputName = filename + '_train.csv'
        print('Reading ' + inputName)

        try:
            in_df = pd.read_csv(inputName, parse_dates=['timestamp'],
                    header=0, skiprows=[1,2], index_col='timestamp')
            in_rm = pd.rolling_mean(in_df.resample("60Min", fill_method="ffill"), 
                    window=3, min_periods=1)
        except:
            print("Error: {0}".format(inputName))
            continue

        ts_list = []
        tr_list = []
        pr_list = []
        an_list = []

        for timestamp, traffic in in_rm.itertuples():
            counter += 1

            result = model.run({ "timestamp": timestamp,
                "traffic": traffic })

            p1 = result.inferences["multiStepBestPredictions"][1]
            #a = result.inferences["anomalyScore"]
            a = result.inferences['multiStepPredictions'][1][result.inferences['multiStepBestPredictions'][1]]

            ts_list.append(timestamp)
            tr_list.append(traffic)
            pr_list.append(p1)
            an_list.append(a)

            if counter % 100 == 0:
                print("pass {0}, {1} records loaded".format(trainCounter, counter))

        a_dict = {'timestamp':ts_list, 'traffic':tr_list,
                'nextPredictedtraffic':pr_list, 'currentConfidence':an_list}

        row_df = pd.DataFrame(a_dict, columns=['traffic',
            'nextPredictedtraffic', 'currentConfidence'], index=ts_list)
        row_df.index.name = 'timestamp'

        row_df.to_csv(outputName, header=True, index=True)

    model.save( os.path.abspath( 'modelSave' ) )


def test(args):

    inputName = ''
    if 'good' in args:
        inputGlob = 'out-2015*.csv'
    elif 'bad' in args:
        inputName = 'out-weeklyBAD.csv'
    else:
        print 'Yer killin me smalls, specify a test (good, bad)'
        return

    # load the previously trained model from disk
    model = Model.load( os.path.abspath( 'modelSave' ))

    # disable learning for testing
    model.disableLearning()

    # setup the metrics handling
    metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(), 
            model.getInferenceType())

    ts_list = []
    tr_list = []
    pr_list = []
    an_list = []

    shifter = InferenceShifter()

    plt.ion()
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

    # Keep the last WINDOW predicted and actual values for plotting.
    actual_history = deque([0.0] * WINDOW, maxlen=360)
    predicted_history = deque([0.0] * WINDOW, maxlen=360)
    anomaly_score_history = deque([0.0] * WINDOW, maxlen=360)
#    ts_history = deque([0.0] * WINDOW, maxlen=360)

    plt.subplot(gs[0])
#    actual_line, = plt.plot(ts_history, actual_history)
#    predicted_line, = plt.plot(ts_history, predicted_history)
    actual_line, = plt.plot(range(WINDOW), actual_history)
    predicted_line, = plt.plot(range(WINDOW), predicted_history)
    plt.legend(tuple(['actual','predicted']), loc='upper center',
            fancybox=True, shadow=True)

    plt.subplot(gs[1])
#    anomaly_score_line, = plt.plot(ts_history, anomaly_score_history, 'r-')
    anomaly_score_line, = plt.plot(range(WINDOW), anomaly_score_history, 'r-')
    plt.legend(tuple(['confidence score']), loc='upper center',
            fancybox=True, shadow=True)

    # Set the y-axis range.
    actual_line.axes.set_ylim(40, 200)
    predicted_line.axes.set_ylim(40, 200)
    anomaly_score_line.axes.set_ylim(0, 1)


    for inputName in glob.glob(inputGlob):
        print("Opening {0}".format(inputName))
        in_df = pd.read_csv(inputName, parse_dates=['timestamp'],
                header=0, skiprows=[1,2], index_col='timestamp')
        in_rm = pd.rolling_mean(in_df.resample("60Min", fill_method="ffill"), 
                window=3, min_periods=1)

        for timestamp, traffic in in_rm.itertuples():
            # run the data
            result = model.run({ "timestamp": timestamp,
                "traffic": traffic })

            # get the metrics data, over-time avarages and such
            # not sure how to interpret these yet... tbd
            result.metrics = metricsManager.update( result )

            # Get shifted result so the predictions line up
            shifted_result = shifter.shift(result)

            # Update the trailing predicted and actual value deques.
            inference = shifted_result.inferences['multiStepBestPredictions'][1]
            if inference is not None:
                # Redraw the chart with the new data.
                actual_history.append(shifted_result.rawInput['traffic'])
                predicted_history.append(inference)
                #anomaly_score = result.inferences['anomalyScore']
                anomaly_score = result.inferences['multiStepPredictions'][1][result.inferences['multiStepBestPredictions'][1]]
                anomaly_score_history.append(anomaly_score)
#                ts_history.append(timestamp)

                ts_list.append(timestamp)
                tr_list.append(shifted_result.rawInput['traffic'])
                pr_list.append(inference)
                #an_list.append(result.inferences['anomalyScore'])
                an_list.append(result.inferences['multiStepPredictions'][1][result.inferences['multiStepBestPredictions'][1]])

                # Redraw the chart with the new data.
                actual_line.set_ydata(actual_history)  # update the data
                predicted_line.set_ydata(predicted_history)  # update the data
                anomaly_score_line.set_ydata(anomaly_score_history)  # update the data

                plt.draw()
                plt.tight_layout()

    plt.ioff()
    plt.show()

    a_dict = {'timestamp':ts_list, 'traffic':tr_list,
            'nextPredictedtraffic':pr_list, 'currentConfidence':an_list}

    row_df = pd.DataFrame(a_dict, columns=['traffic',
        'nextPredictedtraffic', 'currentConfidence'], index=ts_list)
    row_df.index.name = 'timestamp'

    row_df.to_csv('out-march_test.csv', header=True, index=True)

    # save it again, if the recently learned data needs to be updated
    # model.save( os.path.abspath( 'modelSave' ) )

    plt.ion()
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    data_ax = fig.add_subplot(gs[0])
    anom_ax = fig.add_subplot(gs[1], sharex=data_ax)
    data_ax.set_title('Click on the legend to toggles lines on/off')

    plt.title('Click on the legend to toggles lines on/off')
#    plt.subplot(gs[0])
    tr_line, = data_ax.plot(ts_list, row_df['traffic'], 'b-', 
            label='traffic')
    #    line1.axes.set_ylim(0, 100)
    pr_line, = data_ax.plot(ts_list, row_df['nextPredictedtraffic'], 'g-', 
            label='Prediction')
    data_ax.legend(tuple(['actual','predicted']), loc='upper center', 
            fancybox=True, shadow=True)
    #    line2.axes.set_ylim(0, 100)
    #    plt.set_xlabel('Dates')
    #    plt.set_ylabel('HTTP Connections', color='b')
#    for t1 in plt.get_yticklabels():
#        t1.set_color('b')


#    plt.subplot(gs[1])
    an_line, = anom_ax.plot(ts_list, row_df['currentConfidence'], 'r-', 
            label='Confidence')
    an_line.axes.set_ylim(0, 1)
    anom_ax.legend(tuple(['confidence score']), loc='upper center')
#    line3.set_ydata(row_df['currentAnomaly'])  # update the data

#    # handle combined legend for axes
#    plt.subplot(gs[0])
#    lined = {}
#    lines = line1 + line2 + line3
##    lines = line1 + line2
#    labels = [l.get_label() for l in lines]
#    legend = plt.legend(lines, labels, loc='upper center', 
#            fancybox=True, shadow=True)
#    frame = legend.get_frame()
#    frame.set_facecolor('0.90')
#    frame.set_alpha(0.4)
#    for label in legend.get_texts():
#        label.set_fontsize('large')
#    for label in legend.get_lines():
#        label.set_linewidth(1.5)  # the legend line width
#    for legline, origline in zip(legend.get_lines(), lines):
#        legline.set_picker(5)  # 5 pts tolerance
#        lined[legline] = origline
#
#    def onpick(event):
#        # on the pick event, find the orig line corresponding to the
#        # legend proxy line, and toggle the visibility
#        legline = event.artist
#        origline = lined[legline]
#        vis = not origline.get_visible()
#        origline.set_visible(vis)
#        # Change the alpha on the line in the legend so we can see what lines
#        # have been toggled
#        if vis:
#            legline.set_alpha(1.0)
#        else:
#            legline.set_alpha(0.2)
#        fig.canvas.draw()
#
#    fig.canvas.mpl_connect('pick_event', onpick)

    gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    plt.ioff()
    plt.show()


if __name__ == "__main__":

  args = sys.argv[1:]
  if "swarm" in args:
      swarm()

  if "train" in args:
      train()

  if "test" in args:
      test( args )
