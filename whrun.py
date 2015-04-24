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

from options import options

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
        },
        {
            "fieldName": "ps_count",
            "fieldType": "float",
            "maxValue": 750.0,
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
        "predictionSteps": [ 24 ],
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
    SWARM_CONFIG['streamDef']['streams'][0]['source'] = \
            "file://{0}".format(options.swarm_file)
    SWARM_CONFIG['inferenceArgs']['predictedField'] = \
            options.predicted_field
    steps_list = []
    for each in options.prediction_steps_list.split(','): 
        steps_list.append(int(each))
    SWARM_CONFIG['inferenceArgs']['predictionSteps'] = \
            steps_list
    SWARM_CONFIG['swarmSize'] = \
            options.swarm_size
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

def get_files(list_or_mask):
    if [c for c in '*?[]' if c in list_or_mask]:
        files = sorted(glob.glob(list_or_mask))
    else:
        files = list_or_mask.split(',')
    return files

def train():
    # load the params module
    try:
        modelModule = importlib.import_module( MODEL_PARAMS ).MODEL_PARAMS
    except ImportError:
        raise Exception( "FU buddy, no model params found" )

    if options.new_model is True and options.load_last is True:
        print("You can't have --new-model and --load-last - "
                "try again with only one of these options!")
        return
    elif options.new_model is True:
        # create the model
        print("Creating a new model")
        model = ModelFactory.create(modelModule)
        model.enableInference({"predictedField": options.predicted_field})
    elif options.load_last is True:
        print("Loading last model from {0}".format(os.path.abspath('modelSave')))
        model = ModelFactory.loadFromCheckpoint(os.path.abspath('modelSave'))
    else:
        # if neither of them are spec'ed, create the model
        print("Creating a new model")
        model = ModelFactory.create(modelModule)
        model.enableInference({"predictedField": options.predicted_field})

    steps_list = []
    for each in options.prediction_steps_list.split(','): 
        steps_list.append(int(each))

    train_files = get_files(options.train_files)
    #for trainCounter in xrange(1,53):
    if options.train_passes is not None:
        train_passes = options.train_passes
    else:
        train_passes = len(train_files)

    for trainCounter in xrange(0, train_passes):
        counter = 0
        if options.train_random is True:
            file_picker = random.randint(0, len(train_files)-1)
        else:
            file_picker = trainCounter % len(train_files)
        filename = train_files[file_picker]
        inputName = filename
        outputName = os.path.splitext(os.path.basename(filename))[0]\
                + '_train.csv'
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
        ps_list = []
        pr_array = np.zeros(shape=(len(steps_list), 1000), dtype=float)
        an_list = []
        an_array = np.zeros(shape=(len(steps_list), 1000), dtype=float)

        for timestamp, traffic, ps_count in in_rm.itertuples():

            result = model.run({ "timestamp": timestamp,
                "traffic": traffic, "ps_count": ps_count})

            ts_list.append(timestamp)
            tr_list.append(traffic)
            ps_list.append(ps_count)
            step_idx = 0
            for each_step in steps_list:
                p1 = result.inferences["multiStepBestPredictions"][each_step]
                a = result.inferences['multiStepPredictions'][each_step]\
                        [result.inferences['multiStepBestPredictions'][each_step]]
                pr_array[step_idx][counter] = p1
                an_array[step_idx][counter] = a
                step_idx += 1

            counter += 1
            if counter % 100 == 0:
                print("pass {0}, {1} records loaded".format(trainCounter, counter))
        print("Resetting sequence.")
        model.resetSequenceStates()
        print("pass {0}, {1} records loaded".format(trainCounter, counter))

        a_dict = {'timestamp':ts_list, 'traffic':tr_list, 'ps_count':ps_list}
        step_idx = 0
        tmpcolumns=['traffic', 'ps_count']
        for each_step in steps_list:
            a_dict['Step{0}PredictedTraffic'.format(each_step)] = pr_array[step_idx,0:len(ts_list)]
            tmpcolumns.append('Step{0}PredictedTraffic'.format(each_step))
            a_dict['Step{0}Confidence'.format(each_step)] = an_array[step_idx,0:len(ts_list)]
            tmpcolumns.append('Step{0}Confidence'.format(each_step))
            step_idx += 1

        row_df = pd.DataFrame(a_dict, columns=tmpcolumns, index=ts_list)
        row_df.index.name = 'timestamp'

        print("Writing {0}".format(outputName))
        row_df.to_csv(outputName, header=True, index=True)

    model.save( os.path.abspath( 'modelSave' ) )


def test():

    test_files = get_files(options.test_files)

    # load the previously trained model from disk
    model = Model.load( os.path.abspath( 'modelSave' ))

    # disable learning for testing
    model.disableLearning()

    # setup the metrics handling
    metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(), 
            model.getInferenceType())

    steps_list = []
    for each in options.prediction_steps_list.split(','): 
        steps_list.append(int(each))

    ts_list = []
    tr_list = []
    ps_list = []
#    pr_list = []
    pr_array = np.zeros(shape=(len(steps_list), 1000), dtype=float)
    an_list = []
    an_array = np.zeros(shape=(len(steps_list), 1000), dtype=float)

    shifter = InferenceShifter()

    plt.ion()
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

    predicted_history = []
    anomaly_score_history = []
    # Keep the last WINDOW predicted and actual values for plotting.
    #actual_history = deque([0.0] * WINDOW, maxlen=360)
    actual_history = []
    #ps_count_history = deque([0.0] * WINDOW, maxlen=360)
    ps_count_history = []
    for step_idx in range(0,len(steps_list)):
        #predicted_history.append(deque([0.0] * WINDOW, maxlen=360))
        #anomaly_score_history.append(deque([0.0] * WINDOW, maxlen=360))
        predicted_history.append(list())
        anomaly_score_history.append(list())

    data_ax = plt.subplot(gs[0])
    data_ax.set_color_cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    predicted_line = []
    #actual_line, = plt.plot(range(WINDOW), actual_history)
    actual_line, = plt.plot(ts_list, actual_history)
    #ps_count_line, = plt.plot(range(WINDOW), ps_count_history)
    ps_count_line, = plt.plot(ts_list, ps_count_history)
    tmplegend = ['traffic', 'ps_count']
    for step_idx in range(0,len(steps_list)):
        #pr_line, = plt.plot(range(WINDOW), predicted_history[step_idx])
        pr_line, = plt.plot(ts_list, predicted_history[step_idx])
        predicted_line.append(pr_line)
        tmplegend.append("Step{0}".format(steps_list[step_idx]))
    plt.legend(tuple(tmplegend), loc='upper center', fancybox=True, shadow=True)

    anom_ax = plt.subplot(gs[1])
    anom_ax.set_color_cycle(['r', 'c', 'm', 'y', 'k'])
    anomaly_score_line = []
    tmplegend = []
    for step_idx in range(0,len(steps_list)):
        #an_line, = plt.plot(range(WINDOW), anomaly_score_history[step_idx])
        an_line, = plt.plot(ts_list, anomaly_score_history[step_idx])
        anomaly_score_line.append(an_line)
        tmplegend.append("Step{0} Confidence".format(steps_list[step_idx]))
    plt.legend(tuple(tmplegend), loc='upper center',
            fancybox=True, shadow=True)

    # Set the y-axis range.
    actual_line.axes.set_ylim(40, 200)
    ps_count_line.axes.set_ylim(40, 200)
    for step_idx in range(0,len(steps_list)):
        predicted_line[step_idx].axes.set_ylim(40, 200)
        anomaly_score_line[step_idx].axes.set_ylim(0, 1)

    counter = 0
    for inputName in test_files:
        print("Opening {0}".format(inputName))
        in_df = pd.read_csv(inputName, parse_dates=['timestamp'],
                header=0, skiprows=[1,2], index_col='timestamp')
        in_rm = pd.rolling_mean(in_df.resample("60Min", fill_method="ffill"), 
                window=3, min_periods=1)

        for timestamp, traffic, ps_count in in_rm.itertuples():
            # run the data
            result = model.run({ "timestamp": timestamp,
                "traffic": traffic, 'ps_count':ps_count })

            # get the metrics data, over-time avarages and such
            # not sure how to interpret these yet... tbd
            result.metrics = metricsManager.update( result )

            # Get shifted result so the predictions line up
            shifted_result = shifter.shift(result)

            ts_list.append(timestamp)
            tr_list.append(shifted_result.rawInput['traffic'])
            ps_list.append(shifted_result.rawInput['ps_count'])

            step_idx = 0

            actual_history.append(shifted_result.rawInput['traffic'])
            ps_count_history.append(shifted_result.rawInput['ps_count'])
            for each_step in steps_list:

                # Update the trailing predicted and actual value deques.
                inference = shifted_result.inferences['multiStepBestPredictions'][each_step]
                if inference is not None:
                    # Redraw the chart with the new data.
                    predicted_history[step_idx].append(inference)
                    #anomaly_score = result.inferences['anomalyScore']
                    anomaly_score = result.inferences['multiStepPredictions'][each_step][result.inferences['multiStepBestPredictions'][each_step]]
                    anomaly_score_history[step_idx].append(anomaly_score)

                    # pr_list.append(inference)
                    pr_array[step_idx][counter] = inference
                    #an_list.append(result.inferences['anomalyScore'])
                    #an_list.append(result.inferences['multiStepPredictions'][each_step][result.inferences['multiStepBestPredictions'][each_step]])
                    an_array[step_idx][counter] = result.inferences['multiStepPredictions'][each_step][result.inferences['multiStepBestPredictions'][each_step]]
                else:
                    predicted_history[step_idx].append(0)
                    anomaly_score_history[step_idx].append(0)
                    pr_array[step_idx][counter] = 0
                    an_array[step_idx][counter] = 0

                # Redraw the chart with the new data.
                predicted_line[step_idx].set_xdata(ts_list)  # update the data
                predicted_line[step_idx].set_ydata(predicted_history[step_idx])  # update the data
                data_ax.relim()
                data_ax.autoscale_view(True, True, True)

                anomaly_score_line[step_idx].set_xdata(ts_list)  # update the data
                anomaly_score_line[step_idx].set_ydata(anomaly_score_history[step_idx])  # update the data
                anom_ax.relim()
                anom_ax.autoscale_view(True, True, True)

                step_idx += 1

            actual_line.set_xdata(ts_list)  # update the data
            actual_line.set_ydata(actual_history)  # update the data
            ps_count_line.set_xdata(ts_list)  # update the data
            ps_count_line.set_ydata(ps_count_history)  # update the data

#            print("actual_line - {0} [{1}] x {2} [{3}]".format(len(ts_list), 
#                    ts_list[counter], len(actual_history), actual_history[counter]))
#            print("ps_count_line - {0} [{1}] x {2} [{3}]".format(len(ts_list), 
#                    ts_list[counter], len(ps_count_history), ps_count_history[counter]))
#            for step_idx in range(0,len(steps_list)):
#                print("pr_line[{0}] - {1} [{2}] x {3} [{4}]".format(step_idx, 
#                        len(ts_list), ts_list[counter], 
#                        len(predicted_history[step_idx]), predicted_history[step_idx][counter]))
#                print("an_line[{0}] - {1} [{2}] x {3} [{4}]".format(step_idx, 
#                        len(ts_list), ts_list[counter],
#                        len(anomaly_score_history[step_idx]), anomaly_score_history[step_idx][counter]))

            plt.draw()
            plt.tight_layout()
            counter += 1

    plt.ioff()
    plt.show()



    a_dict = {'timestamp':ts_list, 'traffic':tr_list, 'ps_count':ps_list}
    step_idx = 0
    tmpcolumns=['traffic', 'ps_count']

    for each_step in steps_list:
        a_dict['Step{0}PredictedTraffic'.format(each_step)] = pr_array[step_idx,0:len(ts_list)]
        tmpcolumns.append('Step{0}PredictedTraffic'.format(each_step))

        a_dict['Step{0}Confidence'.format(each_step)] = an_array[step_idx,0:len(ts_list)]
        tmpcolumns.append('Step{0}Confidence'.format(each_step))
        step_idx += 1

    row_df = pd.DataFrame(a_dict, columns=tmpcolumns, index=ts_list)
    row_df.index.name = 'timestamp'

    print("Writing {0}".format(options.test_out))
    row_df.to_csv(options.test_out, header=True, index=True)

    # save it again, if the recently learned data needs to be updated
    # model.save( os.path.abspath( 'modelSave' ) )

    plt.ion()
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

    data_ax = fig.add_subplot(gs[0])
    data_ax.set_color_cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    data_ax.set_title(options.predicted_field + ' Prediction')

    anom_ax = fig.add_subplot(gs[1], sharex=data_ax)
    anom_ax.set_color_cycle([          'r', 'c', 'm', 'y', 'k'])

    plt.title('Click on the legend to toggle lines on/off')
    tr_line, = data_ax.plot(ts_list, row_df['traffic'], 
            label='traffic')
    ps_line, = data_ax.plot(ts_list, row_df['ps_count'], 
            label='ps_count')

    predicted_history = []
    tmplegend = ['traffic', 'ps_count']
    for each_step in steps_list:
        pr_line, = data_ax.plot(ts_list, row_df['Step{0}PredictedTraffic'.format(each_step)], 
                label='Step{0}Prediction'.format(each_step))
        predicted_line.append(pr_line)
        tmplegend.append("Step{0} Predicted".format(each_step))
    data_ax.legend(tuple(tmplegend), loc='upper center', 
            fancybox=True, shadow=True)
    #    line2.axes.set_ylim(0, 100)
    #    plt.set_xlabel('Dates')
    #    plt.set_ylabel('HTTP Connections', color='b')
#    for t1 in plt.get_yticklabels():
#        t1.set_color('b')


    anomaly_score_history = []
    tmplegend = []
    for each_step in steps_list:
        an_line, = anom_ax.plot(ts_list, row_df['Step{0}Confidence'.format(each_step)],
                label='Confidence')
        anomaly_score_line.append(an_line)
        tmplegend.append("Step{0} Confidence".format(each_step))
    an_line.axes.set_ylim(0, 1)
    anom_ax.legend(tuple(tmplegend), loc='upper center')
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

    gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    plt.ioff()
    plt.show()


if __name__ == "__main__":

    # TODO: Add optparse for
    # --swarm=<file>
    # --train=<file list or glob>
    # --test=<file list or glob>
  if options.swarm_file is not None:
      swarm()

  if options.train_files is not None:
      train()

  if options.test_files is not None:
      test()
