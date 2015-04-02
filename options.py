import optparse

parser = optparse.OptionParser()
swarm_group = optparse.OptionGroup(parser, 'Swarming Options')
swarm_group.add_option('-s', '--swarm-file', 
        help='Initiate swarming using this file', 
        dest='swarm_file', metavar='<FILE>', 
        action='store')
swarm_group.add_option('-i', '--predicted-field', 
        help='Field to predict', 
        dest='predicted_field', metavar='<CSV FIELD>', 
        action='store')
swarm_group.add_option('-z', '--swarm-size', 
        help='Swarm size - default [%default]', 
        dest='swarm_size', metavar='<small, medium, large>', 
        action='store', default='medium')

train_group = optparse.OptionGroup(parser, 'Training Options')
train_group.add_option('-r', '--train-files', 
        help='Initiate training with these files', 
        dest='train_files', metavar='<FILESMASK|FILE1,FILE2,...>', 
        action='store')
train_group.add_option('-p', '--train-passes', 
        help='Number of training passes - defaults to one pass per file', 
        dest='train_passes', 
        action='store', type='int')
train_group.add_option('-m', '--train-random', 
        help='Randomize the order of the training files - default [%default]', 
        dest='train_random', 
        action='store_true', default=False)
train_group.add_option('-n', '--new-model', 
        help='Reset model and create new - default [%default]', 
        dest='new_model', 
        action='store_true')
train_group.add_option('-l', '--load-last', 
        help='Load last saved model - default [%default]', 
        dest='load_last', 
        action='store_true', default=False)

test_group = optparse.OptionGroup(parser, 'Testing Options')
test_group.add_option('-t', '--test-files', 
        help='Initiate testing with these files', 
        dest='test_files', metavar='<FILEMASK|FILE1,FILE2,...>', 
        action='store')
test_group.add_option('-o', '--test-out', 
        help='Output CSV file for test results - default [%default]', 
        dest='test_out', metavar='<FILE>', 
        action='store', default='out_test.csv')

collect_group = optparse.OptionGroup(parser, 'Collecting Options')
collect_group.add_option('-u', '--url', 
        help='URL to collect data from', 
        dest='collect_url', default=False, metavar='<URL>', 
        action='store')

parser.add_option_group(swarm_group)
parser.add_option_group(train_group)
parser.add_option_group(test_group)
parser.add_option_group(collect_group)
(options, args) = parser.parse_args()
