from modules.builder import builder
from modules.io import dataio
from modules.utils.logger import *
from modules.utils.parser_utils import *

def prepare_model_data(args, conf, device):
    log = logger(args, 'TrainTest')
    trainObj = builder.builder(args, conf, device)
    trainData = dataio.dataio('Train', args, conf, log.outdir)

    # Only load test data if we need it and test directory is specified
    testData = None
    if args.mode in ('TrainAndTest', 'Test'):
        # Check if test directory is the default and exists
        if args.test_dir == 'DefaultTest':
            print("WARNING: No test directory specified. Using default value 'DefaultTest' which doesn't exist.")
            print("If you want to use test data, please specify --test_dir parameter with a valid path.")
            testData = None
        else:
            try:
                testData = dataio.dataio('Test', args, conf, log.outdir)
            except SystemExit as e:
                # Catch the system exit and continue without test data
                print(f"WARNING: Failed to load test data from {args.test_dir}.")
                print("Continuing without test data.")
                testData = None

    save_args(args, log.outdir + '/checkpoint/current/')
    return trainObj, trainData, testData, log
