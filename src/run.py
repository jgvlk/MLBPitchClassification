from datetime import datetime
import sys

from PyPitch.Classification.model import run_model


if __name__ == '__main__':
    start = datetime.now()
    print('||MSG', start, '|| ''PyPitch'' MODULE STARTING')

    args = sys.argv
    repo_dir = str(args[1])
    model_version = str(args[2])
    model_test_ratio = float(args[3])
    model_outlier_std_thresh = int(args[4])

    r = run_model(repo_dir, model_version, model_test_ratio, model_outlier_std_thresh)

    if r == 0:
        end = datetime.now()
        print('||MSG', end, '|| ''PyPitch'' MODULE COMPLETED')
    else:
        end = datetime.now()
        print('||ERR', end, '|| ''PyPitch'' MODULE FINISHED WITH ERRORS')

    runtime = end - start
    print('||MSG', datetime.now(), '|| RUNTIME:', runtime)

