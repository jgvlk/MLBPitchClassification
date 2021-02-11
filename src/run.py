from datetime import datetime

from PyPitch.classification.model import run_model


if __name__ == '__main__':
    start = datetime.now()
    print('||MSG', start, '|| ''PyPitch'' MODULE STARTING')
    r = run_model()
    if r == 0:
        end = datetime.now()
        print('||MSG', end, '|| ''PyPitch'' MODULE COMPLETED')
    else:
        end = datetime.now()
        print('||ERR', end, '|| ''PyPitch'' MODULE FINISHED WITH ERRORS')

    runtime = end - start
    print('||MSG', datetime.now(), '|| RUNTIME:', runtime)

