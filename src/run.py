from datetime import datetime

from PyPitch.classification.model import run_model


try:
    if __name__ == '__main__':
        start = datetime.now()
        print('||MSG', start, '|| ''PyPitch'' MODULE STARTING')
        r = run_model()
        if r == 0:
            end = datetime.now()
            print('||MSG', end, '|| ''PyPitch'' MODULE COMPLETED')
        else:
            raise ('||ERR', datetime.now(), '|| ERROR EXECUTING MODEL')

except:
    print('||ERR', datetime.now(), '|| CHECK ERRORS AND RERUN')

