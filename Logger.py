# Logger class:
# To log special events and append to a file after running
__author__ = "Amin Aghaee"
__copyright__ = "Copyright 2018, Amin Aghaee"

import datetime

class Logger:
    """To log special events and append to a file after running"""

    def __init__(self, fname = './log.txt'):
        self.fname = fname


    def addlog(self, message = None):

        if message is None:
            return 1

        try:
            with open(self.fname, "a") as f:
                fmessage = str(datetime.datetime.now()) + " : " + message + "\n"
                f.write(fmessage)
        except:
            print("Failed to log!")

