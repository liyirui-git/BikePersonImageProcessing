'''
Author: Li, Yirui
Date: 2021-08-15
Description: Create a class named MyLog to help print to termial mean while print to log file
FilePath: /liyirui/PycharmProjects/BikePersonImageProcessing/mylog.py
'''
import os
import utils

class MyLog:

    def __init__(self, folder_path, file_name="log.txt"):
        self.file_path = os.path.join(folder_path, file_name)
        self.file = open(self.file_path, 'w')
    
    def write(self, message, color='black', print=True, space_line=True):
        if print:
            utils.color_print(message, color=color)
        if space_line:
            self.file.write(message + "\n")
        else:
            self.file.write(message + "\n")
    
    def close(self):
        self.file.close()
