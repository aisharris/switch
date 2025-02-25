# determine if the systems needs adaption. Checks if the parameters/ thresholds are violated
from Planner import Planner
import pandas as pd
import time
from Custom_Logger import logger

class Analyzer():
    def __init__(self):

        # setting threshold values obtained from knowledge.csv file.

        self.time = -1

        df = pd.read_csv('knowledge.csv', header=None)

        array = df.to_numpy()
        self.thresholds = {}

        # knowledge.csv format: model_name, rate_min, rate_max
        # add thresholds of models to dict: threshold[model_name_rate_min] = min_threshold
        for i in range(len(array)):
            str_min = array[i][0] + "_rate_min"
            str_max = array[i][0] + "_rate_max"
            self.thresholds[str_min] = array[i][1]
            self.thresholds[str_max] = array[i][2]

        self.count = 0

    def perform_analysis(self, monitor_dict):

        logger.info(    {'Component': "Analyzer" , "Action": "Performing the analysis" }  ) 

        input_rate = monitor_dict["input_rate"]
        model = monitor_dict["model"]

        str_min = model + "_rate_min"
        str_max = model + "_rate_max"
        current_time = time.time()

        # get's the minimum and maximum threshold values for the current working model.

        min_val = self.thresholds.get(str_min)
        max_val = self.thresholds.get(str_max)

        if ((max_val >= input_rate and min_val <= input_rate) == False):

            if (self.time == -1):
                self.time = current_time
            # if threshold sre violated for more than 0.25 sec, we create planner object to obtain the adaptation plan
            elif (current_time - self.time > 0.25):

                self.count += 1
                logger.info(    {'Component': "Analyzer" , "Action": "Creating Planner object" }  ) 
                plan_obj = Planner(input_rate, model)
                plan_obj.generate_adaptation_plan(self.count)

        else:
            self.time = -1
