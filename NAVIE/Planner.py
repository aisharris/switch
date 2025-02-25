from Execute import Executor
import pandas as pd
from Custom_Logger import logger

class Planner():
    def __init__(self,  input_rate , model  ):
        self.input_rate = input_rate
        self.model = model
        logger.info(    {'Component': "Planner" , "Action": "Planner Object created" }  ) 


    def generate_adaptation_plan(self , count):
        
        action = 0
        
        df = pd.read_csv('knowledge.csv', header=None)
        array = df.to_numpy()

        in_rate = self.input_rate

        #check's which model's thershold range is input rate within and accordingly determines the action.
        logger.info(    {'Component': "Planner" , "Action": "Generating the adaptation plan" } )

        # set action to idx + 1 for idx corresponding to appropriate model
        for i in range(len(array)):
            if ( in_rate >= array[i][1] and in_rate <= array[i][2]):
                action = i + 1
        
        if action == 0:
            logger.error(    {'Component': "Planner" , "Action": "No adaptation plan generated" }  ) 
            return
        
        #creates Executor object and call's to perform action.
        exe_obj = Executor()
        exe_obj.perform_action(action)