from Custom_Logger import logger

class Executor():

    def perform_action(self, act, model):

        logger.info(    {'Component': "Executor" , "Action": "Performing Action" }  ) 
        # print('Inside Execute, performing action: ', act)
        logger.data( {"Action": act})
        

        # model switch takes place by changing model name in model.csv file .
        logger.info( {'Component': "Executor" , "Action": f"Switching to model {model}" }  ) 
        f = open("model.csv", "w")
        f.write(model)
        f.close()

        logger.info(    {'Component': "Executor" , "Action": "Finished Action 1" }  ) 

        
        print("Adaptation completed.")
