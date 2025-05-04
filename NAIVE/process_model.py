from ultralytics import YOLO

models = {}
if __name__ == '__main__':

    for m in {'yolov5nu', 'yolov5su', 'yolov5mu', 'yolov5lu', 'yolov5xu'}:
        #models[m] = torch.hub.load('ultralytics/ultralytics', m, force_reload=False, device='cpu')
        z = m + ".pt"
        models[m] = YOLO(z)
    # logger.info(    {'Component': "Process" , "Action": "Model's loaded ready to start processing" }  ) 

    print("Model Loaded")
