from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import zipfile
import os
import subprocess
import time
from elasticsearch import Elasticsearch
from typing import Dict
import csv
from get_data import write_csv, write_json
import platform
import sys

# Modifications Start Here
from VideoToImgZip import video_to_images
# Modifications End Here

es = Elasticsearch(['localhost'])
app = FastAPI()
sys_approch = "NAIVE"
# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

process_running = False
running_processes = []
monitor_directory = ''

def run_in_terminal(command, working_directory=None):
    global running_processes

    try:
        command_list = command.split()
        running_processes.append(subprocess.Popen(command_list, cwd=working_directory))
    except Exception as e:
        print("Couldn't run processes in terminal: ", str(e))

def run_as_background(command):
    try:
        command_list = command.split()
        subprocess.Popen(command_list)
    except Exception as e:
        print("Couldn't run processes in terminal: ", str(e))

# def run_in_new_terminal(command):
#     try:
#         subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', command])
#     except Exception as e:
#         print("new repo")
#         print("Couldn't run processes in terminal: ", str(e))

def run_in_new_terminal(command):
    try:
        # Simple background execution
        process = subprocess.Popen(['bash', '-c', command], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        print(f"Started process with PID: {process.pid}")
    except Exception as e:
        print("\n\n\n\n ---------not fixed----------- \n\n\n\n")
        print("Couldn't run process: ", str(e))

# in case you run into issues with bg process above:
# def run_in_new_terminal(command):
#     # Try different terminal emulators
#     terminals = ['gnome-terminal', 'xterm', 'konsole', 'terminator', 'kitty', 'alacritty']
    
#     for terminal in terminals:
#         try:
#             subprocess.Popen([terminal, '--', 'bash', '-c', command])
#             return  # Exit after successful launch
#         except FileNotFoundError:
#             continue
    
#     # If no terminal emulator is found, run in background
#     print("No terminal emulator found. Running in background.")
#     subprocess.Popen(['bash', '-c', command])

def stop_proccess():
    try: 
        global running_processes
        for script in running_processes:
            script.terminate()
        running_processes.clear()
    except Exception as e:
        print(f"Couldn't stop process in terminal: ",str(e))


def stop_process_in_terminal(file):

    command = f"pgrep -f {file}"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    if error is None:
        output = output.decode().strip()  # Convert bytes to string
        print("Process ID:", output)
        command = f"kill {output}"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    else:
        print("Error:", error.decode())

# gets all the models that are loaded: .pt files in dir
# to add new models, add files
@app.get("/api/models")
async def get_models():
    try:
        models = []
        for filename in os.listdir("."):
            if filename.endswith(".pt"):
                models.append(filename[:-3]) # Remove the .pt
        return {"models": models}
    except Exception as e:
        print("Error getting models:", str(e))
        raise HTTPException(status_code=500, detail="Error getting models")

@app.post("/api/upload")
async def upload_files(zipFile: UploadFile = File(None), videoFile: UploadFile = File(None), csvFile: UploadFile = File(...),  approch: str = Form(...), folder_location: str = Form(None)):

    global sys_approch
    try:
        print(approch)
        sys_approch = approch
        # Create a directory to store the uploaded files
        upload_dir = "uploads"
        shutil.rmtree(upload_dir, ignore_errors=True)
        os.makedirs(upload_dir, exist_ok=True)

        unzip_dir = "unzipped"
        shutil.rmtree(unzip_dir, ignore_errors=True)
        os.makedirs(unzip_dir, exist_ok=True)

        # Save the uploaded files   
        csv_path = os.path.join(upload_dir, csvFile.filename)
        with open(csv_path, "wb") as cf:
            shutil.copyfileobj(csvFile.file, cf)

        unzip_dir = "unzipped"
        print(f"folder location is ->{folder_location}.")

        # Modifications Start Here
        if (videoFile is not None):
            # Save the Uplaoded video file to Uploads folder
            video_path = os.path.join(upload_dir, videoFile.filename)
            with open(video_path, "wb") as vf:
                shutil.copyfileobj(videoFile.file, vf)

            # print("Ehhh")
            video_path = os.path.join(upload_dir, videoFile.filename)
            output_folder = '{}'.format(video_path.split('.')[0])
            zip_name = '{}.zip'.format(output_folder)

            video_to_images(video_path, output_folder=output_folder, zip_name=zip_name)

            zip_path = zip_name

            shutil.rmtree(unzip_dir, ignore_errors=True)
            os.makedirs(unzip_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(unzip_dir)

            print("Folder unzipped successfully.")
            IMAGES_FOLDER = "unzipped/"+videoFile.filename
            IMAGES_FOLDER = IMAGES_FOLDER[:-4]

        elif(zipFile is not None):
        # Modifications End Here
            zip_path = os.path.join(upload_dir, zipFile.filename)
            
            with open(zip_path, "wb") as zf:
                shutil.copyfileobj(zipFile.file, zf)

            # Unzip the uploaded zip file
            
            shutil.rmtree(unzip_dir, ignore_errors=True)
            os.makedirs(unzip_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(unzip_dir)

            print("Folder unzipped successfully.")
            IMAGES_FOLDER = "unzipped/"+zipFile.filename
            IMAGES_FOLDER = IMAGES_FOLDER[:-4]
        else:
            IMAGES_FOLDER = folder_location
        # Perform additional logic with the unzipped folder here

        # Move the CSV file to the unzipped folder
        csv_dest_path = os.path.join(unzip_dir, csvFile.filename)
        shutil.move(csv_path, csv_dest_path)
        print("CSV file saved successfully.")

        CSV_FILE =  "unzipped/" + csvFile.filename
        
        print(CSV_FILE , IMAGES_FOLDER)

        #API to accept user reequests
        run_in_terminal('python3 App.py')
        time.sleep(0.5)
        #Locust to send Request
        run_in_new_terminal(f'export CSV_FILE="{CSV_FILE}" && export IMAGES_FOLDER="{IMAGES_FOLDER}" && locust -f Request_send.py --headless  --host=http://localhost:5000/v1 --users 1 --spawn-rate 1')
        #to start monitoring
        if(approch == "AdaMLs"):   
            print("RUunning monitor_ada.py---------------------")
            run_in_terminal('python3 monitor_ada.py', working_directory='AdaMLs')
        elif(approch == "NAIVE" or approch == "Try Your Own"):
            print("RUunning monitor.py---------------------")
            run_in_terminal('python3 monitor.py')
        elif(approch == "Write Your Own MAPE-K"):
            print("Montior from director: ",monitor_directory)
            run_in_terminal('python3 monitor.py', working_directory=f'{monitor_directory}')
          
        else:
            with open('model.csv', 'w') as file:
                writer = csv.writer(file)
                writer.writerow([approch])
        #upload data to ES
        run_in_terminal('python3 logs_to_es.py')
        run_in_terminal('python3 metrics_to_es.py')
        return {"message": "Files uploaded and processed successfully."}

    except Exception as e:
        print("Error during file upload:", str(e))
        raise HTTPException(status_code=500, detail="An error occurred during file upload.")


@app.post("/execute-python-script")
async def execute_python_script():
    global process_running
    if(process_running):
        return {"message": "Python script already running."}
    try:
        # Start the process.py script
        run_in_terminal('python3 process.py')
        process_running = True
        return {"message": "Python script started successfully."}
    except Exception as e:
        print("Error executing Python script:", str(e))
        raise HTTPException(status_code=500, detail="An error occurred while executing the Python script.")

@app.post("/api/stopProcess")
async def stopProcess():
    try:
        stop_process_in_terminal("Request_send.py")
        stop_proccess()
        return {"message" : "Stoped succesful"}
    except Exception as e:
        print("Error stoping:", str(e))
        raise HTTPException(status_code=500, detail="An error occurred while stoping")

@app.post("/api/newProcess")
async def restartProcess():
    try:
        run_in_terminal('python3 process.py')
        return {"message" : "Process succesful restarted"}
    except Exception as e:
        print("Error stoping:", str(e))
        raise HTTPException(status_code=500, detail="An error occurred while stoping")



@app.post("/api/downloadData")
async def startDownload(data: dict):
    try:
        data_str = data.get("data")
        print(data_str)
        # Define the folder name
        folder_name = "Exported_metrics"

        # Check if the folder exists
        if not os.path.exists(folder_name):
            # Create the folder
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")

        folder_name = "Exported_logs"

        # Check if the folder exists
        if not os.path.exists(folder_name):
            # Create the folder
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")

        # You can use data_str in your script as needed
        # run_as_background('python3 get_data.py')
        write_csv('final_metrics_data' , f'Exported_metrics/exported-data-metrics_{data_str}.csv')
        #saves logs data to json file
        write_json('new_logs' , f'Exported_logs/exported-data-logs_{data_str}.json')
        return {"message": "Downloaded successfully"}
    except Exception as e:
        print("Error stopping:", str(e))
        raise HTTPException(status_code=500, detail="An error occurred while stopping")
    

@app.post("/api/latest_metrics_data")
async def latest_metrics_data():
    try:
        index_name = 'final_metrics_data'
        # Search for the last document created
        search_body = {
            "query": {
                "match_all": {}
            },
            "sort": [
                {
                    "timestamp": {
                        "order": "desc"
                    }
                }
            ],
            "size": 1
        }
        search_result = es.search(index=index_name, body=search_body)
        # Extract the data from the search result
        if search_result['hits']['total']['value'] > 0:
            last_document = search_result['hits']['hits'][0]['_source']
            print(last_document)
            return {'message':last_document}
            
        else:
            return {'message':'No documrnt found'}
            print("No documents found in the index")

    except Exception as e:
        print("Error stoping:", str(e))
        raise HTTPException(status_code=500, detail="An error occurred while stoping")

@app.post("/api/latest_logs")
async def latest_log_data():
    try:
        index_name = 'new_logs'
        # Search for the last document created
        search_body = {
            "query": {
                "match_all": {}
            },
            "sort": [
                {
                    "timestamp": {
                        "order": "desc"
                    }
                }
            ],
            "size": 1
        }
        search_result = es.search(index=index_name, body=search_body)
        # Extract the data from the search result
        if search_result['hits']['total']['value'] > 0:
            last_document = search_result['hits']['hits'][0]['_source']
            print(last_document)
            return {'message':last_document}
            
        else:
            return {'message':'No documrnt found'}
            print("No documents found in the index")

    except Exception as e:
        print("Error stoping:", str(e))
        raise HTTPException(status_code=500, detail="An error occurred while stoping")


@app.post("/api/changeKnowledge")
async def change_knowledge(data: Dict[str, list[Dict[str, str]]]):
    try:
        models_data = data["models"]

        with open('knowledge.csv', 'w', newline='') as file:  # Add newline='' to prevent extra blank lines
            writer = csv.writer(file)
            for model in models_data:
                row = []
                row.append(model["name"])
                row.append(model["lower"])  # Use model name directly
                row.append(model["upper"])  # Use model name directly
                writer.writerow(row)

        return {"message": "Changed knowledge file"}

    except Exception as e:
        print("Error updating knowledge file:", str(e))
        raise HTTPException(status_code=500, detail="An error occurred updating knowledge file")

@app.post("/useNaiveKnowledge")
async def useNaive_knowledge():

    try:
        input_file = 'naive_knowledge.csv'
        output_file = 'knowledge.csv'
        with open(input_file, 'r') as input_csv, open(output_file, 'w', newline='') as output_csv:
            reader = csv.reader(input_csv)
            writer = csv.writer(output_csv)

        # Copy each row from the input CSV to the output CSV
            for row in reader:
                writer.writerow(row)
        return {"message": "Naive knowledge registered"}
        print("CSV data copied successfully.")

    except Exception as e:
        print("Error stoping:", str(e))
        raise HTTPException(status_code=500, detail="An error occurred updating knowledge file")
    

def save_uploaded_file(file: UploadFile, directory: str):
    with open(os.path.join(directory, file.filename), 'wb') as f:
        f.write(file.file.read())

# Define a function to unzip and save the knowledge zip file
def unzip_and_save_knowledge(zip_file: UploadFile, directory: str):
    with open(os.path.join(directory, zip_file.filename), 'wb') as f:
        f.write(zip_file.file.read())
    
    # Unzip the knowledge zip file and save its contents in the same directory
    with zipfile.ZipFile(os.path.join(directory, zip_file.filename), 'r') as zip_ref:
        zip_ref.extractall(directory)

@app.post('/your_mape_k')
async def upload_files(
    monitor: UploadFile,
    analyzer: UploadFile,
    planner: UploadFile,
    execute: UploadFile,
    knowledge: UploadFile,
    id: str = Form(...), 
):
    global monitor_directory
    monitor_directory = f"external_MAPE_K_{id}"
    # Create the "external_MAPE_K_{id}" directory if it doesn't exist
    os.makedirs(f"external_MAPE_K_{id}", exist_ok=True)
    print("Received ID: ", id)
    # Save the uploaded files to the "external_MAPE_K_{id}" directory
    save_uploaded_file(monitor, f"external_MAPE_K_{id}")
    save_uploaded_file(analyzer, f"external_MAPE_K_{id}")
    save_uploaded_file(planner, f"external_MAPE_K_{id}")
    save_uploaded_file(execute, f"external_MAPE_K_{id}")

    # Unzip and save the knowledge zip file
    unzip_and_save_knowledge(knowledge, f"external_MAPE_K_{id}")

    return {"message": "Files uploaded and knowledge unzipped successfully"}




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3001)