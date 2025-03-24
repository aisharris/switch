import React, { useState, useEffect  } from 'react';
import JSZip from 'jszip';
import Dashboard from './Dashboard';
import User_approch from './User_approch'
import User_approch_MAPE_K from './User_Approch_MAPE_K'

const Home = () => {
  const [selectedZipFile, setSelectedZipFile] = useState(null);
  const [selectedCSVFile, setSelectedCSVFile] = useState(null);
  const [showDashBoard ,  setShowDashBoard] = useState(false);
  const [stopProcessing , setstopProcessing] = useState(false);
  const [selectedOption, setSelectedOption] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [models, setModels] = useState([]);
  const [ID, setID] = useState('')
  const [loc, setLoc] = useState('')
  const [selectedVideoFile, setSelectedVideoFile] = useState(null)
  const handleZipFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedZipFile(file);
  };

  useEffect(() => {
    // Fetch the available models from the backend
    const fetchModels = async () => {
      try {
        const response = await fetch('http://localhost:3001/api/models'); // New API endpoint
        if (response.ok) {
          const modelData = await response.json();
          setAvailableModels(modelData.models); // Set the models from the response

          // Initialize models state with fetched model names
          const initialModels = modelData.models.map(modelName => ({
            name: modelName,
            lower: "",
            upper: ""
          }));
          setModels(initialModels);
        } else {
          console.error('Failed to fetch models.');
        }
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    };

    fetchModels(); // Call the function to fetch models
  }, []); // Empty dependency array ensures this runs only once on mount

  const handleCSVFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedCSVFile(file);
  };

  const handleLocationChange=(event)=>{
    setLoc(event.target.value);
    console.log(loc)
  }
  const handleVideoFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedVideoFile(file);
    console.log(file)
  };
  const handleIdChange = (event) => {
    // Update the ID state with the new value from the input field
    setID(event.target.value);
    console.log(ID)
  };

  const handleUpload = async () => {
    if ((!selectedZipFile && !selectedVideoFile && loc==='') || !selectedCSVFile) {
      return;
    }

    try {
      setShowDashBoard(true);
      
      if(selectedZipFile){
        const zip = new JSZip();
        await zip.loadAsync(selectedZipFile);

        const files = [];
        zip.forEach(async (relativePath, file) => {
          const content = await file.async('uint8array');
          files.push({ path: relativePath, content });
        });
      }
      

      const formData = new FormData();
      if (selectedZipFile) {
        formData.append('zipFile', selectedZipFile);
        console.log('Zip file added to foem data')
      }
      if (selectedVideoFile) {
        formData.append('videoFile', selectedVideoFile);
        console.log('Video file added to form data')
      }
      formData.append('csvFile', selectedCSVFile);
      formData.append('approch', selectedOption);
      if (loc) {
        formData.append('folder_location', loc);
      }
  
      console.log(selectedOption, loc)
      const response = await fetch('http://localhost:3001/api/upload', {
        method: 'POST',
        body: formData,
      });

      // Handle the response from the backend
      if (response.ok) {
        console.log('Files uploaded successfully.');
        setShowDashBoard(true);
        // var url = 'http://localhost:5601/app/dashboards#/list?_g=(filters:!(),refreshInterval:(pause:!t,value:0),time:(from:now%2Fd,to:now%2Fd))'
      //   if(window.confirm("View Dashboard")){
      //     window.open(url);
      //  }
        
      } else {
        console.error('Failed to upload files.');
      }
    } catch (error) {
      console.error('Error during file upload:', error);
    }
  };

  const handleModelsUpdate = (updatedModels) => {
    setModels(updatedModels);
  };

  const stopProcess = async () => {
    try{
      const response = await fetch('http://localhost:3001/api/stopProcess', {
          method: 'POST'
        });
      if(response.ok){
        console.log("Stoped succesfully");
        setstopProcessing(true);
      }
      else{
        console.log("Failed to stop program")
      }
    }catch(error){
      console.error('Error during fstoping program:', error);
    }
  };
  
  const newProcess = async ()=>{

    try{
      const response = await fetch('http://localhost:3001/api/newProcess', {
          method: 'POST'
        });
      if(response.ok){
        console.log("Restartes process");
        setstopProcessing(true);
        setID('')
        setSelectedCSVFile(null)
        setSelectedOption('')
        setLoc('')
        setSelectedZipFile(null)

      }
      else{
        console.log("Failed to restart program")
      }
    }catch(error){
      console.error('Error during fstoping program:', error);
    }

    setShowDashBoard(false);
    setstopProcessing(false);
  }

  const downloadData= async ()=>{
    try{
      const response = await fetch('http://localhost:3001/api/downloadData', {
          method: 'POST',
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ data: ID }),
       
        });
      if(response.ok){
        console.log("Downloaded succesfully");
        alert("Downloaded Succesfully")
      }
      else{
        console.log("Failed to stop program")
      }
    }catch(error){
      console.error('Error during fstoping program:', error);
    }
  }

  const handleSelectChange = async(event) => {
    setSelectedOption(event.target.value);
    console.log(event.target.value)
    if(event.target.value === "NAIVE"){
      try{
        const response = await fetch('http://localhost:3001/useNaiveKnowledge', {
          method: 'POST'
        });
        if(response.ok){
          console.log("NAIVE knowledge updated");
          // alert("Downloaded Succesfully")
        }
        else{
          console.log("Failed load NAIVE knowledge")
        }
      }catch(error){
        console.error('Error during loading NAIVE knowledge:', error);
      }
    }
  };

  return (
    <div className="container mt-3 ">
      {!showDashBoard && <div>
        <h1 className="mb-3">SWITCH: An Exemplar for Evaluating Self-Adaptive ML-Enabled Systems</h1>
        <div className="mb-3">
          <label htmlFor="zipFileInput" className="form-label">
            Upload a .zip file, for folder contaning images, the .zip file must have same name as the Image folder.
          </label>
          <input
            type="file"
            className="form-control"
            id="zipFileInput"
            onChange={handleZipFileChange}
          />
        </div>

        <div className="mb-3">
          <label htmlFor="text" className="form-label">
            Upload folder base location if zip size greater than 700MB .
          </label>
          <input
            type="text"
            className="form-control"
            id="textInput"
            onChange={handleLocationChange}
          />
        </div>

        <div className="mb-3">
          <label htmlFor="videoFileInput" className="form-label">
            Or, upload a video file.
          </label>
          <input
            type="file"
            className="form-control"
            id="videoFileInput"
            accept="video/*"
            onChange={handleVideoFileChange}
          />
        </div>

        <div className="mb-3">
          <label htmlFor="csvFileInput" className="form-label">
            Upload a csv file contaning inter arrival rate data.
          </label>
          <input
            type="file"
            className="form-control"
            id="csvFileInput"
            onChange={handleCSVFileChange}
          />
        </div>
        <div>
        
        <div className="container">
      <div className="row justify-content-center">
        <div className="col-md-6">
          <div className="card">
            <div className="card-body">
              <h5 className="card-title">ID your Experiment</h5>
              <div className="form-group">
                <input
                  type="text"
                  id="IdInput"
                  className="form-control"
                  onChange={handleIdChange}
                  value={ID}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

        <div className="mb-3 mt-3 h5">
          <select className="selectpicker" value={selectedOption} onChange={handleSelectChange}>
            <option value="">Select an option</option>
            <option value="NAIVE">NAIVE</option>
            <option value="AdaMLs">AdaMLS</option>
            <option value="Try Your Own">Modify NAIVE</option>
            <option value="Write Your Own MAPE-K">Upload MAPE-K files</option>

            {availableModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
          </select>
        </div>
        
        {selectedOption === "Try Your Own" &&

        <div className="mb-3">
            <User_approch 
            models={models} 
            availableModels={availableModels} // Pass availableModels as prop
            onModelsUpdate={handleModelsUpdate} 
          />
        </div>
        }

        {selectedOption === "Write Your Own MAPE-K" &&

        <div className="mb-3">
            <User_approch_MAPE_K
            id = {ID}/>
        </div>
        }

        </div>
        <button className="btn btn-primary " onClick={handleUpload}>
          Upload Files 
        </button>
      </div>}
      <div>
        {showDashBoard && 
        <div>
          <Dashboard/>
          {!stopProcessing &&
          <button className="btn btn-primary" onClick={stopProcess}>
            Stop Process
          </button>}
          {stopProcessing && 
          <p>
            <button className="btn btn-primary" onClick={downloadData}>
            Download Data
            </button>
          <br />
          <br />
            <button className="btn btn-primary" onClick={newProcess}>
            New Process
            </button> 
          </p>
          }
        </div>
        }
      </div>
    </div>
  );
};

export default Home;
