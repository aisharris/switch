import React, { useState } from "react";
import axios from "axios";

function User_approach({ availableModels, onModelsUpdate }) {
  const [localModels, setLocalModels] = useState([]); // Start with an empty array
  const [finalized, setFinalized] = useState(false);
  const [selectedModel, setSelectedModel] = useState(""); // For the model selection dropdown

  const addModel = () => {
    if (selectedModel) { // Only add if a model is selected
      const modelExists = localModels.some(model => model.name === selectedModel);
      if(!modelExists){
        setLocalModels([
          ...localModels,
          { name: selectedModel, lower: "", upper: "" },
        ]);
        setSelectedModel(""); // Reset the dropdown after adding
      }
      else{
        alert("Model already added")
      }
    }
  };

  const deleteModel = (index) => {
    const updatedModels = [...localModels];
    updatedModels.splice(index, 1); // Remove the model at the specified index
    setLocalModels(updatedModels);
    onModelsUpdate(updatedModels);
  };

  const handleModelChange = (index, field, value) => {
    const updatedModels = [...localModels];
    updatedModels[index][field] = value;
    setLocalModels(updatedModels);
    onModelsUpdate(updatedModels);
  };

  const changeKnowledge = async () => {
    try {
      const response = await axios.post(
        "http://localhost:3001/api/changeKnowledge",
        {
          models: localModels,
        }
      );
      alert("Knowledge finalized");
      setFinalized(true);
      console.log(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const availableModelsToAdd = availableModels.filter(
    (model) => !localModels.find((addedModel) => addedModel.name === model)
  );

  return (
    <div>
      {!finalized && (
        <div>
          {localModels.map((model, index) => (
            <div key={index} className="model-config">
              {/* <h3>Model {index + 1}</h3> */}
              <div className="row mt-2">
                <div className="col-md-4">
                  <select
                    id={`modelName${index}`}
                    className="form-control"
                    value={model.name}
                    onChange={(e) =>
                      handleModelChange(index, "name", e.target.value)
                    }
                  >
                    {/* <option value="">Select a model</option> */}
                    {availableModels.map((availableModel) => (
                      <option key={availableModel} value={availableModel}>
                        {availableModel}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="col-md-4">
                  <input
                    type="text"
                    className="form-control"
                    id={`lower${index}`}
                    placeholder="Lower bound"
                    value={model.lower}
                    onChange={(e) =>
                      handleModelChange(index, "lower", e.target.value)
                    }
                  />
                </div>
                <div className="col-md-4">
                  <input
                    type="text"
                    className="form-control"
                    id={`upper${index}`}
                    placeholder="Upper bound"
                    value={model.upper}
                    onChange={(e) =>
                      handleModelChange(index, "upper", e.target.value)
                    }
                  />
                </div>
                <div className="col-md-1"> {/* Add a column for the delete button */}
                  <button
                    className="btn btn-danger btn-sm mt-2" // Style as a small button
                    onClick={() => deleteModel(index)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}

          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="form-control mt-2"
          >
            <option value="">Select a model to add</option>
            {availableModelsToAdd.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>

          <button className="btn mt-3 btn-primary" onClick={addModel}>
            Add Model
          </button>

          <button
            className="btn mt-3 ml-2 btn-danger"
            onClick={changeKnowledge}
          >
            Finalize Knowledge
          </button>
        </div>
      )}
    </div>
  );
}

export default User_approach;