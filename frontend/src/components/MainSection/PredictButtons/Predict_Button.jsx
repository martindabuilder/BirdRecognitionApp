import { useRef } from "react"
import { useNavigate } from "react-router-dom"
import "./buttons.css"

function PredictButton(){
    const fileInputRef = useRef(null)
    const navigate = useNavigate()
    function handleButtonClick(){fileInputRef.current.click()}

    async function handleFileChange(event){
        const selectedFile = event.target.files[0]
        if (!selectedFile){
            return
        }

        const formData = new FormData()
        formData.append("file", selectedFile)

        try{
            const response = await fetch("http://127.0.0.1:8000/predict", {method: "POST", body: formData})
            const data = await response.json()
            navigate("/results", {state: data})
        }
        catch (error){console.error("Couldn't connect to backend:", error)}
    }

    return (
        <div className="predict-buttons">
            <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                style={{ display: "none" }}
            />

            <button
                className="predict-button"
                onClick={handleButtonClick}>
                Upload an audio file.
            </button>
        </div>
    )
}

export default PredictButton