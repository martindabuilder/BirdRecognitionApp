import { useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { motion } from "motion/react"

import RollingButtonText from "./Rolling_Button_Text.jsx"

import "./buttons.css"

function PredictButton(){
    const fileInputRef = useRef(null)
    const navigate = useNavigate()
    const [rollCount, setRollCount] = useState(0)

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
        <div className = "predict-buttons">
            <input
                ref = {fileInputRef}
                type = "file"
                accept = "audio/*"
                onChange = {handleFileChange}
                style = {{ display: "none" }}
            />

            <motion.button
                className = "predict-button"
                onHoverStart = {() => setRollCount((current) => current + 1)}
                onClick = {handleButtonClick}>

                <RollingButtonText rollCount={rollCount}>
                    Upload an audio file.
                </RollingButtonText>

            </motion.button>
        </div>
    )
}

export default PredictButton