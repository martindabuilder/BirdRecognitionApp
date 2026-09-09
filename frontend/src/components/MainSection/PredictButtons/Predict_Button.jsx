import { useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { motion } from "motion/react"

import RollingButtonText from "../../Shared/Rolling_Button_Text.jsx"
import getBirdPhoto from "../../Shared/get_bird_photo.js"

import "./buttons.css"

function PredictButton({ processingStage, setProcessingStage }){

    const fileInputRef = useRef(null)
    const navigate = useNavigate()
    const [rollCount, setRollCount] = useState(0)

    function handleButtonClick(){fileInputRef.current.click()}

    async function handleFileChange(event){
        const selectedFile = event.target.files[0]

        if (!selectedFile){return}
        setProcessingStage("processing")

        const formData = new FormData()
        formData.append("file", selectedFile)

        try{
            const response = await fetch("http://127.0.0.1:8000/predict", {method: "POST", body: formData})
            const data = await response.json()
            const topSpecies = data.predictions?.[0]
            setProcessingStage("fetching-photo")

            let birdPhoto = null

            if (topSpecies){
                birdPhoto = await getBirdPhoto(topSpecies.species, topSpecies.scientificName)
            }

            setProcessingStage("readying-results")
            navigate("/results", {state: {...data, birdPhoto}})
        }
        catch (error){
            console.error(
                "Couldn't connect to backend:",
                error
            )
            setProcessingStage(null)
        }
    }

    const predicting = processingStage !== null

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
                onClick = {handleButtonClick}

                initial = {{ opacity: 0, scale: 0.8 }}
                animate = {{ opacity: predicting ? 0 : 1, scale: predicting ? 0.9 : 1 }}
                whileHover = {{ scale: 1.03, transition: {duration: 0.2, ease: "ease"} }}

                transition = {{
                    opacity: {duration: 0.3, ease: "ease", delay: predicting ? 0 : 0.2},
                    scale: {type: "spring", stiffness: 300, damping: 12, delay: predicting ? 0 : 0.2}
                }}
            >

                <RollingButtonText rollCount = {rollCount}>
                    Upload an audio file.
                </RollingButtonText>

            </motion.button>
        </div>
    )
}

export default PredictButton