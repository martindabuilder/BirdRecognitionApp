import { useState } from "react"

import MicrophonePredictButton from "../PredictButtons/Microphone_Predict_Button.jsx"
import PredictButton from "../PredictButtons/Predict_Button.jsx"

import "./main_section.css"

function MainSection(){

    const [processingStage, setProcessingStage] = useState(null)

    const processingMessages = {
        processing: "Analysing birdsong...",
        "fetching-photo": "Finding your bird...",
        "readying-results": "Preparing results..."
    }

    return(
        <section className = "main-section">

            <PredictButton
                processingStage = {processingStage}
                setProcessingStage = {setProcessingStage}
            />

            <MicrophonePredictButton
                processingStage = {processingStage}
                setProcessingStage = {setProcessingStage}
            />

            {processingStage && (
                <div className = "processing">
                    {processingMessages[processingStage]}
                </div>
            )}

        </section>
    )
}

export default MainSection