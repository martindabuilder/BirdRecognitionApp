import { useState } from "react"

import MicrophonePredictButton from "../PredictButtons/Microphone_Predict_Button.jsx"
import PredictButton from "../PredictButtons/Predict_Button.jsx"

import "./main_section.css"

function MainSection(){

    const [predicting, setPredicting] = useState(false)

    return(
        <section className = "main-section">
            <PredictButton predicting = {predicting} setPredicting = {setPredicting} />
            <MicrophonePredictButton predicting = {predicting} setPredicting = {setPredicting} />

            {predicting && (
                <div className = "processing">
                    Processing audio..
                </div>
            )}
        </section>
    )
}

export default MainSection