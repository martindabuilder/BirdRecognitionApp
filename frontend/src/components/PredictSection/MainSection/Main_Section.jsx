import "./main_section.css"

import MicrophonePredictButton from "../MicrophonePredictButton/Microphone_Predict_Button.jsx"
import PredictButton from "../PredictButton/Predict_Button.jsx"

function MainSection(){
    return(
        <section className = "main-section">
            <p> Birds yaaayyy </p>
            <PredictButton />
            <MicrophonePredictButton />
        </section>

    )
}

export default MainSection