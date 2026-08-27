import MicrophonePredictButton from "../PredictButtons/Microphone_Predict_Button.jsx"
import PredictButton from "../PredictButtons/Predict_Button.jsx"

import "./main_section.css"

function MainSection(){
    return(
        <section className = "main-section">
            <PredictButton />
            <MicrophonePredictButton />
        </section>

    )
}

export default MainSection