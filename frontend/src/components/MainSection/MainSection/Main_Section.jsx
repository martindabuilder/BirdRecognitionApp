import "./main_section.css"

import TitleBar from "../TitleBar/TitleBar.jsx"

import MicrophonePredictButton from "../PredictButtons/Microphone_Predict_Button.jsx"
import PredictButton from "../PredictButtons/Predict_Button.jsx"


function MainSection(){
    return(
        <section className = "main-section">
            <TitleBar />
            <PredictButton />
            <MicrophonePredictButton />
        </section>

    )
}

export default MainSection