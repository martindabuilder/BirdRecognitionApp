import "../PredictButton/predict_buttons.css"
import "./microphone_predict_button.css"

function MicrophonePredictButton(){
    return (
        <div className = "predict-buttons">
            <button className = "microphone-predict-button">
                Record audio.
            </button>
        </div>
    )
}

export default MicrophonePredictButton