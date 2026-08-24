import "../../styles/Predict_Section_Styles/predict_buttons.css"

function Buttons(){
    return (
        <div className = "predict-buttons">
            <button className = "predict-button">
                Predict audio file
            </button>

            <button className = "audio-predict-button">
                Predict by microphone
            </button>
        </div>
    )
}

export default Buttons