import "./predict_buttons.css"

function PredictButton(){
    async function handlePredict(){
        try{
            const response = await fetch("http://127.0.0.1:8000/predict");
            const data = await response.json();
            console.log(data);
        } 
        catch (error) {
            console.error("Couldn't connect to backend: ", error);
        }
    }

    return (
        <div className = "predict-buttons">
            <button className = "predict-button" onClick={handlePredict}>
                Upload an audio file.
            </button>
        </div>
    )
}

export default PredictButton