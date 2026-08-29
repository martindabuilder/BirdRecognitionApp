import { useLocation, useNavigate } from "react-router-dom"
import BirdPhoto from "../Shared/BirdPhoto.jsx"

import "./results_section.css"

function ResultsSection(){
    const navigate = useNavigate()
    const location = useLocation()
    const result = location.state

    function goHome(){navigate("/")}

    if (!result) {
        return (
            <section className="results-section">
                <h1>No results found</h1>
                <button className="escape-button" onClick={goHome}>
                    go back lol
                </button>
            </section>
        )
    }

    const predictions = result.predictions || []
    const topSpecies = predictions[0]

    return (
        <section className="results-section">

            <button className="escape-button" onClick={goHome}>
                <span></span>
                <span></span>
            </button>
            <h3> Prediction results </h3>
            <div className="bird-photo-container">
                {topSpecies && (
                    <BirdPhoto
                        commonName = {topSpecies.species}
                        scientificName = {topSpecies.scientificName}
                    />
                )}
            </div>

            <div className="total-results-container">
                {topSpecies && (
                    <div className="main-confidence">
                        <h3>{topSpecies.species}</h3>
                        <p>Confidence: {(topSpecies.probabilities * 100).toFixed(2)}%</p>
                    </div>
                )}

                <div className="top4-predictions">
                    {predictions.slice(1, 5).map((prediction, index) => (
                        <div className="prediction" key = {prediction.species}>
                            <span> {index + 1}. {prediction.species}</span>
                            <span> {(prediction.probabilities * 100).toFixed(2)}%</span>
                        </div>
                    ))}
                </div>

                <div className="spectrogram-segments">
                    {result.spectrograms.map((spectrogram, index) => (
                        <img key = {index} src = {`data:image/png;base64,${spectrogram}`} alt = {`Spectrogram segment ${index + 1}`}/>
                    ))}
                </div>

                <div className = "habitat-map">
                    map info
                </div>
            </div>

        </section>
    )
}

export default ResultsSection