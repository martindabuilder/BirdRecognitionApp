import { useLocation } from "react-router-dom"
import { useEffect, useState } from "react"

import BirdPhoto from "../Shared/BirdPhoto.jsx"
import HabitatMap from "../Shared/HabitatMap.jsx"
import AudioPlayer from "./Audio_Player.jsx"
import EscapeButton from "../Shared/Escape_Button.jsx"

import "./results_section.css"

function ResultsSection(){
    const location = useLocation()
    const result = location.state
    const [entryAnimation, setEntryAnimation] = useState("initial")

    function handleClose() {setEntryAnimation("closing")}

     useEffect(() => {
        if (!result) return

        /* constants for the opening animation */
        const titleTimer = setTimeout(() => {setEntryAnimation("title-visible")}, 200)
        const popIntimer = setTimeout(() => {setEntryAnimation("bird-visible")}, 800)
        const moveTimer = setTimeout(() => {setEntryAnimation("bird-moved")}, 1300)
        const resultsTimer = setTimeout(() => setEntryAnimation("results-visible"), 2000)
        
        return () => {
            clearTimeout(titleTimer)
            clearTimeout(popIntimer)
            clearTimeout(moveTimer)
            clearTimeout(resultsTimer)
        }
    }, [result])

    /* if the result is missing the user gets warned and redirected back */
    if (!result) {
        return (
            <section className = "warning-section">
                <h1 className = "stroke-text">No uploaded results.</h1>
                <h1 className = "stroke-text">Go back to the main page, and try again.</h1>

                <EscapeButton />
            </section>
        )
    }

    /* spectrogram segments output function */
    function handleSpectrogramScroll(e) {
        e.stopPropagation()
        e.currentTarget.scrollLeft += e.deltaY
        e.currentTarget.scrollLeft += e.deltaX
    }

    /* prediction result related constants */
    const predictions = result.predictions || []
    const topSpecies = predictions[0]

    return (
        <section className={`results-section entrance-${entryAnimation}`}>
            <EscapeButton onClick = {handleClose}/>

            <h3 className = "results-title">Predicted as</h3>

            <div className = "bird-photo-container">
                {topSpecies && (
                    <BirdPhoto
                        commonName = {topSpecies.species}
                        scientificName = {topSpecies.scientificName}
                    />
                )}
            </div>

            <div className = "total-results-container stroke-text">
                {topSpecies && (
                    <div className = "main-confidence stroke-text">
                        <h3 className = "main-confidence-header"> {topSpecies.species} </h3>
                        <p className = "scientific-name"> <i> {topSpecies.scientificName} </i> </p>
                        <p>Confidence: {(topSpecies.probabilities * 100).toFixed(2)}%</p>
                    </div>
                )}

                <div className = "top4-predictions stroke-text">
                    <h4 className = "top4-header"> Closest predictions </h4>
                    {predictions.slice(1, 5).map((prediction) => (
                        <div className = "prediction" key = {prediction.species}>
                            <span> {prediction.species} </span>
                            <span> {(prediction.probabilities * 100).toFixed(2)}% </span>
                        </div>
                    ))}
                </div>
                
                <div className="custom-audio-player stroke-text">
                    <h3 className="custom-audio-player-header">
                        Uploaded audio
                    </h3>

                    {result.audio && (<AudioPlayer src={`data:${result.audioType};base64,${result.audio}`}/>)}
                </div>

                <div className = "spectrogram-segments-wrapper stroke-text">
                    <h4 className = "spectrogram-header"> {result.spectrograms.length} spectrogram segments </h4>
                    <div className = "spectrogram-segments" onWheel={handleSpectrogramScroll}>
                        {result.spectrograms.map((spectrogram, index) => (
                            <img key = {index} src = {`data:image/png;base64,${spectrogram}`} alt = {`Spectrogram segment ${index + 1}`}/>
                        ))}
                    </div>
                </div>

                <div className = "habitat-map">
                    <h3 className = "habitat-header stroke-text">
                        Expected habitat location
                    </h3>

                    {topSpecies && (<HabitatMap scientificName = {topSpecies.scientificName}/>)}
                </div>
            </div>

        </section>
    )
}

export default ResultsSection