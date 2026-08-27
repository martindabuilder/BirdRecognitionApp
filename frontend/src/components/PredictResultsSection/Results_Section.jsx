import { useLocation, useNavigate } from "react-router-dom"

import "./results_section.css"

function ResultsSection(){
    const navigate = useNavigate()
    const location = useLocation()
    const result = location.state

    function goHome(){navigate("/")}

    if(!result){}

    const predictions = result.predictions || []

    return (
        <section className="results-section">

            <button className="escape-button" onClick={goHome}>
                go back lol
            </button>

            <div className="bird-photo-container">
                birb
            </div>

            <div className="total-results-container">
                <div className="main-confidence">
                    main result
                </div>

                <div className="top4-predictions">
                    top 4
                </div>

                <div className="spectrogram-segments">
                    spectrograms
                </div>

                <div className="habitat-map">
                    map info
                </div>
            </div>
            
        </section>
    )
} 

export default ResultsSection