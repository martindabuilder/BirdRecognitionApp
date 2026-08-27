import { useLocation, useNavigate } from "react-router-dom"

import "./results_section.css"

function ResultsSection(){
    const navigate = useNavigate()
    const location = useLocation()
    const result = location.state

    function goHome(){navigate("/")}

    if(!result){}

    return (
        <section className="results-section">
            <button className = "escape-button" onClick = {goHome}>
                go back lol
            </button>

        </section>
    )
} 

export default ResultsSection