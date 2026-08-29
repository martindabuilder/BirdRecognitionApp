import { useNavigate } from "react-router-dom"

import "./info_section.css"


function InfoSection(){
    const navigate = useNavigate()
    return (
        <section className="info-section">
            <button className = "back-button" onClick = {() => navigate("/")}>
                go back home
            </button>
            ifnromation 
        </section>
    )
}

export default InfoSection