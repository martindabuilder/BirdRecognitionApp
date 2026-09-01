import { useNavigate } from "react-router-dom"

import "./info_section.css"
import EscapeButton from "../Shared/Escape_Button"


function InfoSection(){
    const navigate = useNavigate()
    return (
        <section className="info-section">

            <EscapeButton />
            
            <h1>Sources and things ive used</h1> 
        </section>
    )
}

export default InfoSection