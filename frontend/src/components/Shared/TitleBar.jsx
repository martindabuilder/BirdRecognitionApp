import { useState } from "react"
import { useNavigate } from "react-router-dom"

import "./title-bar.css"


function TitleBar(){
    const [menuOpen, setMenuOpen] = useState(false)
    const navigate = useNavigate()

    function goToBirdList(){
        setMenuOpen(false)
        navigate("/birdlist")
    }

    function goToSources(){
        setMenuOpen(false)
        navigate("/information")
    }

    return (
        <>
        <div className = "title-bar">
            <button className={`sidebar-button ${menuOpen ? "open" : ""}`} onClick = {() => setMenuOpen(!menuOpen)}>
                <span></span>
                <span></span>
                <span></span>
            </button>

            <h2 className="project-title">
                Birds Recognition Project
            </h2>
        </div>

        <div className = {`menu-blur ${menuOpen ? "open" : ""}`}></div>

        <div className={`side-menu-section ${menuOpen ? "open" : ""}`}>
            <button className="list-button" onClick = {goToBirdList}>
                List of birds.
            </button>

            <button className="info-button" onClick = {goToSources}>
                Sources.
            </button>
        </div>

        </>
    )
}

export default TitleBar