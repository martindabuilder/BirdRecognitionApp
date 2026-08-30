import { useState } from "react"
import { useNavigate, useLocation } from "react-router-dom"

import "./title-bar.css"


function TitleBar(){
    const [menuOpen, setMenuOpen] = useState(false)
    const navigate = useNavigate()
    const currentPage =
        location.pathname === "/results"
            ? "results"
            : location.pathname === "/birdlist"
            ? "birdlist"
            : location.pathname === "/information"
            ? "info"
            : "main"

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
        <div className = {`title-bar ${currentPage}-theme`}>

            <div className = "title-gradient main-gradient"></div>
            <div className = "title-gradient results-gradient"></div>
            <div className = "title-gradient birdlist-gradient"></div>
            <div className = "title-gradient info-gradient"></div>

            <button className = {`sidebar-button ${menuOpen ? "open" : ""}`} onClick = {() => setMenuOpen(!menuOpen)}>
                <span></span>
                <span></span>
                <span></span>
            </button>

            <h2 className = "project-title">
                Birds Recognition Project
            </h2>
        </div>

        <div className = {`menu-blur ${menuOpen ? "open" : ""}`}></div>

        <div className = {`side-menu-section ${menuOpen ? "open" : ""}`}>
            <button className = "list-button" onClick = {goToBirdList}>
                List of birds.
            </button>

            <button className = "info-button" onClick = {goToSources}>
                Sources.
            </button>
        </div>

        </>
    )
}

export default TitleBar