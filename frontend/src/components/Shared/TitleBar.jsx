import { useState } from "react"
import { useNavigate, useLocation } from "react-router-dom"

import RollingButtonText from "./Rolling_Button_Text.jsx"

import "./title-bar.css"


function TitleBar(){
    const [menuOpen, setMenuOpen] = useState(false)
    const navigate = useNavigate()
    const location = useLocation()

    /*rolling text constants */
    const [sourceButtonRoll, setSourceButtonRoll] = useState(0)
    const [listButtonRoll, setListButtonRoll] = useState(0)


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

            <button className = {`sidebar-button ${menuOpen ? "open" : ""}`} onClick = {() => setMenuOpen(!menuOpen)}>
                <span></span>
                <span></span>
                <span></span>
            </button>

            <h2 className = "project-title"> Birds Recognition Project </h2>
        </div>

        <div className = {`menu-blur ${menuOpen ? "open" : ""}`}></div>

        <div className = {`side-menu-section ${currentPage}-theme ${menuOpen ? "open" : ""}`}>            
            <button 
                className = "list-button" onClick = {goToBirdList}
                onMouseEnter = {() => setListButtonRoll((current) => current + 1)}
            >

                <RollingButtonText rollCount = {listButtonRoll}>
                    List of birds.
                </RollingButtonText>

            </button>

        
            <button
                className = "sources-button" onClick = {goToSources}
                onMouseEnter = {() => setSourceButtonRoll((current) => current + 1)}
            >

                <RollingButtonText rollCount = {sourceButtonRoll}>
                    Sources.
                </RollingButtonText>

            </button>
        </div>

        </>
    )
}

export default TitleBar