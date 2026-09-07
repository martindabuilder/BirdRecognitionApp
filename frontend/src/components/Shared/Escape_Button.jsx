import { useNavigate } from "react-router-dom"
import { useState } from "react"

import "./escape_button.css"

function EscapeButton({ to = "/", className = "", onClick }) {
    const navigate = useNavigate()
    const [exiting, setExiting] = useState(false)

    function handleClick() {
        setExiting(true)
        onClick?.()
        setTimeout(() => {navigate(to)}, 1000)
    }

    return (
        <button
            className = {`escape-button ${className} ${exiting ? "escape-button-exit" : ""}`}
            onClick = {handleClick}
            aria-label = "Go back"
        >
            <span></span>
            <span></span>
        </button>
    )
}

export default EscapeButton