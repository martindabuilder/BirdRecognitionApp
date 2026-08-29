import { useState } from "react" 

import "./title-bar.css"


function TitleBar(){
    const [menuOpen, setMenuOpen] = useState(false)

    return (
        <div className = "title-bar">
            <button className = "sidebar-button" onClick = {() => setMenuOpen(!menuOpen)}>
                <span></span>
                <span></span>
                <span></span>
            </button>

            <h2 className = "project-title">Birds Recognition Project</h2>

            {menuOpen && (
                <div className = {`side-menu-section ${menuOpen ? "open" : ""}`}>
                    <button className = "info-button">
                        button1
                    </button>

                    <button className = "list-button">
                        button2
                    </button>
                </div>
            )}
        </div>
    )
}

export default TitleBar