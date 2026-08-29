import { useNavigate } from "react-router-dom"

import BirdPhoto from "../Shared/BirdPhoto.jsx"

import "./bird_section.css"

function BirdList(){
    const navigate = useNavigate()
    return (
        <section className="list-section">
            <button className = "back-button" onClick = {() => navigate("/")}>
                go back home
            </button>
            bird list yippee
        </section>
    )
}

export default BirdList