import { useNavigate } from "react-router-dom"
import { useEffect, useState, useRef } from "react"

import BirdPhoto from "../Shared/BirdPhoto.jsx"
import EscapeButton from "../Shared/Escape_Button.jsx"
import CustomScrollBar from "../Shared/Scrollbar.jsx"

import "./bird_section.css"

/* Function that fetches corresponding photos for all the bird classes availabe */
function BirdList(){
    const listRef = useRef(null)
    const [birds, setBirds] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(false)

    useEffect(() => {
        fetch("http://127.0.0.1:8000/birds")
            .then(response => {
                if (!response.ok) {throw new Error("Failed to fetch birds")}
                return response.json()
            })
            .then(data => {
                setBirds(data)
                setLoading(false)
            })
            .catch(error => {
                console.error("Error loading birds: ", error)
                setError(true)
                setLoading(false)
            })
    }, [])

    return (
        <section className = "bird-list-section" ref = {listRef}>
            <EscapeButton />

            <CustomScrollBar scrollRef={listRef} />

            <h1 className = "bird-list-title"> Available Birds. </h1>

            {loading && ( <p>Loading birds...</p> )}

            {error && (<p> Failed to load bird information. </p>)}

            {!loading && !error && (
                <div className = "bird-list">
                    {birds.map((bird) => (
                        <div className = "bird-card" key={bird.label}>
                            <div className = "bird-card-photo">
                                <BirdPhoto 
                                    commonName = {bird.commonName}
                                    scientificName = {bird.scientificName}
                                />
                            </div>

                            <h2> {bird.commonName} </h2> 
                            <p> <i>{bird.scientificName}</i> </p>

                        </div>
                    ))}
                </div>
            )}
        </section>
    )
}

export default BirdList