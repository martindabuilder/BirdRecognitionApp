import { MapContainer, TileLayer } from "react-leaflet"
import { useEffect, useState } from "react"

import "./habitat_map.css"
import "leaflet/dist/leaflet.css"

function HabitatMap({scientificName}) {
    const [loading, SetLoading] = useState(true)
    const [taxonID, setTaxonID] = useState(null)
    const [error, setError] = useState(false)

    useEffect(() => {
        async function getHabitat() {
            SetLoading(true)
            setTaxonID(null)
            setError(false)
            
            try {
                const params = new URLSearchParams({ q: scientificName, rank: "species" })
                const response = await fetch(`https://api.inaturalist.org/v1/taxa?${params}`)

                if (!response.ok) { throw new Error("Failed to find bird species.") }
                
                const data = await response.json()
                const taxon = data.results?.[0]

                if (!taxon) { throw new Error("Bird species not found.")}

                setTaxonID(taxon.id)
            }
            catch(error) {setError(true) }
            finally { SetLoading(false) }
        }

        if (scientificName) { getHabitat() }
    }, [scientificName])

    if (loading) {}

    if (error || !taxonID) {
        return (
            <div className = "habitat-fallback">
                Placeholder text, will be image
            </div>
        )
    }

    return (
        <div className="habitat-map-container">
            <MapContainer
            center={[30, 10]}
            zoom={2}
            minZoom={2}
            maxZoom={8}
            scrollWheelZoom={true}
            className="leaflet-habitat-map"
            >

            <TileLayer
                attribution='&copy; OpenStreetMap contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            <TileLayer
                attribution='&copy; iNaturalist'
                url={`https://api.inaturalist.org/v1/taxon_ranges/${taxonID}/{z}/{x}/{y}.png`}
                opacity={0.6}
            />

            </MapContainer>
        </div>
    )
}

export default HabitatMap