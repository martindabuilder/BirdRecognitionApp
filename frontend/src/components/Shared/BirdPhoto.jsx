import { useEffect, useState } from "react";

import "./bird_photo.css";

function BirdPhoto({ commonName, scientificName }) {
    const [photo, setPhoto] = useState(null)
    const [loading, setLoading] = useState(null)

    useEffect(() => {
        async function getBirdPhoto() {
            setPhoto(null)
            setLoading(null)

            try {
                const searchBird = scientificName || commonName

                const params = new URLSearchParams({})
            }
            catch (error) {
                
            }
        }
        
    })

}

export default BirdPhoto;