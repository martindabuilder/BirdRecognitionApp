import { useEffect, useState } from "react";

import fallBackImage from "../../assets/fallbackimage.jpg"
import "./bird_photo.css";

function BirdPhoto({ commonName, scientificName }) {
    const [photo, setPhoto] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        async function getBirdPhoto() {
            setPhoto(null)
            setLoading(true)

            try {
                const searchBird = scientificName || commonName
                const params = new URLSearchParams({
                    action: "query",
                    generator: "search",
                    gsrsearch: searchBird,
                    gsrnamespace: "6",
                    gsrlimit: "10",
                    prop: "imageinfo",
                    iiprop: "url|mime|size",
                    iiurlwidth: "800",
                    format: "json",
                    origin: "*"
                })

                const response = await fetch(`https://commons.wikimedia.org/w/api.php?${params}`)
                if (!response.ok) {throw new Error("Failed to search Wikimedia Commons")}

                const data = await response.json()
                const pages = Object.values(data.query?.pages || {})
                const image = pages.find(page => {
                    const info = page.imageinfo?.[0]
                    if (!info?.thumburl || !info.mime?.startsWith("image/")) { return false }
                    const aspectRatio = info.width / info.height
                    return aspectRatio >= 0.55 && aspectRatio <= 1.55
                })


                if (image) {setPhoto((image.imageinfo)[0].thumburl)}
            } 
            catch (error) { } 
            finally { setLoading(false) }
        }
        if (commonName || scientificName) { getBirdPhoto() } 
    }, [commonName, scientificName])

    if (loading) {return ( <div className = "loading"> Loading photo... </div>)}
    if (!photo) { return ( <img className="bird-photo" src={fallBackImage} alt="No bird photo available"/>)}

    /* returns the corresponding image */
    return (
        <img className = "bird-photo" 
        src = {photo} 
        alt = {commonName} />
    )
}

export default BirdPhoto;