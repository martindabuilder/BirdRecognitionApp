async function getBirdPhoto(commonName, scientificName) {
    try {
        const searchBird = scientificName || commonName

        const params = new URLSearchParams({
            action: "query",
            generator: "search",
            gsrsearch: searchBird,
            gsrnamespace: "6",
            gsrlimit: "30",
            prop: "imageinfo",
            iiprop: "url|mime|size",
            iiurlwidth: "800",
            format: "json",
            origin: "*"
        })

        const response = await fetch(`https://commons.wikimedia.org/w/api.php?${params}`)
        if (!response.ok) {throw new Error("Failed to find bird photo.")}

        const data = await response.json()
        const pages = Object.values(data.query?.pages || {})

        const image = pages.find(page => {
            const info = page.imageinfo?.[0]

            if (!info?.thumburl || !info.mime?.startsWith("image/")) {
                return false
            }

            const aspectRatio = info.width / info.height
            return aspectRatio >= 0.55 && aspectRatio <= 2
        })

        if (!image) {return null}

        const photoUrl = image.imageinfo[0].thumburl
        const loadedImage = new Image()
        loadedImage.src = photoUrl

        await loadedImage.decode()

        return photoUrl
    }

    catch (error) {
        console.error("Couldn't load bird photo:", error)
        return null
    }
}

export default getBirdPhoto