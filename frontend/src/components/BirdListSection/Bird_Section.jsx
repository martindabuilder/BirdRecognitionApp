import { useEffect, useState, useRef } from "react"

import BirdPhoto from "../Shared/BirdPhoto.jsx"
import EscapeButton from "../Shared/Escape_Button.jsx"
import CustomScrollBar from "../Shared/Scrollbar.jsx"
import getBirdPhoto from "../Shared/get_bird_photo.js"

import "./bird_section.css"


function BirdList(){
    const listRef = useRef(null)
    const loadingMoreRef = useRef(false)
    const batchRef = useRef(0)
    const loadedBatchesRef = useRef(new Set())

    const [birds, setBirds] = useState([])
    const [allBirds, setAllBirds] = useState([])

    const [loading, setLoading] = useState(true)
    const [loadingMore, setLoadingMore] = useState(false)
    const [error, setError] = useState(false)

    const [exiting, setExiting] = useState(false)

    function handleClose() { setExiting(true) }

    useEffect(() => {
        async function loadBirdList() {
            try {
                const response = await fetch("http://127.0.0.1:8000/birds")
                if (!response.ok) {throw new Error("Failed to fetch birds")}

                const data = await response.json()
                setAllBirds(data)
                setLoading(false)
            }

            catch (error) {
                console.error("Error loading birds: ", error)
                setError(true)
                setLoading(false)
            }
        }

        loadBirdList()
    }, [])

    async function loadBatch(batchNumber) {
        if ( loadingMoreRef.current || loadedBatchesRef.current.has(batchNumber)) { return }

        const start = batchNumber * 50
        const end = start + 50
        const batchBirds = allBirds.slice(start, end)

        if (batchBirds.length === 0) { return }

        loadedBatchesRef.current.add(batchNumber)

        loadingMoreRef.current = true
        setLoadingMore(true)

        let nextIndex = 0

        async function worker() {
            while (nextIndex < batchBirds.length) {
                const index = nextIndex
                nextIndex += 1

                const bird = batchBirds[index]
                const photo = await getBirdPhoto(bird.commonName, bird.scientificName)

                setBirds(previousBirds => [...previousBirds, {...bird, photo}])
            }
        }

        const workers = []

        for ( let i = 0;  i < Math.min(10, batchBirds.length); i += 1) { 
            workers.push(worker()) 
        }

        await Promise.all(workers)

        loadingMoreRef.current = false
        setLoadingMore(false)
    }

    useEffect(() => {
        if (allBirds.length === 0) { return }
        batchRef.current = 0
        loadBatch(0)
        }, [allBirds])

    useEffect(() => {
        const list = listRef.current

        if (!list) { return }

        function handleScroll() {
            const distanceFromBottom =
                list.scrollHeight -
                list.scrollTop -
                list.clientHeight

            if (distanceFromBottom < 500 && !loadingMoreRef.current && birds.length < allBirds.length) 
            {
                const nextBatch = batchRef.current + 1
                if (loadedBatchesRef.current.has(nextBatch)) { return }
                batchRef.current = nextBatch
                loadBatch(nextBatch)
            }
        }

        list.addEventListener("scroll", handleScroll)

        return () => {
            list.removeEventListener("scroll", handleScroll)
        }
    }, [birds.length, allBirds.length])

    return (
        <section className = {`bird-list-section ${exiting ? "bird-list-section-exit" : ""}`} ref = {listRef}>            <EscapeButton onClick = {handleClose} />
            <CustomScrollBar scrollRef = {listRef} />
            <h1 className = "bird-list-title"> Available Birds. </h1>

            {loading && ( <p> Loading birds... </p> )}
            {error && ( <p> Failed to load bird information. </p> )}

            {!error && birds.length > 0 && (
                <div className = "bird-list">
                    {birds.map((bird, index) => (
                        <BirdCard key = {bird.label} bird = {bird} index = {index}/>
                    ))}
                </div>
            )}

            {loadingMore && ( <p className = "bird-loading-more"> Loading more birds... </p> )}
        </section>
    )
}


function BirdCard({ bird, index }) {
    const cardRef = useRef(null)
    const [visible, setVisible] = useState(false)

    useEffect(() => {
        const card = cardRef.current

        if (!card) { return }

        const observer = new IntersectionObserver(
            entries => {
                if (entries[0].isIntersecting) {
                    setVisible(true)
                    observer.disconnect()
                }}, { rootMargin: "100px" }
        )

        observer.observe(card)

        return () => { observer.disconnect() }
    }, [])

    return (
        <div
            ref = {cardRef}
            className = {`bird-card ${visible ? "bird-card-visible" : ""}`}
            style = {{ animationDelay: `${(index % 50) * 0.05}s`
            }}
        >
            <div className = "bird-card-photo">
                <BirdPhoto
                    commonName = {bird.commonName}
                    photo = {bird.photo}
                />
            </div>

            <h2> {bird.commonName} </h2>
            <p> <i> {bird.scientificName} </i> </p>
        </div>
    )
}

export default BirdList