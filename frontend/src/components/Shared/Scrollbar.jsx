import { useState, useEffect } from "react"

import "./custom-scroll-bar.css"

function CustomScrollBar({ scrollRef, className = "" }) {

    const [scrollPercent, setScrollPercent] = useState(0)
    const [thumbHeight, setThumbHeight] = useState(100)

    function updateScrollBar() {

        const element = scrollRef.current

        if (!element) return

        const {scrollTop, scrollHeight, clientHeight} = element

        const maxScroll = scrollHeight - clientHeight

        if (maxScroll <= 0) {
            setScrollPercent(0)
            setThumbHeight(100)
            return
        }

        setScrollPercent(scrollTop / maxScroll)
        setThumbHeight((clientHeight / scrollHeight) * 100)
    }

    function handleTrackClick(e) {
        const element = scrollRef.current

        if (!element) return

        const track = e.currentTarget.getBoundingClientRect()
        const clickPosition = e.clientY - track.top
        const percentage = clickPosition / track.height

        element.scrollTop = percentage * (element.scrollHeight - element.clientHeight)
    }

    useEffect(() => {

        const element = scrollRef.current

        if (!element) return

        element.addEventListener("scroll", updateScrollBar)
        window.addEventListener("resize", updateScrollBar)

        updateScrollBar()

        return () => {
            element.removeEventListener("scroll", updateScrollBar)
            window.removeEventListener("resize", updateScrollBar)
        }

    }, [scrollRef])

    return (
        <div className = {`custom-scroll-bar-track ${className}`} onClick={handleTrackClick}>
            <div
                className = "custom-scroll-bar-thumb"
                style = {{height: `${thumbHeight}%`, top: `${scrollPercent * (100 - thumbHeight)}%`}}
            />
        </div>
    )
}

export default CustomScrollBar