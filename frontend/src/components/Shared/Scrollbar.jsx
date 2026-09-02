import { useState, useEffect, useRef } from "react"

import "./custom-scroll-bar.css"

function CustomScrollBar({ scrollRef, className = "" }) {

    const [scrollPercent, setScrollPercent] = useState(0)
    const [thumbHeight, setThumbHeight] = useState(100)

    const trackRef = useRef(null)
    const isDragging = useRef(false)

    const [visible, setVisible] = useState(true)
    const hideTimeout = useRef(null)

    function updateScrollBar() {
        const element = scrollRef.current
        if (!element) return

        setVisible(true)
        clearTimeout(hideTimeout.current)
        hideTimeout.current = setTimeout(() => {setVisible(false)}, 1500)


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


    /*handles holding down and dragging the scroll thumb */
    function handleThumbMouseDown(e) {
        e.preventDefault()
        isDragging.current = true

        function handleMouseMove(e) {
            const element = scrollRef.current
            const track = trackRef.current

            if (!element || !track) return

            const trackRect = track.getBoundingClientRect()
            const percentage = (e.clientY - trackRect.top) / trackRect.height
            const clampedPercentage = Math.max(0, Math.min(1, percentage))

            element.scrollTop = clampedPercentage * (element.scrollHeight - element.clientHeight)
        }

        function handleMouseUp() {
            isDragging.current = false
            document.removeEventListener("mousemove", handleMouseMove)
            document.removeEventListener("mouseup", handleMouseUp)
        }

        document.addEventListener("mousemove", handleMouseMove)
        document.addEventListener("mouseup", handleMouseUp)
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
            clearTimeout(hideTimeout.current)
        }

    }, [scrollRef])

    return (
        <div
            ref = {trackRef}
            className={`custom-scroll-bar-track ${className} ${visible ? "visible" : "hidden"}`}
            onClick={handleTrackClick}
        >
            <div
                className = "custom-scroll-bar-thumb"
                style = {{height: `${thumbHeight}%`, top: `${scrollPercent * (100 - thumbHeight)}%`}}
                onMouseDown={handleThumbMouseDown}
            />
        </div>
    )
}

export default CustomScrollBar