import { useRef, useState, useEffect } from "react"

import "./audio_player.css"


function AudioPlayer({ src }) {
    
    const audioRef = useRef(null)

    /* audio player related constants */
    const [isPlaying, setIsPlaying] = useState(false)
    const [currentTime, setCurrentTime] = useState(0)
    const [audioDuration, setAudioDuration] = useState(0)
    const [volume, setVolume] = useState(1)
    const [audioSrc, setAudioSrc] = useState("")

    useEffect(() => {
        if (!src) return

        const [metadata, base64Data] = src.split(",")
        const mimeType = metadata.match(/data:(.*?);base64/)?.[1] || "audio/mpeg"
        const binaryData = atob(base64Data)
        const bytes = new Uint8Array(binaryData.length)

        for (let i = 0; i < binaryData.length; i++) {
            bytes[i] = binaryData.charCodeAt(i)
        }

        const audioBlob = new Blob([bytes], { type: mimeType })
        const blobUrl = URL.createObjectURL(audioBlob)

        setAudioSrc(blobUrl)

        return () => {
            URL.revokeObjectURL(blobUrl)
        
        }
    }, [src])


    useEffect(() => {
        setCurrentTime(0)
        setAudioDuration(0)
        setIsPlaying(false)
    }, [src])


    function handleMetadata() {
        const duration = audioRef.current.duration

        if (Number.isFinite(duration) && duration > 0) {
            setAudioDuration(duration)
        }
    }

    function startPauseAudio() {
        if (audioRef.current.paused) {
            audioRef.current.play()
            setIsPlaying(true)
        }
        else {
            audioRef.current.pause()
            setIsPlaying(false)
        }
    }

    function timeUpdate() {
        setCurrentTime(audioRef.current.currentTime)
    }

    function audioEnd() {
        setIsPlaying(false)
        setCurrentTime(0)
    }

    function handleProgressChange(e) {
        const newTime = Number(e.target.value)
        audioRef.current.currentTime = newTime
        setCurrentTime(newTime)
    }

    function formatTime(time) {
        const minutes = Math.floor(time / 60)
        const seconds = Math.floor(time % 60)
        return `${minutes}:${seconds.toString().padStart(2, "0")}`
    }

    function handleVolumeChange(e) {
        const newVolume = Number(e.target.value)
        audioRef.current.volume = newVolume
        setVolume(newVolume)
    }

    return(
        <div className = "audio-player-container">
            <audio 
                ref = {audioRef}
                src = {audioSrc}
                onTimeUpdate = {timeUpdate}
                onLoadedMetadata = {handleMetadata}
                onDurationChange={handleMetadata}
                onEnded = {audioEnd}
            />

            <button 
                className = {`audio-play-button ${isPlaying ? "playing" : ""}`}
                onClick = {startPauseAudio}
                aria-label={isPlaying ? "Pause audio" : "Play audio"}>
                
                <span className = "button-icons">
                    {isPlaying ? "❚❚" : "▶"}
                </span>
            </button>

            <span className = "audio-time">
                {formatTime(currentTime)}
            </span>

            <div className = "audio-progress-wrapper">
                <input
                    className="audio-progress"
                    type="range"
                    min="0"
                    max={audioDuration || 0}
                    value={currentTime}
                    onChange={handleProgressChange}
                    style={{
                        "--progress": audioDuration > 0
                        ? `${(currentTime / audioDuration) * 100}%`
                        : "0%"
                    }}                
                />
            </div>

            <span className = "audio-time">
                {formatTime(audioDuration)}
            </span>

            <span className = "volume-icon">
                volume
            </span>

            <div className = "volume-wrapper">
                <input
                    className = "volume-slider"
                    type = "range"
                    min = "0"
                    max = "1"
                    step = "0.01"
                    value = {volume}
                    onChange = {handleVolumeChange}
                    style = {{
                        "--volume": `${volume * 100}%`
                    }}
                    aria-label = "Volume"
                />
            </div>
        </div>
    )

}

export default AudioPlayer