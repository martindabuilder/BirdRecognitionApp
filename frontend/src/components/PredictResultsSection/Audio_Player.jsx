import { useRef, useState } from "react"

import "./audio_player.css"


function AudioPlayer({ src }) {
    
    const audioRef = useRef(null)

    /* audio player related constants */
    const [isPlaying, setIsPlaying] = useState(false)
    const [currentTime, setCurrentTime] = useState(0)
    const [audioDuration, setAudioDuration] = useState(0)
    const [volume, setVolume] = useState(1)


    function handleMetadata(){
        setAudioDuration(audioRef.current.duration)
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
        setIsPlaying(0)
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
                src = {src}
                onTimeUpdate = {timeUpdate}
                onLoadedMetadata = {handleMetadata}
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