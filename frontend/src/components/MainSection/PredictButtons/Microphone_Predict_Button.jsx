import { useState, useRef, useEffect } from "react"
import { motion } from "motion/react"
import { useNavigate } from "react-router-dom"

import getBirdPhoto from "../../Shared/get_bird_photo.js"

import RollingButtonText from "../../Shared/Rolling_Button_Text.jsx"

import "./buttons.css"

function MicrophonePredictButton({ processingStage, setProcessingStage }){

    const [rollCount, setRollCount] = useState(0)
    const [rolling, setRolling] = useState(false)

    const navigate = useNavigate()

    const chunksRef = useRef([])
    const streamRef = useRef(null)  
    const sourceRef = useRef(null)
    const processorRef = useRef(null)
    const recordingRef = useRef(false)
    const pauseRecordingRef = useRef(false)

    const audioContextRef = useRef(null)
    const gainRef = useRef(null)

    const [recording, setRecording] = useState(false)
    const [recordingTime, setRecordingTime] = useState(0)
    const [pauseRecording, setPauseRecording] = useState(false)
    
    const predicting = processingStage !== null

    useEffect(() => {
        if (!recording || pauseRecording) { return }

        const timer = setInterval(() => {
            setRecordingTime((current) => current + 1)
        }, 1000)

        return () => clearInterval(timer)
    }, [recording, pauseRecording])


    function mergeChunks(chunks) {
        const totalLength = chunks.reduce((total, chunk) => total + chunk.length, 0)
        const result = new Float32Array(totalLength)

        let offset = 0

        for (const chunk of chunks) {
            result.set(chunk, offset)
            offset += chunk.length
        }

        return result
    }


    function resampleAudio(audioData, inputSampleRate, outputSampleRate) {
        if (inputSampleRate === outputSampleRate)
            return audioData

        const ratio = inputSampleRate / outputSampleRate
        const outputLength = Math.round(audioData.length / ratio)
        const output = new Float32Array(outputLength)
        
        for (let i = 0; i < outputLength; i++) {
            const position = i * ratio
            const index = Math.floor(position)
            const nextIndex = Math.min(index + 1, audioData.length - 1)
            const fraction = position - index

            output[i] = audioData[index] * (1 - fraction) + audioData[nextIndex] * fraction
        }

        return output
    }


    function encodeToWav(audioData, sampleRate) {
        const buffer = new ArrayBuffer(44 + audioData.length * 2)
        const view = new DataView(buffer)

        function writeString(offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i))
            }
        }

        writeString(0, "RIFF")
        view.setUint32(4, 36 + audioData.length * 2, true)

        writeString(8, "WAVE")
        writeString(12, "fmt ")

        view.setUint32(16, 16, true)
        view.setUint16(20, 1, true)
        view.setUint16(22, 1, true)
        view.setUint32(24, sampleRate, true)
        view.setUint32(28, sampleRate * 2, true)
        view.setUint16(32, 2, true)
        view.setUint16(34, 16, true)

        writeString(36, "data")
        view.setUint32(40, audioData.length * 2, true)

        let offset = 44

        for (let i = 0; i < audioData.length; i++){
            const sample = Math.max(-1, Math.min(1, audioData[i]))
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
            offset += 2
        }

        return new Blob([view], { type: "audio/wav" })
    }


    function restartRecording() {
        if (processorRef.current){
            processorRef.current.onaudioprocess = null
            processorRef.current.disconnect()
            processorRef.current = null
        }

        if (sourceRef.current){
            sourceRef.current.disconnect()
            sourceRef.current = null
        }

        if (gainRef.current){
            gainRef.current.disconnect()
            gainRef.current = null
        }

        if (streamRef.current){
            streamRef.current.getTracks().forEach((track) => track.stop())
            streamRef.current = null
        }

        if (audioContextRef.current){
            audioContextRef.current.close()
            audioContextRef.current = null
        }
    }


    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({audio: true})
            const audioContext = new AudioContext()
            const source = audioContext.createMediaStreamSource(stream)
            const processor = audioContext.createScriptProcessor(4096, 1, 1)

             const gain = audioContext.createGain()

            gain.gain.value = 0

            chunksRef.current = []

            streamRef.current = stream
            audioContextRef.current = audioContext
            sourceRef.current = source
            processorRef.current = processor
            gainRef.current = gain

            recordingRef.current = true
            pauseRecordingRef.current = false

            source.connect(processor)
            processor.connect(gain)
            gain.connect(audioContext.destination)

            processor.onaudioprocess = (event) => {
                if (!recordingRef.current || pauseRecordingRef.current){ return }

                const inputData =event.inputBuffer.getChannelData(0)
                chunksRef.current.push(new Float32Array(inputData))
            }

            setRecordingTime(0)
            setPauseRecording(false)
            setRecording(true)

        }
        catch(error){
            console.error("Couldn't access microphone: ", error)
        }
    }

    async function stopRecording(){
        if (!audioContextRef.current) { return }

        const inputSampleRate = audioContextRef.current.sampleRate

        recordingRef.current = false
        pauseRecordingRef.current = false

        const samples = mergeChunks(chunksRef.current)

        restartRecording()

        setRecording(false)
        setPauseRecording(false)
        setRecordingTime(0)

        if (samples.length === 0) { 
            console.error("No audio was recorded.") 
            return
        }

        const resampledAudio = resampleAudio(samples, inputSampleRate, 32000)
        const finishedRecording = encodeToWav(resampledAudio, 32000)

        await sendRecording(finishedRecording)
    }

    function pauseRecordingAudio() {
            pauseRecordingRef.current = true
            setPauseRecording(true)
    }


    function resumeRecording() {
        pauseRecordingRef.current = false
        setPauseRecording(false)
    }


    function cancelRecording() {
        recordingRef.current = false
        pauseRecordingRef.current = false

        restartRecording()
        chunksRef.current = []

        setRecording(false)
        setPauseRecording(false)
        setRecordingTime(0)
    }

    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60)
        const remainingSeconds = seconds % 60

        return `${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`
    }


    async function sendRecording(finishedRecording) {
        setProcessingStage("processing")

        const audioFile = new File([finishedRecording], "microphone_recording.wav", { type: "audio/wav"})
        const formData = new FormData()

        formData.append("file", audioFile)

        try {
            const response = await fetch("http://127.0.0.1:8000/predict", {method: "POST", body: formData})

            if (!response.ok) { throw new Error(`Prediction failed: ${response.status}`) }

            const data = await response.json()
            const topSpecies = data.predictions?.[0]

            setProcessingStage("fetching-photo")

            let birdPhoto = null

            if (topSpecies){
                birdPhoto = await getBirdPhoto(topSpecies.species, topSpecies.scientificName)
            }

            setProcessingStage("readying-results")
            navigate("/results", { state: {...data, birdPhoto} })
        }
        catch(error){
            console.error("Couldn't connect to backend:", error)
            setProcessingStage(null)
        }
    }

    return (
        <div className = "predict-buttons">
            <motion.button
                className = "predict-button"
                onClick = {startRecording}
                onHoverStart = { () => {
                    if (rolling) return

                    setRolling(true)
                    setRollCount((current) => current + 1)
                }}
                initial = {{ opacity: 0, scale: 0.8 }}
                animate = {{ opacity: predicting ? 0 : 1, scale: predicting ? 0.9 : 1 }}
                whileHover = {{ scale: 1.03, transition: {duration: 0.2, ease: "ease"} }}

                transition = {{
                    opacity: { duration: 0.3, ease: "ease", delay: predicting ? 0 : 0.2},
                    scale: { type: "spring", stiffness: 300, damping: 12, delay: predicting ? 0 : 0.2}
                }}
            >

                <RollingButtonText rollCount = {rollCount} onAnimationComplete = {() => setRolling(false)}>
                    Record audio.
                </RollingButtonText>

            </motion.button>

             {recording && (
            <div className = "recording-overlay">

                <div className = "recording-popup">

                    <div className = "recording-title">
                        Recording Birdsong
                    </div>

                    <div className = "recording-timer">
                        {formatTime(recordingTime)}
                    </div>

                    <div className = "recording-status">
                        {pauseRecording ? "recording paused" : "recording..."}
                    </div>

                    <div className = "recording-buttons">

                        {!pauseRecording ? (
                            <button
                                className = "record-button pause-button"
                                onClick = {pauseRecordingAudio}
                            >
                                Pause
                            </button>
                        ) : (
                            <button
                                className = "record-button resume-button"
                                onClick = {resumeRecording}
                            >
                                Resume
                            </button>
                        )}

                        <button
                            className = "record-button stop-button"
                            onClick = {stopRecording}
                        >
                            Stop
                        </button>

                        <button
                            className = "record-button cancel-button"
                            onClick = {cancelRecording}
                        >
                            Cancel
                        </button>

                    </div>

                </div>

            </div>
        )}
        </div>
    )
}

export default MicrophonePredictButton