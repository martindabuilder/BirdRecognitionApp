import { useState } from "react"
import { motion } from "motion/react"

import RollingButtonText from "./Rolling_Button_Text.jsx"

import "./buttons.css"

function MicrophonePredictButton(){
    const [rollCount, setRollCount] = useState(0)

    return (
        <div className = "predict-buttons">
            <motion.button
                className = "predict-button"
                onHoverStart = {() => setRollCount((current) => current + 1)}>

                <RollingButtonText rollCount = {rollCount}>
                    Record audio.
                </RollingButtonText>

            </motion.button>
        </div>
    )
}

export default MicrophonePredictButton