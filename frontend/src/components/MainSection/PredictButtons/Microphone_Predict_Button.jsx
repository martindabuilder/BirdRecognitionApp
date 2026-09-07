import { useState } from "react"
import { motion } from "motion/react"

import RollingButtonText from "./Rolling_Button_Text.jsx"

import "./buttons.css"

function MicrophonePredictButton({ predicting }) {
    const [rollCount, setRollCount] = useState(0)

    return (
        <div className = "predict-buttons">
            <motion.button
                className = "predict-button"
                onHoverStart = {() => setRollCount((current) => current + 1)}
                initial = {{ opacity: 0, scale: 0.8 }}
                animate = {{ opacity: predicting ? 0 : 1, scale: predicting ? 0.9 : 1 }}
                whileHover = {{ scale: 1.03, transition: { duration: 0.2, ease: "ease"}}} 

                transition = {{ opacity: {
                    duration: 0.3,
                    ease: "ease",
                    delay: predicting ? 0 : 0.2
                    },
                    scale: { 
                        type: "spring", 
                        stiffness: 300, 
                        damping: 12, delay: predicting ? 0 : 0.2
                    }
                }}>
                    <RollingButtonText rollCount = {rollCount}>
                        Record audio.
                    </RollingButtonText>
            </motion.button>
        </div>
    )
}

export default MicrophonePredictButton