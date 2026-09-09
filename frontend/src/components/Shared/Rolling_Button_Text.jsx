import { motion } from "motion/react"

import "./rolling-button.css"

function RollingButtonText({ children, rollCount }){
    return (
        <span className = "rolling-text" aria-hidden = "true">
            <span className = "rolling-text-line">
                {children.split("").map((character, index) => (
                    <motion.span
                        className = "rolling-text-character"
                        key = {`${character}-${index}`}
                        animate = {{ y: `-${rollCount * 1.2}em` }}
                        transition = {{duration: 0.35, delay: index * 0.02, ease: "ease",}}
                    >
                        {Array.from({ length: 100 }, (_, repetition) => (
                            <span key = {repetition}>
                                {character === " " ? "\u00a0" : character}
                            </span>
                        ))}
                    </motion.span>
                ))}
            </span>
        </span>
    )
}

export default RollingButtonText