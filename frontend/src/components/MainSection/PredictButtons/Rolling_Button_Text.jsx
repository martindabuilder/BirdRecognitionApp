import { motion } from "motion/react"

function RollingButtonText({ children, rollCount }){
    return (
        <span className = "rolling-text" aria-hidden = "true">
            <span className = "rolling-text-line">
                {children.split("").map((character, index) => (
                    <motion.span
                        className = "rolling-text-character"
                        key = {`${character}-${index}`}
                        animate = {{ y: `-${rollCount * 1.2}em` }}
                        transition = {{
                            duration: 0.45,
                            delay: index * 0.02,
                            ease: [0.22, 1, 0.36, 1],
                        }}
                    >
                        {Array.from({ length: 20 }, (_, repetition) => (
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