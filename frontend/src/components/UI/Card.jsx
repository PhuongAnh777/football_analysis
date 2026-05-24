import { forwardRef } from 'react'
import { motion } from 'framer-motion'

const Card = forwardRef(function Card({ children, className = '', glow, animate = false, delay = 0, ...props }, ref) {
  const glowClass = glow ? `glow-${glow}` : ''
  if (animate) {
    return (
      <motion.div ref={ref}
        initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay, ease: 'easeOut' }}
        className={`card ${glowClass} ${className}`} {...props}>
        {children}
      </motion.div>
    )
  }
  return <div ref={ref} className={`card ${glowClass} ${className}`} {...props}>{children}</div>
})

export default Card
