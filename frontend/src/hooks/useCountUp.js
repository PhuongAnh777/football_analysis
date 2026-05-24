import { useState, useEffect, useRef } from 'react'

export function useCountUp(target, duration = 1200, inView = true, decimals = 0) {
  const [value, setValue] = useState(0)
  const frameRef = useRef(null)
  const startRef = useRef(null)

  useEffect(() => {
    if (!inView) return
    const end = Number(target) || 0
    if (frameRef.current) cancelAnimationFrame(frameRef.current)

    function animate(timestamp) {
      if (!startRef.current) startRef.current = timestamp
      const progress = Math.min((timestamp - startRef.current) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setValue(end * eased)
      if (progress < 1) frameRef.current = requestAnimationFrame(animate)
      else setValue(end)
    }

    startRef.current = null
    frameRef.current = requestAnimationFrame(animate)
    return () => { if (frameRef.current) cancelAnimationFrame(frameRef.current) }
  }, [target, duration, inView])

  return value.toFixed(decimals)
}
