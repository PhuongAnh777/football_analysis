import { useEffect, useRef, useCallback } from 'react'

export function usePolling(fn, interval = 2000, active = true) {
  const fnRef = useRef(fn)
  fnRef.current = fn
  const timerRef = useRef(null)
  const runningRef = useRef(false)

  const stop = useCallback(() => {
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null }
  }, [])

  useEffect(() => {
    if (!active) { stop(); return }
    let cancelled = false

    async function tick() {
      if (cancelled || runningRef.current) return
      runningRef.current = true
      try { await fnRef.current() } finally { runningRef.current = false }
      if (!cancelled) timerRef.current = setTimeout(tick, interval)
    }

    tick()
    return () => { cancelled = true; stop() }
  }, [active, interval, stop])
}
