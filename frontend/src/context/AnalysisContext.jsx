import { createContext, useContext, useReducer, useEffect, useCallback } from 'react'

const AnalysisContext = createContext(null)
const STORAGE_KEY = 'football_analysis_state'

const initialState = {
  jobId: null,
  status: 'idle',
  uploadProgress: 0,
  processingProgress: 0,
  currentStep: null,
  message: '',
  results: null,
  error: null,
}

function reducer(state, action) {
  switch (action.type) {
    case 'SET_JOB':        return { ...state, jobId: action.jobId, status: 'processing', error: null }
    case 'SET_UPLOADING':  return { ...state, status: 'uploading', uploadProgress: action.progress }
    case 'SET_PROCESSING': return { ...state, status: 'processing',
      processingProgress: action.progress ?? state.processingProgress,
      currentStep: action.currentStep ?? state.currentStep,
      message: action.message ?? state.message }
    case 'SET_DONE':       return { ...state, status: 'done', processingProgress: 100, results: action.results }
    case 'SET_ERROR':      return { ...state, status: 'error', error: action.error }
    case 'SET_RESULTS':    return { ...state, results: action.results }
    case 'RESET':          return { ...initialState }
    default:               return state
  }
}

export function AnalysisProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState, (init) => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const p = JSON.parse(saved)
        return { ...init, jobId: p.jobId || null, status: p.status === 'done' ? 'done' : init.status }
      }
    } catch { /* ignore */ }
    return init
  })

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ jobId: state.jobId, status: state.status }))
    } catch { /* ignore */ }
  }, [state.jobId, state.status])

  const setUploading  = useCallback((progress) => dispatch({ type: 'SET_UPLOADING', progress }), [])
  const setJob        = useCallback((jobId) => dispatch({ type: 'SET_JOB', jobId }), [])
  const setProcessing = useCallback((data) => dispatch({ type: 'SET_PROCESSING', ...data }), [])
  const setDone       = useCallback((results) => dispatch({ type: 'SET_DONE', results }), [])
  const setError      = useCallback((error) => dispatch({ type: 'SET_ERROR', error }), [])
  const setResults    = useCallback((results) => dispatch({ type: 'SET_RESULTS', results }), [])
  const reset         = useCallback(() => { localStorage.removeItem(STORAGE_KEY); dispatch({ type: 'RESET' }) }, [])

  return (
    <AnalysisContext.Provider value={{ ...state, setUploading, setJob, setProcessing, setDone, setError, setResults, reset }}>
      {children}
    </AnalysisContext.Provider>
  )
}

export function useAnalysis() {
  const ctx = useContext(AnalysisContext)
  if (!ctx) throw new Error('useAnalysis must be inside AnalysisProvider')
  return ctx
}
