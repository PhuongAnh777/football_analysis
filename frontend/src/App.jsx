import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AnalysisProvider } from './context/AnalysisContext'
import MainLayout from './components/Layout/MainLayout'
import UploadPage from './pages/UploadPage'
import DashboardPage from './pages/DashboardPage'
import ReportPage from './pages/ReportPage'
import VideoPlayerPage from './pages/VideoPlayerPage'

export default function App() {
  return (
    <AnalysisProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<MainLayout />}>
            <Route index element={<Navigate to="/upload" replace />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/report" element={<ReportPage />} />
            <Route path="/video" element={<VideoPlayerPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AnalysisProvider>
  )
}
