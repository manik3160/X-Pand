import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { SpeedInsights } from '@vercel/speed-insights/react'
import { AppProvider } from '@/hooks/useApp'
import HomePage from '@/pages/HomePage'
import DashboardPage from '@/pages/DashboardPage'

export default function App() {
  return (
    <BrowserRouter>
      <AppProvider>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
        </Routes>
      </AppProvider>
      <SpeedInsights />
    </BrowserRouter>
  )
}
