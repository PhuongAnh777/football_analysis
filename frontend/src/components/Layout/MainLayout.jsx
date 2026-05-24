import { Outlet, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Sidebar from './Sidebar'

const pageVariants = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
  exit:    { opacity: 0, y: -8,  transition: { duration: 0.2 } },
}

export default function MainLayout() {
  const location = useLocation()
  return (
    <div className="flex min-h-screen" style={{ background: 'var(--color-background)' }}>
      <Sidebar />
      <main className="flex-1 ml-64 min-h-screen overflow-x-hidden">
        <AnimatePresence mode="wait">
          <motion.div key={location.pathname} variants={pageVariants}
            initial="initial" animate="animate" exit="exit" className="min-h-screen">
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
