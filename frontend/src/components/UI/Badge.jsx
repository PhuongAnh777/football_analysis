export default function Badge({ children, variant = 'default', size = 'md', className = '', ...props }) {
  const sizes = { sm: 'text-xs px-2 py-0.5', md: 'text-xs px-2.5 py-1', lg: 'text-sm px-3 py-1.5' }
  const variants = {
    default: 'bg-surface-2 text-text-secondary border border-border',
    team1:   'bg-blue-500/15 text-blue-400 border border-blue-500/30',
    team2:   'bg-red-500/15 text-red-400 border border-red-500/30',
    accent:  'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30',
    outline: 'bg-transparent text-text-secondary border border-border',
    ghost:   'bg-surface-2 text-text-primary border-none',
    ai:      'bg-gradient-to-r from-blue-500/20 to-emerald-500/20 text-emerald-300 border border-emerald-500/30',
  }
  return (
    <span className={`inline-flex items-center gap-1 rounded-full font-medium ${sizes[size]} ${variants[variant]} ${className}`} {...props}>
      {children}
    </span>
  )
}
