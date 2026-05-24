/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background:      'var(--color-background)',
        surface:         'var(--color-surface)',
        'surface-2':     'var(--color-surface-2)',
        border:          'var(--color-border)',
        'border-hover':  'var(--color-border-hover)',
        'team-1':        'var(--color-team-1)',
        'team-2':        'var(--color-team-2)',
        accent:          'var(--color-accent)',
        'text-primary':  'var(--color-text-primary)',
        'text-secondary':'var(--color-text-secondary)',
        'grade-a':       'var(--color-grade-a)',
        'grade-b':       'var(--color-grade-b)',
        'grade-c':       'var(--color-grade-c)',
        'grade-d':       'var(--color-grade-d)',
        'grade-f':       'var(--color-grade-f)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
