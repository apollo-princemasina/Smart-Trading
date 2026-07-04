/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:        '#0D1117',
        surface:   '#161B22',
        border:    '#21262D',
        muted:     '#6B7280',
        bull:      '#10B981',
        bear:      '#EF4444',
        hold:      '#F59E0B',
        manip:     '#8B5CF6',
        expand:    '#3B82F6',
        consolidate: '#6B7280',
        gold:      '#F0B429',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
