/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Background depth scale
        navy: {
          950: '#04080F',
          900: '#070C14',
          850: '#0A1220',
          800: '#0D1827',
          750: '#101E30',
          700: '#111D2E',
          600: '#162236',
          500: '#1A2D44',
          400: '#1E3450',
        },
        // UI layer aliases
        bg:      '#070C14',
        surface: '#0D1827',
        elevated:'#111D2E',
        hover:   '#162236',
        // Borders
        border:  '#1E2D40',
        'border-subtle': '#111D2E',
        // Text
        primary:   '#EDF2F7',
        secondary: '#7C8EA6',
        muted:     '#4B5D74',
        // Brand
        gold:     '#F0B429',
        'gold-dim':'#7A5C14',
        'gold-glow':'#F0B42920',
        // Decision semantic
        buy:       '#10B981',
        'buy-dim': '#065F46',
        'buy-bg':  '#022C22',
        sell:      '#F87171',
        'sell-dim':'#991B1B',
        'sell-bg': '#2D0A0A',
        wait:      '#FBBF24',
        'wait-dim':'#92400E',
        'wait-bg': '#2D1A00',
        // Regime
        manipulation: '#A78BFA',
        expansion:    '#60A5FA',
        consolidation:'#6B7280',
        // System
        signal:   '#38BDF8',
        success:  '#10B981',
        warning:  '#FBBF24',
        danger:   '#F87171',
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['var(--font-mono)', 'JetBrains Mono', 'Fira Code', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.875rem' }],
      },
      boxShadow: {
        'glow-gold':  '0 0 20px rgba(240, 180, 41, 0.15)',
        'glow-buy':   '0 0 20px rgba(16, 185, 129, 0.2)',
        'glow-sell':  '0 0 20px rgba(248, 113, 113, 0.2)',
        'glow-wait':  '0 0 20px rgba(251, 191, 36, 0.2)',
        'card':       '0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3)',
        'card-lg':    '0 4px 16px rgba(0,0,0,0.5)',
        'inner-gold': 'inset 0 0 0 1px rgba(240, 180, 41, 0.2)',
      },
      animation: {
        'pulse-slow':  'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
        'fade-in':     'fadeIn 0.3s ease-out',
        'slide-up':    'slideUp 0.3s ease-out',
        'ping-slow':   'ping 2s cubic-bezier(0,0,0.2,1) infinite',
      },
      keyframes: {
        fadeIn:  { from: { opacity: '0' }, to: { opacity: '1' } },
        slideUp: { from: { opacity: '0', transform: 'translateY(8px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'noise': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.05'/%3E%3C/svg%3E\")",
      },
    },
  },
  plugins: [],
}
