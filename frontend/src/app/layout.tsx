import type { Metadata, Viewport } from 'next'
import { Providers } from '@/providers/Providers'
import './globals.css'

export const metadata: Metadata = {
  title:       'MFIP — Moonshot Forex Intelligence Platform',
  description: 'AI Decision Intelligence Platform for institutional-grade forex analysis',
  icons: { icon: '/favicon.ico' },
}

export const viewport: Viewport = {
  themeColor:  '#070C14',
  colorScheme: 'dark',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  )
}
