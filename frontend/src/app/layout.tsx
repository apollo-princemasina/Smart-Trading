import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'MFIP — Moonshot Forex Intelligence Platform',
  description: 'Live EURUSD M15 AI signal dashboard',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
