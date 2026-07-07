'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/utils/cn'
import { useUIStore } from '@/stores/uiStore'

interface NavItem {
  label:    string
  href:     string
  icon:     React.ReactNode
  disabled?: boolean
}

function GridIcon()    { return <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5"><rect x="1" y="1" width="6" height="6" rx="1"/><rect x="9" y="1" width="6" height="6" rx="1"/><rect x="1" y="9" width="6" height="6" rx="1"/><rect x="9" y="9" width="6" height="6" rx="1"/></svg> }
function BrainIcon()   { return <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5"><path d="M8 2C5.8 2 4 3.8 4 6c0 1.1.4 2 1.1 2.7-.7.6-1.1 1.5-1.1 2.3 0 1.7 1.3 3 3 3s3-1.3 3-3c0-.8-.4-1.7-1.1-2.3C9.6 8 10 7.1 10 6c0-2.2-1.8-4-2-4z"/><path d="M5.5 7.5H4M10.5 7.5H12"/></svg> }
function ClockIcon()   { return <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5"><circle cx="8" cy="8" r="6"/><path d="M8 5v3l2 2"/></svg> }
function ListIcon()    { return <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5"><path d="M2 4h12M2 8h12M2 12h12"/></svg> }
function CubeIcon()    { return <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5"><path d="M8 1.5L14 5v6L8 14.5 2 11V5L8 1.5z"/><path d="M8 1.5V8M2 5l6 3M14 5l-6 3"/></svg> }
function PulseIcon()   { return <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5"><path d="M1 8h3l2-5 3 10 2-5h4"/></svg> }
function GearIcon()    { return <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5"><circle cx="8" cy="8" r="2.5"/><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41"/></svg> }

const NAV_ITEMS: NavItem[] = [
  { label: 'Dashboard',          href: '/',            icon: <GridIcon /> },
  { label: 'Market Intelligence', href: '/intelligence', icon: <BrainIcon />, disabled: true },
  { label: 'Decision History',   href: '/decisions',   icon: <ClockIcon /> },
  { label: 'Predictions',        href: '/predictions', icon: <ListIcon /> },
  { label: 'Models',             href: '/models',      icon: <CubeIcon /> },
  { label: 'System Health',      href: '/system',      icon: <PulseIcon /> },
]

const BOTTOM_ITEMS: NavItem[] = [
  { label: 'Settings', href: '/settings', icon: <GearIcon /> },
]

function NavLink({ item, collapsed }: { item: NavItem; collapsed: boolean }) {
  const pathname = usePathname()
  const active = pathname === item.href || (item.href !== '/' && pathname.startsWith(item.href))

  return (
    <Link
      href={item.disabled ? '#' : item.href}
      aria-disabled={item.disabled}
      title={collapsed ? item.label : undefined}
      className={cn(
        'group flex items-center rounded-md transition-all duration-150 relative',
        collapsed ? 'justify-center p-2.5' : 'gap-3 px-3 py-2',
        active && !item.disabled
          ? 'bg-navy-600 text-primary'
          : item.disabled
            ? 'text-muted cursor-not-allowed opacity-40'
            : 'text-secondary hover:text-primary hover:bg-navy-700',
      )}
    >
      {/* Active indicator */}
      {active && !item.disabled && (
        <span className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-gold rounded-r" />
      )}

      <span className={cn('shrink-0 transition-colors', active ? 'text-gold' : '')}>
        {item.icon}
      </span>

      {!collapsed && (
        <span className="text-xs font-medium truncate">{item.label}</span>
      )}

      {/* Tooltip for collapsed */}
      {collapsed && (
        <span className="pointer-events-none absolute left-full ml-2 z-50 bg-elevated border border-border text-primary text-xs rounded px-2 py-1 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity shadow-card-lg">
          {item.label}
        </span>
      )}
    </Link>
  )
}

export function NavSidebar() {
  const { sidebarOpen } = useUIStore()
  const collapsed = !sidebarOpen

  return (
    <aside
      className={cn(
        'flex flex-col bg-surface border-r border-border transition-all duration-200 shrink-0',
        collapsed ? 'w-14' : 'w-52',
      )}
    >
      {/* Logo */}
      <div className={cn(
        'flex items-center border-b border-border h-14 shrink-0',
        collapsed ? 'justify-center px-0' : 'px-4 gap-2.5',
      )}>
        <div className="w-7 h-7 rounded bg-gold/10 border border-gold/30 flex items-center justify-center shrink-0">
          <span className="text-gold font-black text-sm">M</span>
        </div>
        {!collapsed && (
          <div>
            <p className="text-primary font-bold text-sm leading-none">MFIP</p>
            <p className="text-muted text-[10px] tracking-wide">Intelligence</p>
          </div>
        )}
      </div>

      {/* Main nav */}
      <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map(item => (
          <NavLink key={item.href} item={item} collapsed={collapsed} />
        ))}
      </nav>

      {/* Bottom nav */}
      <div className="px-2 pb-3 pt-2 border-t border-border space-y-0.5">
        {BOTTOM_ITEMS.map(item => (
          <NavLink key={item.href} item={item} collapsed={collapsed} />
        ))}
      </div>
    </aside>
  )
}
