import { useState, useEffect } from 'react'

export default function FadeIn({
  children,
  delay = 0,
  duration = 800,
  className = '',
}: {
  children: React.ReactNode
  delay?: number
  duration?: number
  className?: string
}) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), delay)
    return () => clearTimeout(timer)
  }, [delay])

  return (
    <div
      className={className}
      style={{
        opacity: visible ? 1 : 0,
        transition: `opacity ${duration}ms ease-in-out`,
      }}
    >
      {children}
    </div>
  )
}
