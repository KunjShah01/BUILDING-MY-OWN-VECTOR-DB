import { useState, useEffect } from 'react'

export default function AnimatedHeading({
  text,
  className = '',
  delay = 200,
  charDelay = 30,
}: {
  text: string
  className?: string
  delay?: number
  charDelay?: number
}) {
  const [visible, setVisible] = useState(false)
  const lines = text.split('\n')

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), delay)
    return () => clearTimeout(timer)
  }, [delay])

  let globalCharIndex = 0

  return (
    <h1 className={className} style={{ letterSpacing: '-0.04em' }}>
      {lines.map((line, lineIndex) => {
        const startIndex = globalCharIndex
        globalCharIndex += line.length
        return (
          <div key={lineIndex} className="flex flex-wrap justify-center">
            {line.split('').map((char, charIndex) => (
              <span
                key={charIndex}
                className="inline-block transition-all duration-500"
                style={{
                  opacity: visible ? 1 : 0,
                  transform: visible ? 'translateX(0)' : 'translateX(-18px)',
                  transitionDelay: `${lineIndex * line.length * charDelay + charIndex * charDelay}ms`,
                }}
              >
                {char === ' ' ? '\u00A0' : char}
              </span>
            ))}
          </div>
        )
      })}
    </h1>
  )
}
