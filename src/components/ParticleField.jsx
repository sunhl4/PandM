import { useEffect, useRef } from 'react'

export default function ParticleField() {
  const canvasRef = useRef(null)
  const animRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    let W = canvas.offsetWidth
    let H = canvas.offsetHeight
    canvas.width = W
    canvas.height = H

    // Particle class
    class Particle {
      constructor() { this.reset() }
      reset() {
        this.x = Math.random() * W
        this.y = Math.random() * H
        this.vx = (Math.random() - 0.5) * 0.3
        this.vy = (Math.random() - 0.5) * 0.3
        this.radius = Math.random() * 1.5 + 0.3
        this.alpha = Math.random() * 0.6 + 0.1
        this.color = Math.random() > 0.5 ? '0,212,255' : '139,92,246'
        this.pulse = Math.random() * Math.PI * 2
        this.pulseSpeed = Math.random() * 0.02 + 0.005
      }
      update() {
        this.x += this.vx
        this.y += this.vy
        this.pulse += this.pulseSpeed
        const pulseAlpha = this.alpha * (0.7 + 0.3 * Math.sin(this.pulse))
        if (this.x < -10 || this.x > W + 10 || this.y < -10 || this.y > H + 10) this.reset()
        return pulseAlpha
      }
      draw(alpha) {
        ctx.beginPath()
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${this.color},${alpha})`
        ctx.fill()
      }
    }

    // Wave line class (quantum interference lines)
    class WaveLine {
      constructor() { this.reset() }
      reset() {
        this.y = Math.random() * H
        this.amplitude = Math.random() * 20 + 5
        this.frequency = Math.random() * 0.015 + 0.003
        this.phase = Math.random() * Math.PI * 2
        this.speed = Math.random() * 0.008 + 0.002
        this.alpha = Math.random() * 0.06 + 0.01
        this.color = Math.random() > 0.5 ? '0,212,255' : '139,92,246'
        this.width = Math.random() * 1 + 0.3
      }
      update() { this.phase += this.speed }
      draw() {
        ctx.beginPath()
        ctx.strokeStyle = `rgba(${this.color},${this.alpha})`
        ctx.lineWidth = this.width
        for (let x = 0; x < W; x += 3) {
          const y = this.y + Math.sin(x * this.frequency + this.phase) * this.amplitude
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
        }
        ctx.stroke()
      }
    }

    const PARTICLE_COUNT = Math.min(Math.floor((W * H) / 6000), 120)
    const WAVE_COUNT = 8
    const particles = Array.from({ length: PARTICLE_COUNT }, () => new Particle())
    const waves = Array.from({ length: WAVE_COUNT }, () => new WaveLine())

    const CONNECTION_DIST = 80

    const render = () => {
      ctx.clearRect(0, 0, W, H)

      // Draw wave lines
      waves.forEach((w) => { w.update(); w.draw() })

      // Draw connections
      ctx.lineWidth = 0.4
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x
          const dy = particles[i].y - particles[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < CONNECTION_DIST) {
            const alpha = (1 - dist / CONNECTION_DIST) * 0.12
            ctx.beginPath()
            ctx.strokeStyle = `rgba(0,212,255,${alpha})`
            ctx.moveTo(particles[i].x, particles[i].y)
            ctx.lineTo(particles[j].x, particles[j].y)
            ctx.stroke()
          }
        }
      }

      // Draw particles
      particles.forEach((p) => {
        const a = p.update()
        p.draw(a)
      })

      animRef.current = requestAnimationFrame(render)
    }

    render()

    const handleResize = () => {
      W = canvas.offsetWidth
      H = canvas.offsetHeight
      canvas.width = W
      canvas.height = H
    }
    window.addEventListener('resize', handleResize)

    return () => {
      cancelAnimationFrame(animRef.current)
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ opacity: 0.65 }}
    />
  )
}
