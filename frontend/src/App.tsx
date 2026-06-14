import { useEffect, useRef, useState } from 'react'
import { motion, useInView, AnimatePresence } from 'framer-motion'
import { ArrowDown, GitForkIcon, BarChart3, Brain, Search, Globe, Layers, Shield, Zap, Cpu, TestTube, ExternalLink, ChevronRight, Sparkles, Code2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'

const NAV_ITEMS = ['Features', 'SDKs', 'Benchmarks']
const FEATURES = [
  { icon: Zap, title: '9 ANN Algorithms', desc: 'HNSW, IVF+PQ, Vamana/DiskANN, Int8, LSH, KD-Tree, VP-Tree, BM25, Hybrid RRF', color: '#06b6d4' },
  { icon: Brain, title: 'Agentic Memory', desc: 'pgvector-backed persistent memory with CRUD, semantic search, LLM chat, and SSE streaming', color: '#8b5cf6' },
  { icon: Search, title: 'SPLADE + ColBERT', desc: 'Learned sparse retrieval via transformers and late-interaction MaxSim scoring', color: '#10b981' },
  { icon: Globe, title: 'Polyglot SDKs', desc: 'Python, TypeScript, Go, Java, Rust, .NET — type-safe clients for every language', color: '#eab308' },
  { icon: BarChart3, title: 'Self-Tuning Indexes', desc: 'AI-recommended parameters and adaptive per-query index routing', color: '#f97316' },
  { icon: Shield, title: 'Enterprise Compliance', desc: 'SOC2/GDPR reports, retention policies, query budgets, AES-256 encryption', color: '#ec4899' },
  { icon: Layers, title: 'Real-Time Streaming', desc: 'SSE subscriptions, webhooks, lock-free HNSW writes, materialized views', color: '#22d3ee' },
  { icon: Cpu, title: 'Ecosystem Integrations', desc: 'Haystack, Semantic Kernel, LangChain, LlamaIndex, Arrow Flight, MCP Server', color: '#a855f7' },
  { icon: TestTube, title: '660+ Tests', desc: 'Full coverage: API, indexes, services, durability, ML models, infrastructure', color: '#facc15' },
]
const SDKS = [
  { icon: '🐍', name: 'Python', stable: true },
  { icon: '🔷', name: 'TypeScript', stable: true },
  { icon: '🐹', name: 'Go', stable: true },
  { icon: '☕', name: 'Java', stable: true },
  { icon: '🦀', name: 'Rust', stable: true },
  { icon: '💧', name: '.NET', stable: true },
]
const BENCHMARKS = [
  { index: 'HNSW', params: '(M=16, ef=200)', recall: '0.981', latency: '0.129 s', p99: '0.789 s', build: '45.2 s', highlight: true },
  { index: 'IVF', params: '(nlist=100, nprobe=10)', recall: '0.940', latency: '0.350 s', p99: '1.200 s', build: '1.8 s', highlight: false },
  { index: 'Brute Force', params: '', recall: '1.000', latency: '4.200 s', p99: '9.100 s', build: '—', highlight: false },
  { index: 'Vamana/DiskANN', params: '(L=75, R=50)', recall: '0.970', latency: '0.150 s', p99: '0.850 s', build: '38.0 s', highlight: false },
]

function useCountUp(ref: React.RefObject<HTMLElement | null>, target: number, suffix = '') {
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return
      let current = 0
      const increment = Math.max(1, Math.floor(target / 40))
      const timer = setInterval(() => {
        current += increment
        if (current >= target) { current = target; clearInterval(timer) }
        el.textContent = current + suffix
      }, 30)
      observer.disconnect()
    }, { threshold: 0.5 })
    observer.observe(el)
    return () => observer.disconnect()
  }, [ref, target, suffix])
}

function FadeIn({ children, className = '', delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.6, delay, ease: [0.175, 0.885, 0.32, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

function StatItem({ target, label, suffix = '' }: { target: number; label: string; suffix?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  useCountUp(ref, target, suffix)
  return (
    <div className="text-center">
      <div ref={ref} className="text-5xl font-extrabold bg-gradient-to-br from-white to-cyan-400 bg-clip-text text-transparent">
        0
      </div>
      <div className="text-sm text-muted mt-2">{label}</div>
    </div>
  )
}

function Particles() {
  const [particles, setParticles] = useState<{ x: number; y: number; size: number; duration: number; delay: number }[]>([])
  useEffect(() => {
    setParticles(Array.from({ length: 30 }, () => ({
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 3 + 1,
      duration: Math.random() * 10 + 10,
      delay: Math.random() * 5,
    })))
  }, [])
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {particles.map((p, i) => (
        <motion.div
          key={i}
          className="absolute rounded-full bg-white/10"
          style={{ left: `${p.x}%`, top: `${p.y}%`, width: p.size, height: p.size }}
          animate={{ y: [0, -30, 0], opacity: [0, 0.5, 0] }}
          transition={{ duration: p.duration, repeat: Infinity, delay: p.delay, ease: 'easeInOut' }}
        />
      ))}
    </div>
  )
}

export default function App() {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <div className="min-h-screen bg-background text-foreground overflow-x-hidden">
      {/* NAV */}
      <motion.nav
        className={`fixed top-0 left-0 right-0 z-50 px-6 py-4 flex items-center justify-between transition-all duration-300 ${
          scrolled ? 'bg-background/80 backdrop-blur-xl border-b border-white/5' : ''
        }`}
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
      >
        <a href="/" className="flex items-center gap-2 text-lg font-bold">
          <span className="size-2 rounded-full bg-cyan-400" />
          VectorDB
        </a>
        <div className="hidden md:flex items-center gap-8">
          {NAV_ITEMS.map(item => (
            <a key={item} href={`#${item.toLowerCase()}`} className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              {item}
            </a>
          ))}
          <a href="https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB" target="_blank" rel="noopener noreferrer" className="text-sm text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1.5">
            <GitForkIcon className="size-4" /> GitHub
          </a>
          <a href="/dashboard"><Button size="sm">Dashboard <ExternalLink className="size-3.5" /></Button></a>
        </div>
      </motion.nav>

      {/* HERO */}
      <section className="relative min-h-screen flex flex-col items-center justify-center text-center px-6 overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute size-[600px] rounded-full bg-cyan-500/10 blur-[120px] -top-32 -left-32 animate-pulse" style={{ animationDuration: '8s' }} />
          <div className="absolute size-[500px] rounded-full bg-purple-500/10 blur-[120px] -bottom-32 -right-32 animate-pulse" style={{ animationDuration: '10s', animationDelay: '2s' }} />
          <div className="absolute size-[400px] rounded-full bg-emerald-500/10 blur-[120px] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-pulse" style={{ animationDuration: '12s', animationDelay: '4s' }} />
          <div className="absolute inset-0" style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)', backgroundSize: '60px 60px' }} />
        </div>
        <Particles />
        <div className="relative z-10 max-w-4xl mx-auto">
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="mb-6">
            <Badge variant="outline" className="border-cyan-500/20 text-cyan-400 bg-cyan-500/5 px-4 py-1.5 text-xs gap-1.5">
              <Sparkles className="size-3" /> v1.0.0 · Open Source
            </Badge>
          </motion.div>
          <motion.h1
            className="text-5xl sm:text-7xl md:text-8xl font-extrabold leading-[0.95] tracking-tight mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8, ease: [0.175, 0.885, 0.32, 1] }}
          >
            <span className="bg-gradient-to-r from-white via-slate-300 to-cyan-400 bg-clip-text text-transparent">
              Build intelligent
            </span>
            <br />
            <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-white bg-clip-text text-transparent">
              search at scale
            </span>
          </motion.h1>
          <motion.p
            className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.6 }}
          >
            A production-grade vector database with 9+ ANN algorithms, agentic memory, multi-vector search, and enterprise compliance — built for AI workloads.
          </motion.p>
          <motion.div
            className="flex flex-wrap gap-4 justify-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.6 }}
          >
            <a href="/dashboard"><Button size="lg" className="bg-gradient-to-r from-cyan-500 to-cyan-600 hover:from-cyan-400 hover:to-cyan-500 text-white shadow-lg shadow-cyan-500/20">
              Launch Dashboard <ExternalLink className="size-4" />
            </Button></a>
            <a href="#features"><Button variant="outline" size="lg" className="border-white/10 hover:bg-white/5">
              Explore Features <ChevronRight className="size-4" />
            </Button></a>
          </motion.div>
        </div>
        <motion.div className="absolute bottom-8 z-10" animate={{ y: [0, 8, 0] }} transition={{ duration: 2, repeat: Infinity }}>
          <ArrowDown className="size-5 text-muted" />
        </motion.div>
      </section>

      {/* FEATURES */}
      <section id="features" className="px-6 py-24 max-w-6xl mx-auto">
        <FadeIn>
          <Badge variant="outline" className="border-cyan-500/20 text-cyan-400 bg-cyan-500/5 mb-4">Capabilities</Badge>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">Everything you need for vector search</h2>
          <p className="text-muted-foreground max-w-xl mb-12">9 ANN algorithms, multi-modal embeddings, real-time streaming, enterprise compliance, and SDKs for every major language.</p>
        </FadeIn>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map((f, i) => (
            <FadeIn key={f.title} delay={i * 0.05}>
              <Card className="group border-white/5 bg-white/[0.02] hover:bg-white/[0.04] hover:border-cyan-500/20 transition-all duration-500 cursor-default overflow-hidden relative">
                <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <CardContent className="p-6">
                  <div className="size-11 rounded-xl flex items-center justify-center mb-4 text-lg" style={{ background: `${f.color}15`, color: f.color }}>
                    <f.icon className="size-5" />
                  </div>
                  <h3 className="font-semibold mb-1.5">{f.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
                </CardContent>
              </Card>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* SDKs */}
      <section id="sdks" className="px-6 py-24 max-w-6xl mx-auto">
        <FadeIn>
          <Badge variant="outline" className="border-purple-500/20 text-purple-400 bg-purple-500/5 mb-4">SDKs & Clients</Badge>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">Ship in any language</h2>
          <p className="text-muted-foreground max-w-xl mb-12">Idiomatic, type-safe clients for every major ecosystem.</p>
        </FadeIn>
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
          {SDKS.map((s, i) => (
            <FadeIn key={s.name} delay={i * 0.08}>
              <div className="p-5 rounded-xl bg-white/[0.02] border border-white/5 hover:border-cyan-500/20 hover:bg-white/[0.04] transition-all text-center cursor-default group">
                <div className="text-3xl mb-2">{s.icon}</div>
                <div className="text-sm font-semibold">{s.name}</div>
                <div className="text-[10px] text-emerald-400 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">✓ Stable</div>
              </div>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* STATS */}
      <section className="px-6 py-24 max-w-4xl mx-auto">
        <FadeIn>
          <Badge variant="outline" className="border-amber-500/20 text-amber-400 bg-amber-500/5 mb-4">By the Numbers</Badge>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3 text-center">Built for production</h2>
        </FadeIn>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-8 mt-12">
          <StatItem target={9} label="ANN Algorithms" />
          <StatItem target={11} label="SDKs & Integrations" />
          <StatItem target={660} label="Tests" suffix="+" />
          <StatItem target={80} label="REST Endpoints" suffix="+" />
        </div>
      </section>

      {/* BENCHMARKS */}
      <section id="benchmarks" className="px-6 py-24 max-w-5xl mx-auto">
        <FadeIn>
          <Badge variant="outline" className="border-cyan-500/20 text-cyan-400 bg-cyan-500/5 mb-4">Performance</Badge>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">Benchmark snapshot</h2>
          <p className="text-muted-foreground max-w-xl mb-12">10K vectors, 128-dim — real numbers from the built-in benchmark suite.</p>
        </FadeIn>
        <FadeIn>
          <div className="overflow-x-auto rounded-2xl border border-white/5">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/5 bg-white/[0.02]">
                  <th className="text-left p-4 font-semibold text-muted text-xs uppercase tracking-wider">Index</th>
                  <th className="text-left p-4 font-semibold text-muted text-xs uppercase tracking-wider">Recall@10</th>
                  <th className="text-left p-4 font-semibold text-muted text-xs uppercase tracking-wider">Avg Latency</th>
                  <th className="text-left p-4 font-semibold text-muted text-xs uppercase tracking-wider">p99 Latency</th>
                  <th className="text-left p-4 font-semibold text-muted text-xs uppercase tracking-wider">Build Time</th>
                </tr>
              </thead>
              <tbody>
                {BENCHMARKS.map(b => (
                  <tr key={b.index} className="border-b border-white/[0.02] last:border-0 hover:bg-cyan-500/5 transition-colors">
                    <td className="p-4">
                      <span className="font-semibold">{b.index}</span>
                      {b.params && <span className="text-muted ml-1 text-xs">{b.params}</span>}
                    </td>
                    <td className={`p-4 font-mono ${b.highlight ? 'text-cyan-400' : ''}`}>{b.recall}</td>
                    <td className="p-4 font-mono text-muted-foreground">{b.latency}</td>
                    <td className="p-4 font-mono text-muted-foreground">{b.p99}</td>
                    <td className="p-4 font-mono text-muted-foreground">{b.build}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </FadeIn>
      </section>

      {/* CTA */}
      <section className="px-6 py-24 text-center">
        <FadeIn>
          <div className="max-w-2xl mx-auto p-12 rounded-3xl bg-gradient-to-br from-cyan-500/5 via-purple-500/5 to-cyan-500/5 border border-cyan-500/10">
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-4">Ready to build?</h2>
            <p className="text-muted-foreground mb-8 max-w-md mx-auto">Get started with the dashboard or explore the codebase on GitHub.</p>
            <div className="flex flex-wrap gap-4 justify-center">
              <a href="/dashboard"><Button size="lg" className="bg-gradient-to-r from-cyan-500 to-cyan-600 hover:from-cyan-400 hover:to-cyan-500 text-white shadow-lg shadow-cyan-500/20">
                Launch Dashboard <ExternalLink className="size-4" />
              </Button></a>
              <a href="https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB" target="_blank" rel="noopener noreferrer">
                <Button variant="outline" size="lg" className="border-white/10 hover:bg-white/5">
                  <GitForkIcon className="size-4" /> GitHub
                </Button>
              </a>
            </div>
          </div>
        </FadeIn>
      </section>

      {/* FOOTER */}
      <footer className="border-t border-white/5 px-6 py-10 text-center text-sm text-muted">
        <div className="flex justify-center gap-6 mb-4 flex-wrap">
          {['Features', 'SDKs', 'Benchmarks', 'GitHub', 'Issues'].map(item => (
            <a key={item} href={item === 'GitHub' ? 'https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB' : item === 'Issues' ? 'https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/issues' : `#${item.toLowerCase()}`}
              className="hover:text-foreground transition-colors">
              {item}
            </a>
          ))}
        </div>
        <p>Built with Rust, Python, and TypeScript. Open source under MIT License.</p>
      </footer>
    </div>
  )
}
