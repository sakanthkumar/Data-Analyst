import React, { useState, useEffect, useRef } from 'react';
import GlassCard from '../components/GlassCard';
import FadeIn from '../components/animation/FadeIn';
import { useTheme } from '../ThemeContext';

export default function LandingPage({ onNavigateToAuth }) {
  const { theme, toggleTheme } = useTheme();
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [hoveredLayer, setHoveredLayer] = useState(null);
  const carouselRef = useRef(null);

  // Subtle Parallax Event Listener
  useEffect(() => {
    const handleMouseMove = (e) => {
      const x = (e.clientX / window.innerWidth) - 0.5;
      const y = (e.clientY / window.innerHeight) - 0.5;
      setMousePos({ x, y });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const getHeroLayerStyle = (layerId, baseTop, baseRight, baseRotate, w, h, depth, baseBlur = 0) => {
    const isHovered = hoveredLayer === layerId;
    const isAnyHovered = hoveredLayer !== null;
    
    // Calculate Parallax shift
    const moveX = mousePos.x * depth;
    const moveY = mousePos.y * depth;
    
    // Scale, opacity, and blur values based on hover state
    let scale = 1;
    let opacity = 0.85;
    let blurVal = baseBlur;
    let zIndex = depth;
    
    if (isHovered) {
      scale = 1.08;
      opacity = 1.0;
      blurVal = 0;
      zIndex = 100; // Bring to front
    } else if (isAnyHovered) {
      scale = 0.94;
      opacity = 0.6;
      blurVal = baseBlur + 1; // Extra blur on inactive layers
    } else {
      // Normal state overrides
      if (layerId === 'front') {
        scale = 1.03;
        opacity = 1.0;
      } else if (layerId === 'back') {
        scale = 0.9;
        opacity = 0.7;
      }
    }
    
    return {
      position: 'absolute',
      top: baseTop,
      right: baseRight,
      width: `${w}px`,
      height: `${h}px`,
      transform: `translate(${moveX}px, ${moveY}px) rotateX(${mousePos.y * 6}deg) rotateY(${mousePos.x * 6}deg) rotate(${baseRotate}) scale(${scale})`,
      transition: 'transform 0.25s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.25s ease, filter 0.25s ease',
      filter: blurVal > 0 ? `blur(${blurVal}px)` : 'none',
      opacity: opacity,
      zIndex: zIndex,
      cursor: 'pointer'
    };
  };

  const handleScroll = (dir) => {
    if (carouselRef.current) {
      const scrollAmt = dir === 'next' ? 624 : -624;
      carouselRef.current.scrollBy({ left: scrollAmt, behavior: 'smooth' });
    }
  };

  return (
    <FadeIn>
      <div style={{ 
        minHeight: '100vh', 
        backgroundColor: 'var(--bg-body)', 
        color: 'var(--text-main)',
        fontFamily: 'Inter, sans-serif',
        overflowX: 'hidden'
      }}>
        {/* Navigation Bar */}
        <header style={{
          position: 'sticky',
          top: 0,
          zIndex: 100,
          backdropFilter: 'blur(20px)',
          backgroundColor: 'var(--bg-header)',
          borderBottom: '1px solid var(--border-color)',
          height: '64px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 40px',
          boxSizing: 'border-box'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ 
              width: '32px', 
              height: '32px', 
              borderRadius: '4px', 
              backgroundColor: 'var(--primary-color)', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              color: 'white',
              boxShadow: '0 0 10px rgba(59, 130, 246, 0.4)'
            }}>
              <span className="material-symbols-outlined" style={{ fontSize: '20px', fontVariationSettings: "'FILL' 1" }}>
                analytics
              </span>
            </div>
            <div>
              <h1 style={{ fontSize: '1.25rem', fontWeight: 'bold', color: 'var(--text-main)', margin: 0, fontFamily: 'Geist', lineHeight: '1.1' }}>
                Analyst.AI
              </h1>
              <span style={{ fontSize: '9px', letterSpacing: '0.12em', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                Enterprise Suite
              </span>
            </div>
          </div>

          <nav style={{ display: 'flex', gap: '30px', alignItems: 'center' }} className="landing-nav-links">
            <a href="#features" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.9rem', fontWeight: 500, transition: 'color 0.2s' }}>Features</a>
            <a href="#workflow" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.9rem', fontWeight: 500, transition: 'color 0.2s' }}>Workflow</a>
            <a href="#showcase" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.9rem', fontWeight: 500, transition: 'color 0.2s' }}>Showcase</a>
            <a href="#about" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.9rem', fontWeight: 500, transition: 'color 0.2s' }}>About</a>
          </nav>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <button 
              onClick={toggleTheme}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-muted)',
                display: 'flex',
                alignItems: 'center',
                padding: '6px',
                borderRadius: '50%',
                cursor: 'pointer'
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>
                {theme === 'dark' ? 'light_mode' : 'dark_mode'}
              </span>
            </button>
            <button 
              onClick={() => onNavigateToAuth('login')}
              style={{ background: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-main)', fontSize: '0.85rem', padding: '6px 14px', borderRadius: '6px' }}
            >
              Log In
            </button>
            <button 
              onClick={() => onNavigateToAuth('register')}
              style={{ backgroundColor: 'var(--primary-color)', color: '#fff', fontSize: '0.85rem', padding: '6px 16px', borderRadius: '6px' }}
            >
              Get Started
            </button>
          </div>
        </header>

        {/* Hero Section */}
        <section style={{ 
          padding: '80px 40px 100px 40px',
          maxWidth: 'var(--spacing-container-max)',
          margin: '0 auto',
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '40px',
          alignItems: 'center'
        }} className="hero-responsive">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div style={{ 
              display: 'inline-flex', 
              alignItems: 'center', 
              gap: '6px', 
              backgroundColor: 'rgba(59, 130, 246, 0.08)', 
              border: '1px solid rgba(59, 130, 246, 0.2)',
              borderRadius: '20px',
              padding: '4px 12px',
              width: 'fit-content'
            }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--secondary-color)' }}></span>
              <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--secondary-color)', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                Enterprise Intelligence V2.0
              </span>
            </div>
            
            <h2 style={{ 
              fontSize: '3.2rem', 
              fontWeight: 800, 
              color: 'var(--text-main)', 
              lineHeight: 1.1, 
              fontFamily: 'Geist', 
              letterSpacing: '-0.03em',
              margin: 0
            }}>
              Transform Enterprise Data Into <span style={{ color: 'var(--primary-color)' }}>Actionable Intelligence</span>
            </h2>
            
            <p style={{ 
              fontSize: '1.15rem', 
              color: 'var(--text-muted)', 
              lineHeight: 1.55, 
              margin: 0,
              maxWidth: '560px'
            }}>
              Upload datasets. Discover hidden insights. Generate executive reports. Chat with your data using AI-powered contextual engines.
            </p>

            <div style={{ display: 'flex', gap: '16px', marginTop: '8px' }}>
              <button 
                onClick={() => onNavigateToAuth('register')}
                style={{ 
                  backgroundColor: 'var(--primary-color)', 
                  color: 'white', 
                  fontSize: '0.95rem', 
                  padding: '12px 24px', 
                  borderRadius: '6px', 
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  boxShadow: '0 4px 14px rgba(59, 130, 246, 0.3)'
                }}
              >
                Start Free Analysis
                <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>arrow_forward</span>
              </button>
              <button 
                onClick={() => onNavigateToAuth('login')}
                style={{ 
                  border: '1px solid var(--border-color)', 
                  color: 'var(--text-main)', 
                  fontSize: '0.95rem', 
                  padding: '12px 24px', 
                  borderRadius: '6px',
                  fontWeight: 500,
                  backgroundColor: 'var(--bg-card)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: '18px', fontVariationSettings: "'FILL' 1" }}>play_circle</span>
                Watch Demo
              </button>
            </div>
          </div>

          {/* Hero Stacked Perspective Panels */}
          <div style={{ position: 'relative', height: '520px', perspective: '1000px' }} className="hero-stack-container">
            
            {/* Layer 1: Back (Executive Reports) */}
            <div 
              style={getHeroLayerStyle('back', '10px', '120px', '8deg', 380, 214, 12, 1.5)}
              onMouseEnter={() => setHoveredLayer('back')}
              onMouseLeave={() => setHoveredLayer(null)}
            >
              <GlassCard style={{ width: '100%', height: '100%', padding: '4px', overflow: 'hidden', position: 'relative', border: '1px solid rgba(255, 255, 255, 0.08)', boxShadow: '0 15px 30px rgba(0, 0, 0, 0.4)' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '1px', background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.25), transparent)', zIndex: 5 }} />
                <img 
                  src="/assets/dashboard.png" 
                  alt="Executive Reports" 
                  style={{ 
                    width: '100%', 
                    height: '100%', 
                    objectFit: 'cover',
                    transform: hoveredLayer === 'back' ? 'scale(1.05)' : 'scale(1)',
                    transition: 'transform 0.4s ease'
                  }}
                />
              </GlassCard>
            </div>

            {/* Layer 2: Left (Dataset Library) */}
            <div 
              style={getHeroLayerStyle('left', '240px', '180px', '-6deg', 380, 214, 25, 0)}
              onMouseEnter={() => setHoveredLayer('left')}
              onMouseLeave={() => setHoveredLayer(null)}
            >
              <GlassCard style={{ width: '100%', height: '100%', padding: '4px', overflow: 'hidden', position: 'relative', border: '1px solid rgba(255, 255, 255, 0.08)', boxShadow: '0 20px 40px rgba(0, 0, 0, 0.5)' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '1px', background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.25), transparent)', zIndex: 5 }} />
                <img 
                  src="/assets/showcase_metadata.png" 
                  alt="Dataset Library" 
                  style={{ 
                    width: '100%', 
                    height: '100%', 
                    objectFit: 'cover',
                    transform: hoveredLayer === 'left' ? 'scale(1.05)' : 'scale(1)',
                    transition: 'transform 0.4s ease'
                  }}
                />
              </GlassCard>
            </div>

            {/* Layer 3: Right (AI Copilot) */}
            <div 
              style={getHeroLayerStyle('right', '140px', '-20px', '4deg', 380, 214, 28, 0.5)}
              onMouseEnter={() => setHoveredLayer('right')}
              onMouseLeave={() => setHoveredLayer(null)}
            >
              <GlassCard style={{ width: '100%', height: '100%', padding: '4px', overflow: 'hidden', position: 'relative', border: '1px solid rgba(255, 255, 255, 0.08)', boxShadow: '0 20px 40px rgba(0, 0, 0, 0.5)' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '1px', background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.25), transparent)', zIndex: 5 }} />
                <img 
                  src="/assets/dashboard.png" 
                  alt="Executive Reports" 
                  style={{ 
                    width: '100%', 
                    height: '100%', 
                    objectFit: 'cover',
                    transform: hoveredLayer === 'right' ? 'scale(1.05)' : 'scale(1)',
                    transition: 'transform 0.4s ease'
                  }}
                />
              </GlassCard>
            </div>

            {/* Layer 4: Front (Overview Dashboard) */}
            <div 
              style={getHeroLayerStyle('front', '110px', '60px', '-2deg', 430, 242, 42, 0)}
              onMouseEnter={() => setHoveredLayer('front')}
              onMouseLeave={() => setHoveredLayer(null)}
            >
              <GlassCard style={{ 
                width: '100%', 
                height: '100%', 
                padding: '4px', 
                overflow: 'hidden', 
                position: 'relative', 
                border: hoveredLayer === 'front' ? '1px solid rgba(173, 198, 255, 0.5)' : '1px solid rgba(173, 198, 255, 0.25)', 
                boxShadow: hoveredLayer === 'front' 
                  ? '0 0 30px rgba(59, 130, 246, 0.35), 0 25px 50px rgba(0, 0, 0, 0.7)' 
                  : '0 0 20px rgba(59, 130, 246, 0.15), 0 25px 50px rgba(0, 0, 0, 0.6)',
                transition: 'border-color 0.25s, box-shadow 0.25s'
              }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '1px', background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.35), transparent)', zIndex: 5 }} />
                <img 
                  src="/assets/dashboard.png" 
                  alt="Overview Dashboard" 
                  style={{ 
                    width: '100%', 
                    height: '100%', 
                    objectFit: 'cover',
                    transform: hoveredLayer === 'front' ? 'scale(1.05)' : 'scale(1)',
                    transition: 'transform 0.4s ease'
                  }}
                />
              </GlassCard>
            </div>

            {/* Atmospheric light glows */}
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: '380px',
              height: '380px',
              backgroundColor: 'var(--primary-color)',
              filter: 'blur(100px)',
              opacity: 0.08,
              zIndex: -1,
              borderRadius: '50%'
            }} />
          </div>
        </section>

        {/* Section: Inside Analyst.AI (Platform Overview Strip) */}
        <section id="platform-overview" style={{
          padding: '80px 40px',
          borderTop: '1px solid var(--border-color)',
          backgroundColor: 'var(--bg-sidebar)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{ maxWidth: 'var(--spacing-container-max)', margin: '0 auto' }}>
            <div style={{ textAlign: 'center', marginBottom: '60px' }}>
              <h2 style={{ fontSize: '2.2rem', fontWeight: 700, margin: '0 0 12px 0', fontFamily: 'Geist', color: 'var(--text-main)' }}>
                Inside Analyst.AI
              </h2>
              <p style={{ maxWidth: '600px', margin: '0 auto', color: 'var(--text-muted)', fontSize: '1.05rem', lineHeight: 1.5 }}>
                Explore the complete intelligence workflow.
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '24px' }}>
              {/* Card 1: Dataset Library */}
              <PlatformCard
                screenshot="/assets/dashboard.png"
                title="Dataset Library"
                explanation="Ingest CSV, JSON, and database streams securely. Review schema auto-detection and data quality scores."
              />

              {/* Card 2: Analysis Workspace */}
              <PlatformCard
                screenshot="/assets/showcase_correlation.png"
                title="Analysis Workspace"
                explanation="Run multi-variable correlation diagnostics, statistical drivers, and anomaly scans to map structural behaviors."
              />

              {/* Card 3: AI Copilot */}
               <PlatformCard
                screenshot="/assets/dashboard.png"
                title="Dataset Library"
                explanation="Converse with your datasets in natural language. Ask questions, build quick segments, and generate prompt-driven insights."
              />

              {/* Card 4: Executive Reports */}
              <PlatformCard
                screenshot="/assets/showcase_reports.png"
                title="Executive Reports"
                explanation="Compile statistical highlights and linear visualizations into executive summaries. Export boardroom-ready PDF dossiers in one click."
              />
            </div>
          </div>
        </section>

        {/* Feature Grid: Comprehensive Capability Stack */}
        <section id="features" style={{ 
          padding: '100px 40px',
          borderTop: '1px solid var(--border-color)',
          backgroundColor: 'rgba(255,255,255,0.01)'
        }}>
          <div style={{ maxWidth: 'var(--spacing-container-max)', margin: '0 auto' }}>
            <div style={{ textAlign: 'center', marginBottom: '60px' }}>
              <h2 style={{ fontSize: '2.2rem', fontWeight: 700, margin: '0 0 16px 0', fontFamily: 'Geist' }}>
                Comprehensive Capability Stack
              </h2>
              <p style={{ maxWidth: '600px', margin: '0 auto', color: 'var(--text-muted)', fontSize: '1rem' }}>
                Built for speed and precision. Leverage our modular architecture to dissect complex data environments in seconds.
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px' }}>
              {/* Feature 1 */}
              <GlassCard style={{ padding: '24px' }} className="landing-feature-card">
                <div style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '8px',
                  backgroundColor: 'var(--bg-input)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--primary-color)',
                  marginBottom: '24px'
                }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '24px' }}>upload_file</span>
                </div>
                <h4 style={{ fontSize: '1.2rem', marginBottom: '8px', fontFamily: 'Geist' }}>Dataset Analysis</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: 0 }}>
                  Multi-format CSV, JSON, and Parquet loader with automated schema auto-detection.
                </p>
              </GlassCard>

              {/* Feature 2 */}
              <GlassCard style={{ padding: '24px' }} className="landing-feature-card">
                <div style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '8px',
                  backgroundColor: 'var(--bg-input)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--secondary-color)',
                  marginBottom: '24px'
                }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '24px' }}>smart_toy</span>
                </div>
                <h4 style={{ fontSize: '1.2rem', marginBottom: '8px', fontFamily: 'Geist' }}>AI Copilot</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: 0 }}>
                  Conversational context assistant that understands complex corporate business logic.
                </p>
              </GlassCard>

              {/* Feature 3 */}
              <GlassCard style={{ padding: '24px' }} className="landing-feature-card">
                <div style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '8px',
                  backgroundColor: 'var(--bg-input)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--primary-color)',
                  marginBottom: '24px'
                }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '24px' }}>query_stats</span>
                </div>
                <h4 style={{ fontSize: '1.2rem', marginBottom: '8px', fontFamily: 'Geist' }}>Correlation Discovery</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: 0 }}>
                  Automatic statistical mapping to find non-obvious relationships and linear indicators.
                </p>
              </GlassCard>

              {/* Feature 4 */}
              <GlassCard style={{ padding: '24px' }} className="landing-feature-card">
                <div style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '8px',
                  backgroundColor: 'var(--bg-input)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--secondary-color)',
                  marginBottom: '24px'
                }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '24px' }}>cleaning_services</span>
                </div>
                <h4 style={{ fontSize: '1.2rem', marginBottom: '8px', fontFamily: 'Geist' }}>Missing Value Detection</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: 0 }}>
                  Completeness profiling and intelligent validation vector suggestions.
                </p>
              </GlassCard>

              {/* Feature 5 */}
              <GlassCard style={{ padding: '24px' }} className="landing-feature-card">
                <div style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '8px',
                  backgroundColor: 'var(--bg-input)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--primary-color)',
                  marginBottom: '24px'
                }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '24px' }}>picture_as_pdf</span>
                </div>
                <h4 style={{ fontSize: '1.2rem', marginBottom: '8px', fontFamily: 'Geist' }}>Automated Reports</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: 0 }}>
                  Dynamic executive PDF summary generator with auto-captioned statistical results.
                </p>
              </GlassCard>

              {/* Feature 6 */}
              <GlassCard style={{ padding: '24px' }} className="landing-feature-card">
                <div style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '8px',
                  backgroundColor: 'var(--bg-input)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--secondary-color)',
                  marginBottom: '24px'
                }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '24px' }}>dashboard_customize</span>
                </div>
                <h4 style={{ fontSize: '1.2rem', marginBottom: '8px', fontFamily: 'Geist' }}>Executive Dashboards</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: 0 }}>
                  Cognitive KPI overview blocks specifically aligned for high-level enterprise analysis.
                </p>
              </GlassCard>

              {/* Feature 7 */}
              <GlassCard style={{ padding: '24px' }} className="landing-feature-card">
                <div style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '8px',
                  backgroundColor: 'var(--bg-input)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--primary-color)',
                  marginBottom: '24px'
                }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '24px' }}>menu_book</span>
                </div>
                <h4 style={{ fontSize: '1.2rem', marginBottom: '8px', fontFamily: 'Geist' }}>Knowledge Base</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: 0 }}>
                  Centralized document RAG search index linking terms to operation manuals.
                </p>
              </GlassCard>

              {/* Feature 8 */}
              <GlassCard style={{ padding: '24px' }} className="landing-feature-card">
                <div style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '8px',
                  backgroundColor: 'var(--bg-input)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--secondary-color)',
                  marginBottom: '24px'
                }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '24px' }}>monitoring</span>
                </div>
                <h4 style={{ fontSize: '1.2rem', marginBottom: '8px', fontFamily: 'Geist' }}>Model Insights</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: 0 }}>
                  Real-time LLM selector and system validation metrics monitoring.
                </p>
              </GlassCard>
            </div>
          </div>
        </section>

        {/* Precision Workflow */}
        <section id="workflow" style={{ 
          padding: '100px 40px',
          borderTop: '1px solid var(--border-color)'
        }}>
          <div style={{ maxWidth: 'var(--spacing-container-max)', margin: '0 auto' }}>
            <div style={{ textAlign: 'center', marginBottom: '80px' }}>
              <h2 style={{ fontSize: '2.2rem', fontWeight: 700, margin: '0 0 16px 0', fontFamily: 'Geist' }}>
                Precision Workflow
              </h2>
              <div style={{ width: '96px', height: '4px', backgroundColor: 'var(--primary-color)', margin: '0 auto', borderRadius: '4px' }}></div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '24px', position: 'relative' }} className="story-responsive">
              {/* Horizontal line for desktop flow */}
              <div style={{
                position: 'absolute',
                top: '40px',
                left: 0,
                width: '100%',
                height: '2px',
                background: 'linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%)',
                opacity: 0.2,
                zIndex: 0
              }} className="landing-workflow-line" />

              {/* Step 1 */}
              <div style={{ display: 'flex', flexDirection: 'column', itemsAlign: 'center', textAlign: 'center', gap: '24px', zIndex: 1 }} className="workflow-item-responsive">
                <div style={{
                  width: '80px',
                  height: '80px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--bg-sidebar)',
                  border: '2px solid var(--primary-color)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 'bold',
                  fontSize: '1.5rem',
                  color: 'var(--primary-color)',
                  margin: '0 auto'
                }}>
                  01
                </div>
                <div>
                  <h4 style={{ fontSize: '1.2rem', margin: '0 0 8px 0', fontFamily: 'Geist' }}>Upload Dataset</h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.45, margin: 0 }}>
                    Connect your secure workspace or drag-and-drop CSV datasets directly.
                  </p>
                </div>
              </div>

              {/* Step 2 */}
              <div style={{ display: 'flex', flexDirection: 'column', itemsAlign: 'center', textAlign: 'center', gap: '24px', zIndex: 1 }} className="workflow-item-responsive">
                <div style={{
                  width: '80px',
                  height: '80px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--bg-sidebar)',
                  border: '2px solid var(--primary-color)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 'bold',
                  fontSize: '1.5rem',
                  color: 'var(--text-muted)',
                  margin: '0 auto'
                }}>
                  02
                </div>
                <div>
                  <h4 style={{ fontSize: '1.2rem', margin: '0 0 8px 0', fontFamily: 'Geist' }}>Analyze Data</h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.45, margin: 0 }}>
                    The AI engine automatically profiles schemas, structures, and missing factors.
                  </p>
                </div>
              </div>

              {/* Step 3 */}
              <div style={{ display: 'flex', flexDirection: 'column', itemsAlign: 'center', textAlign: 'center', gap: '24px', zIndex: 1 }} className="workflow-item-responsive">
                <div style={{
                  width: '80px',
                  height: '80px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--bg-sidebar)',
                  border: '2px solid var(--primary-color)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 'bold',
                  fontSize: '1.5rem',
                  color: 'var(--text-muted)',
                  margin: '0 auto'
                }}>
                  03
                </div>
                <div>
                  <h4 style={{ fontSize: '1.2rem', margin: '0 0 8px 0', fontFamily: 'Geist' }}>Generate Insights</h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.45, margin: 0 }}>
                    Receive cognitive summaries, drivers, and visual correlation charts.
                  </p>
                </div>
              </div>

              {/* Step 4 */}
              <div style={{ display: 'flex', flexDirection: 'column', itemsAlign: 'center', textAlign: 'center', gap: '24px', zIndex: 1 }} className="workflow-item-responsive">
                <div style={{
                  width: '80px',
                  height: '80px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--bg-sidebar)',
                  border: '2px solid var(--primary-color)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 'bold',
                  fontSize: '1.5rem',
                  color: 'var(--text-muted)',
                  margin: '0 auto'
                }}>
                  04
                </div>
                <div>
                  <h4 style={{ fontSize: '1.2rem', margin: '0 0 8px 0', fontFamily: 'Geist' }}>Export Reports</h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.45, margin: 0 }}>
                    Single-click export of synthesized results to boardroom PDF dossiers.
                  </p>
                </div>
              </div>

            </div>
          </div>
        </section>

        {/* Screenshot Showcase Carousel Gallery */}
        <section id="showcase" style={{ 
          padding: '100px 40px',
          borderTop: '1px solid var(--border-color)',
          backgroundColor: 'var(--bg-sidebar)'
        }}>
          <div style={{ maxWidth: 'var(--spacing-container-max)', margin: '0 auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '48px', flexWrap: 'wrap', gap: '24px' }}>
              <div>
                <h2 style={{ fontSize: '2.2rem', fontWeight: 700, margin: '0 0 16px 0', fontFamily: 'Geist' }}>
                  Modern Interface for Modern Teams
                </h2>
                <p style={{ maxWidth: '600px', margin: 0, color: 'var(--text-muted)', fontSize: '1rem' }}>
                  Every screen is meticulously designed to reduce cognitive load and prioritize data clarity.
                </p>
              </div>

              <div style={{ display: 'flex', gap: '16px' }}>
                <button 
                  onClick={() => handleScroll('prev')}
                  style={{ 
                    width: '48px', 
                    height: '48px', 
                    borderRadius: '50%', 
                    backgroundColor: 'var(--bg-card)', 
                    border: '1px solid var(--border-color)', 
                    color: 'var(--text-main)', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    cursor: 'pointer'
                  }}
                >
                  <span className="material-symbols-outlined">chevron_left</span>
                </button>
                <button 
                  onClick={() => handleScroll('next')}
                  style={{ 
                    width: '48px', 
                    height: '48px', 
                    borderRadius: '50%', 
                    backgroundColor: 'var(--bg-card)', 
                    border: '1px solid var(--border-color)', 
                    color: 'var(--text-main)', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    cursor: 'pointer'
                  }}
                >
                  <span className="material-symbols-outlined">chevron_right</span>
                </button>
              </div>
            </div>

            {/* Horizontal Scrollable Row */}
            <div 
              ref={carouselRef}
              style={{
                display: 'flex',
                gap: '24px',
                overflowX: 'auto',
                paddingBottom: '24px',
                boxSizing: 'border-box',
                scrollBehavior: 'smooth'
              }}
              className="no-scrollbar"
            >
              {/* Showcase 1: Dashboard Overview */}
              <ShowcaseCard src="/assets/showcase_dashboard.png" alt="Dashboard Overview" />

              {/* Showcase 2: Report Generation */}
              <ShowcaseCard src="/assets/showcase_reports.png" alt="Report Generation" />

              {/* Showcase 3: Visual Analytics */}
              <ShowcaseCard src="/assets/showcase_analytics.png" alt="Visual Analytics" />

              {/* Showcase 4: AI Chat Interface */}
              <ShowcaseCard src="/assets/showcase_chat.png" alt="AI Chat Interface" />
            </div>

          </div>
        </section>

        {/* About Section: Why Analyst.AI? */}
        <section id="about" style={{ 
          padding: '100px 40px',
          borderTop: '1px solid var(--border-color)'
        }}>
          <div style={{ maxWidth: 'var(--spacing-container-max)', margin: '0 auto' }}>
            <div style={{ 
              display: 'grid',
              gridTemplateColumns: '1fr 1.1fr',
              gap: '80px',
              alignItems: 'center',
              marginBottom: '56px'
            }} className="hero-responsive">
              
              {/* Left Column: Real Analyst.AI Screen */}
              <div style={{ position: 'relative' }}>
                <WhyImageCard src="/assets/showcase_dashboard.png" alt="Analyst.AI Overview Dashboard" />
              </div>

              {/* Right Column: About details */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
                <div>
                  <h2 style={{ fontSize: '2.2rem', fontWeight: 700, margin: '0 0 16px 0', fontFamily: 'Geist' }}>
                    Why Analyst.AI?
                  </h2>
                  <p style={{ color: 'var(--text-muted)', fontSize: '1rem', lineHeight: 1.5, margin: 0 }}>
                    Analyst.AI simplifies the journey from raw datasets to actionable insights.
                    Upload structured data, explore statistical relationships, uncover hidden patterns, interact with an AI Copilot, and generate executive-ready reports from a single workspace.
                  </p>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                  {/* Dataset Intelligence */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="material-symbols-outlined" style={{ color: 'var(--primary-color)', fontSize: '1.2rem' }}>table_chart</span>
                      <h4 style={{ fontSize: '1.05rem', fontWeight: 600, margin: 0, fontFamily: 'Geist' }}>Dataset Intelligence</h4>
                    </div>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.4, margin: 0 }}>
                      Auto-profile structures, formats, and quality flags dynamically.
                    </p>
                  </div>

                  {/* Exploratory Analysis */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="material-symbols-outlined" style={{ color: 'var(--secondary-color)', fontSize: '1.2rem' }}>insights</span>
                      <h4 style={{ fontSize: '1.05rem', fontWeight: 600, margin: 0, fontFamily: 'Geist' }}>Exploratory Analysis</h4>
                    </div>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.4, margin: 0 }}>
                      Map multi-variable correlations and target drivers instantly.
                    </p>
                  </div>

                  {/* Conversational Insights */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="material-symbols-outlined" style={{ color: 'var(--primary-color)', fontSize: '1.2rem' }}>chat</span>
                      <h4 style={{ fontSize: '1.05rem', fontWeight: 600, margin: 0, fontFamily: 'Geist' }}>Conversational Insights</h4>
                    </div>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.4, margin: 0 }}>
                      Query datasets in plain English with our context-aware AI Copilot.
                    </p>
                  </div>

                  {/* Executive Reports */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="material-symbols-outlined" style={{ color: 'var(--secondary-color)', fontSize: '1.2rem' }}>description</span>
                      <h4 style={{ fontSize: '1.05rem', fontWeight: 600, margin: 0, fontFamily: 'Geist' }}>Executive Reports</h4>
                    </div>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.4, margin: 0 }}>
                      Generate boardroom-grade summaries and PDF dossiers in one click.
                    </p>
                  </div>
                </div>

                <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '24px' }}>
                  <button 
                    onClick={() => onNavigateToAuth('register')}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: 'var(--primary-color)',
                      fontWeight: 600,
                      fontSize: '0.95rem',
                      padding: 0,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}
                  >
                    Get started with Analyst.AI
                    <span className="material-symbols-outlined">arrow_right_alt</span>
                  </button>
                </div>
              </div>

            </div>

            {/* Horizontal Workflow Strip */}
            <div style={{ marginTop: '56px' }}>
              <GlassCard style={{ 
                padding: '24px 32px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '16px',
                flexWrap: 'wrap',
                border: '1px solid var(--border-color)',
                backgroundColor: 'var(--bg-sidebar)'
              }}>
                {/* Step 1 */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{
                    width: '36px',
                    height: '36px',
                    borderRadius: '50%',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    border: '1px solid var(--primary-color)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--primary-color)'
                  }}>
                    <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>upload_file</span>
                  </div>
                  <span style={{ fontWeight: 600, fontSize: '0.95rem', fontFamily: 'Geist' }}>Upload Dataset</span>
                </div>

                <span className="material-symbols-outlined" style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>arrow_right_alt</span>

                {/* Step 2 */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{
                    width: '36px',
                    height: '36px',
                    borderRadius: '50%',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    border: '1px solid var(--secondary-color)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--secondary-color)'
                  }}>
                    <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>analytics</span>
                  </div>
                  <span style={{ fontWeight: 600, fontSize: '0.95rem', fontFamily: 'Geist' }}>Analyze Data</span>
                </div>

                <span className="material-symbols-outlined" style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>arrow_right_alt</span>

                {/* Step 3 */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{
                    width: '36px',
                    height: '36px',
                    borderRadius: '50%',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    border: '1px solid var(--primary-color)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--primary-color)'
                  }}>
                    <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>chat</span>
                  </div>
                  <span style={{ fontWeight: 600, fontSize: '0.95rem', fontFamily: 'Geist' }}>Ask AI Questions</span>
                </div>

                <span className="material-symbols-outlined" style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>arrow_right_alt</span>

                {/* Step 4 */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{
                    width: '36px',
                    height: '36px',
                    borderRadius: '50%',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    border: '1px solid var(--secondary-color)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--secondary-color)'
                  }}>
                    <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>description</span>
                  </div>
                  <span style={{ fontWeight: 600, fontSize: '0.95rem', fontFamily: 'Geist' }}>Generate Reports</span>
                </div>
              </GlassCard>
            </div>

          </div>
        </section>

        {/* CTA section */}
        <section style={{ 
          padding: '0 40px 100px 40px',
          maxWidth: 'var(--spacing-container-max)',
          margin: '0 auto'
        }}>
          <GlassCard style={{ padding: '60px 40px', textAlign: 'center', display: 'flex', flexDirection: 'column', gap: '24px', alignItems: 'center' }}>
            <h2 style={{ fontSize: '2.4rem', fontWeight: 800, margin: 0, fontFamily: 'Geist' }}>
              Ready to democratize your data?
            </h2>
            <p style={{ maxWidth: '640px', color: 'var(--text-muted)', fontSize: '1.05rem', margin: 0 }}>
              Join over 200+ enterprise teams using Analyst.AI to drive growth and operational efficiency.
            </p>
            <div style={{ display: 'flex', gap: '16px', marginTop: '8px' }}>
              <button 
                onClick={() => onNavigateToAuth('register')}
                style={{ 
                  backgroundColor: 'var(--primary-color)', 
                  color: 'white', 
                  fontSize: '0.95rem', 
                  padding: '12px 28px', 
                  borderRadius: '6px', 
                  fontWeight: 600,
                  boxShadow: '0 4px 14px rgba(59, 130, 246, 0.3)'
                }}
              >
                Get Started Now
              </button>
              <button 
                onClick={() => onNavigateToAuth('login')}
                style={{ 
                  border: '1px solid var(--border-color)', 
                  color: 'var(--text-main)', 
                  fontSize: '0.95rem', 
                  padding: '12px 28px', 
                  borderRadius: '6px',
                  fontWeight: 600,
                  backgroundColor: 'transparent'
                }}
              >
                Book a Demo
              </button>
            </div>
          </GlassCard>
        </section>

        {/* Footer */}
        <footer style={{ 
          borderTop: '1px solid var(--border-color)', 
          padding: '40px 40px',
          backgroundColor: 'var(--bg-sidebar)'
        }}>
          <div style={{ 
            maxWidth: 'var(--spacing-container-max)', 
            margin: '0 auto',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            flexWrap: 'wrap',
            gap: '40px'
          }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '24px', height: '24px', borderRadius: '4px', backgroundColor: 'var(--primary-color)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff' }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '14px', fontVariationSettings: "'FILL' 1" }}>analytics</span>
                </div>
                <span style={{ fontSize: '1.1rem', fontWeight: 'bold', color: 'var(--text-main)', fontFamily: 'Geist' }}>Analyst.AI</span>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0, maxWidth: '280px', lineHeight: 1.5 }}>
                © 2026 Analyst.AI Enterprise. All rights reserved. Precision data intelligence for modern business.
              </p>
            </div>

            <div style={{ display: 'flex', gap: '60px', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>Product</span>
                <a href="#features" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textDecoration: 'none' }}>Features</a>
                <a href="#workflow" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textDecoration: 'none' }}>Workflow</a>
                <a href="#about" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textDecoration: 'none' }}>About</a>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>Resources</span>
                <a href="#docs" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textDecoration: 'none' }}>Documentation</a>
                <a href="#api" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textDecoration: 'none' }}>API Docs</a>
                <a href="#support" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textDecoration: 'none' }}>Support</a>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>Legal</span>
                <a href="#privacy" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textDecoration: 'none' }}>Privacy Policy</a>
                <a href="#terms" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textDecoration: 'none' }}>Terms of Service</a>
                <a href="#security" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textDecoration: 'none' }}>Security</a>
              </div>
            </div>
          </div>
        </footer>

      </div>
    </FadeIn>
  );
}

function PlatformCard({ screenshot, title, explanation }) {
  const [isHovered, setIsHovered] = useState(false);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    setTilt({ x: x * 6, y: y * -6 }); // subtle 3D tilt
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    setTilt({ x: 0, y: 0 });
  };

  return (
    <div 
      onMouseEnter={() => setIsHovered(true)}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        background: 'rgba(16, 32, 52, 0.4)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '12px',
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        transform: isHovered 
          ? `translateY(-8px) scale(1.03) rotateX(${tilt.y}deg) rotateY(${tilt.x}deg)` 
          : 'translateY(0px) scale(1) rotateX(0deg) rotateY(0deg)',
        boxShadow: isHovered 
          ? '0 0 25px rgba(59, 130, 246, 0.25), 0 20px 40px -10px rgba(0, 0, 0, 0.6)' 
          : '0 4px 20px rgba(0, 0, 0, 0.3)',
        borderColor: isHovered ? 'rgba(173, 198, 255, 0.45)' : 'rgba(255, 255, 255, 0.1)',
        transition: 'transform 0.2s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.3s, border-color 0.3s',
        position: 'relative',
        overflow: 'hidden',
        transformStyle: 'preserve-3d',
        perspective: '1000px'
      }}
    >
      {/* Top light reflection shine */}
      <div style={{ 
        position: 'absolute', 
        top: 0, 
        left: 0, 
        right: 0, 
        height: '1px', 
        background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent)' 
      }} />

      {/* Screenshot Frame */}
      <div style={{ 
        width: '100%', 
        aspectRatio: '16/9', 
        borderRadius: '8px', 
        overflow: 'hidden', 
        border: '1px solid rgba(255, 255, 255, 0.05)',
        backgroundColor: '#000f21'
      }}>
        <img 
          src={screenshot} 
          alt={title} 
          style={{ 
            width: '100%', 
            height: '100%', 
            objectFit: 'cover', 
            opacity: isHovered ? 0.95 : 0.8,
            filter: isHovered ? 'none' : 'grayscale(0.15)',
            transform: isHovered ? 'scale(1.05)' : 'scale(1)',
            transition: 'opacity 0.3s, filter 0.3s, transform 0.4s ease' 
          }} 
        />
      </div>

      {/* Content */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <h4 style={{ 
          fontSize: '1.2rem', 
          fontWeight: 600, 
          margin: 0, 
          fontFamily: 'Geist', 
          color: isHovered ? 'var(--primary-color)' : 'var(--text-main)',
          transition: 'color 0.3s'
        }}>
          {title}
        </h4>
        <p style={{ 
          fontSize: '0.88rem', 
          color: 'var(--text-muted)', 
          lineHeight: 1.5, 
          margin: 0 
        }}>
          {explanation}
        </p>
      </div>
    </div>
  );
}

function ShowcaseCard({ src, alt }) {
  const [isHovered, setIsHovered] = useState(false);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    setTilt({ x: x * 6, y: y * -6 }); // subtle 3D tilt
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    setTilt({ x: 0, y: 0 });
  };

  return (
    <div 
      onMouseEnter={() => setIsHovered(true)}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ 
        minWidth: '600px', 
        width: '600px', 
        aspectRatio: '16/9', 
        borderRadius: '12px', 
        overflow: 'hidden', 
        flexShrink: 0,
        perspective: '1000px'
      }}
    >
      <GlassCard 
        style={{ 
          width: '100%', 
          height: '100%', 
          padding: '0', 
          border: '1px solid var(--border-color)',
          borderColor: isHovered ? 'rgba(173, 198, 255, 0.4)' : 'var(--border-color)',
          transform: isHovered ? `scale(1.02) rotateX(${tilt.y}deg) rotateY(${tilt.x}deg)` : 'scale(1) rotateX(0deg) rotateY(0deg)',
          boxShadow: isHovered ? '0 15px 30px rgba(0, 0, 0, 0.5), 0 0 20px rgba(59, 130, 246, 0.15)' : 'none',
          transition: 'transform 0.2s cubic-bezier(0.25, 0.8, 0.25, 1), border-color 0.3s, box-shadow 0.3s'
        }}
      >
        <img 
          src={src} 
          alt={alt} 
          style={{ 
            width: '100%', 
            height: '100%', 
            objectFit: 'cover',
            transform: isHovered ? 'scale(1.04)' : 'scale(1)',
            transition: 'transform 0.4s ease'
          }} 
        />
      </GlassCard>
    </div>
  );
}

function WhyImageCard({ src, alt }) {
  const [isHovered, setIsHovered] = useState(false);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    setTilt({ x: x * 6, y: y * -6 }); // subtle 3D tilt
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    setTilt({ x: 0, y: 0 });
  };

  return (
    <div 
      onMouseEnter={() => setIsHovered(true)}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ 
        width: '100%', 
        aspectRatio: '16/9', 
        borderRadius: '12px', 
        overflow: 'hidden',
        perspective: '1000px'
      }}
    >
      <GlassCard 
        style={{ 
          width: '100%', 
          height: '100%', 
          padding: '0', 
          border: '1px solid var(--border-color)',
          borderColor: isHovered ? 'rgba(173, 198, 255, 0.4)' : 'var(--border-color)',
          transform: isHovered ? `scale(1.02) rotateX(${tilt.y}deg) rotateY(${tilt.x}deg)` : 'scale(1) rotateX(0deg) rotateY(0deg)',
          boxShadow: isHovered ? '0 15px 30px rgba(0, 0, 0, 0.5), 0 0 20px rgba(59, 130, 246, 0.15)' : 'none',
          transition: 'transform 0.2s cubic-bezier(0.25, 0.8, 0.25, 1), border-color 0.3s, box-shadow 0.3s'
        }}
      >
        <img 
          src={src} 
          alt={alt} 
          style={{ 
            width: '100%', 
            height: '100%', 
            objectFit: 'cover',
            transform: isHovered ? 'scale(1.04)' : 'scale(1)',
            transition: 'transform 0.4s ease'
          }} 
        />
      </GlassCard>
    </div>
  );
}
