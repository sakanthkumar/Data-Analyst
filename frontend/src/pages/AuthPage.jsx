import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../components/GlassCard';
import AmbientAIOrb3D from '../three/AmbientAIOrb3D';
import { useTheme } from '../ThemeContext';

export default function AuthPage({ defaultState = 'login', onAuthSuccess, onNavigateHome }) {
  const { theme, toggleTheme } = useTheme();
  const [authState, setAuthState] = useState(defaultState); // 'login' | 'register'
  const [direction, setDirection] = useState(1); // 1 for slide-right, -1 for slide-left
  
  // Credentials States
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [role, setRole] = useState('Business Analyst');
  const [confirmPassword, setConfirmPassword] = useState('');

  const toggleState = () => {
    if (authState === 'login') {
      setDirection(1);
      setAuthState('register');
    } else {
      setDirection(-1);
      setAuthState('login');
    }
  };

  const handleLoginSubmit = (e) => {
    e.preventDefault();
    if (!email || !password) {
      alert('Please enter your credentials.');
      return;
    }
    // Perform mock authentication, saving user session
    const mockUser = {
      name: email.split('@')[0].toUpperCase(),
      email: email,
      role: 'Enterprise Administrator'
    };
    onAuthSuccess(mockUser);
  };

  const handleRegisterSubmit = (e) => {
    e.preventDefault();
    if (!name || !email || !password || !confirmPassword) {
      alert('Please fill all registration fields.');
      return;
    }
    if (password !== confirmPassword) {
      alert('Passwords do not match.');
      return;
    }
    // Register success, save user session
    const mockUser = {
      name: name,
      email: email,
      role: role
    };
    onAuthSuccess(mockUser);
  };

  // Slide Animation configurations
  const slideVariants = {
    enter: (dir) => ({
      x: dir > 0 ? 100 : -100,
      opacity: 0
    }),
    center: {
      x: 0,
      opacity: 1,
      transition: {
        x: { type: 'spring', stiffness: 160, damping: 18 },
        opacity: { duration: 0.25 }
      }
    },
    exit: (dir) => ({
      x: dir < 0 ? 100 : -100,
      opacity: 0,
      transition: {
        x: { type: 'spring', stiffness: 160, damping: 18 },
        opacity: { duration: 0.2 }
      }
    })
  };

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '1.1fr 0.9fr',
      minHeight: '100vh',
      backgroundColor: 'var(--bg-body)',
      color: 'var(--text-main)',
      fontFamily: 'Inter, sans-serif'
    }} className="auth-page-container hero-responsive">
      
      {/* LEFT COLUMN: BRANDING, 3D ORB, METRICS, PREVIEWS */}
      <div style={{
        backgroundColor: 'var(--bg-sidebar)',
        borderRight: '1px solid var(--border-color)',
        padding: '40px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Top brand indicator */}
        <div 
          onClick={onNavigateHome}
          style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', zIndex: 10 }}
        >
          <div style={{ 
            width: '28px', 
            height: '28px', 
            borderRadius: '4px', 
            backgroundColor: 'var(--primary-color)', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            color: 'white'
          }}>
            <span className="material-symbols-outlined" style={{ fontSize: '18px', fontVariationSettings: "'FILL' 1" }}>
              analytics
            </span>
          </div>
          <div>
            <h1 style={{ fontSize: '1.1rem', fontWeight: 'bold', color: 'var(--text-main)', margin: 0, fontFamily: 'Geist' }}>
              Analyst.AI
            </h1>
            <span style={{ fontSize: '8px', letterSpacing: '0.1em', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
              Enterprise Platform
            </span>
          </div>
        </div>

        {/* Floating Previews & 3D Orb Parallax Panel */}
        <div style={{ 
          position: 'relative', 
          height: '420px', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          margin: '20px 0'
        }}>
          {/* 3D Ambient AI Orb in center */}
          <div style={{ 
            position: 'absolute', 
            width: '260px', 
            height: '260px', 
            zIndex: 1, 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center' 
          }}>
            <AmbientAIOrb3D height={250} isThinking={false} />
          </div>

          {/* Floating Card 1: Overview Preview */}
          <motion.div 
            style={{
              position: 'absolute',
              top: '10px',
              left: '20px',
              zIndex: 3,
              width: '180px'
            }}
            animate={{ y: [0, -6, 0] }}
            transition={{ repeat: Infinity, duration: 6, ease: "easeInOut" }}
          >
            <GlassCard style={{ padding: '12px', fontSize: '0.75rem', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', color: 'var(--primary-color)' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>query_stats</span>
                <span>Active</span>
              </div>
              <strong style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-main)' }}>Analysis Workspace</strong>
              <span style={{ color: 'var(--text-muted)' }}>7-tab deep analytics</span>
            </GlassCard>
          </motion.div>

          {/* Floating Card 2: Dataset Library */}
          <motion.div 
            style={{
              position: 'absolute',
              top: '40px',
              right: '20px',
              zIndex: 3,
              width: '180px'
            }}
            animate={{ y: [0, -8, 0] }}
            transition={{ repeat: Infinity, duration: 5, ease: "easeInOut", delay: 0.5 }}
          >
            <GlassCard style={{ padding: '12px', fontSize: '0.75rem', border: '1px solid rgba(139, 92, 246, 0.2)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', color: 'var(--secondary-color)' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>database</span>
                <span>Ready</span>
              </div>
              <strong style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-main)' }}>Dataset Library</strong>
              <span style={{ color: 'var(--text-muted)' }}>Upload CSV, XLSX, JSON</span>
            </GlassCard>
          </motion.div>

          {/* Floating Card 3: AI Copilot Message bubble */}
          <motion.div 
            style={{
              position: 'absolute',
              bottom: '40px',
              left: '10px',
              zIndex: 3,
              width: '190px'
            }}
            animate={{ y: [0, -5, 0] }}
            transition={{ repeat: Infinity, duration: 4.5, ease: "easeInOut", delay: 1 }}
          >
            <GlassCard style={{ padding: '12px', fontSize: '0.75rem', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
              <div style={{ display: 'flex', gap: '6px', alignItems: 'center', marginBottom: '6px' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '14px', color: 'var(--accent-color)' }}>auto_awesome</span>
                <span style={{ fontWeight: 600, color: 'var(--accent-color)' }}>Copilot Response</span>
              </div>
              <span style={{ color: 'var(--text-main)', fontSize: '0.75rem', lineHeight: 1.3 }}>
                "Calculated correlation: 0.82 with target failure status."
              </span>
            </GlassCard>
          </motion.div>

          {/* Floating Card 4: Analysis Workspace */}
          <motion.div 
            style={{
              position: 'absolute',
              bottom: '20px',
              right: '30px',
              zIndex: 3,
              width: '180px'
            }}
            animate={{ y: [0, -7, 0] }}
            transition={{ repeat: Infinity, duration: 5.5, ease: "easeInOut", delay: 1.5 }}
          >
            <GlassCard style={{ padding: '12px', fontSize: '0.75rem', border: '1px solid rgba(245, 158, 11, 0.2)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', color: 'var(--warning-color)' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>checklist</span>
                <span>AI Report</span>
              </div>
              <strong style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-main)' }}>Executive Insights</strong>
              <span style={{ color: 'var(--text-muted)' }}>Auto-generated PDF</span>
            </GlassCard>
          </motion.div>

        </div>

        {/* Bottom indicators: Metrics & Trust */}
        <div style={{ zIndex: 10 }}>
          {/* Trust Indicators bar */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            borderBottom: '1px solid var(--border-color)',
            paddingBottom: '16px',
            marginBottom: '16px',
            fontSize: '0.75rem',
            color: 'var(--text-muted)',
            flexWrap: 'wrap',
            gap: '12px'
          }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span className="material-symbols-outlined" style={{ fontSize: '14px', color: 'var(--accent-color)' }}>verified</span>
              Secure Uploads
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span className="material-symbols-outlined" style={{ fontSize: '14px', color: 'var(--accent-color)' }}>lock</span>
              AES-256 Encryption
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span className="material-symbols-outlined" style={{ fontSize: '14px', color: 'var(--accent-color)' }}>admin_panel_settings</span>
              RBAC Governance
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span className="material-symbols-outlined" style={{ fontSize: '14px', color: 'var(--accent-color)' }}>receipt_long</span>
              Audit Logging
            </span>
          </div>

          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Local · No cloud sync
            </span>
            <button 
              onClick={onNavigateHome}
              style={{ background: 'transparent', color: 'var(--primary-color)', fontSize: '0.8rem', border: 'none', padding: 0 }}
            >
              ← Back to homepage
            </button>
          </div>
        </div>

      </div>

      {/* RIGHT COLUMN: CREDENTIALS GLASS CARD FORM */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px',
        position: 'relative'
      }}>
        {/* Toggle Theme buttons absolute on top right */}
        <div style={{ position: 'absolute', top: '30px', right: '40px', display: 'flex', gap: '16px' }}>
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
        </div>

        {/* Sliding Card Frame */}
        <div style={{ width: '100%', maxWidth: '440px', overflow: 'hidden' }}>
          <AnimatePresence initial={false} custom={direction} mode="wait">
            {authState === 'login' ? (
              <motion.div
                key="login"
                custom={direction}
                variants={slideVariants}
                initial="enter"
                animate="center"
                exit="exit"
              >
                <GlassCard style={{ 
                  padding: '40px', 
                  border: '1.5px solid var(--border-color)',
                  boxShadow: 'var(--shadow-md)'
                }}>
                  <h3 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '6px', fontFamily: 'Geist' }}>
                    Welcome Back
                  </h3>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '24px' }}>
                    Enter credentials to access your enterprise workspace logs.
                  </p>

                  <form onSubmit={handleLoginSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div>
                      <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
                        Work Email
                      </label>
                      <input 
                        type="email"
                        required
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="you@example.com"
                        style={{
                          backgroundColor: 'var(--bg-input)',
                          border: '1px solid var(--border-color)',
                          color: 'var(--text-main)',
                          borderRadius: '6px',
                          padding: '10px 12px',
                          fontSize: '0.9rem',
                          outline: 'none',
                          width: '100%',
                          boxSizing: 'border-box'
                        }}
                      />
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                        <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                          Password
                        </label>
                        <a href="#forgot" style={{ fontSize: '0.75rem', color: 'var(--primary-color)', textDecoration: 'none' }}>Forgot?</a>
                      </div>
                      <input 
                        type="password"
                        required
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Enter your password"
                        style={{
                          backgroundColor: 'var(--bg-input)',
                          border: '1px solid var(--border-color)',
                          color: 'var(--text-main)',
                          borderRadius: '6px',
                          padding: '10px 12px',
                          fontSize: '0.9rem',
                          outline: 'none',
                          width: '100%',
                          boxSizing: 'border-box'
                        }}
                      />
                    </div>

                    <button 
                      type="submit"
                      style={{
                        backgroundColor: 'var(--primary-color)',
                        color: 'white',
                        fontWeight: 600,
                        fontSize: '0.9rem',
                        padding: '12px',
                        borderRadius: '6px',
                        marginTop: '8px',
                        border: 'none',
                        cursor: 'pointer'
                      }}
                    >
                      Authenticate Credentials
                    </button>
                  </form>

                  <div style={{ display: 'flex', alignItems: 'center', margin: '24px 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    <div style={{ flex: 1, height: '1px', backgroundColor: 'var(--border-color)' }}></div>
                    <span style={{ padding: '0 10px' }}>OR CONTINUE AS GUEST</span>
                    <div style={{ flex: 1, height: '1px', backgroundColor: 'var(--border-color)' }}></div>
                  </div>

                  <button 
                    onClick={() => onAuthSuccess({ name: 'Guest', email: 'guest@local', role: 'Viewer' })}
                    style={{
                      width: '100%',
                      backgroundColor: 'transparent',
                      border: '1px solid var(--border-color)',
                      color: 'var(--text-main)',
                      fontSize: '0.85rem',
                      padding: '10px',
                      borderRadius: '6px',
                      fontWeight: 500,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px',
                      cursor: 'pointer'
                    }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>person</span>
                    Continue without account
                  </button>

                  <div style={{ marginTop: '24px', textAlign: 'center', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>New to Analyst.AI? </span>
                    <button 
                      onClick={toggleState}
                      style={{ background: 'transparent', color: 'var(--primary-color)', border: 'none', padding: 0, fontWeight: 600 }}
                    >
                      Create enterprise account
                    </button>
                  </div>
                </GlassCard>
              </motion.div>
            ) : (
              <motion.div
                key="register"
                custom={direction}
                variants={slideVariants}
                initial="enter"
                animate="center"
                exit="exit"
              >
                <GlassCard style={{ 
                  padding: '40px', 
                  border: '1.5px solid var(--border-color)',
                  boxShadow: 'var(--shadow-md)'
                }}>
                  <h3 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '6px', fontFamily: 'Geist' }}>
                    Create Account
                  </h3>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '24px' }}>
                    Create your profile to start uploading and analyzing datasets.
                  </p>

                  <form onSubmit={handleRegisterSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div>
                      <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
                        Full Name
                      </label>
                      <input 
                        type="text"
                        required
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="Your full name"
                        style={{
                          backgroundColor: 'var(--bg-input)',
                          border: '1px solid var(--border-color)',
                          color: 'var(--text-main)',
                          borderRadius: '6px',
                          padding: '10px 12px',
                          fontSize: '0.9rem',
                          outline: 'none',
                          width: '100%',
                          boxSizing: 'border-box'
                        }}
                      />
                    </div>

                    <div>
                      <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
                        Work Email
                      </label>
                      <input 
                        type="email"
                        required
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="your@email.com"
                        style={{
                          backgroundColor: 'var(--bg-input)',
                          border: '1px solid var(--border-color)',
                          color: 'var(--text-main)',
                          borderRadius: '6px',
                          padding: '10px 12px',
                          fontSize: '0.9rem',
                          outline: 'none',
                          width: '100%',
                          boxSizing: 'border-box'
                        }}
                      />
                    </div>

                    <div>
                      <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
                        Organizational Role
                      </label>
                      <select 
                        value={role}
                        onChange={(e) => setRole(e.target.value)}
                        style={{
                          backgroundColor: 'var(--bg-input)',
                          border: '1px solid var(--border-color)',
                          color: 'var(--text-main)',
                          borderRadius: '6px',
                          padding: '10px 12px',
                          fontSize: '0.9rem',
                          outline: 'none',
                          width: '100%',
                          boxSizing: 'border-box',
                          fontFamily: 'Inter'
                        }}
                      >
                        <option value="Business Analyst">Business Analyst</option>
                        <option value="Data Scientist">Data Scientist</option>
                        <option value="Research Team Member">Research Team Member</option>
                        <option value="Enterprise Leader">Enterprise Leader</option>
                      </select>
                    </div>

                    <div>
                      <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
                        Password
                      </label>
                      <input 
                        type="password"
                        required
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Create a password"
                        style={{
                          backgroundColor: 'var(--bg-input)',
                          border: '1px solid var(--border-color)',
                          color: 'var(--text-main)',
                          borderRadius: '6px',
                          padding: '10px 12px',
                          fontSize: '0.9rem',
                          outline: 'none',
                          width: '100%',
                          boxSizing: 'border-box'
                        }}
                      />
                    </div>

                    <div>
                      <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
                        Confirm Password
                      </label>
                      <input 
                        type="password"
                        required
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        placeholder="Repeat your password"
                        style={{
                          backgroundColor: 'var(--bg-input)',
                          border: '1px solid var(--border-color)',
                          color: 'var(--text-main)',
                          borderRadius: '6px',
                          padding: '10px 12px',
                          fontSize: '0.9rem',
                          outline: 'none',
                          width: '100%',
                          boxSizing: 'border-box'
                        }}
                      />
                    </div>

                    <button 
                      type="submit"
                      style={{
                        backgroundColor: 'var(--primary-color)',
                        color: 'white',
                        fontWeight: 600,
                        fontSize: '0.9rem',
                        padding: '12px',
                        borderRadius: '6px',
                        marginTop: '8px',
                        border: 'none',
                        cursor: 'pointer'
                      }}
                    >
                      Initialize Account
                    </button>
                  </form>

                  <div style={{ marginTop: '24px', textAlign: 'center', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Already registered? </span>
                    <button 
                      onClick={toggleState}
                      style={{ background: 'transparent', color: 'var(--primary-color)', border: 'none', padding: 0, fontWeight: 600 }}
                    >
                      Sign in instead
                    </button>
                  </div>
                </GlassCard>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

      </div>

    </div>
  );
}
