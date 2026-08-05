import React, { useEffect, useState } from 'react';

export default function LoadingAnalysis({ apiFinished, onComplete }) {
  const [currentStep, setCurrentStep] = useState(0);
  
  const steps = [
    'Reading schema...',
    'Detecting missing values...',
    'Evaluating correlations...',
    'Generating explanation...',
    'Preparing recommendations...'
  ];

  const stepDelay = 300; // 300ms per step -> 1.5 seconds total minimum duration

  // Auto-advance step index up to the last step (index 4)
  useEffect(() => {
    let timer;
    const tick = () => {
      setCurrentStep(prev => {
        if (prev < steps.length - 1) {
          timer = setTimeout(tick, stepDelay);
          return prev + 1;
        }
        return prev;
      });
    };
    timer = setTimeout(tick, stepDelay);
    return () => clearTimeout(timer);
  }, [steps.length]);

  // Sync completion when BOTH the API has finished and steps have completed
  useEffect(() => {
    if (apiFinished && currentStep === steps.length - 1) {
      const completionTimer = setTimeout(() => {
        onComplete();
      }, 200); // Tiny visual pause to show the final step completed
      return () => clearTimeout(completionTimer);
    }
  }, [apiFinished, currentStep, onComplete, steps.length]);

  return (
    <div 
      className="glass-card fade-in" 
      style={{ 
        padding: '24px', 
        background: 'var(--bg-card)', 
        border: '1px solid var(--border-color)', 
        borderRadius: '12px',
        margin: '16px 0',
        maxWidth: '480px',
        boxShadow: 'var(--shadow-md)'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
        <div className="spinner-small" style={{ width: '20px', height: '20px', margin: 0 }} />
        <h4 
          style={{ 
            margin: 0, 
            fontFamily: 'Geist', 
            fontSize: '1.05rem', 
            color: 'var(--text-main)',
            fontWeight: 600
          }}
        >
          Analyzing dataset...
        </h4>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {steps.map((step, idx) => {
          const isCompleted = idx < currentStep || (idx === currentStep && currentStep === steps.length - 1 && apiFinished);
          const isActive = idx === currentStep && !isCompleted;
          
          return (
            <div 
              key={idx} 
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '12px',
                opacity: isCompleted || isActive ? 1 : 0.4,
                transition: 'opacity 0.3s ease'
              }}
            >
              {isCompleted ? (
                <span 
                  className="material-symbols-outlined" 
                  style={{ 
                    fontSize: '18px', 
                    color: 'var(--accent-color)', // Green check
                    fontWeight: 'bold' 
                  }}
                >
                  check_circle
                </span>
              ) : isActive ? (
                <div 
                  style={{ 
                    width: '8px', 
                    height: '8px', 
                    borderRadius: '50%', 
                    backgroundColor: 'var(--primary-color)',
                    boxShadow: '0 0 8px var(--primary-color)',
                    animation: 'pulse 1.5s infinite alternate',
                    margin: '5px'
                  }} 
                />
              ) : (
                <div 
                  style={{ 
                    width: '6px', 
                    height: '6px', 
                    borderRadius: '50%', 
                    backgroundColor: 'var(--text-muted)',
                    margin: '6px'
                  }} 
                />
              )}
              
              <span 
                style={{ 
                  fontSize: '0.875rem', 
                  color: isActive ? 'var(--text-main)' : 'var(--text-muted)',
                  fontWeight: isActive ? 600 : 400
                }}
              >
                {step}
              </span>
            </div>
          );
        })}
      </div>

      <style>{`
        @keyframes pulse {
          from { transform: scale(0.8); opacity: 0.5; }
          to { transform: scale(1.2); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
