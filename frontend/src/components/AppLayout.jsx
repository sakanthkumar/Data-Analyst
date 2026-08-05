import React from 'react';

export default function AppLayout({ sidebar, header, children }) {
  return (
    <div className="app-layout" style={{ display: 'flex', height: '100vh', overflow: 'hidden', backgroundColor: 'var(--bg-body)' }}>
      {/* Sidebar container */}
      {sidebar}
      
      {/* Main app panel */}
      <div 
        className="main-content-area" 
        style={{ 
          flex: 1, 
          display: 'flex', 
          flexDirection: 'col', 
          overflow: 'hidden',
          position: 'relative'
        }}
      >
        {/* Header container */}
        {header}
        
        {/* Children scroll area */}
        <main 
          className="scrollable-content"
          style={{ 
            flex: 1, 
            overflowY: 'auto', 
            padding: 'var(--spacing-gutter-md)',
            maxWidth: 'var(--spacing-container-max)',
            margin: '0 auto',
            width: '100%',
            boxSizing: 'border-box'
          }}
        >
          {children}
        </main>
      </div>
    </div>
  );
}
