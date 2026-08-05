import React from 'react';

const Sidebar = ({ activeTab, onTabChange, domainProfile }) => {
    // Menu items mapping the Stitch design navigation labels to the existing functional tabs
    const menuItems = [
        { id: 'dashboard', label: 'Dashboard', icon: 'dashboard' },
        { id: 'copilot',   label: 'AI Copilot', icon: 'chat' },
        { id: 'logs',      label: 'Dataset Library', icon: 'query_stats' },
        { id: 'reports',   label: 'Advanced Insights', icon: 'analytics' },
        { id: 'manuals',   label: 'Reference Documents', icon: 'description' },
        { id: 'settings',  label: 'Settings', icon: 'settings' },
    ];

    return (
        <aside 
            className="sidebar"
            style={{
                width: '260px',
                backgroundColor: 'var(--bg-sidebar)',
                borderRight: '1px solid var(--border-color)',
                display: 'flex',
                flexDirection: 'column',
                height: '100vh',
                boxSizing: 'border-box',
                padding: 'var(--spacing-gutter-md) 1rem',
                transition: 'background-color 0.3s ease, border-color 0.3s ease'
            }}
        >
            {/* Header logo / branding */}
            <div style={{ padding: '0 0.75rem 2rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div 
                    style={{ 
                        width: '32px', 
                        height: '32px', 
                        borderRadius: '4px', 
                        backgroundColor: 'var(--primary-color)', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        color: 'white',
                        boxShadow: '0 0 10px rgba(59, 130, 246, 0.4)'
                    }}
                >
                    <span className="material-symbols-outlined" style={{ fontSize: '20px', fontVariationSettings: "'FILL' 1" }}>
                        analytics
                    </span>
                </div>
                <div>
                    <h1 
                        style={{ 
                            fontSize: '1.25rem', 
                            fontWeight: 'bold', 
                            color: 'var(--primary-color)', 
                            margin: 0,
                            lineHeight: 1.1,
                            fontFamily: 'Geist'
                        }}
                    >
                        Analyst.AI
                    </h1>
                    <div 
                        style={{ 
                            fontSize: '9px', 
                            letterSpacing: '0.12em', 
                            color: 'var(--text-muted)', 
                            textTransform: 'uppercase',
                            fontWeight: 600
                        }}
                    >
                        Enterprise Suite
                    </div>
                </div>
            </div>

            {/* Navigation links */}
            <nav style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {menuItems.map(item => {
                    const isActive = activeTab === item.id;
                    return (
                        <div
                            key={item.id}
                            className={`nav-item ${isActive ? 'active' : ''}`}
                            onClick={() => onTabChange(item.id)}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '12px',
                                padding: '10px 14px',
                                borderRadius: '8px',
                                cursor: 'pointer',
                                transition: 'all 0.2s ease',
                                color: isActive ? 'var(--primary-color)' : 'var(--text-muted)',
                                backgroundColor: isActive ? 'rgba(59, 130, 246, 0.08)' : 'transparent',
                                borderRight: isActive ? '3px solid var(--primary-color)' : 'none',
                                fontWeight: isActive ? 600 : 500
                            }}
                        >
                            <span 
                                className="material-symbols-outlined" 
                                style={{ 
                                    fontSize: '20px',
                                    fontVariationSettings: isActive ? "'FILL' 1" : "'FILL' 0"
                                }}
                            >
                                {item.icon}
                            </span>
                            <span style={{ fontSize: '0.9rem', fontFamily: 'Inter' }}>{item.label}</span>
                        </div>
                    );
                })}
            </nav>

            {/* Action panel & help links */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: 'auto', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                {/* New Analysis Trigger Button */}
                <button
                    className="primary-btn"
                    onClick={() => onTabChange('analysis')}
                    style={{
                        width: '100%',
                        padding: '10px',
                        borderRadius: '8px',
                        backgroundColor: 'var(--primary-color)',
                        color: 'var(--text-on-primary, white)',
                        border: 'none',
                        fontWeight: 600,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '6px',
                        marginBottom: '8px',
                        boxShadow: '0 4px 10px rgba(59, 130, 246, 0.2)'
                    }}
                >
                    <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>add</span>
                    New Analysis
                </button>

                <div 
                    onClick={() => onTabChange('manuals')}
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        padding: '6px 12px',
                        cursor: 'pointer',
                        color: 'var(--text-muted)',
                        fontSize: '0.85rem'
                    }}
                >
                    <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>description</span>
                    <span>Documentation</span>
                </div>

                {/* System Status info */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    <span style={{ color: '#10b981', display: 'inline-block', fontSize: '12px' }}>●</span>
                    <span>System Core: Operational</span>
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;
