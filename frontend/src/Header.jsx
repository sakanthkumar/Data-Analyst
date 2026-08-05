import React from 'react';
import { useTheme } from './ThemeContext';

const Header = ({ title, domainProfile, user, onLogout }) => {
    const { theme, toggleTheme } = useTheme();

    const getInitials = (fullName) => {
        if (!fullName) return "EA";
        const parts = fullName.trim().split(/\s+/);
        if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
        return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    };

    return (
        <header 
            className="top-bar"
            style={{
                height: '64px',
                backgroundColor: 'var(--bg-header)',
                backdropFilter: 'blur(20px)',
                borderBottom: '1px solid var(--border-color)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0 var(--spacing-gutter-md)',
                transition: 'background-color 0.3s ease, border-color 0.3s ease',
                position: 'sticky',
                top: 0,
                zIndex: 30,
                width: '100%',
                boxSizing: 'border-box'
            }}
        >
            {/* Left side: Page Title and Navigation Sub-links */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <h2 
                        style={{ 
                            fontSize: '1.25rem', 
                            fontWeight: 'bold', 
                            color: 'var(--text-main)', 
                            margin: 0,
                            fontFamily: 'Geist'
                        }}
                    >
                        {title}
                    </h2>
                    {domainProfile && domainProfile.domain && (
                        <div 
                            style={{ 
                                fontSize: '0.75rem', 
                                color: 'var(--accent-color)', 
                                fontWeight: 500,
                                display: 'flex',
                                alignItems: 'center',
                                gap: '4px',
                                marginTop: '2px'
                            }}
                        >
                            <span className="material-symbols-outlined" style={{ fontSize: '12px' }}>language</span>
                            <span>{domainProfile.domain}</span>
                        </div>
                    )}
                </div>
            </div>

            {/* Right side: Search, Actions, Profile */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                {/* Simulated Quick Search */}
                <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                    <span 
                        className="material-symbols-outlined" 
                        style={{ 
                            position: 'absolute', 
                            left: '10px', 
                            fontSize: '18px', 
                            color: 'var(--text-muted)',
                            pointerEvents: 'none'
                        }}
                    >
                        search
                    </span>
                    <input 
                        type="text" 
                        placeholder="Search data..." 
                        style={{
                            backgroundColor: 'var(--bg-input)',
                            border: '1px solid var(--border-color)',
                            color: 'var(--text-main)',
                            borderRadius: '20px',
                            padding: '6px 12px 6px 32px',
                            fontSize: '0.8rem',
                            width: '200px',
                            fontFamily: 'Inter',
                            outline: 'none',
                            transition: 'all 0.2s ease'
                        }}
                    />
                </div>

                {/* Quick actions row */}
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button 
                        className="icon-btn" 
                        style={{
                            padding: '6px',
                            background: 'transparent',
                            border: 'none',
                            color: 'var(--text-muted)',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            borderRadius: '50%'
                        }}
                        title="Notifications"
                    >
                        <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>notifications</span>
                    </button>
                    <button 
                        className="icon-btn" 
                        onClick={toggleTheme} 
                        style={{
                            padding: '6px',
                            background: 'transparent',
                            border: 'none',
                            color: 'var(--text-muted)',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            borderRadius: '50%'
                        }}
                        title="Toggle Theme"
                    >
                        <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>
                            {theme === 'dark' ? 'light_mode' : 'dark_mode'}
                        </span>
                    </button>
                    {onLogout && (
                        <button 
                            className="icon-btn" 
                            onClick={onLogout} 
                            style={{
                                padding: '6px',
                                background: 'transparent',
                                border: 'none',
                                color: 'var(--text-muted)',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                borderRadius: '50%'
                            }}
                            title="Sign Out"
                        >
                            <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>logout</span>
                        </button>
                    )}
                </div>

                {/* User avatar chip */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', paddingLeft: '8px', borderLeft: '1px solid var(--border-color)' }}>
                    <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-main)' }}>{user?.name || "Enterprise Analyst"}</span>
                        <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>{user?.role || "Lead Analyst"}</span>
                    </div>
                    <div 
                        style={{ 
                            width: '32px', 
                            height: '32px', 
                            borderRadius: '50%', 
                            border: '1.5px solid var(--primary-color)',
                            backgroundColor: 'var(--bg-input)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: 'var(--primary-color)',
                            fontWeight: 'bold',
                            fontSize: '0.8rem',
                            fontFamily: 'Geist'
                        }}
                    >
                        {getInitials(user?.name)}
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header;
