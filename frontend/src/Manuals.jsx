import React, { useState, useEffect } from 'react';
import { api } from './services/api';

function Manuals() {
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [message, setMessage] = useState('');
    const [manuals, setManuals] = useState([]);

    useEffect(() => {
        fetchManuals();
    }, []);

    const fetchManuals = async () => {
        try {
            const response = await api.getManuals();
            setManuals(Array.isArray(response.data) ? response.data : []);
        } catch (error) {
            console.error('Error fetching manuals:', error);
        }
    };

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setUploading(true);
        setMessage('Uploading and indexing...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await api.uploadManual(formData);
            setMessage('Success: ' + response.data.message);
            setFile(null);
            fetchManuals(); // Refresh list
        } catch (error) {
            const errMsg = error.response?.data?.error || error.message;
            setMessage('Error: ' + errMsg);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div style={{ padding: '20px', color: '#fff' }}>
            <h2>Document Library (Reference Manuals)</h2>
            <p>Upload reference manuals (PDF) here. The agent will read them to enrich the analysis context.</p>

            <div style={{ marginBottom: '30px', background: '#333', padding: '20px', borderRadius: '8px' }}>
                <h3>Add New Document</h3>
                <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    style={{ marginBottom: '10px', display: 'block' }}
                />
                <button
                    onClick={handleUpload}
                    disabled={!file || uploading}
                    style={{
                        padding: '10px 20px',
                        backgroundColor: uploading ? '#666' : '#007bff',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: uploading ? 'not-allowed' : 'pointer'
                    }}
                >
                    {uploading ? 'Processing...' : 'Upload & Index'}
                </button>
                {message && <p style={{ marginTop: '10px', color: message.startsWith('Error') ? '#ff6b6b' : '#51cf66' }}>{message}</p>}
            </div>

            <div>
                <h3>Available Documents</h3>
                {manuals.length === 0 ? (
                    <p>No reference documents uploaded yet.</p>
                ) : (
                    <ul style={{ listStyle: 'none', padding: 0 }}>
                        {manuals.map((manual, index) => (
                            <li key={index} style={{ padding: '10px', borderBottom: '1px solid #444' }}>
                                📄 {manual}
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

export default Manuals;
