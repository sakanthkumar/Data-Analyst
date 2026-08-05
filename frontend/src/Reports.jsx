import { useState, useEffect } from 'react';
import { api } from './services/api';

export default function Reports() {
    const [reports, setReports] = useState([]);
    const [selectedReport, setSelectedReport] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchReports = async () => {
            try {
                const res = await api.getReportsList();
                setReports(res.data);
            } catch (e) { console.error(e); }
        };
        fetchReports();
    }, []);

    const viewReport = async (id) => {
        setLoading(true);
        try {
            const res = await api.getReportDetail(id);
            setSelectedReport(res.data);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    if (loading) return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            <div className="spinner"></div>
            <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>Loading report details...</p>
        </div>
    );

    return (
        <div className="container">
            {!selectedReport ? (
                <>
                    <h2>Saved Analysis Reports</h2>
                    <div className="table-wrapper">
                        <table>
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Dataset/Context</th>
                                    <th>Analysis Type</th>
                                    <th>Highlighted Records</th>
                                    <th>Action</th>
                                </tr>
                            </thead>
                            <tbody>
                                {reports.length === 0 && <tr><td colSpan="5">No saved reports found.</td></tr>}
                                {reports.map((r) => (
                                    <tr key={r.id}>
                                        <td>{new Date(r.timestamp).toLocaleString()}</td>
                                        <td>{r.machine_name}</td>
                                        <td><span className="badge">{r.analysis_type}</span></td>
                                        <td>{r.total_failures}</td>
                                        <td>
                                            <button className="secondary-btn" onClick={() => viewReport(r.id)}>View Details</button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            ) : (
                <div className="report-details">
                    <button className="secondary-btn" onClick={() => setSelectedReport(null)} style={{ marginBottom: '1rem' }}>← Back to List</button>
                    <div className="card">
                        <header style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '1rem', marginBottom: '1rem' }}>
                            <h2>{selectedReport.analysis_type} Report</h2>
                            <p>Dataset/Context: {selectedReport.machine_name} | Date: {new Date(selectedReport.timestamp).toLocaleString()}</p>
                        </header>

                        {selectedReport.failures && selectedReport.failures.length > 0 && (
                            <div className="table-wrapper" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                                <h4>Highlighted Records ({selectedReport.total_failures})</h4>
                                <table>
                                    <thead>
                                        <tr>{Object.keys(selectedReport.failures[0]).map(k => <th key={k}>{k}</th>)}</tr>
                                    </thead>
                                    <tbody>
                                        {selectedReport.failures.map((row, i) => (
                                            <tr key={i}>{Object.values(row).map((v, j) => <td key={j}>{v}</td>)}</tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
