import { useState } from "react";
import { api } from "./services/api";

function Upload({ onUploadSuccess, onUploadStart }) {
  const [uploading, setUploading] = useState(false);
  const [machineName, setMachineName] = useState("");

  const upload = async (e, currentMachineName) => {
    const file = e.target.files[0];
    if (!file) return;

    if (onUploadStart) onUploadStart();
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    if (currentMachineName) {
      form.append("machine_name", currentMachineName);
    }
    try {
      const res = await api.uploadDataset(form);
      if (onUploadSuccess) onUploadSuccess(res.data);
    } catch (e) {
      console.error(e);
      alert("Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-box">
      <div style={{ marginBottom: "10px" }}>
        <input
          type="text"
          placeholder="Dataset Name/Context (e.g. Customer Churn)"
          id="machine-name"
          value={machineName}
          onChange={(e) => setMachineName(e.target.value)}
          style={{ padding: "8px", borderRadius: "4px", border: "1px solid #444", background: "#333", color: "#fff" }}
        />
      </div>
      <label className="upload-btn">
        {uploading ? "Uploading..." : "Click to Upload CSV"}
        <input
          type="file"
          accept=".csv"
          onChange={(e) => upload(e, machineName)}
          style={{ display: 'none' }}
          disabled={uploading}
        />
      </label>
    </div>
  );
}
export default Upload;

