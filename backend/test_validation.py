import unittest
import os
os.environ["TESTING"] = "true"
import io
import pandas as pd
from fastapi.testclient import TestClient

# Adjust path to import backend modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app, DATASTORE, ANALYSIS_CACHE

class TestValidation(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Clear DATASTORE before each test
        DATASTORE.clear()
        DATASTORE["acronyms"] = {}
        DATASTORE["profiling_status"] = "idle"
        DATASTORE["report_generation_status"] = "idle"
        DATASTORE["domain_profile"] = None
        DATASTORE["dataset_session_id"] = None
        DATASTORE["executive_report"] = None
        DATASTORE["highlighted_records"] = None
        DATASTORE["target_analysis"] = None
        DATASTORE["driver_analysis"] = None
        DATASTORE["report_cache"] = {}
        DATASTORE["chat_history"] = []
        DATASTORE["background_analysis_results"] = None
        DATASTORE["target_column"] = None
        ANALYSIS_CACHE.clear()

    def test_csv_upload_titanic(self):
        # Path to Titanic dataset in workspace
        titanic_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "Titanic-Dataset.csv")
        self.assertTrue(os.path.exists(titanic_path), f"Titanic dataset not found at {titanic_path}")
        
        with open(titanic_path, "rb") as f:
            response = self.client.post(
                "/upload",
                files={"file": ("Titanic-Dataset.csv", f, "text/csv")},
                data={"machine_name": "Titanic Survivors Study"}
            )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["detected_target"], "Survived")
        self.assertEqual(data["confidence"], 0.95)
        self.assertIn("Survived", data["candidate_targets"])
        self.assertEqual(data["filename"], "Titanic-Dataset.csv") # sanitized filename

    def test_target_confirmation_workflow(self):
        # Load titanic df mock into memory
        titanic_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "Titanic-Dataset.csv")
        df = pd.read_csv(titanic_path)
        DATASTORE["df"] = df
        DATASTORE["dataset_session_id"] = "test-session-uuid"
        
        # 1. Confirm default target (Survived)
        response = self.client.post(
            "/analysis/confirm_target",
            json={"target_column": "Survived"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["target_column"], "Survived")
        self.assertIn("status", data)
        self.assertEqual(DATASTORE["target_column"], "Survived")

        # 2. Confirm manual override target (Pclass)
        response = self.client.post(
            "/analysis/confirm_target",
            json={"target_column": "Pclass"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["target_column"], "Pclass")
        self.assertEqual(DATASTORE["target_column"], "Pclass")

        # 3. Confirm invalid target column
        response = self.client.post(
            "/analysis/confirm_target",
            json={"target_column": "InvalidColumnName"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Column 'InvalidColumnName' not found", response.json()["detail"])

    def test_background_profiling_lifecycle(self):
        import unittest.mock as mock
        
        # Load titanic df mock into memory
        titanic_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "Titanic-Dataset.csv")
        df = pd.read_csv(titanic_path)
        DATASTORE["df"] = df
        DATASTORE["dataset_session_id"] = "test-session-uuid"
        
        # Mock profile_dataset to return quickly
        mock_profile = {"domain": "Predictive Survival", "confidence": 0.95, "recommended_kpis": []}
        
        with mock.patch("main.threading.Thread") as mock_thread, \
            mock.patch("main.agent.profile_dataset", return_value=mock_profile) as mock_profile_func, \
            mock.patch("main.run_background_analysis") as mock_analysis_func:
            
            # Start profiling/analysis
            response = self.client.post("/analysis/start")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "started")
            
            # Call background target function synchronously to test transition logic
            from main import run_background_profiling_and_analysis
            run_background_profiling_and_analysis(df, None, True, "test-session-uuid")
            
            # Verify profiling completed
            self.assertEqual(DATASTORE["profiling_status"], "completed")
            self.assertEqual(DATASTORE["domain_profile"], mock_profile)
            self.assertGreaterEqual(mock_profile_func.call_count, 1)
            mock_analysis_func.assert_called_once_with(df, None, "test-session-uuid")


    def test_secure_uploads_validation(self):
        # 1. Test invalid CSV extension
        bad_file = io.BytesIO(b"dummy,header\n1,2")
        response = self.client.post(
            "/upload",
            files={"file": ("malicious_script.sh", bad_file, "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only .csv files are supported", response.json()["detail"])

        # 2. Test malformed/corrupted CSV headers (e.g. binary image)
        corrupted_csv = io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
        response = self.client.post(
            "/upload",
            files={"file": ("corrupt.csv", corrupted_csv, "text/csv")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid CSV format or corrupted file headers", response.json()["detail"])

        # 3. Test manuals upload invalid extension
        bad_manual = io.BytesIO(b"dummy pdf body")
        response = self.client.post(
            "/manuals/upload",
            files={"file": ("guide.txt", bad_manual, "text/plain")}
        )
        self.assertEqual(response.status_code, 200) # manual upload returns json block, check for error key
        self.assertIn("error", response.json())
        self.assertIn("Only PDF files are supported", response.json()["error"])

        # 4. Test manuals upload invalid PDF signature
        bad_sig_manual = io.BytesIO(b"HTML body instead of PDF")
        response = self.client.post(
            "/manuals/upload",
            files={"file": ("guide.pdf", bad_sig_manual, "application/pdf")}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("error", response.json())
        self.assertIn("Invalid PDF format or corrupted file headers", response.json()["error"])

    def test_pdf_export_route(self):
        # 1. Test export when no report exists
        response = self.client.get("/reports/export/pdf")
        self.assertEqual(response.status_code, 400)
        self.assertIn("No executive report generated yet", response.json()["detail"])

        # 2. Test successful export when report is populated
        DATASTORE["executive_report"] = (
            "# Executive Summary\n"
            "This is a test summary for PDF generation.\n\n"
            "# Key Findings\n"
            "- Finding 1. One sentence.\n"
            "- Finding 2. One sentence.\n\n"
            "# Statistical Evidence\n"
            "- 95% survival rate. One sentence.\n"
            "- Positive correlation between Fare and Survival. One sentence.\n\n"
            "# Recommendations\n"
            "- Adjust safety parameters. One sentence.\n"
            "- Schedule regular drills. One sentence.\n"
        )
        DATASTORE["machine_name"] = "Test Context"
        DATASTORE["filename"] = "test.csv"
        DATASTORE["target_column"] = "Survived"
        DATASTORE["domain_profile"] = {"domain": "Predictive Demographics"}

        response = self.client.get("/reports/export/pdf")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        # Verify content has PDF header bytes
        self.assertTrue(response.content.startswith(b"%PDF"))

if __name__ == "__main__":
    unittest.main()
