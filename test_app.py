import io
import os
import unittest

os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key-12345")
os.environ.setdefault("GEMINI_API_KEY", "test-api-key")

import app as ai_app


class AIAppUnitTests(unittest.TestCase):
    def setUp(self):
        ai_app.app.config["TESTING"] = True
        self.client = ai_app.app.test_client()

    def test_split_text(self):
        sample = "data " * 1000
        chunks = ai_app.split_text(sample, chunk_size=400, overlap=80)
        self.assertTrue(len(chunks) >= 2)
        self.assertEqual(ai_app.split_text(""), [])

    def test_extract_text_from_txt_file(self):
        content = "Project Alpha is a quantum computing framework built for scalability."
        parsed = ai_app.extract_file_content(content.encode("utf-8"), "alpha.txt")
        self.assertIn("quantum computing", parsed["text"])
        self.assertEqual(parsed["file_type"], "Text Document")
        self.assertEqual(parsed["icon"], "📝")

    def test_extract_text_from_py_file(self):
        code = "def calculate_sum(a, b):\n    return a + b\n"
        parsed = ai_app.extract_file_content(code.encode("utf-8"), "math_utils.py")
        self.assertIn("calculate_sum", parsed["text"])
        self.assertEqual(parsed["file_type"], "Python Code")
        self.assertEqual(parsed["icon"], "🐍")

    def test_build_document_index_and_retrieve(self):
        chunks = [
            "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
            "The stock market experienced gains driven by tech equities.",
            "Convolutional neural networks are commonly used for image classification.",
        ]
        doc_entry = ai_app.build_document_index(chunks, full_text="\n\n".join(chunks))
        self.assertEqual(len(doc_entry["chunks"]), 3)
        self.assertIsNotNone(doc_entry["vectorizer"])
        self.assertIsNotNone(doc_entry["tfidf_matrix"])

        # Semantic query
        results = ai_app.retrieve_chunks("How do neural networks classify images?", doc_entry, top_k=1)
        self.assertTrue(results)
        self.assertIn("neural", results[0].lower())

        # General query fallback
        general_results = ai_app.retrieve_chunks("Can you summarize this?", doc_entry, top_k=2)
        self.assertTrue(general_results)

    def test_home_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Helix AI", response.data)

    def test_health_route(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data["status"], "online")
        self.assertIn("models", json_data)

    def test_document_routes(self):
        # Empty doc
        response = self.client.get("/document")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["has_document"])

        # Preview when no doc
        prev_resp = self.client.get("/document/preview")
        self.assertEqual(prev_resp.status_code, 404)

        # Detach doc
        del_resp = self.client.delete("/document")
        self.assertEqual(del_resp.status_code, 200)

    def test_upload_text_file(self):
        # Upload .txt file
        sample_txt = b"Artificial intelligence is transforming software engineering and data science."
        data = {
            "file": (io.BytesIO(sample_txt), "sample_notes.txt")
        }
        resp = self.client.post("/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        json_data = resp.get_json()
        self.assertEqual(json_data["filename"], "sample_notes.txt")
        self.assertEqual(json_data["file_type"], "Text Document")
        self.assertTrue(json_data["chunks"] >= 1)

    def test_upload_unsupported_file(self):
        data = {
            "file": (io.BytesIO(b"fake exe"), "malicious.exe")
        }
        resp = self.client.post("/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported file type", resp.get_json()["error"])

    def test_chat_validation(self):
        resp = self.client.post("/chat", json={})
        self.assertEqual(resp.status_code, 400)

        resp = self.client.post("/chat", json={"message": "   "})
        self.assertEqual(resp.status_code, 400)

    def test_history_and_clear_routes(self):
        with self.client.session_transaction() as sess:
            sess["history"] = [{"user": "Hi", "assistant": "Hello there!"}]

        resp = self.client.get("/history")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.get_json()["history"]), 1)

        resp = self.client.post("/clear")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/history")
        self.assertEqual(resp.get_json()["history"], [])

    def test_export_route(self):
        with self.client.session_transaction() as sess:
            sess["history"] = [{"user": "Question", "assistant": "Answer", "timestamp": "14:30"}]

        resp = self.client.get("/export")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Helix AI Conversation Export", resp.data)

    def test_thread_management(self):
        # Create thread
        resp = self.client.post("/threads", json={"title": "Test AI Project"})
        self.assertEqual(resp.status_code, 200)
        t_data = resp.get_json()
        self.assertTrue(t_data["success"])
        thread_id = t_data["thread"]["thread_id"]

        # List threads
        resp = self.client.get("/threads")
        self.assertEqual(resp.status_code, 200)
        threads = resp.get_json()["threads"]
        self.assertTrue(any(t["thread_id"] == thread_id for t in threads))

        # Rename thread
        resp = self.client.patch(f"/threads/{thread_id}", json={"title": "Renamed Thread"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["title"], "Renamed Thread")

        # Switch thread
        resp = self.client.post("/threads/switch", json={"thread_id": thread_id})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["thread_id"], thread_id)

        # Delete thread
        resp = self.client.delete(f"/threads/{thread_id}")
        self.assertEqual(resp.status_code, 200)

    def test_sql_studio_api(self):
        # Tables route
        resp = self.client.get("/api/sql/tables")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["success"])
        table_names = [t["table_name"] for t in resp.get_json()["tables"]]
        self.assertIn("threads", table_names)
        self.assertIn("chat_history", table_names)

        # Custom SQL query
        query_payload = {"query": "SELECT 42 AS answer, 'helix' AS name"}
        resp = self.client.post("/api/sql/query", json=query_payload)
        self.assertEqual(resp.status_code, 200)
        q_data = resp.get_json()
        self.assertTrue(q_data["success"])
        self.assertEqual(q_data["columns"], ["answer", "name"])
        self.assertEqual(q_data["rows"][0], [42, "helix"])

        # SQL stats
        resp = self.client.get("/api/sql/stats")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("total_tables", resp.get_json())

    def test_sql_slash_command(self):
        resp = self.client.post("/chat", json={"message": "/sql SELECT 1 + 1 AS sum_res;"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("is_sql"))
        self.assertIn("sum_res", data.get("response"))

    def test_sql_csv_import(self):
        csv_sample = "name,role,department\nAlice,Engineer,AI\nBob,Designer,UI\n"
        resp = self.client.post("/api/sql/import-csv", json={
            "table_name": "employees_test",
            "csv_data": csv_sample
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("table_name"), "employees_test")
        self.assertEqual(data.get("row_count"), 2)

        # Query the newly imported table
        q_resp = self.client.post("/api/sql/query", json={"query": "SELECT * FROM employees_test"})
        self.assertEqual(q_resp.status_code, 200)
        self.assertEqual(len(q_resp.get_json()["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
