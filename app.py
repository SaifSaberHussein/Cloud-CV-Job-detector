from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from model_loader import load_model
from inference import run_inference
from storage import save_to_gcs, bucket
from docx import Document
import PyPDF2
import os, json, io, re, time
import google.cloud.logging

app = Flask(__name__)

# ── DIRECT CLOUD LOGGING (Fix from v2) ────────────────────────────────────
# Uses logging_client.logger() directly instead of setup_logging(),
# which was the root cause of logs not appearing in Cloud Logging.
logging_client = google.cloud.logging.Client()
cloud_logger = logging_client.logger("resume-processor-logs")

# Startup confirmation log
cloud_logger.log_text("--- API STARTUP: Direct Cloud Logging Active ---", severity="INFO")

# Load model
model, tokenizer, device = load_model("./flan-t5-lora-finetuned")

CATEGORIES = [
    "engineering", "finance", "healthcare", "education", "tech",
    "sales-marketing", "business-hr", "creative-media-design",
    "hospitality-food", "aviation-transport", "construction", "other"
]

# ── GUI STYLES ─────────────────────────────────────────────────────────────
CSS_STYLE = """
<style>
    body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #f1f5f9; padding: 40px; margin: 0; }
    .container { max-width: 900px; margin: auto; background: #1e293b; padding: 30px; border-radius: 16px; border: 1px solid #334155; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    .nav { margin-bottom: 30px; display: flex; justify-content: center; gap: 20px; }
    .nav a { color: #38bdf8; text-decoration: none; font-weight: 600; padding: 8px 16px; border-radius: 8px; transition: background 0.2s; }
    .nav a:hover { background: #334155; }
    h2 { color: #38bdf8; margin-top: 0; }
    textarea { width: 100%; height: 120px; background: #0f172a; color: white; border: 1px solid #334155; border-radius: 8px; padding: 12px; box-sizing: border-box; margin-top: 10px; }
    button { background: #38bdf8; color: #0f172a; border: none; padding: 10px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; transition: opacity 0.2s; }
    button:hover { opacity: 0.9; }
    .view-btn { background: #94a3b8; color: #0f172a; margin-right: 10px; }
    .verify-btn { background: #4fffb0; color: #0f172a; margin-left: 10px; }
    .card { background: #334155; padding: 20px; border-radius: 12px; margin-top: 20px; border-left: 5px solid #38bdf8; text-align: left; }
    select { background: #0f172a; color: white; border: 1px solid #475569; padding: 8px; border-radius: 6px; min-width: 150px; }
    .file-item { display: flex; justify-content: space-between; align-items: center; background: #1e293b; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #334155; }

    /* Modal Styles */
    .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); backdrop-filter: blur(5px); }
    .modal-content { background: #1e293b; margin: 5% auto; padding: 30px; width: 70%; border-radius: 16px; border: 1px solid #38bdf8; max-height: 80vh; overflow-y: auto; position: relative; }
    .close { color: #94a3b8; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }
    pre { white-space: pre-wrap; word-wrap: break-word; color: #e2e8f0; font-family: 'Consolas', monospace; background: #0f172a; padding: 15px; border-radius: 8px; text-align: left; }
</style>
"""

# ── CORE LOGIC: PROCESS & LOG ──────────────────────────────────────────────
def process_chunk(text):
    if len(text.strip()) < 50:
        return None

    # Log attempt directly to Cloud
    cloud_logger.log_text(f"ATTEMPTING EXTRACTION: Text length {len(text)}", severity="INFO")

    start_time = time.time()
    try:
        res = run_inference(text, model, tokenizer, device)

        # Confidence Gate — route to review queue if below threshold
        folder = "extractions" if res["confidence"] >= 0.60 else "review-queue"
        save_to_gcs(text, res["extracted"], folder=folder)

        latency = round(time.time() - start_time, 2)
        conf    = round(res["confidence"], 2)
        cat     = res["extracted"].get("Main Job", "unknown")
        status  = "flagged" if folder == "review-queue" else "success"

        # Log completion directly to Cloud (structured string)
        cloud_logger.log_text(
            f"Extraction completed | Category: {cat} | Conf: {conf} | Latency: {latency}s | Status: {status}",
            severity="INFO"
        )

        return {"category": cat, "conf": res["confidence"]}

    except Exception as e:
        cloud_logger.log_text(f"EXTRACTION ERROR: {str(e)}", severity="ERROR")
        return None


# ── PAGE: EXTRACTOR ────────────────────────────────────────────────────────
@app.route("/health", methods=["GET", "POST"])
def health():
    batch_results, last_val = [], ""
    if request.method == "POST":
        file = request.files.get('cv_file')
        if file and file.filename != '':
            ext = file.filename.rsplit('.', 1)[1].lower()
            text = ""
            if ext == 'docx':
                doc = Document(io.BytesIO(file.read()))
                text = "\n".join([p.text for p in doc.paragraphs])
            elif ext == 'pdf':
                reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                text = "\n".join([page.extract_text() for page in reader.pages])

            chunks = re.split(r'---|\nRESUME\n|\nCV\n', text)
            for c in chunks:
                r = process_chunk(c.strip())
                if r:
                    batch_results.append(r)

        elif request.form.get("cv_text"):
            last_val = request.form.get("cv_text")
            r = process_chunk(last_val)
            if r:
                batch_results.append(r)

    HTML = """
    <html><head>{{ css|safe }}</head><body>
        <div class="nav"><a href="/health">Extractor</a> | <a href="/review">Review Queue</a></div>
        <div class="container">
            <h2>AI CV Batch Processor</h2>
            <form method="POST" enctype="multipart/form-data">
                <div style="border: 2px dashed #334155; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                    <p style="margin:0 0 10px 0; font-size:14px; color:#94a3b8;">Upload Word or PDF</p>
                    <input type="file" name="cv_file" accept=".docx,.pdf">
                </div>
                <p style="font-size:12px; color:#94a3b8;">OR Paste Single CV Text:</p>
                <textarea name="cv_text" placeholder="Paste text here...">{{ last_val }}</textarea>
                <button type="submit" style="width:100%; margin-top:15px;">Process Content</button>
            </form>
            {% if results %}
            <div class="card" style="border-left-color: #4fffb0;">
                <h3 style="margin-top:0">Processing Results:</h3>
                {% for r in results %}
                <div style="margin-bottom:8px;">
                    • {{ r.category }}
                    <span style="color:{{ '#4fffb0' if r.conf >= 0.6 else '#fbbf24' }}">
                        ({{ (r.conf*100)|round(1) }}%)
                    </span>
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
    </body></html>
    """
    return render_template_string(HTML, results=batch_results, last_val=last_val, css=CSS_STYLE)


# ── PAGE: REVIEW QUEUE ─────────────────────────────────────────────────────
@app.route("/review", methods=["GET"])
def review():
    blobs = bucket.list_blobs(prefix="review-queue/")
    files = [b.name for b in blobs]

    HTML = """
    <html><head>{{ css|safe }}</head><body>
        <div class="nav"><a href="/health">Extractor</a> | <a href="/review">Review Queue</a></div>
        <div class="container">
            <h2>Human Review Queue</h2>
            <p style="color:#94a3b8">Flagged extractions (Confidence &lt; 60%).</p>
            {% for f in files %}
            <div class="file-item">
                <div style="font-size:12px; color:#94a3b8; font-family:monospace;">{{ f.split('/')[-1][:25] }}...</div>
                <div style="display:flex; align-items:center;">
                    <button class="view-btn" onclick="showResume('{{ f }}')">View Resume</button>
                    <form action="/verify" method="POST" style="margin:0; display:flex; align-items:center;">
                        <input type="hidden" name="file_path" value="{{ f }}">
                        <select name="correct_category">
                            {% for c in cats %}<option value="{{ c }}">{{ c }}</option>{% endfor %}
                        </select>
                        <button type="submit" class="verify-btn">Verify</button>
                    </form>
                </div>
            </div>
            {% endfor %}
            {% if not files %}
                <p style="margin-top:30px; color:#64748b;">The review queue is empty.</p>
            {% endif %}
        </div>

        <div id="resModal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeModal()">&times;</span>
                <h3 style="color:#38bdf8; margin-top:0;">Resume Original Input</h3>
                <pre id="resContent">Loading content from GCS...</pre>
            </div>
        </div>

        <script>
            function showResume(path) {
                document.getElementById('resModal').style.display = "block";
                document.getElementById('resContent').innerText = "Fetching...";
                fetch('/get_content?path=' + path)
                    .then(r => r.json())
                    .then(data => { document.getElementById('resContent').innerText = data.text; });
            }
            function closeModal() { document.getElementById('resModal').style.display = "none"; }
            window.onclick = function(e) { if (e.target.id == 'resModal') closeModal(); }
        </script>
    </body></html>
    """
    return render_template_string(HTML, files=files, cats=CATEGORIES, css=CSS_STYLE)


# ── LOGIC: FETCH & VERIFY ──────────────────────────────────────────────────
@app.route("/get_content")
def get_content():
    path = request.args.get('path')
    blob = bucket.blob(path)
    data = json.loads(blob.download_as_text())
    return jsonify({"text": data.get("resume_input", "No text content found.")})


@app.route("/verify", methods=["POST"])
def verify():
    path = request.form.get("file_path")
    cat  = request.form.get("correct_category")
    blob = bucket.blob(path)
    try:
        data = json.loads(blob.download_as_text())
        data["extracted"]["Main Job"] = cat
        data["human_verified"] = True
        new_blob = bucket.blob(path.replace("review-queue", "extractions"))
        new_blob.upload_from_string(json.dumps(data), content_type="application/json")
        blob.delete()
        cloud_logger.log_text(f"VERIFIED: {path} -> {cat}", severity="INFO")
    except Exception as e:
        cloud_logger.log_text(f"VERIFICATION ERROR: {str(e)}", severity="ERROR")
    return redirect(url_for('review'))


# ── API ENDPOINT ───────────────────────────────────────────────────────────
@app.route("/extract", methods=["POST"])
def extract():
    body = request.get_json(silent=True)
    if not body or "resume_text" not in body:
        return jsonify({"error": "Missing resume_text"}), 400
    res = process_chunk(body["resume_text"])
    return jsonify(res), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)