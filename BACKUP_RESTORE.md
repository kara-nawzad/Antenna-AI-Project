# Backup & Restore Guide — Before Formatting Your Laptop

This guide ensures **nothing is lost** when you format your machine. After the
wipe, you can restore everything (code, models, data, website) in ~10 minutes.

---

## 📍 Where everything lives after backup

| Place | Contents | Why? |
|---|---|---|
| **GitHub** — `kara-nawzad/Antenna-AI-Project` | All code, scalers, assets, Dockerfile, fly.toml, this file | Code belongs on GitHub |
| **Hugging Face** — `kara-nawzad/antenna-models` | ALL big files (models, data) — see list below | No file-size limits; fast CDN; used by the app at runtime |
| **fly.io** (optional) | The running website | Hosts the live app |
| **Groq** | Your API key stays in your Groq account — you just re-enter it after restore | Never commit API keys |

> ✅ The only places you **must** push to before formatting are **GitHub** and **Hugging Face**.

---

## 🚨 Files that are NOT yet in this folder — find them on your laptop BEFORE formatting

The `find` command in this repo shows only the small code files (~380 KB). The
following big files are in your `.gitignore` (not tracked by Git) and are **NOT
in this workspace**. You MUST find copies on your laptop and upload them to HF.
Check every folder where you trained/saved models:

Files you MUST find:

- [ ] `forward_model_v2_curve.keras` (the 115 MB main model — already on HF ✅)
- [ ] `s11_dip_specialist.pkl` (⚠️ was DELETED from HF; only exists on your
      laptop if you still have it — look in your training folder!)
- [ ] `antenna_data_cleaned.csv` (7.76 MB — already on HF ✅)
- [ ] `inverse_model_final.keras` (if you ever trained it — optional)
- [ ] `shrink_specialist.py`, `forward_model_v2_report.json`,
      `s11_dip_specialist_report.json`, `processed_data.npz` (training scripts
      / reports — optional but nice to keep)
- [ ] Any other `.keras` / `.h5` / `.pkl.gz` model checkpoints you care about
- [ ] `.streamlit/secrets.toml` (contains your `GROQ_API_KEY` — keep a copy of
      the **key value** somewhere safe like a password manager; you don't need
      to upload the file itself)

**Tip:** On Windows, use File Explorer search in your project/Google Drive for
`*.keras`, `*.pkl`, `*.csv`. On Linux/Mac, run in your project root:
```bash
find ~ -name "*.keras" -o -name "*specialist*.pkl" -o -name "antenna_data_cleaned.csv" 2>/dev/null
```

---

## Step 1 — Push code to GitHub (DONE automatically by the agent if you push)

The code in this folder is ready. From a terminal in this repo:

```bash
git status        # review changes
git add -A
git commit -m "Fix HF_REPO, add Dockerfile + fly.toml + backup guide"
git push origin arena/01a04465-antenna-ai-project
# then merge to main (or just keep on this branch — your call)
```

---

## Step 2 — Upload ALL big files to Hugging Face (EASIEST way: web UI)

This is the safest option before formatting. You just drag & drop.

1. Go to https://huggingface.co/kara-nawzad/antenna-models
2. Click **"Add file" → "Upload files"**
3. Drag in EVERY big file from the list above (including `s11_dip_specialist.pkl`
   if you still have it — re-uploading it will restore the deleted file).
4. Add a commit message like `"backup all models & data before format"` and click **Commit**.
5. Wait for the upload — the 115 MB `.keras` file may take a few minutes.

After this, HF has a complete copy of every model and data file.

### Alternative — via git CLI (more reliable for big files)
```bash
# (from any folder on your personal machine, NOT this sandbox)
pip install -U huggingface_hub
# login once with a token from https://huggingface.co/settings/tokens
huggingface-cli login

git lfs install                       # one-time
git clone https://huggingface.co/kara-nawzad/antenna-models
cd antenna-models

# Copy every file you found (models, data, scripts, reports) into this folder, e.g.:
# copy /path/to/forward_model_v2_curve.keras ./
# copy /path/to/s11_dip_specialist.pkl ./
# copy /path/to/antenna_data_cleaned.csv ./
# ...

git add .
git commit -m "full backup before laptop format"
git push
```

---

## Step 3 — (Optional) Save your Groq key

Your Groq API key starts with `gsk_`. Find it in one of these places:
- Streamlit Cloud → your app → ⚙️ Settings → Secrets
- Your old `.streamlit/secrets.toml` file
- https://console.groq.com/keys (you can always create a new one for free)

Save it in a password manager — you'll need it after restore.

---

## Step 4 — Format safely 💾

At this point, three independent copies exist:
1. GitHub — code
2. Hugging Face — models & data
3. Groq — API key

You can format.

---

## Step 5 — Restore on your new/clean machine

```bash
# 1. Install Python 3.11, git, (optional) flyctl
#    Python: https://www.python.org   flyctl: https://fly.io/docs/hands-on/install-flyctl/

# 2. Clone code from GitHub
git clone https://github.com/kara-nawzad/Antenna-AI-Project.git
cd Antenna-AI-Project

# 3. Install dependencies
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

# 4. Run locally — the app AUTO-DOWNLOADS the big models/CSVs from Hugging Face on first run
streamlit run 10_web_app.py
# That's it. The browser opens automatically.
```

> The first startup takes ~30–90 seconds while it downloads the 115 MB model
> from HF; subsequent starts are instant because the files are cached locally.

---

## Step 6 — Deploy to fly.io (optional replacement for Streamlit Cloud)

```bash
fly auth login          # opens browser, log into your existing fly account
fly launch              # asks a few questions — answer NO to "deploy now?" and NO to "create a Postgres DB?"
                        # (fly.toml is already in the repo so it will use it)
fly secrets set GROQ_API_KEY=gsk_your_actual_key_here
fly deploy              # builds Docker image and deploys
fly open                # opens https://antenna-ai.fly.dev (or whatever app name you chose)
```

Notes:
- This will NOT interfere with your other fly.io apps — each app is independent.
- Free tier = 3 shared-CPU VMs. As long as your existing app + this one ≤ 3, it's free.
- `auto_stop_machines = "stop"` in fly.toml means this app sleeps when idle
  (no credits burned), and wakes up on first visit (~15-30 s cold start).
- The Docker image does NOT bake in the 115 MB model — it downloads from HF
  on first container start. HF bandwidth is free/unlimited, so no costs.

---

## Step 7 — (If needed) Get models back directly from HF

If you ever want the raw files back locally (e.g., to retrain), you can download
them from the HF web UI, or run in Python:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="kara-nawzad/antenna-models", filename="forward_model_v2_curve.keras", local_dir=".")
hf_hub_download(repo_id="kara-nawzad/antenna-models", filename="s11_dip_specialist.pkl",       local_dir=".")
hf_hub_download(repo_id="kara-nawzad/antenna-models", filename="antenna_data_cleaned.csv",   local_dir=".")
```

Or clone the HF repo like any git repo (see "Alternative — via git CLI" above).

---

## ✅ Quick checklist before you hit "Format"

- [ ] Code pushed to GitHub (Step 1)
- [ ] All big files uploaded to Hugging Face (Step 2) — especially
      `s11_dip_specialist.pkl` if you still have it!
- [ ] Groq API key saved somewhere safe (Step 3)
- [ ] Any other personal files (papers, CST files, Google Drive link still works
      for research papers — see the hero button in the web app) backed up

After formatting, follow Step 5 (and optionally 6). You're done! 🚀
