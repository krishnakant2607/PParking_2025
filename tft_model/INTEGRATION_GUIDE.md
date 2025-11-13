# 🔗 Integration Guide - Adding TFT Model to Existing Repo

## Overview

This guide explains how to add this TFT model folder to your existing PParking_2025 repository **without affecting your live website**.

---

## ✅ It's Safe Because:

1. **Vercel only deploys the web app** - Vercel looks for `package.json`, `next.config.js`, and the `app/` folder. It completely ignores this `tft_model/` folder.

2. **Separate stacks** - Your website uses Next.js/TypeScript, this model uses Python - no conflicts.

3. **No build interference** - The `tft_model/` folder contains only Python code and data files. It won't be included in your Next.js build.

4. **Optional `.vercelignore`** - You can explicitly tell Vercel to ignore this folder (though it already does).

---

## 📋 Steps to Add This Folder

### Option 1: Direct Copy (Recommended)

```bash
# 1. Navigate to your local PParking_2025 repo
cd /path/to/your/PParking_2025

# 2. Copy the tft_model folder
cp -r /Users/krishnakant/Documents/lastwihs/lastone/tft_model .

# 3. Add to git
git add tft_model/

# 4. Commit
git commit -m "Add TFT forecasting model (Python ML component)"

# 5. Push to GitHub
git push origin master
```

### Option 2: Use Git Submodule (Advanced)

If you want to keep the model in a separate repository:

```bash
cd /path/to/your/PParking_2025
git submodule add https://github.com/yourusername/tft-model.git tft_model
git commit -m "Add TFT model as submodule"
git push origin master
```

---

## 🔒 Ensure Vercel Ignores the Model (Optional)

Add this to your `.vercelignore` file (create it in your repo root if it doesn't exist):

```
# .vercelignore
tft_model/
*.pt
*.pkl
```

This explicitly tells Vercel to ignore the TFT model folder during deployment.

---

## 📂 Expected Repo Structure After Adding

```
PParking_2025/                    (your repo root)
│
├── app/                          ← Next.js web app (unchanged)
├── components/                   ← Next.js components (unchanged)
├── lib/                          ← Next.js lib (unchanged)
├── package.json                  ← Next.js package (unchanged)
├── next.config.js                ← Next.js config (unchanged)
├── .gitignore                    ← Your existing gitignore
├── README.md                     ← Your main README
│
└── tft_model/                    ← NEW: ML model folder
    ├── README.md                 ← Model documentation
    ├── requirements.txt          ← Python dependencies
    ├── src/                      ← Python source code
    ├── models/                   ← Trained model files
    ├── configs/                  ← Model configs
    ├── data/                     ← Sample data
    └── docs/                     ← Model reports
```

---

## 🧪 Test Locally Before Pushing

### 1. Test Your Website Still Works

```bash
cd /path/to/your/PParking_2025
npm install
npm run dev
```

Visit `http://localhost:3000` - your website should work normally.

### 2. Test the Model Works

```bash
cd tft_model
pip install -r requirements.txt
python src/tft_model.py  # Should run without errors
```

---

## 🚀 After Pushing to GitHub

1. **Your website will still deploy normally** on Vercel
2. **The model folder will be visible** on GitHub (but Vercel ignores it)
3. **No impact on your live site**

---

## 📝 Update Your Main README

Add a section to your main `README.md` (in repo root) to mention the model:

```markdown
## 📊 ML Forecasting Model

This repository includes a Python-based Temporal Fusion Transformer (TFT) model for parking occupancy forecasting.

📁 **Model Location**: `tft_model/`  
📖 **Documentation**: See [tft_model/README.md](./tft_model/README.md)  
🎯 **Performance**: R² = 0.8262, MAE = 0.2924

The model is separate from the web application and requires Python to run.
```

---

## ⚠️ Important Notes

### Model Checkpoint Size (27 MB)

The `best_model.pt` file is 27 MB. GitHub allows files up to 100 MB, so you're fine. However:

**Option A: Keep it in the repo** (recommended for now)
- ✅ Simple, works immediately
- ❌ Increases repo size

**Option B: Use Git LFS** (if size becomes an issue)
```bash
git lfs install
git lfs track "tft_model/models/*.pt"
git add .gitattributes
```

**Option C: Host separately**
- Upload to Google Drive, Dropbox, or Hugging Face
- Add download link in the model README
- Add to `.gitignore`: `tft_model/models/*.pt`

---

## 🐛 Troubleshooting

### Issue: Vercel build fails after adding folder

**Solution**: Add `.vercelignore` with:
```
tft_model/
```

### Issue: GitHub rejects large file

**Solution**: Use Git LFS or remove `best_model.pt` and host it externally

### Issue: Want to keep repos separate

**Solution**: Create a separate repo for the model and link to it in your main README

---

## ✅ Quick Checklist

Before pushing:

- [ ] Website still runs locally (`npm run dev`)
- [ ] Model files copied to `tft_model/` folder
- [ ] No conflicts in `.gitignore`
- [ ] Optional: Added `.vercelignore` 
- [ ] Optional: Updated main README to mention the model
- [ ] Committed with clear message
- [ ] Pushed to GitHub

After pushing:

- [ ] Check GitHub - folder appears correctly
- [ ] Check Vercel - website still deploys and works
- [ ] No build errors

---

## 📧 Questions?

If you encounter any issues, check:
1. Vercel build logs
2. GitHub file sizes
3. `.gitignore` and `.vercelignore` files

**The model folder is completely independent and won't affect your web deployment!** ✅
