# Before You Publish - Checklist ✅

**Important:** Complete these steps before making the repository public or submitting to AIES.

---

## 1. Personalize Information (15 minutes)

### Update Author Information

**Files to edit:**
- [ ] `README.md` - Replace `[Your Name]` and `[your.email@domain.com]`
- [ ] `setup.py` - Update author, author_email, url
- [ ] `LICENSE` - Replace `[Your Name]`
- [ ] `INSTALLATION.md` - Update contact email
- [ ] `data/README.md` - Update contact information
- [ ] `docs/TECHNICAL_REPORT.md` - Add author name in header

**Search and replace:**
```bash
# Find all instances
grep -r "Your Name" .
grep -r "your.email@domain.com" .
grep -r "your-username" .
```

### Update URLs

**Replace `your-username` with your GitHub username in:**
- [ ] README.md (3 instances)
- [ ] setup.py (1 instance)
- [ ] INSTALLATION.md (2 instances)
- [ ] data/README.md (1 instance)

---

## 2. Verify Installation (10 minutes)

### Test on Fresh Environment

```bash
# Create new virtual environment
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# Install from scratch
cd mirrorfield_publication
pip install -r requirements.txt
pip install -e .

# Run tests
python experiments/test_phase_e_bundle.py
```

**Expected:** All 6 tests pass

### Generate Figures

```bash
python experiments/generate_publication_plots_v2.py
```

**Expected:** 6 files created in `plots/` (3 PNG, 3 PDF)

---

## 3. Review Documentation (20 minutes)

### Read Through Each File

- [ ] `README.md` - Does it make sense to someone new?
- [ ] `INSTALLATION.md` - Can someone follow these steps?
- [ ] `docs/TECHNICAL_REPORT.md` - Any typos or broken references?
- [ ] `data/README.md` - Clear data access instructions?

### Check All Links

```bash
# Find all markdown links
grep -r "\[.*\](" *.md docs/*.md
```

**Verify:**
- [ ] GitHub URLs point to correct repository
- [ ] Figure references exist (figure1_*.png, etc.)
- [ ] Internal document links work (docs/TECHNICAL_REPORT.md, etc.)

---

## 4. Add Co-Authors (if applicable)

### If You Have Collaborators

**Update these files:**
- [ ] `README.md` - Add co-authors to citation
- [ ] `setup.py` - Add to author field
- [ ] `LICENSE` - Add co-author names
- [ ] `docs/TECHNICAL_REPORT.md` - Update author list

**Example format:**
```
**Authors:** Jane Doe¹, John Smith²
¹University of Example, ²Example Institute
```

---

## 5. Data Compliance (10 minutes)

### Verify Data Privacy

- [ ] No personally identifiable information (PII) in any files
- [ ] No API keys committed (check .env is in .gitignore)
- [ ] No proprietary data without permission
- [ ] Data source properly credited

### Check Embeddings

**If you include actual embeddings:**
- [ ] Confirm OpenAI ToS allows redistribution
- [ ] Consider sharing only aggregated statistics
- [ ] Document embedding version and parameters

**Current status:** Only aggregated statistics embedded in scripts ✅

---

## 6. Code Quality (15 minutes)

### Run Basic Checks

**Check for common issues:**
```bash
# Find print statements (may be debug code)
grep -r "print(" experiments/*.py | grep -v "# "

# Find TODOs
grep -r "TODO" .

# Check for absolute paths (should be relative)
grep -r "C:/" experiments/*.py
grep -r "/Users/" experiments/*.py
```

### Clean Up Comments

- [ ] Remove any "FIXME" or "HACK" comments
- [ ] Remove personal notes like "remember to..."
- [ ] Keep useful docstrings and explanations

---

## 7. Citation & arXiv (5 minutes)

### Prepare for arXiv Upload

**Once paper is ready:**
1. Upload to arXiv.org
2. Get arXiv ID (e.g., 2401.12345)
3. Update citation in README.md:

```bibtex
@article{yourname2026geometric,
  title={Boundary-Stratified Evaluation of k-NN Geometric Features for AI Safety Detection},
  author={Your Name},
  journal={arXiv preprint arXiv:2401.12345},  # Update with real ID
  year={2026}
}
```

---

## 8. GitHub Setup (10 minutes)

### Initialize Repository

```bash
cd mirrorfield_publication
git init
git add .
git commit -m "Initial commit: Geometric safety features for AI"
```

### Create GitHub Repo

1. Go to https://github.com/new
2. Name: `geometric-safety-features` (or similar)
3. Description: "k-NN Geometric Features for AI Safety Detection"
4. **Keep private until paper accepted** (optional)
5. Don't initialize with README (you already have one)

### Push to GitHub

```bash
git remote add origin https://github.com/your-username/geometric-safety-features.git
git branch -M main
git push -u origin main
```

### Set Up Repository Settings

**On GitHub:**
- [ ] Add topics: `ai-safety`, `machine-learning`, `embeddings`, `uncertainty-quantification`
- [ ] Add description and website (if you have project page)
- [ ] Enable issues for questions
- [ ] Add `CITATION.cff` file (optional, GitHub auto-generates)

---

## 9. Add Badges (5 minutes)

### Update README Badges

**Replace placeholders in README.md with real badges:**

```markdown
[![arXiv](https://img.shields.io/badge/arXiv-2401.12345-b31b1b.svg)](https://arxiv.org/abs/2401.12345)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
```

**After acceptance:**
```markdown
[![AIES 2026](https://img.shields.io/badge/AIES-2026-green)]()
```

---

## 10. Final Review (10 minutes)

### Pre-Publication Checklist

- [ ] Author names and emails updated everywhere
- [ ] GitHub URLs point to correct repository
- [ ] All tests pass on fresh install
- [ ] All figures render correctly
- [ ] No API keys or secrets in code
- [ ] LICENSE file has correct year and names
- [ ] README citation has correct information
- [ ] Data availability statement is clear
- [ ] No broken links in documentation
- [ ] .gitignore excludes sensitive files

### Test As External User

**Pretend you're downloading this for the first time:**
1. Clone your GitHub repo in new folder
2. Follow INSTALLATION.md exactly
3. Run test suite
4. Generate figures
5. Read README start to finish

**If anything confuses you, fix it!**

---

## 11. Optional Enhancements

### Nice-to-Have (Not Required)

- [ ] Create `CHANGELOG.md` for version tracking
- [ ] Add GitHub Actions for CI/CD (run tests automatically)
- [ ] Create Jupyter notebook demo
- [ ] Add example usage in `examples/` directory
- [ ] Create project website (GitHub Pages)
- [ ] Record video walkthrough

---

## 12. Publication Timeline

### Recommended Order

1. **Today:** Complete steps 1-6 above (personalize, verify, review)
2. **This weekend:** Final proofread of technical report
3. **Monday:** Create GitHub repo (step 8)
4. **Tuesday:** Upload to arXiv, get ID
5. **Wednesday:** Update all citations with arXiv ID
6. **Thursday:** Submit to AIES (or target conference)
7. **After acceptance:** Make repo public, add badges, announce

---

## Need Help?

### Common Issues

**"Git won't initialize"**
- Solution: Make sure you're in `mirrorfield_publication/` directory

**"pip install fails"**
- Solution: Update pip: `pip install --upgrade pip`

**"Tests fail on fresh install"**
- Solution: Check Python version (need 3.9+)

**"Figures don't render"**
- Solution: Install matplotlib: `pip install matplotlib seaborn`

---

## Quick Reference: Files to Customize

**Must update:**
1. README.md (author, email, GitHub URL)
2. setup.py (author, email, url)
3. LICENSE (copyright holder name)
4. docs/TECHNICAL_REPORT.md (author name)

**Should update:**
5. INSTALLATION.md (contact email)
6. data/README.md (contact info)

**After arXiv upload:**
7. README.md (citation with arXiv ID)
8. All badge URLs

---

**Estimated total time to complete checklist:** 1.5-2 hours

**Ready to publish?** Follow steps 1-10, then proceed to step 11 (GitHub setup)!

Good luck with your submission! 🚀
