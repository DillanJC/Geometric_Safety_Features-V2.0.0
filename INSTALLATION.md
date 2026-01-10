# Installation Guide

## Quick Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Detailed Setup

### 1. Clone Repository

```bash
git clone https://github.com/DillanJC/mirrorfield_publication.git
cd mirrorfield_publication
```

### 2. Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n mirrorfield python=3.9
conda activate mirrorfield
```

### 3. Install Dependencies

**Core dependencies only:**
```bash
pip install numpy scipy scikit-learn
```

**With plotting support:**
```bash
pip install -r requirements.txt
```

**Install package:**
```bash
pip install -e .  # Development mode
# or
pip install .     # Standard installation
```

### 4. Verify Installation

```bash
python -c "from mirrorfield.geometry import GeometryBundle; print('Success!')"
```

### 5. Run Tests

```bash
python experiments/test_phase_e_bundle.py
```

Expected output:
```
Test 1: GeometryBundle initialization... PASS
Test 2: Feature computation... PASS
Test 3: Feature matrix shape... PASS
Test 4: Feature values... PASS
Test 5: Summarize method... PASS
Test 6: Batch-order invariance... PASS

All tests passed!
```

---

## Optional: Behavioral Flip Experiment

To run the behavioral flip experiment, you need an OpenAI API key:

### 1. Install Additional Dependencies

```bash
pip install openai python-dotenv
```

### 2. Set API Key

**Option A: Environment Variable**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option B: .env File**
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 3. Run Behavioral Flip Pipeline

```bash
python experiments/behavioral_flip_sample_selection.py
python experiments/behavioral_flip_generate_paraphrases.py  # Cost: ~$0.90
python experiments/behavioral_flip_compute_flips.py
python experiments/behavioral_flip_analyze.py
```

---

## Troubleshooting

### ImportError: No module named 'mirrorfield'

**Solution:** Install the package in development mode:
```bash
pip install -e .
```

### NumPy version conflicts

**Solution:** Upgrade NumPy:
```bash
pip install --upgrade numpy>=1.24.0
```

### Plotting issues (matplotlib)

**Solution:** Install with display backend:
```bash
# Linux/Mac
pip install matplotlib

# Windows (if using WSL)
export DISPLAY=:0
```

### OpenAI API errors

**Check API key:**
```bash
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

**Common issues:**
- Key not set: Export environment variable
- Invalid key: Check for typos, regenerate key
- Rate limits: Wait and retry (behavioral flip uses ~180 API calls)

---

## System Requirements

### Minimum
- Python 3.9+
- 2GB RAM
- 100MB disk space

### Recommended
- Python 3.10+
- 8GB RAM (for larger datasets)
- 500MB disk space (including data)

### Platform Support
- ✅ Linux (tested on Ubuntu 20.04+)
- ✅ macOS (tested on macOS 12+)
- ✅ Windows 10/11 (tested on Windows 10)

---

## Docker (Optional)

Coming soon: Dockerfile for containerized deployment.

---

## Need Help?

- **Documentation:** See `docs/TECHNICAL_REPORT.md`
- **Issues:** [GitHub Issues](https://github.com/DillanJC/mirrorfield_publication/issues)
- **Email:** DillanJC91@Gmail.com
