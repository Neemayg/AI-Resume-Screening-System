# 🚀 Setup Guide for Friend

## First Time Setup - Follow Step by Step

### Step 1: Clone the Repository
```bash
cd ~/Desktop
git clone https://github.com/Neemayg/AI-Resume-Screening-System.git
cd AI-Resume-Screening-System
```

### Step 2: Give Execute Permissions to Scripts
```bash
chmod +x setup.sh start.sh sync.sh
```

### Step 3: Run Setup (One Time Only)
```bash
./setup.sh
```

This will:
- ✅ Check Python installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Set everything up

### Step 4: Start the Server
```bash
./start.sh
```

---

## When Your Friend Pushes New Code

### Just Run This (Automatic Everything!)
```bash
./sync.sh
./start.sh
```

That's it! ✅

---

## If You Get Errors

### Error: "Permission denied"
```bash
chmod +x setup.sh start.sh sync.sh
```

### Error: "Python not found"
Install Python 3.9 or higher:
- Mac: `brew install python3`
- Download from: https://www.python.org/downloads/

### Error: "No module named uvicorn"
```bash
cd backend
source venv/bin/activate
pip3 install -r requirements.txt
```

### Error: "Could not import module 'app'"
You're using wrong command! Use:
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

OR just use the script:
```bash
./start.sh
```

### Error: "Port 8000 already in use"
Kill the existing process:
```bash
lsof -ti:8000 | xargs kill -9
```

Then start again:
```bash
./start.sh
```

---

## Complete Manual Steps (If Scripts Don't Work)

### 1. Fresh Start
```bash
cd AI-Resume-Screening-System
rm -rf backend/venv
```

### 2. Create Virtual Environment
```bash
cd backend
python3 -m venv venv
```

### 3. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Start Server
```bash
uvicorn main:app --reload --port 8000
```

### 6. Open Frontend
Open `frontend/index.html` in your browser

---

## Quick Reference

| What to Do | Command |
|------------|---------|
| First time setup | `./setup.sh` |
| Start server | `./start.sh` |
| Get latest code | `./sync.sh` then `./start.sh` |
| Stop server | Press `CTRL+C` |
| Manual start | `cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000` |

---

## Still Getting Errors?

1. **Take a screenshot of the error**
2. **Copy the full error message**
3. **Send it to your friend**

Common things to check:
- ✅ Are you in the right directory? (`pwd` should show `.../AI-Resume-Screening-System`)
- ✅ Did you pull latest code? (`git pull origin main`)
- ✅ Did you run setup? (`./setup.sh`)
- ✅ Is Python 3.9+ installed? (`python3 --version`)
