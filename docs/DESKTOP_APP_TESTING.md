# Testing Guide - Cyber Scout Desktop

## Applied Fixes (v2.0.0)

### A) WebView2 Pre-check (no loop)
- ✅ Windows registry check for WebView2 Runtime
- ✅ pywebview import verification
- ✅ msedgewebview2.exe existence check
- ✅ A SINGLE dialog if missing ("WebView2 Required")
- ✅ Download option + exit (no retry)

### B) Limited Retry for Streamlit
- ✅ Maximum 3 attempts
- ✅ Exponential backoff: 2s, 5s, 10s
- ✅ 90s timeout per attempt
- ✅ Full subprocess kill between attempts
- ✅ A SINGLE final dialog on failure

### C) Zero Popup Spam
- ✅ Single-instance lock (mutex on Windows)
- ✅ Rate-limiting: max 1 dialog per run
- ✅ "Already Running" dialog if already open

### D) Logging for Debug
- ✅ Log file: `%TEMP%/cyberscout_launcher.log` (frozen) or `outputs/cyberscout_launcher.log` (dev)
- ✅ `--debug` flag for console output
- ✅ Logging: port, cmd, retries, health-check, stderr, WebView2 failure reason

---

## How to Test

### 1. Testing in Development (without build)

```bash
cd C:\Users\Mafia\Downloads\disertatie\imagetrust

# Normal testing
python -m imagetrust desktop

# Testing with debug logging
python -m imagetrust desktop --debug

# Testing with specific port
python -m imagetrust desktop --debug --port 8502
```

### 2. Testing Cyber Scout (Streamlit only, without native window)

```bash
# Run only the Streamlit server
python -m imagetrust cyber

# Then open in browser: http://localhost:8501
```

### 3. Testing EXE (after build)

```bash
# Build
cd C:\Users\Mafia\Downloads\disertatie\imagetrust
pyinstaller CyberScout.spec --clean

# Run
dist\CyberScout\CyberScout.exe

# If you have issues, check the log
type %TEMP%\cyberscout_launcher.log
```

---

## Test Scenarios

### Scenario 1: Normal Startup
1. Run `imagetrust desktop`
2. **Expected:**
   - Displays "Checking for existing instances..."
   - Displays "Checking WebView2 runtime..."
   - Displays "WebView2 OK"
   - Displays "Finding available port..."
   - Displays "Using port: 8501"
   - Displays "Starting analysis server..."
   - Displays "Server ready!"
   - The Cyber Scout window opens

### Scenario 2: Duplicate Instance
1. Run `imagetrust desktop` (first instance)
2. In another terminal, run `imagetrust desktop` again
3. **Expected:**
   - The second instance displays A SINGLE "Already Running" dialog
   - The second instance closes immediately (exit 0)
   - The first instance continues to function

### Scenario 3: Missing WebView2 (simulation)
1. Uninstall WebView2 Runtime (or test on a machine without it)
2. Run `imagetrust desktop`
3. **Expected:**
   - A SINGLE "WebView2 Required" dialog
   - Download option (YES opens browser)
   - The application closes (exit 1)
   - Multiple dialogs do **NOT** appear

### Scenario 4: Port Occupied
1. Start another server on port 8501: `python -m http.server 8501`
2. Run `imagetrust desktop`
3. **Expected:**
   - Automatically finds another port (8502, 8503, etc.)
   - Functions normally

### Scenario 5: Streamlit Crash (simulation)
1. Temporarily modify `cyber_app.py` to crash on import
2. Run `imagetrust desktop`
3. **Expected:**
   - Makes 3 attempts with backoff
   - A SINGLE "Server Start Failed" dialog
   - No dialog spam appears

### Scenario 6: Debug Mode
1. Run `imagetrust desktop --debug`
2. **Expected:**
   - Detailed output in the console
   - Each step is logged
   - Log file contains complete information

---

## Log File Verification

### Log Location
- **Development:** `imagetrust/outputs/cyberscout_launcher.log`
- **Frozen (EXE):** `%TEMP%/cyberscout_launcher.log`

### Normal Log Example
```
2025-01-21 15:30:00 [INFO] ======================================================================
2025-01-21 15:30:00 [INFO] Cyber Scout Desktop Launcher v2.0.0
2025-01-21 15:30:00 [INFO] Timestamp: 2025-01-21T15:30:00.123456
2025-01-21 15:30:00 [INFO] Debug mode: False
2025-01-21 15:30:00 [INFO] Frozen (PyInstaller): False
2025-01-21 15:30:00 [INFO] Python: 3.10.11 (main, ...)
2025-01-21 15:30:00 [INFO] ======================================================================
2025-01-21 15:30:00 [INFO] Step 1: Checking for existing instances...
2025-01-21 15:30:00 [INFO] Acquiring lock: C:\Users\...\Temp\cyberscout_desktop.lock
2025-01-21 15:30:00 [INFO] Lock acquired (Windows msvcrt)
2025-01-21 15:30:00 [INFO] Step 2: Checking WebView2...
2025-01-21 15:30:00 [INFO] Checking WebView2 availability...
2025-01-21 15:30:00 [INFO] WebView2 found in registry: 120.0.2210.91
2025-01-21 15:30:00 [INFO] pywebview imported successfully, version: 4.4.1
2025-01-21 15:30:00 [INFO] WebView2 available: version=120.0.2210.91
2025-01-21 15:30:00 [INFO] Step 4: Finding available port...
2025-01-21 15:30:00 [INFO] Finding free port starting from 8501...
2025-01-21 15:30:00 [INFO] Found free port: 8501
2025-01-21 15:30:00 [INFO] Step 5: Starting analysis server...
2025-01-21 15:30:00 [INFO] Streamlit start attempt 1/3
2025-01-21 15:30:00 [INFO] Starting Streamlit server
2025-01-21 15:30:00 [INFO]   App path: ...\cyber_app.py
2025-01-21 15:30:00 [INFO]   Port: 8501
2025-01-21 15:30:00 [INFO] Streamlit process started with PID 12345
2025-01-21 15:30:00 [INFO] Waiting for server on port 8501 (timeout: 90s)
2025-01-21 15:30:15 [INFO] Server ready after 15.2s (15 checks)
2025-01-21 15:30:15 [INFO] Streamlit started successfully!
2025-01-21 15:30:15 [INFO] Step 6: Creating application window...
2025-01-21 15:30:15 [INFO] Window URL: http://127.0.0.1:8501
2025-01-21 15:30:15 [INFO] Starting webview event loop...
```

---

## Fix Confirmation

### ✅ Infinite loop no longer exists
- Maximum 3 attempts for Streamlit
- WebView2 check makes a SINGLE pass

### ✅ Popup spam no longer exists
- Rate limit: `_dialog_shown` flag
- Maximum 1 dialog per run

### ✅ OOM no longer exists
- Proper process cleanup
- `taskkill /F /T` for Windows
- Timeouts on each operation

---

## Modified Files

| File | Changes |
|------|---------|
| `src/imagetrust/frontend/desktop_launcher.py` | Completely rewritten v2.0.0 |
| `src/imagetrust/cli.py` | Added `--debug` and `--port` for desktop command |
| `CyberScout.spec` | Additional modules, comments, icon support |

---

## Troubleshooting

### "WebView2 Required" but it is installed
1. Check WebView2 version: `reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"`
2. Reinstall WebView2 Evergreen: https://developer.microsoft.com/microsoft-edge/webview2/

### "Server Start Failed" after 3 attempts
1. Check the log for specific errors
2. Test Streamlit separately: `python -m imagetrust cyber`
3. Check if the port is free: `netstat -ano | findstr 8501`

### Window does not open
1. Run with `--debug` for more details
2. Check `%TEMP%\cyberscout_launcher.log`
3. Try running as Administrator

### Lock file stuck (after crash)
1. Delete manually: `del %TEMP%\cyberscout_desktop.lock`
2. Close all python processes: `taskkill /F /IM python.exe`

---

*Document updated for Cyber Scout Desktop v2.0.0*
