# Ghid de Testare - Cyber Scout Desktop

## Fix-uri Aplicate (v2.0.0)

### A) WebView2 Pre-check (fără loop)
- ✅ Verificare registry Windows pentru WebView2 Runtime
- ✅ Verificare import pywebview
- ✅ Verificare existență msedgewebview2.exe
- ✅ UN SINGUR dialog dacă lipsește ("WebView2 Required")
- ✅ Opțiune de download + exit (nu retry)

### B) Retry Limitat pentru Streamlit
- ✅ Maximum 3 încercări
- ✅ Backoff exponențial: 2s, 5s, 10s
- ✅ Timeout 90s per încercare
- ✅ Kill complet subprocess între încercări
- ✅ UN SINGUR dialog final la eșec

### C) Zero Popup Spam
- ✅ Single-instance lock (mutex pe Windows)
- ✅ Rate-limiting: max 1 dialog per run
- ✅ Dialog "Already Running" dacă e deschis deja

### D) Logging pentru Debug
- ✅ Log file: `%TEMP%/cyberscout_launcher.log` (frozen) sau `outputs/cyberscout_launcher.log` (dev)
- ✅ Flag `--debug` pentru output în consolă
- ✅ Logare: port, cmd, retries, health-check, stderr, motiv WebView2 failure

---

## Cum Testezi

### 1. Testare în Development (fără build)

```bash
cd C:\Users\Mafia\Downloads\disertatie\imagetrust

# Testare normală
python -m imagetrust desktop

# Testare cu debug logging
python -m imagetrust desktop --debug

# Testare cu port specific
python -m imagetrust desktop --debug --port 8502
```

### 2. Testare Cyber Scout (doar Streamlit, fără fereastră nativă)

```bash
# Rulează doar serverul Streamlit
python -m imagetrust cyber

# Apoi deschide în browser: http://localhost:8501
```

### 3. Testare EXE (după build)

```bash
# Build
cd C:\Users\Mafia\Downloads\disertatie\imagetrust
pyinstaller CyberScout.spec --clean

# Rulare
dist\CyberScout\CyberScout.exe

# Dacă ai probleme, verifică log-ul
type %TEMP%\cyberscout_launcher.log
```

---

## Scenarii de Test

### Scenariu 1: Pornire Normală
1. Rulează `imagetrust desktop`
2. **Așteptat:**
   - Afișează "Checking for existing instances..."
   - Afișează "Checking WebView2 runtime..."
   - Afișează "WebView2 OK"
   - Afișează "Finding available port..."
   - Afișează "Using port: 8501"
   - Afișează "Starting analysis server..."
   - Afișează "Server ready!"
   - Se deschide fereastra Cyber Scout

### Scenariu 2: Instanță Dublă
1. Rulează `imagetrust desktop` (prima instanță)
2. Într-un alt terminal, rulează din nou `imagetrust desktop`
3. **Așteptat:**
   - A doua instanță afișează UN SINGUR dialog "Already Running"
   - A doua instanță se închide imediat (exit 0)
   - Prima instanță continuă să funcționeze

### Scenariu 3: WebView2 Lipsă (simulare)
1. Dezinstalează WebView2 Runtime (sau testează pe o mașină fără)
2. Rulează `imagetrust desktop`
3. **Așteptat:**
   - UN SINGUR dialog "WebView2 Required"
   - Opțiune de download (YES deschide browser)
   - Aplicația se închide (exit 1)
   - **NU** apar mai multe dialoguri

### Scenariu 4: Port Ocupat
1. Pornește alt server pe portul 8501: `python -m http.server 8501`
2. Rulează `imagetrust desktop`
3. **Așteptat:**
   - Găsește automat alt port (8502, 8503, etc.)
   - Funcționează normal

### Scenariu 5: Streamlit Crash (simulare)
1. Modifică temporar `cyber_app.py` pentru a da crash la import
2. Rulează `imagetrust desktop`
3. **Așteptat:**
   - Face 3 încercări cu backoff
   - UN SINGUR dialog "Server Start Failed"
   - Nu apare spam de dialoguri

### Scenariu 6: Debug Mode
1. Rulează `imagetrust desktop --debug`
2. **Așteptat:**
   - Output detaliat în consolă
   - Fiecare pas este logat
   - Log file conține informații complete

---

## Verificare Log File

### Locație Log
- **Development:** `imagetrust/outputs/cyberscout_launcher.log`
- **Frozen (EXE):** `%TEMP%/cyberscout_launcher.log`

### Exemplu Log Normal
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

## Confirmare Fix-uri

### ✅ NU mai există loop infinit
- Maximum 3 încercări pentru Streamlit
- WebView2 check face UN SINGUR pass

### ✅ NU mai există popup spam
- Rate limit: `_dialog_shown` flag
- Maximum 1 dialog per run

### ✅ NU mai există OOM
- Cleanup corect al proceselor
- `taskkill /F /T` pentru Windows
- Timeout-uri la fiecare operație

---

## Fișiere Modificate

| Fișier | Modificări |
|--------|------------|
| `src/imagetrust/frontend/desktop_launcher.py` | Rescris complet v2.0.0 |
| `src/imagetrust/cli.py` | Adăugat `--debug` și `--port` pentru desktop command |
| `CyberScout.spec` | Module adiționale, comentarii, icon support |

---

## Troubleshooting

### "WebView2 Required" dar e instalat
1. Verifică versiunea WebView2: `reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"`
2. Reinstalează WebView2 Evergreen: https://developer.microsoft.com/microsoft-edge/webview2/

### "Server Start Failed" după 3 încercări
1. Verifică log-ul pentru erori specifice
2. Testează Streamlit separat: `python -m imagetrust cyber`
3. Verifică dacă portul e liber: `netstat -ano | findstr 8501`

### Fereastra nu se deschide
1. Rulează cu `--debug` pentru mai multe detalii
2. Verifică `%TEMP%\cyberscout_launcher.log`
3. Încearcă să rulezi ca Administrator

### Lock file blocat (după crash)
1. Șterge manual: `del %TEMP%\cyberscout_desktop.lock`
2. Închide toate procesele python: `taskkill /F /IM python.exe`

---

*Document actualizat pentru Cyber Scout Desktop v2.0.0*
