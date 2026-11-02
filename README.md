Python environment setup for this project

Windows (PowerShell):

1. Create virtual environment (if needed):

   ```powershell
   python -m venv .venv
   ```

2. Activate the venv:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Run tests:

   ```powershell
   .\.venv\Scripts\python.exe -m pytest
   ```

VS Code: select the interpreter at `.venv\Scripts\python.exe` or the workspace will use the configured interpreter by the included settings file.
