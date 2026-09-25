# Autonomous Agent Safety Guardrails & Ground Limits

> [!CAUTION]
> **CRITICAL OPERATIONAL BOUNDARIES**
> These rules are permanently active across this workspace to prevent catastrophic system damage, irreversible data loss, repository corruption, and hardware crashes.

---

## 1. Absolute File & System Protection
1. **Never Execute Destructive System Commands**:
   - Strictly banned: `rm -rf /`, `rm -rf ~`, `rm -rf *`, `mkfs`, `dd if=...`, `shred`, `chmod -R 777 /`.
   - Never modify or touch system root files (`/etc`, `/boot`, `/usr`, `/lib`, `/var`, `/sys`).
   - Never execute destructive commands with `sudo` unless explicitly requested and confirmed by the user.
2. **Never Bulk Delete User Project Files**:
   - Never delete source code files (`.py`, `.kt`, `.java`, `.cpp`, `.c`, `.h`), reports (`.pdf`, `.docx`, `.md`), Jupyter notebooks (`.ipynb`), or diagrams without explicit, specific confirmation.
   - Never delete or wipe project directories (`Smart system/`, `Smart_system_ass3/`, `Smart_system_ass4/`, `Mobile app/`, `PTTK/`, `VPProjects/`, `MyDataProjects/`).
   - For temporary cleanup, only target known safe build caches (`__pycache__`, `.pytest_cache`, `.gradle/caches`) and verify paths before removing.

---

## 2. Git & Version Control Guardrails
1. **Zero History Destruction**:
   - Strictly banned: `git push --force`, `git push -f`, `git clean -fdx`, `git reset --hard` (unless explicitly requested to undo a specific commit).
   - Before performing any git branch switch, stash or commit pending work to ensure zero work is lost.
2. **Pre-Push Large File Safety Check**:
   - Always run or verify through `sync_checkpoint.sh check-large-files`.
   - Never stage or commit files > 10MB (such as raw datasets, archive files `.tar.gz`, or model weights `.pth`, `.h5`, `.joblib`).
   - Never commit sensitive secrets, API keys, `.env` files, or personal access tokens.

---

## 3. Hardware Meltdown Protection (Laptop vs PC)
1. **Low-Resource Laptop Mode Safeguards**:
   - When running on `PORTABLE_LAPTOP` (CPU < 6 cores, RAM <= 8GB, or no discrete NVIDIA GPU):
     - **NEVER** launch full deep learning training sweeps, large epochs, or large batch sizes. Enforce `--debug --samples 16` mini-batch mode only.
     - **NEVER** launch an Android Studio AVD emulator. Always use a physical phone connected via ADB USB/Wi-Fi.
     - Never spawn unbounded multiprocessing pools that peg CPU at 100% and freeze the system.
2. **Orphan Process Cleanup**:
   - Whenever starting local servers (e.g. socket servers, Flask/FastAPI dev servers, test runners), ensure clean shutdown on exit. Never leave zombie processes running in the background consuming memory.

---

## 4. Verification Before Action Protocol
Before executing any file deletion, Git push, or multi-threaded command:
- **Inspect**: Confirm exact file paths and target destinations.
- **Isolate**: Limit changes strictly to the relevant project folder.
- **Inform**: Clearly state the non-destructive nature of operations to the user.
