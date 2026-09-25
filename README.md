# Multi-Project Portfolio Manager & ECC Agent System

An autonomous, multi-agent portfolio management framework built with Google Antigravity and fused with **Everything Claude Code (ECC)**. It orchestrates 5 concurrent software engineering university courses, routes tasks to specialized review lanes, and intelligently adapts compute workloads between a **portable, low-spec laptop** and a **high-performance desktop PC**.

> [!TIP]
> **Per-Project Agents**: Each project folder has its own `AGENTS.md` and `.agents/` directory with domain-specific skills, rules, and subagents. When you open a project folder, the agent automatically becomes a specialist for that domain.

---

## 🚀 Quick Start: How to Use Everyday

### 1. Daily Standup (Every Morning / Start of Session)
Just type in the chat:
> **"What should I work on today?"** *(or `/portfolio status`)*

The Master Orchestrator applies the **ECC 4-Tier Standup Triage**:
1. Scans your hardware via `check_device.sh` to classify host as `PORTABLE_LAPTOP` or `STRONG_PC`.
2. Reads [`PROJECTS_DASHBOARD.md`](./PROJECTS_DASHBOARD.md) to check milestones and sprint deadlines.
3. Recommends the 3 highest-priority tasks that fit your current machine's capabilities.

---

### 2. Working on a Specific Subject (Fused with ECC Specialists)
You can directly command the specialized subagents in plain English:

| If you want to... | Type this prompt: | Responsible Agent (ECC Fused) | Active Skill |
| :--- | :--- | :--- | :--- |
| **Design UML / Architecture** | *"Draft a Sequence Diagram for checkout flow in ISAD"* | `isad-architect` / `code-architect` | [`backend-patterns`](./.agents/skills/backend-patterns/) |
| **Review Database / ERD** | *"Audit foreign keys, indexing, and normalization"* | `database-reviewer` | [`api-design`](./.agents/skills/api-design/) |
| **Train AI / Deep Learning** | *"Build PyTorch CNN with mini-batch debug mode"* | `ai-model-engineer` | [`mle-workflow`](./.agents/skills/mle-workflow/) |
| **Review ML & Prevent Leakage** | *"Audit train/val/test split and feature leakage"* | `mle-reviewer` | [`mle-workflow`](./.agents/skills/mle-workflow/) |
| **Resolve PyTorch / CUDA Errors** | *"Fix CUDA device mismatch and DataLoader error"* | `pytorch-build-resolver` | [`mle-workflow`](./.agents/skills/mle-workflow/) |
| **Develop Android App** | *"Build Compose screen with ViewModel & StateFlow"* | `android-dev` | [Kotlin Rules](./.agents/rules/kotlin/) |
| **Review Kotlin & Coroutines** | *"Audit Compose recomposition and Coroutines safety"* | `kotlin-reviewer` | [Kotlin Rules](./.agents/rules/kotlin/) |
| **Fix Gradle Build Errors** | *"Resolve Gradle dependency conflict and memory limits"* | `kotlin-build-resolver` | [`android_guide.md`](./.agents/skills/portfolio-manager/references/android_guide.md) |
| **Build Sockets / Server** | *"Write async TCP server with length-prefixed framing"* | `network-socket-dev` | [`api-design`](./.agents/skills/api-design/) |
| **Troubleshoot Network Sockets** | *"Diagnose ECONNRESET and port reuse in socket server"* | `network-troubleshooter` | [`network_programming_guide.md`](./.agents/skills/portfolio-manager/references/network_programming_guide.md) |
| **Plan Sprints & WBS** | *"Generate WBS and Mermaid Gantt chart for SPM"* | `spm-planner` | [`plan-canvas`](./.agents/skills/plan-canvas/) |
| **Audit Code & Find Bugs** | *"Perform pre-commit review and hunt silent failures"* | `code-reviewer` / `silent-failure-hunter` | [`verification-loop`](./.agents/skills/verification-loop/) |

---

### 3. Switching Between Laptop & Desktop PC (Handoff)

#### When Leaving Your Laptop (e.g., leaving university/cafe):
Run in your terminal:
```bash
bash .agents/skills/portfolio-manager/scripts/sync_checkpoint.sh save "Done drafting model and UML on laptop"
```
*(This automatically scans for accidental large files > 10MB, commits your code, and pushes to GitHub).*

#### When Sitting at Your Strong PC (e.g., at home):
Run in your terminal:
```bash
bash .agents/skills/portfolio-manager/scripts/sync_checkpoint.sh load
```
*(This pulls your latest work. The agent on your PC will automatically detect your NVIDIA GPU and take over heavy CUDA training, Android emulators, and full builds).*

---

## 🛡️ Built-in Safety Guardrails & Ground Limits

The workspace is protected by an **always-on safety contract** ([`AGENTS.md`](./AGENTS.md)) and the **`safety-guard` skill**:

1. **No Catastrophic File Deletion**:
   - The agent is **strictly forbidden** from running destructive commands (`rm -rf /`, `rm -rf ~`, `rm -rf *`).
   - Source code (`.py`, `.kt`, `.java`), notebooks (`.ipynb`), reports (`.pdf`, `.docx`), and project directories will **never** be deleted without your explicit confirmation.
2. **Git History Protection**:
   - Forced pushes (`git push --force`) and destructive hard resets (`git reset --hard`) are blocked.
   - Files > 10MB (model weights, raw datasets) are intercepted before committing to protect your GitHub repository.
3. **Hardware Meltdown Protection**:
   - When on your weak laptop, the agent will **never** launch heavy CUDA training loops or Android Virtual Device (AVD) emulators that could freeze or crash your machine.
4. **ECC Verification Loop**:
   - Every modification is validated through tests, syntax checks, and workspace audits before being marked as done.
5. **Safety Audit Command**:
   - Ask the agent: *"Audit workspace safety"* to check disk space, runaway processes, and folder integrity at any time.

---

## 💻 Hardware Workload Matrix

| Task Category | Weak Laptop (`dung-HP-Notebook`) | Strong PC (`Desktop RTX 3060`) |
| :--- | :--- | :--- |
| **AI / Deep Learning** | Write code, verify mini-batch (`--debug --samples 16`) | Full GPU training (`--device cuda --epochs 50`), generate plots |
| **Android Studio** | Kotlin coding, test on **Physical Phone via USB/Wi-Fi ADB** | Android Studio AVD Emulators, clean release builds |
| **Network Sockets** | Localhost TCP/UDP testing, protocol framing | Docker network topologies, traffic load testing |
| **ISAD & SPM** | Write specs, render Mermaid UML diagrams, WBS | Review documents, export final presentation slide decks |

---

## 📂 Repository Layout (Fused with ECC)

```text
/home/dung/Documents/
├── README.md                                     # Fused documentation guide
├── AGENTS.md                                     # Always-on Safety Guardrails & Ground Limits
├── PROJECTS_DASHBOARD.md                         # Master 5-course tracking board
├── .gitignore                                    # Excludes datasets & weights from Git
├── .agents/
│   ├── rules/
│   │   ├── safety_guardrails.md                  # System protection rules
│   │   ├── python/                               # ECC Python coding & security rules
│   │   ├── kotlin/                               # ECC Kotlin / Android rules
│   │   └── common/                               # ECC Git, testing & review rules
│   ├── subagents/                                # Fused ECC Specialist Subagents
│   │   ├── isad-architect.md                     # Systems & UML Architect
│   │   ├── code-architect.md                     # Modular Code Architect
│   │   ├── database-reviewer.md                  # ERD & Schema Normalization
│   │   ├── mle-reviewer.md                       # Data Leakage & ML Pipeline Reviewer
│   │   ├── pytorch-build-resolver.md             # CUDA / Tensor Shape Resolver
│   │   ├── kotlin-reviewer.md                    # Compose & Coroutines Reviewer
│   │   ├── kotlin-build-resolver.md              # Gradle & Build Cache Resolver
│   │   ├── network-architect.md                  # Sockets & Protocol Architect
│   │   ├── network-troubleshooter.md             # Connection & Packet Troubleshooter
│   │   ├── spm-planner.md                        # WBS & Sprint Planner
│   │   ├── code-reviewer.md                      # Pre-commit Quality Reviewer
│   │   └── silent-failure-hunter.md              # Exception & Bug Hunter
│   └── skills/
│       ├── portfolio-manager/                    # Master Orchestration Skill
│       ├── safety-guard/                         # System Integrity & Safety Skill
│       ├── mle-workflow/                         # ECC ML Engineering Workflow
│       ├── backend-patterns/                     # ECC Clean Architecture Patterns
│       ├── api-design/                           # ECC API & Protocol Design
│       ├── verification-loop/                    # ECC Verification Loop
│       ├── tdd-workflow/                         # ECC Test-Driven Development
│       ├── security-review/                      # ECC Security & Secrets Review
│       ├── strategic-compact/                    # ECC Context Window Optimization
│       └── plan-canvas/                          # ECC Milestone & Plan Canvas
├── PTTK/                                         # Project 1: ISAD / PTTK
├── Smart system/                                 # Project 2: AI / Smart Systems
├── Mobile app/                                   # Project 3: Android Studio
├── Network Programming/                          # Project 4: Socket Programming
└── SPM/                                          # Project 5: Software Project Management
```

---

## 🔗 Syncing on Your Strong PC

To sync all these fused ECC capabilities to your desktop PC, run:

```bash
cd ~/Documents
git pull origin main
```

*Antigravity on your PC will automatically detect the updated `.agents/` directory, loading all 11 subagents, 10 skills, and safety rules instantly!*
