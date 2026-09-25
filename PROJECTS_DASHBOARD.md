# Master Project Portfolio Dashboard

> **Last Updated**: 2026-09-23  
> **Active Host**: `dung-HP-Notebook` (PORTABLE_LAPTOP)  
> **Master Orchestrator**: Active via [`.agents/skills/portfolio-manager/SKILL.md`](file:///home/dung/Documents/.agents/skills/portfolio-manager/SKILL.md)  
> **Framework**: Fused with **Everything Claude Code (ECC)**

---

## 1. Executive Summary & Projects Overview

| # | Project / Course | Milestone / Focus | Status | Assigned Specialists (ECC Fused) | Next Deadline | Hardware Req |
| :-: | :--- | :--- | :---: | :--- | :---: | :---: |
| **1** | **ISAD / PTTK** (Information Systems Analysis & Design) | UML Architecture, ERD & Sequence Diagrams | 🟡 `IN_PROGRESS` | `isad-architect`<br/>`code-architect`<br/>`database-reviewer` | TBD | 💻 Laptop-Friendly |
| **2** | **Smart System** (AI / ML / DL / CNN) | Assignment 4 / LeNet, CNN & PyTorch Pipelines | 🟡 `IN_PROGRESS` | `ai-model-engineer`<br/>`mle-reviewer`<br/>`pytorch-build-resolver` | TBD | 🖥️ PC (GPU Training) / 💻 Laptop (Mini-batch) |
| **3** | **Mobile App Development** (Android Studio) | Java, XML Layouts, MVVM + LiveData, Room Database | ⚪ `NOT_STARTED` | `java-reviewer`<br/>`java-build-resolver` | TBD | 💻 Laptop (Code & Phone ADB) / 🖥️ PC (AVD) |
| **4** | **Network Programming** (L3/L4 Protocols & Sockets) | Asyncio TCP Server, Framing & Scapy Analysis | ⚪ `NOT_STARTED` | `network-socket-dev`<br/>`network-architect`<br/>`network-troubleshooter` | TBD | 💻 Laptop-Friendly |
| **5** | **Software Project Management** (SPM) | Project Charter, WBS & Verification Loop | 🟡 `IN_PROGRESS` | `spm-planner`<br/>`code-reviewer`<br/>`silent-failure-hunter` | TBD | 💻 Laptop-Friendly |

*Status Legend: 🟢 `COMPLETED` | 🟡 `IN_PROGRESS` | 🔴 `BLOCKED` | ⚪ `NOT_STARTED`*

---

## 2. Immediate Next Actions (Ranked by Device Fit)

### 💻 Laptop-Friendly Tasks (Execute Today on Laptop)
1. **[ISAD / PTTK]**: Use `isad-architect` and `database-reviewer` to refine database schema normalization and generate Mermaid Sequence Diagrams.
2. **[SPM]**: Use `spm-planner` to establish the Sprint 1 WBS and risk severity matrix with [`spm_guide.md`](file:///home/dung/Documents/.agents/skills/portfolio-manager/references/spm_guide.md).
3. **[Smart System]**: Use `mle-reviewer` to audit data split leakage, and test PyTorch CNN logic locally with `--debug --samples 16`.
4. **[Network]**: Use `network-architect` to build binary length-prefixed packet framing (`struct.pack('!I', len)`) in `asyncio`.
5. **[Mobile App]**: Apply low-spec `gradle.properties` limits and test Compose layouts directly on a physical phone via `adb connect`.

### 🖥️ High-Compute Tasks (Queue for Desktop PC)
1. **[Smart System]**: Pull git branch on Desktop PC and run full CUDA training sweeps with `pytorch-build-resolver`:
   ```bash
   python train.py --device cuda --epochs 50 --batch-size 64
   ```
2. **[Mobile App]**: Run full clean release builds (`./gradlew assembleRelease`) and launch Android Studio AVD emulators for screen recording.

---

## 3. Active Sprint Goals & Active ECC Skills

| Course | Sprint Focus | Active Skills |
| :--- | :--- | :--- |
| **ISAD / PTTK** | Sequence & Class Diagrams, Normalization | `api-design`, `backend-patterns` |
| **Smart System** | CNN Model Evaluation, Confusion Matrix | `mle-workflow`, `security-review` |
| **Mobile App** | Java MVVM Scaffold, Room DB, XML Layouts | `java-coding-standards`, `verification-loop` |
| **Network** | TCP Server Broadcast, Protocol Framing | `api-design`, `backend-patterns` |
| **SPM** | WBS Breakdown, Sprint Backlog, Risk Matrix | `plan-canvas`, `verification-loop`, `tdd-workflow` |

---

## 4. Cross-Device Handoff Log
- **2026-09-24** (`PORTABLE_LAPTOP`): Deployed **per-project agent architecture** — each of the 5 project folders now has its own `AGENTS.md` (specialist identity), `.agents/skills/` (domain-specific skills from ECC-main), `.agents/rules/` (language rules), and `.agents/subagents/` (specialist subagents). 12 skills copied from ECC-main; existing portfolio skills symlinked.
- **2026-09-23** (`PORTABLE_LAPTOP`): Fused Everything Claude Code (ECC) framework: imported 11 specialized subagents, 8 core skills (`mle-workflow`, `api-design`, `backend-patterns`, `verification-loop`, etc.), and Python/Kotlin rules.
- **2026-09-22** (`PORTABLE_LAPTOP`): Initialized Portfolio Management Skill and Central Dashboard. Pushed to GitHub.
