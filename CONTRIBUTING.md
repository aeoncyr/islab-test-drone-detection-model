# 🤝 Contributing to Vanilla Drone Detection

> [!NOTE]
> **Project Context**: This repository originated as part of the AI Engineering Researcher evaluation test for **ISLab, Pusan National University**. To support the long-term continuation of this research and facilitate open-source collaboration, these guidelines establish a clean, safe, and reproducible CI/CD development workflow.

Thank you for your interest in contributing to the **Vanilla Drone Detection** project! To maintain high code quality, scientific rigor, and a clean Git history, all contributors must adhere to our standardized development workflow.

---

## 🔄 Standard Contribution Lifecycle

We follow a strict **Issue ➔ Branch ➔ PR ➔ Review ➔ Merge** workflow:

```text
1. Issue Creation ────► 2. Topic Branch ────► 3. Development & Tests ────► 4. Pull Request ────► 5. Code Review ────► 6. Merge
 (Describe task)        (feat/ or fix/)         (Pass test_pipeline)         (Fill template)       (Maintainer approval)    (Squash/Merge)
```

---

### Step 1: Open an Issue 📝
- Before starting any development or modifying code, search existing issues to avoid duplication.
- Create a new issue using our structured templates:
  - **Bug Report**: For broken logic, memory leaks, or execution failures.
  - **Feature / Model Proposal**: For new architectural components, loss functions, or evaluation scripts.
- Wait for task discussion or alignment before investing extensive implementation time.

---

### Step 2: Create a Feature Branch 🌿
- Always branch off the latest `main` branch:
  ```bash
  git checkout main
  git pull origin main
  git checkout -b feat/your-feature-name   # For new features or models
  # or
  git checkout -b fix/issue-description   # For bug fixes
  ```
- **Branch Naming Conventions**:
  - `feat/<short-description>`: New architectures, loss functions, or scripts.
  - `fix/<short-description>`: Bug fixes or memory patches.
  - `docs/<short-description>`: Documentation or paper updates.
  - `perf/<short-description>`: Speed or memory optimizations.

---

### Step 3: Local Development & Verification ✅
- Write clean, modular, object-oriented code following the existing repository design patterns.
- Ensure all docstrings, type annotations, and unit tests are included.
- **Mandatory Verification**: Run the automated test suite before pushing:
  ```bash
  python test_pipeline.py
  ```
  *Your pull request will be blocked if any verification check fails.*

---

### Step 4: Commit Conventions (Conventional Commits) 💬
We enforce clean, conventional commit messages:
```bash
feat(models): implement dual-domain CBAM attention in CSPDarknet
fix(engine): resolve CUDA memory fragmentation during validation
docs(readme): add ONNX edge deployment benchmark instructions
perf(dataloader): optimize memory caching for multi-GPU training
```

---

### Step 5: Submit a Pull Request (PR) 🚀
- Push your branch to GitHub:
  ```bash
  git push -u origin feat/your-feature-name
  ```
- Open a Pull Request targeting `main`.
- Fill out all sections of the [Pull Request Template](.github/pull_request_template.md):
  - Link the corresponding Issue (`Closes #<issue_number>`).
  - Describe the problem and proposed solution.
  - Provide test execution logs.

---

### Step 6: Code Review & Merging 🔍
- All PRs require:
  1. Passing GitHub Actions CI automated pipeline tests.
  2. Formal review and approval by the repository maintainer ([@aeoncyr](https://github.com/aeoncyr)).
  3. Resolution of all review comments and discussion threads.
- Once approved, the maintainer will merge the PR via squash-and-merge or rebase-merge to maintain a clean linear commit history. Direct pushes to `main` are strictly prohibited.
