# Repository Guidelines

## Project Structure & Module Organization
- Notebooks: Top-level `T3_Exercise_*.ipynb` files contain exercises (e.g., tensor fundamentals, activation functions).
- Rubric: `Simple_Lab_Assessment_Rubric.md` describes evaluation criteria.
- Optional folders (if added): place datasets in `data/`, images in `assets/`, and reusable helpers in `utils/`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install basics: `pip install jupyter numpy matplotlib` (plus your chosen ML stack, e.g., PyTorch).
- Launch UI: `jupyter lab` (or `jupyter notebook`) to edit and run exercises.
- Execute headless: `jupyter nbconvert --to notebook --execute T3_Exercise_1_Tensor_Fundamentals.ipynb --output exec_out.ipynb`

## Coding Style & Naming Conventions
- Language: Python in notebooks; follow PEP 8 with 4‑space indentation.
- Names: `snake_case` for variables/functions, `CapWords` for classes.
- Notebook names: `T3_Exercise_<N>_<Topic>.ipynb` (keep sequential numbers and clear topics).
- Cells: prefer small, focused cells; add brief markdown headers for sections.

## Testing Guidelines
- Run-all: use “Restart & Run All” to ensure order-independent execution.
- Assertions: include `assert` checks for key tensors/shapes/values.
- Determinism: set seeds where relevant (e.g., `torch.manual_seed(0)`).
- Automated run: validate notebooks execute via the nbconvert command above; investigate and fix any cell failures.

## Commit & Pull Request Guidelines
- Commits: short, imperative, scoped to a notebook/topic.
  - Examples: `exercise: fix softmax edge case`, `nb: add shape asserts to reduction ops`.
- Hygiene: avoid committing large data, secrets, or bulky outputs; clear noisy outputs before commit when practical.
- PRs: include a summary, changed notebooks list, screenshots of key plots/tables, and any environment notes (Python/Lib versions).

## Security & Configuration Tips
- Do not embed credentials or tokens in notebooks.
- Use a `.gitignore` to exclude `.ipynb_checkpoints/`, large datasets, and temporary outputs.
