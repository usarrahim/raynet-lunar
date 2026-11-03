---
name: next-task
description: Do the next unchecked task from docs/08_ROADMAP.md step by step, test it, log it, and report in Turkish.
disable-model-invocation: true
argument-hint: "[task id, for example P1.3] (optional)"
---
Do exactly one roadmap task.

1. Read `PROGRESS.md` and `docs/08_ROADMAP.md`. If "$ARGUMENTS" names a task id, do that task.
   Otherwise take the first unchecked task whose earlier tasks are done.
2. Read the docs that the task points to, and follow the rules in `CLAUDE.md`.
3. Before editing any file, write the plan in Turkish: goal, files to create or change, how you
   will test it, and whether it needs GPU hours or the owner's hands.
   If the task is marked [GPU] or [Owner], stop after the plan and wait for the owner's "tamam".
4. Implement in small steps. For pure functions (physics, metrics) write the tests first.
5. Run `pytest -q`. If training code changed, also run the CPU smoke test. Fix until green.
6. Update `PROGRESS.md` (date, task, what changed, commands, results or TBD) and tick the box in
   `docs/08_ROADMAP.md`. Commit with a short English message, for example `P1.3: physics module + tests`.
7. Answer in Turkish with three short sections:
   **Ne yaptık?**, **Nasıl çalıştırılır?**, **Ne öğrendik?** (3-5 simple bullets that explain the
   key idea, so the owner can present it in class). End with the next task id.
