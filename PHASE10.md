# Phase 10: Controlled Code Modification

`ModificationService` extends Phase 9's `CHANGE_PLAN` action. It requests a
plan through the existing action service, then asks the configured
`LLMProvider` for JSON change records only. The LLM has no shell interface and
cannot select test commands.

Changes are applied only to `RepositoryModel.local_path`, which is the
temporary clone supplied by the existing ingestion/cloning lifecycle. Paths
must be relative, remain within that workspace after symlink resolution, use
an allowed text extension, and avoid sensitive-looking filenames. The original
repository is never written, committed, pushed, or deployed.

Each run returns proposed and applied changes, actual unified diffs, validation
and test results, warnings, and errors. Python files receive syntax validation.
Tests are selected deterministically only for recognized Python test layouts
and run as `sys.executable -m pytest`, with `shell=False`, a timeout, capped
output, and a scrubbed environment. Unsupported layouts report `not_available`.

Validation failure, malformed LLM output, unsafe changes, and unsuccessful
tests restore the controlled workspace from the in-memory before-state while
retaining the generated diff for human review. No action commits, pushes,
creates pull requests, installs dependencies, or executes LLM-generated
commands.
