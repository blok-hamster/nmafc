# Contributing to NMAFC

Thanks for your interest. NMAFC is developed by a very small team, so set
expectations early: we review thoughtfully but slowly, and we are strict about
things that change retrieval results.

## Before you start

- **Read [`RESEARCH.md`](RESEARCH.md).** It documents what has been measured,
  what failed, and what we deliberately left unwired. It is the fastest way to
  understand where the design boundaries are and why they exist.
- **Open an issue before a large change.** If you are proposing something that
  alters retrieval, scoring, decay, or the storage schema, we want to discuss
  the design before you spend the effort. Several of the most valuable things in
  this repo came out of a design conversation in an issue thread.
- **Contributions require a CLA signature.** See
  [`CONTRIBUTOR_LICENSE_AGREEMENT.md`](CONTRIBUTOR_LICENSE_AGREEMENT.md). This is
  a low-friction sign-off, not a legal negotiation, but it is required for
  inbound contributions.

## Development setup

Requires Python 3.11+ (3.12 recommended; CI runs 3.11 and 3.12).

```bash
git clone https://github.com/blok-hamster/nmafc.git
cd nmafc

# uv is the project's package manager; the lockfile is authoritative
uv sync --extra all

# quick verification
uv run pytest            # 625 tests, ~30s, zero network
uv run ruff check .
```

### Running the test suite

```bash
NMAFC_RUN_LIVE=0 uv run pytest          # default: skips paid provider calls
uv run pytest -k decay                  # single area
uv run pytest tests/integration         # integration-shaped tests
```

Tests are deterministic and run fully offline. Anything that needs real provider
credentials is marked `live` and is skipped unless `NMAFC_RUN_LIVE=1`.

**Please keep the suite offline.** The mocking pattern in `tests/` — real
LanceDB, real SQLite, real SQL, fake providers subclassing the real ABCs — is
deliberate. Integration-shaped unit tests catch more than shallow mocks, and
they run in 30 seconds instead of requiring an API budget.

## Code standards

- **Ruff** for lint and import order (`line-length = 100`). CI enforces it.
- **Type hints** on new public functions. `mypy` is configured strict but is not
  yet enforced in CI — it reports outstanding findings, and new code should not
  add to them.
- **Comments should explain *why*, not *what*.** The existing docstrings carry
  measured negative results and reasoning about trade-offs. That is the standard
  here: if you tried two approaches and one was worse, say so in the code, with
  the number. A future maintainer will thank you and a reviewer will trust you.
- **Do not add silent failure paths.** If a parse can fail, log it or raise.
  Swallowing an exception so a turn "succeeds" is the failure mode this project
  most needs to avoid — a memory system that silently loses facts is worse than
  one that errors.

## Changing retrieval or scoring

This area has extra expectations:

1. **A change must come with measurement.** `scripts/benchmarks/` has the paired
   harness. State what you expect to move and by how much, then show it did.
2. **Negative results belong in the repo.** `RESEARCH.md` and
   `README.md` document features that were measured and rejected. Adding yours
   keeps the project honest and is genuinely valuable.
3. **Do not tune against the benchmark and ship it as a general improvement.**
   Several components were A/B-selected against paired answers. If a default
   changes, justify it on mechanism, not on the scorecard.
4. **Configuration defaults must be documented.** `schemas/memory.py` fields are
   merged flat from `[decay]` and `[retrieval]` in `configs/default.toml`; a new
   field needs a description that says what it does at zero.

## Pull requests

- One logical change per PR. Split refactors from behaviour changes.
- Update `README.md` and/or `RESEARCH.md` if you change documented behaviour.
- Add a `CHANGELOG.md` entry under `## [Unreleased]`.
- CI must be green: `uv lock --check`, `ruff`, and `pytest`.

## Adding a provider

See `examples/custom_provider.py` and `src/nmafc/integration/base.py`. The
extension surface is two abstract methods. Note that registration is currently a
hand-written dispatch chain in `factory.py` — there is no plugin registry — so a
new provider requires editing that file.

## Reporting security issues

Do **not** open a public issue for a security vulnerability. See
[`SECURITY.md`](SECURITY.md) for the reporting channel.

## Licence

By contributing you agree your contribution is licensed under the PolyForm
Noncommercial License 1.0.0, per the CLA. See [`LICENSE`](LICENSE).