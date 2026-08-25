# Release process

This document is the maintainer checklist for SubBake releases. Release builds
come from GitHub Actions, never from a developer workstation.

## 1. Prepare the release commit

1. Work from a clean branch based on `main`.
2. Update the workspace version in `Cargo.toml` and keep `Cargo.lock` in sync.
3. Update `CHANGELOG.md`, `README.md`, and `docs/release-notes/vVERSION.md`.
4. Confirm configuration and persisted-data changes follow
   `docs/compatibility.md`.
5. Run the complete local verification suite:

   ```bash
   cargo fmt --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace --no-fail-fast
   cargo build --locked --release -p subbake-cli
   cargo audit
   cargo deny check licenses bans sources
   ```

   The release workflow pins the exact toolchain in `rust-toolchain.toml`. The
   lower `rust-version` declared in the manifests is the separately tested MSRV.

6. Merge or push the release commit and wait for every required GitHub check on
   that exact commit.

## 2. Build a candidate without publishing

Run the `Release` workflow manually from the target commit. The manual run builds
the same GNU and musl Linux x64 archives as a tag build, verifies the GNU glibc
baseline and musl static-link contract, runs smoke checks, and uploads a workflow
artifact without creating a GitHub Release.

Download the workflow artifact and verify the GNU archive on clean Ubuntu 22.04
and Ubuntu 24.04 installations, and the musl archive on a clean Alpine system.
At minimum, test `--version`, `--help`, completion generation, provider
configuration failure, FFmpeg discovery, bubblewrap discovery, and managed
whisper.cpp status/install behavior.

## 3. Create the tag

Ensure the local checkout is clean and is the exact commit that passed CI:

```bash
git status --short
git rev-parse HEAD
git tag -s -a v0.2.0-alpha.1 -m "SubBake 0.2.0-alpha.1"
git push origin v0.2.0-alpha.1
```

Use the actual release version in place of the example. Never move or reuse a
published tag; fix a bad release with a new version.

## 4. Inspect the draft

A tag push makes the `Release` workflow rebuild the archive and create a draft
pre-release. Download the draft assets and verify:

```bash
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf subbake-v0.2.0-alpha.1-x86_64-unknown-linux-gnu.tar.gz
./subbake-v0.2.0-alpha.1-x86_64-unknown-linux-gnu/sbake --version
tar -xzf subbake-v0.2.0-alpha.1-x86_64-unknown-linux-musl.tar.gz
./subbake-v0.2.0-alpha.1-x86_64-unknown-linux-musl/sbake --version
```

Confirm that the reported version and Git revision match the release tag, the
release notes list external dependencies and known limitations, the provenance
attestation is present, and the exact tagged source remains downloadable.

## 5. Publish and verify

Publish the GitHub draft while leaving alpha, beta, and release-candidate builds
marked as pre-releases. Verify the release page and assets from a signed-out
browser session. Record newly discovered problems as issues and publish fixes as
a new version instead of replacing existing assets.
