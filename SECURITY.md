# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue.
2. Email **security@flashvision.dev** with a detailed description.
3. Include steps to reproduce, impact assessment, and any suggested fix.

We will acknowledge receipt within 48 hours and aim to release a patch within 7 days for critical issues.

## Scope

- Code execution vulnerabilities in model loading / checkpoint deserialization
- Path traversal in file I/O utilities
- Arbitrary code execution via video file parsing
- Dependency vulnerabilities (please check upstream first)

## Out of Scope

- Adversarial attacks on generated videos (model robustness)
- Denial of service via large inputs (resource limits are the user's responsibility)
