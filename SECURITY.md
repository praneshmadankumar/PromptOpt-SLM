# Security Policy

## Supported Versions

PromptOpt-SLM provides active security support for the following releases:

| Version | Supported          | Python Target       |
| ------- | ------------------ | ------------------- |
| 1.0.x   | :white_check_mark: | 3.10, 3.11, 3.12    |
| < 1.0   | :x:                | N/A                 |

---

## Security Philosophy & Scope

PromptOpt-SLM is a local-first, privacy-preserving research application designed to operate completely offline:

1. **No External Telemetry**: The application transmits zero user data, prompt text, or model weights to third-party endpoints.
2. **Offline Demarcation**: The bundled DEMO profile executes entirely in local CPU memory without network access.
3. **Adversarial Robustness Testing**: The included `RobustnessLab` explores controlled text perturbations (typos, word dropout, synonym replacement, and benign injection probes) strictly for reliability assessment. It does not contain exploit tools or malicious payloads.
4. **Local Database Security**: The SQLite experiment logger (`results/experiments.db`) uses strictly parameterized SQL queries to prevent SQL injection vulnerabilities.

---

## Reporting a Vulnerability

If you discover a security vulnerability or credential leak within this repository, please do **NOT** open a public issue.

Instead, please report the vulnerability privately:

- **Email**: Send detailed disclosure reports to `praneshmadankumar@gmail.com`
- **GitHub Security Advisories**: Submit a report directly via the GitHub Security Advisory portal for this repository.

Please include:
- A clear description of the vulnerability.
- Steps to reproduce the issue or a proof-of-concept script.
- Potential impact on local system execution or model weights.

### Response Timeline

- **Initial Acknowledgement**: Within 48 hours.
- **Triage & Status Update**: Within 5 business days.
- **Patch Release**: Released promptly following verification and resolution.
