# Security Policy

## Supported Versions

Security fixes are provided for the latest released minor version.

## Reporting

Do not open a public issue for vulnerabilities involving arbitrary expression execution, path traversal, credential exposure or corrupted artifact reuse. Report the issue privately through the repository's GitHub security advisory page and include a minimal reproduction, affected version and expected impact.

The expression language is intentionally restricted. Any path that allows an expression to execute Python, import modules or access undeclared fields is treated as a security vulnerability.
