# Security policy

## Reporting a vulnerability

If you find a security issue in `trtcheck`, please **do not** open a
public issue. Use one of:

- GitHub's [private vulnerability reporting](https://github.com/sohams25/trtcheck/security/advisories/new)
  on this repository. Preferred.
- Email the maintainer at `sohams.web@gmail.com` with the details.

You should get an acknowledgement within five business days. Fixes
typically ship in the next patch release; severe issues get a release
out-of-cycle.

## Supported versions

trtcheck follows semver from v1.0 onward. Only the current minor
receives security fixes:

| Version | Supported |
|---|---|
| 1.x (latest minor) | yes |
| 0.x | no |

If you're running 0.x for a reason that prevents upgrading, mention it
in your report and we'll discuss backports.

## Scope

In scope:

- The `trtcheck` CLI and library, including all bundled checkers,
  fixers, and reporters.
- The GitHub Action shipped from this repo (`action.yml` and `action/`).
- The bundled operator matrix and remediation database.

Out of scope:

- Third-party plugins discovered via the entry-point mechanism. Those
  are their own packages with their own maintainers.
- TensorRT itself or NVIDIA's `trtexec`. Forward those to NVIDIA.
- ONNX files supplied by users. trtcheck only reads them; if a
  malformed ONNX file crashes `onnx.load`, that's an issue to file
  against `onnx`.

## What we promise

- No untrusted code execution. trtcheck loads ONNX files via `onnx.load`
  and inspects the resulting graph in memory. It does not deserialize
  pickle, eval strings, or execute model code.
- HTML reports are self-contained; all dynamic strings from ONNX are
  HTML-escaped before being written.
- GitHub Action sticky comments escape ONNX-derived text before
  rendering it as markdown.
- The GitHub Action workflow has strict allowlists on `inputs.paths` and
  `inputs.version`.
