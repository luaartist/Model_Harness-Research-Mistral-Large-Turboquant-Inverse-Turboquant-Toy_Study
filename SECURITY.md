# Security Policy

Do not commit private model weights, API tokens, local credentials, VS Code workspace storage, chat transcripts, or generated decode artifacts.

The harness can call a local bridge endpoint through `BRIDGE_URL`, defaulting to `http://127.0.0.1:8504`. Keep bridge services bound to localhost unless you have a separate authentication and network-control layer.

If you find a secret, credential, or private artifact in this repository, rotate the affected credential or remove the artifact from history before public reuse.