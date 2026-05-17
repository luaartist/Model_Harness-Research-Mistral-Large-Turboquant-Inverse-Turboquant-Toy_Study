# Manifest Schema Notes

The runtime manifest is a hash contract for a shardlet package. At minimum it records:

- schema version
- package identity
- artifact path or URI, byte count, and SHA-256
- selected tensor prefix
- kernel path, function, byte count, and SHA-256
- tensor names, shapes, dtypes, byte counts, and hashes
- generated input names, paths, shapes, dtypes, byte counts, and hashes
- validation metrics and claim boundary

Use package-relative paths for files committed to the repository and URI-style paths for external artifacts. Runtime endpoints such as `BRIDGE_URL` belong in environment/config, not in release-lock identity.