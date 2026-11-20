"""Salted hashing for customer identifiers."""

from __future__ import annotations

import hashlib
import os

import fsspec
import polars as pl


def get_salt(env_var: str) -> str:
    salt = os.environ.get(env_var, "")
    if not salt:
        raise RuntimeError(f"PII salt missing: set environment variable {env_var}")
    return salt


def hash_ids(col: pl.Expr, salt: str) -> pl.Expr:
    """Replace each ID with sha256(salt + id) as hex; nulls stay null."""

    def _h(value: str) -> str:
        return hashlib.sha256(f"{salt}{value}".encode()).hexdigest()

    return col.map_elements(_h, return_dtype=pl.String, skip_nulls=True)


FINGERPRINT_FILE = "_salt_fingerprint"


class SaltMismatchError(RuntimeError):
    """The lake was built with a different salt; mixing salts silently breaks customer joins."""


def salt_fingerprint(salt: str) -> str:
    """One-way identifier of the salt, safe to store: reveals nothing about the salt itself."""
    return hashlib.sha256(f"late-delivery-salt-fingerprint:{salt}".encode()).hexdigest()[:16]


def ensure_lake_salt(root: str, salt: str) -> None:
    """Record the salt's fingerprint on first use; refuse any later run with a different salt."""
    fs, base = fsspec.core.url_to_fs(root)
    path = f"{base}/{FINGERPRINT_FILE}"
    current = salt_fingerprint(salt)
    if not fs.exists(path):
        fs.makedirs(base, exist_ok=True)
        with fs.open(path, "w") as f:
            f.write(current + "\n")
        return
    with fs.open(path, "r") as f:
        recorded = f.read().strip()
    if recorded != current:
        raise SaltMismatchError(
            f"this lake was built with a different PII salt (fingerprint {recorded}, current "
            f"{current}). Use the original salt, or rebuild the lake from day 0 with the new one."
        )
