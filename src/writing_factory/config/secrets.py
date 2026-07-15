"""OS-backed credential storage for external service API secrets."""

from __future__ import annotations

from typing import Protocol

import keyring

SERVICE_NAME = "WritingFactory"
SILICONFLOW_CREDENTIAL = "siliconflow-api-key"
MINERU_CREDENTIAL = "mineru-api-token"


class CredentialStore(Protocol):
    """Minimal secret-store contract used by configuration and tests."""

    def get(self, name: str) -> str | None: ...

    def set(self, name: str, value: str) -> None: ...

    def delete(self, name: str) -> None: ...


class KeyringCredentialStore:
    """Persist credentials in the operating system credential vault."""

    def get(self, name: str) -> str | None:
        return keyring.get_password(SERVICE_NAME, name)

    def set(self, name: str, value: str) -> None:
        secret = value.strip()
        if not secret:
            raise ValueError("API 凭据不能为空")
        keyring.set_password(SERVICE_NAME, name, secret)

    def delete(self, name: str) -> None:
        try:
            keyring.delete_password(SERVICE_NAME, name)
        except keyring.errors.PasswordDeleteError:
            return
