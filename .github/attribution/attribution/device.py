"""Credential and HTTP client for a computer that is connected to an account.

The credential lives in the telemetry state directory. A test that points
``ATTRIBUTION_TELEMETRY_DIR`` at a temporary folder never touches a real one.
Only the token travels, in the Authorization header. Session text never does.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from . import __version__
from .install_code import InstallCodeError, _NoRedirect, _origin

_TOKEN = re.compile(r"^jrd_[A-Za-z0-9_-]{32}$")
_TIMEOUT_SECONDS = 20
_MAX_RESPONSE_BYTES = 65536


class DeviceError(ValueError):
    """The hosted service refused a device request or did not answer."""


class DeviceRevoked(DeviceError):
    """The service no longer knows this device. The caller clears the credential."""


class DeviceHTTPError(DeviceError):
    """The service answered with an error status that the job reports."""

    def __init__(self, status: int, code: str | None, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def credential_path(state_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the credential file path without creating it."""

    from .telemetry import _state_dir_path

    return _state_dir_path(state_dir) / "device.json"


def load_credential(state_dir: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """Return the stored credential, or ``None`` when none is valid."""

    path = credential_path(state_dir)
    try:
        if path.is_symlink():
            return None
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    token, hosted, device_id = value.get("token"), value.get("hosted_url"), value.get("device_id")
    if not isinstance(token, str) or not _TOKEN.fullmatch(token) or not isinstance(hosted, str):
        return None
    if type(device_id) is not int or device_id <= 0:
        return None
    try:
        origin = _origin(hosted)
    except InstallCodeError:
        return None
    login, paired_at = value.get("login"), value.get("paired_at")
    return {
        "hosted_url": origin, "device_id": device_id, "token": token,
        "login": login if isinstance(login, str) else None,
        "paired_at": paired_at if isinstance(paired_at, str) else None,
    }


def save_credential(
    hosted_url: str, device_id: int, token: str, login: str | None,
    state_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Write the credential with mode 0600 and return its path."""

    from .telemetry import secure_state_dir

    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        raise DeviceError("The hosted service sent a device credential that Joyride cannot read.")
    if type(device_id) is not int or device_id <= 0:
        raise DeviceError("The hosted service sent a device credential that Joyride cannot read.")
    directory = secure_state_dir(state_dir)
    payload = {
        "hosted_url": _origin(hosted_url), "device_id": device_id, "token": token,
        "login": login if isinstance(login, str) else None,
        "paired_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    handle, temporary = tempfile.mkstemp(dir=directory, prefix="device-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        os.chmod(temporary, 0o600)
        os.replace(temporary, directory / "device.json")
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary)
        raise
    return directory / "device.json"


def clear_credential(state_dir: str | os.PathLike[str] | None = None) -> bool:
    """Remove the credential and say whether one existed."""

    try:
        credential_path(state_dir).unlink()
    except FileNotFoundError:
        return False
    return True


def device_descriptor() -> dict[str, str]:
    """Describe this computer for pairing: host name, platform, and CLI version."""

    name = "".join(char for char in socket.gethostname() if char.isprintable()).strip()
    return {"name": name[:64] or "computer", "platform": sys.platform[:32],
            "client_version": __version__[:32]}


class DeviceClient:
    """Send the device routes of the hosted service with the stored credential."""

    def __init__(self, credential: Mapping[str, Any]) -> None:
        self.origin = _origin(str(credential["hosted_url"]))
        self._token = str(credential["token"])
        self._opener = build_opener(_NoRedirect())

    def _post(self, path: str, body: dict[str, Any] | bytes) -> dict[str, Any]:
        data = body if isinstance(body, bytes) else json.dumps(body, separators=(",", ":")).encode("utf-8")
        request = Request(
            self.origin + path, data=data, method="POST",
            headers={"Authorization": "Bearer " + self._token, "Accept": "application/json",
                     "Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=_TIMEOUT_SECONDS) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            payload = _payload(exc.read(_MAX_RESPONSE_BYTES))
            code = payload.get("code") if isinstance(payload, dict) else None
            if exc.code == 401 and code == "device_revoked":
                raise DeviceRevoked("This computer is no longer connected.") from exc
            if exc.code in {429, 500, 502, 503, 504}:
                # The shared upload retry rule handles these statuses.
                raise
            message = payload.get("error") if isinstance(payload, dict) else None
            raise DeviceHTTPError(
                exc.code, code if isinstance(code, str) else None,
                message.strip()[:300] if isinstance(message, str) and message.strip()
                else f"The hosted service refused the request (HTTP {exc.code}).",
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise DeviceError("The hosted service did not answer.") from exc
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise DeviceError("The hosted service sent an answer that Joyride cannot read.")
        payload = _payload(raw)
        if not isinstance(payload, dict):
            raise DeviceError("The hosted service sent an answer that Joyride cannot read.")
        return payload

    def poll(self, busy: bool) -> dict[str, Any]:
        # The poll refreshes the CLI version. The name and platform stay as paired.
        return self._post("/v1/devices/poll", {"busy": bool(busy),
                                               "client_version": device_descriptor()["client_version"]})

    def advance_pull_requests(self, job_id: int) -> dict[str, Any]:
        return self._post(f"/v1/devices/jobs/{int(job_id)}/pull-requests", {})

    def upload_sessions(self, job_id: int, data: bytes) -> dict[str, Any]:
        return self._post(f"/v1/devices/jobs/{int(job_id)}/sessions", data)

    def progress(self, job_id: int, phase: str, progress: Mapping[str, Any],
                 attempt: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"phase": phase, "progress": dict(progress)}
        if attempt is not None:
            body["attempt"] = attempt
        return self._post(f"/v1/devices/jobs/{int(job_id)}/progress", body)

    def finish(
        self, job_id: int, status: str, *, error_code: str | None = None,
        error: str | None = None, progress: Mapping[str, Any] | None = None,
        attempt: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"status": status}
        if attempt is not None:
            body["attempt"] = attempt
        if error_code is not None:
            body["error_code"] = error_code
        if error is not None:
            body["error"] = error
        if progress is not None:
            body["progress"] = dict(progress)
        return self._post(f"/v1/devices/jobs/{int(job_id)}/finish", body)


def _payload(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return None
