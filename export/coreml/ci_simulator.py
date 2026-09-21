#!/usr/bin/env python3
"""Create and boot one explicit iOS Simulator on an ephemeral GitHub runner.

Print only its UDID to stdout; write setup evidence on success or failure.
No model artifacts or simulator runtimes are downloaded.

Simulator command reference: `xcrun simctl help` from the selected Xcode.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import uuid


class SetupError(RuntimeError):
    pass


class SimulatorSetup:
    def __init__(self, arguments: argparse.Namespace):
        self.arguments = arguments
        self.report_path = Path(arguments.report)
        self.report = {
            "scope": "GitHub-hosted iOS Simulator setup; not physical-device evidence",
            "status": "starting",
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "requested_runtime_version": arguments.runtime_version,
            "requested_device_type": arguments.device_type,
            "developer_dir": os.environ.get("DEVELOPER_DIR"),
            "runner_environment": os.environ.get("RUNNER_ENVIRONMENT"),
            "runner_image": os.environ.get("ImageOS"),
            "runner_image_version": os.environ.get("ImageVersion"),
            "host_architecture": platform.machine(),
            "commands": [],
        }

    def save(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.report_path.parent,
            prefix=self.report_path.name + ".", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(self.report, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        try:
            temporary_path.replace(self.report_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def command(self, *arguments: str, timeout: int = 60) -> str:
        print("Simulator setup: " + " ".join(arguments), file=sys.stderr, flush=True)
        started = time.monotonic()
        entry = {"arguments": list(arguments), "timeout_seconds": timeout}
        self.report["commands"].append(entry)
        self.save()
        try:
            result = subprocess.run(
                arguments, check=False, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            entry.update({"status": "timed_out", "elapsed_seconds": time.monotonic() - started})
            self.save()
            raise SetupError(f"Command exceeded {timeout}s: {' '.join(arguments)}") from error
        except OSError as error:
            entry.update({"status": "launch_failed", "error": str(error)})
            self.save()
            raise SetupError(f"Could not execute {' '.join(arguments)}: {error}") from error
        entry.update({
            "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "stdout_tail": result.stdout[-16_000:],
            "stderr_tail": result.stderr[-16_000:],
            "output_truncated": len(result.stdout) > 16_000 or len(result.stderr) > 16_000,
        })
        self.save()
        if result.returncode:
            raise SetupError(
                f"Command exited {result.returncode}: {' '.join(arguments)}\n"
                + (result.stderr or result.stdout)[-4_000:]
            )
        return result.stdout

    def inventory(self, kind: str) -> dict:
        output = self.command("xcrun", "simctl", "list", kind, "--json")
        try:
            result = json.loads(output)
        except json.JSONDecodeError as error:
            raise SetupError(f"simctl {kind} inventory was not valid JSON") from error
        if not isinstance(result, dict):
            raise SetupError(f"simctl {kind} inventory was not a JSON object")
        self.report[f"{kind}_inventory"] = result
        self.save()
        return result

    def runtime(self, inventory: dict) -> dict | None:
        matches = [
            item for item in inventory.get("runtimes", [])
            if item.get("version") == self.arguments.runtime_version
            and item.get("identifier", "").startswith("com.apple.CoreSimulator.SimRuntime.iOS-")
            and item.get("isAvailable") is True
        ]
        if len(matches) > 1:
            raise SetupError(f"Multiple available runtimes match iOS {self.arguments.runtime_version}")
        return matches[0] if matches else None

    def run(self) -> str:
        # Keep creation and boot confined to the explicitly authorized ephemeral
        # CI; do not change existing simulators or download runtime components.
        if (os.environ.get("GITHUB_ACTIONS") != "true"
                or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
                or platform.system() != "Darwin"):
            raise SetupError("Run this setup only on an ephemeral GitHub-hosted macOS runner")
        self.report["xcode_version"] = self.command("xcodebuild", "-version").strip()
        runtimes = self.inventory("runtimes")
        runtime = self.runtime(runtimes)
        if runtime is None:
            raise SetupError(
                f"No available iOS {self.arguments.runtime_version} runtime in the selected Xcode. "
                f"See runtimes_inventory in {self.report_path}. No runtime was downloaded."
            )
        self.report["selected_runtime"] = runtime
        device_types = self.inventory("devicetypes")
        matching_types = [
            item for item in device_types.get("devicetypes", [])
            if item.get("name") == self.arguments.device_type
            and item.get("identifier", "").startswith("com.apple.CoreSimulator.SimDeviceType.")
        ]
        if len(matching_types) != 1:
            raise SetupError(
                f"Expected one device type named {self.arguments.device_type!r}; "
                f"found {len(matching_types)}. See devicetypes_inventory."
            )
        device_type = matching_types[0]
        self.report["selected_device_type"] = device_type
        name = f"Orukeet CI {self.arguments.runtime_version} {uuid.uuid4().hex[:12]}"
        output = self.command(
            "xcrun", "simctl", "create", name, device_type["identifier"], runtime["identifier"],
        ).strip()
        try:
            identifier = str(uuid.UUID(output)).upper()
        except ValueError as error:
            raise SetupError(f"simctl create did not return a single valid UDID: {output!r}") from error
        self.report.update({"created_device_name": name, "udid": identifier, "status": "created"})
        self.save()
        self.command("xcrun", "simctl", "boot", identifier)
        self.command("xcrun", "simctl", "bootstatus", identifier, "-b", timeout=240)
        devices = self.inventory("devices").get("devices", {}).get(runtime["identifier"], [])
        matching_devices = [item for item in devices if item.get("udid", "").upper() == identifier]
        if (len(matching_devices) != 1
                or matching_devices[0].get("state") != "Booted"
                or matching_devices[0].get("isAvailable") is not True):
            raise SetupError("The created device is not available and Booted under the exact requested runtime")
        observed_type = matching_devices[0].get("deviceTypeIdentifier")
        if observed_type is not None and observed_type != device_type["identifier"]:
            raise SetupError("The created device's observed type differs from the selected device type")
        self.report.update({
            "status": "ready",
            "verified_device": matching_devices[0],
            "destination": f"platform=iOS Simulator,id={identifier}",
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        self.save()
        return identifier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Write JSON setup evidence, including failures")
    parser.add_argument("--runtime-version", default="18.5")
    parser.add_argument("--device-type", default="iPhone 16")
    arguments = parser.parse_args()
    setup = SimulatorSetup(arguments)
    try:
        identifier = setup.run()
    except (SetupError, OSError) as error:
        setup.report.update({"status": "failed", "error": str(error)})
        try:
            setup.save()
        except OSError as write_error:
            print(f"Could not save simulator setup evidence: {write_error}", file=sys.stderr)
        print(f"Simulator setup failed: {error}", file=sys.stderr)
        return 1
    print(identifier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
