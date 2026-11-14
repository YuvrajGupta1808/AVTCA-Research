#!/usr/bin/env python3
"""Download the EngageNet dataset from the public OneDrive share."""

import argparse
import asyncio
import json
import math
import os
import re
import time
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from pyppeteer import chromium_downloader
from pyppeteer import launch
from pyppeteer.errors import BrowserError
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "EngageNet"
ODC_REFERER = "https://onedrive.live.com/"
CHILDREN_QUERY = "?%24top=200&select=id%2Cname%2Csize%2Cfolder%2Cfile&ump=1"
AUTH_RETRIES = 5
REQUEST_RETRIES = 5
DOWNLOAD_RETRIES = 5
RETRY_BACKOFF_SECONDS = 3
ENV_SHARE_URL = "ENGAGENET_SHARE_URL"
ENV_DRIVE_ID = "ENGAGENET_DRIVE_ID"
ENV_ROOT_ITEM_ID = "ENGAGENET_ROOT_ITEM_ID"


def load_repo_env_file() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{timestamp}] {message}", flush=True)


def build_multipart_body(boundary: str, token: str) -> bytes:
    return (
        f"--{boundary}\r\n"
        "Content-Disposition: form-data;name=data\r\n"
        "Prefer: HonorNonIndexedQueriesWarningMayFailRandomly, allowthrottleablequeries, "
        "Include-Feature=AddToOneDrive;Vault\r\n"
        "X-ClientService-ClientTag: ODC Web\r\n"
        "Application: ODC Web\r\n"
        "Scenario: BrowseFiles\r\n"
        "ScenarioType: AUO\r\n"
        "X-HTTP-Method-Override: GET\r\n"
        "Content-Type: application/json\r\n"
        f"Authorization: {token}\r\n"
        "\r\n"
        f"--{boundary}--"
    ).encode("utf-8")


async def capture_auth_context(share_url: str) -> Dict[str, str]:
    token_holder: Dict[str, str] = {}
    executable_path = chromium_downloader.chromium_executable()
    browser = await launch(
        executablePath=executable_path,
        headless=True,
        args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        handleSIGINT=False,
        handleSIGTERM=False,
        handleSIGHUP=False,
    )
    try:
        page = await browser.newPage()
        await page.setViewport({"width": 1440, "height": 1100})

        def on_request(req):
            if "/_api/v2.0/drives/" not in req.url or "/children" not in req.url:
                return
            post_data = req.postData or ""
            match = re.search(r"Authorization: badger\s+([^\r\n]+)", post_data)
            if not match:
                return
            token_holder["token"] = f"badger {match.group(1)}"
            token_holder["user_agent"] = req.headers.get("user-agent", "")

        page.on("request", on_request)
        await page.goto(share_url, {"waitUntil": "networkidle2", "timeout": 180000})
        for _ in range(60):
            if token_holder.get("token"):
                break
            await asyncio.sleep(1)
        if not token_holder.get("token"):
            raise RuntimeError("Failed to capture OneDrive authorization token from page load")
        return token_holder
    finally:
        await browser.close()


class EngageNetDownloader:
    def __init__(self, share_url: str, drive_id: str, root_item_id: str, output_dir: Path):
        self.share_url = share_url
        self.drive_id = drive_id
        self.root_item_id = root_item_id
        self.output_dir = output_dir
        self.session = requests.Session()
        self.token: Optional[str] = None
        self.user_agent: str = "Mozilla/5.0"
        self.failures: List[Dict[str, str]] = []
        self.completed_files = 0
        self.skipped_files = 0
        self.failure_path = self.output_dir / "_download_failures.json"
        self.state_path = self.output_dir / "_download_state.json"

    def write_state(self) -> None:
        state = {
            "completed_files": self.completed_files,
            "skipped_files": self.skipped_files,
            "failures": len(self.failures),
            "last_update_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def record_failure(self, *, item_id: str, rel_path: Path, stage: str, error: Exception) -> None:
        failure = {
            "id": item_id,
            "path": str(rel_path),
            "stage": stage,
            "error": str(error),
        }
        self.failures.append(failure)
        self.failure_path.write_text(json.dumps(self.failures, indent=2), encoding="utf-8")
        log(f"failed {rel_path} during {stage}: {error}")

    @staticmethod
    def backoff(attempt: int) -> float:
        return RETRY_BACKOFF_SECONDS * math.pow(2, max(0, attempt - 1))

    def ensure_auth(self) -> None:
        if self.token:
            return
        for attempt in range(1, AUTH_RETRIES + 1):
            try:
                log(f"capturing OneDrive auth token (attempt {attempt}/{AUTH_RETRIES})")
                ctx = asyncio.run(capture_auth_context(self.share_url))
                self.token = ctx["token"]
                self.user_agent = ctx.get("user_agent") or self.user_agent
                return
            except (BrowserError, RuntimeError) as exc:
                if attempt == AUTH_RETRIES:
                    raise
                delay = self.backoff(attempt)
                log(f"auth capture failed: {exc}; retrying in {delay:.0f}s")
                time.sleep(delay)

    def odc_post(self, url: str, *, stream: bool = False) -> requests.Response:
        for attempt in range(1, REQUEST_RETRIES + 1):
            self.ensure_auth()
            assert self.token is not None
            boundary = f"----CodexBoundary{uuid.uuid4().hex}"
            try:
                response = self.session.post(
                    url,
                    headers={
                        "referer": ODC_REFERER,
                        "user-agent": self.user_agent,
                        "content-type": f"multipart/form-data;boundary={boundary}",
                    },
                    data=build_multipart_body(boundary, self.token),
                    allow_redirects=False,
                    stream=stream,
                    timeout=180,
                )
                if response.status_code in (401, 403):
                    self.token = None
                    raise requests.HTTPError(f"auth rejected with {response.status_code}", response=response)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                if getattr(exc.response, "status_code", None) in (401, 403):
                    self.token = None
                if attempt == REQUEST_RETRIES:
                    raise
                delay = self.backoff(attempt)
                log(f"request failed for {url}: {exc}; retrying in {delay:.0f}s")
                time.sleep(delay)
        raise RuntimeError(f"request retry loop exhausted for {url}")

    def list_children(self, item_id: str) -> List[Dict]:
        url = (
            f"https://my.microsoftpersonalcontent.com/_api/v2.0/drives/"
            f"{self.drive_id}/items/{item_id}/children{CHILDREN_QUERY}"
        )
        results: List[Dict] = []
        while url:
            response = self.odc_post(url)
            payload = response.json()
            results.extend(payload.get("value", []))
            url = payload.get("@odata.nextLink")
        return results

    def file_redirect_url(self, item_id: str) -> str:
        url = (
            f"https://my.microsoftpersonalcontent.com/_api/v2.0/drives/"
            f"{self.drive_id}/items/{item_id}/content?ump=1"
        )
        response = self.odc_post(url, stream=True)
        if response.status_code != 302:
            raise RuntimeError(f"Expected redirect for file content, got {response.status_code}")
        redirect_url = response.headers.get("Location")
        if not redirect_url:
            raise RuntimeError("Missing download redirect URL")
        return redirect_url

    def download_file(self, item: Dict, rel_dir: Path) -> None:
        filename = item["name"]
        size = int(item.get("size") or 0)
        target_dir = self.output_dir / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        if target_path.exists() and target_path.stat().st_size == size:
            self.skipped_files += 1
            log(f"skip  {rel_dir / filename}")
            self.write_state()
            return

        tmp_path = target_path.with_suffix(target_path.suffix + ".part")
        rel_path = rel_dir / filename
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
                redirect_url = self.file_redirect_url(item["id"])
                with self.session.get(
                    redirect_url,
                    headers={"user-agent": self.user_agent, "referer": ODC_REFERER},
                    stream=True,
                    timeout=180,
                ) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("Content-Length") or size or 0)
                    with open(tmp_path, "wb") as fh, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=str(rel_path),
                        leave=False,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            pbar.update(len(chunk))
                os.replace(tmp_path, target_path)
                self.completed_files += 1
                log(f"saved {rel_path}")
                self.write_state()
                return
            except (requests.RequestException, OSError, RuntimeError) as exc:
                if attempt == DOWNLOAD_RETRIES:
                    self.record_failure(item_id=item["id"], rel_path=rel_path, stage="download_file", error=exc)
                    return
                delay = self.backoff(attempt)
                log(f"download failed for {rel_path}: {exc}; retrying in {delay:.0f}s")
                time.sleep(delay)

    def download_tree(self, item_id: str, rel_dir: Path = Path(".")) -> List[Dict]:
        try:
            children = self.list_children(item_id)
        except Exception as exc:
            self.record_failure(item_id=item_id, rel_path=rel_dir, stage="list_children", error=exc)
            return []
        manifest_entries: List[Dict] = []
        for child in children:
            rel_path = rel_dir / child["name"]
            manifest_entries.append(
                {
                    "id": child["id"],
                    "name": child["name"],
                    "path": str(rel_path),
                    "size": child.get("size"),
                    "is_folder": bool(child.get("folder")),
                }
            )
            if child.get("folder"):
                (self.output_dir / rel_path).mkdir(parents=True, exist_ok=True)
                log(f"dir   {rel_path}")
                manifest_entries.extend(self.download_tree(child["id"], rel_path))
            else:
                self.download_file(child, rel_dir)
        return manifest_entries


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    load_repo_env_file()
    parser = argparse.ArgumentParser(description="Download EngageNet from a public OneDrive share")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Destination directory for the mirrored dataset",
    )
    parser.add_argument(
        "--share-url",
        default=os.environ.get(ENV_SHARE_URL, ""),
        help=f"OneDrive share URL. Can also be provided via ${ENV_SHARE_URL}.",
    )
    parser.add_argument(
        "--drive-id",
        default=os.environ.get(ENV_DRIVE_ID, ""),
        help=f"OneDrive drive ID. Can also be provided via ${ENV_DRIVE_ID}.",
    )
    parser.add_argument(
        "--root-item-id",
        default=os.environ.get(ENV_ROOT_ITEM_ID, ""),
        help=f"OneDrive root item ID. Can also be provided via ${ENV_ROOT_ITEM_ID}.",
    )
    args = parser.parse_args(list(argv))

    missing = []
    if not args.share_url:
        missing.append(f"--share-url or ${ENV_SHARE_URL}")
    if not args.drive_id:
        missing.append(f"--drive-id or ${ENV_DRIVE_ID}")
    if not args.root_item_id:
        missing.append(f"--root-item-id or ${ENV_ROOT_ITEM_ID}")
    if missing:
        parser.error("Missing required OneDrive download inputs: " + ", ".join(missing))
    return args


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloader = EngageNetDownloader(
        share_url=args.share_url,
        drive_id=args.drive_id,
        root_item_id=args.root_item_id,
        output_dir=output_dir,
    )
    downloader.write_state()
    manifest = downloader.download_tree(args.root_item_id)
    manifest_path = output_dir / "_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    downloader.write_state()
    log(f"wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
