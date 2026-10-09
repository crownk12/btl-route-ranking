"""Capture today's UN Comtrade dataset catalog and, with a premium key, bulk files.

Every invocation creates a new UTC snapshot.  The public getDa catalog identifies
current country/month datasets; only a fully downloaded premium bulk file is a
candidate complete source.  Neither can recreate an earlier value vintage.

Raw API bytes and bulk files go under data/comtrade-vintages/raw/ (gitignored).
Public, key-free manifests and SHA-256 digests go under manifests/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));import btl_paths
from adapters.contracts import COUNTRIES, ROUTE_SUPPLIERS

BASE = ROOT / "data/comtrade-vintages"
CATALOG_ENDPOINT = "https://comtradeapi.un.org/public/v1/getDa/C/M/HS"
BULK_ENDPOINT = "https://comtradeapi.un.org/bulk/v1/get/C/M/HS"
USER_AGENT = "before-the-list-local-research/1.0"


def parse_period(value: str) -> str:
    if not re.fullmatch(r"[0-9]{6}", value) or not 1 <= int(value[4:]) <= 12:
        raise argparse.ArgumentTypeError("period must be YYYYMM")
    return value


def parse_reporter(value: str) -> str:
    code = value.upper()
    if code not in ROUTE_SUPPLIERS:
        raise argparse.ArgumentTypeError("reporter must be one of the nine route candidates")
    return code


def months(first: str, last: str) -> tuple[str, ...]:
    year, month = int(first[:4]), int(first[4:])
    end = (int(last[:4]), int(last[4:]))
    periods = []
    while (year, month) <= end:
        periods.append(f"{year:04d}{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return tuple(periods)


def safe_url(url: str) -> str:
    parts = urlsplit(url)
    query = [(name, value) for name, value in parse_qsl(parts.query)
             if name.lower() not in ("subscription-key", "subscriptionkey", "ocp-apim-subscription-key")]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def manifest_file_url(url: str) -> str:
    """Keep only the documented endpoint, never a signed query or fragment."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def keyed_url(url: str, key: str) -> str:
    clean = safe_url(url)
    parts = urlsplit(clean)
    query = list(parse_qsl(parts.query)) + [("subscription-key", key)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_json(url: str, *, key: str | None = None) -> tuple[dict, bytes, int]:
    # Official documentation shows the subscription key in the query.  It is
    # applied only in memory; no keyed URL, exception string, or payload is logged.
    request_url = keyed_url(url, key) if key else url
    with urlopen(Request(request_url, headers={"User-Agent": USER_AGENT}), timeout=45) as response:
        blob = response.read(20_000_001)
        if len(blob) > 20_000_000:
            raise ValueError("JSON API response exceeded the 20 MB bound")
        payload = json.loads(blob)
        if not isinstance(payload, dict):
            raise ValueError("API response was not a JSON object")
        return payload, blob, response.status


def dataset_catalog(reporters: tuple[str, ...], periods: tuple[str, ...], raw: Path,
                    manifest: Path, captured_at: str) -> tuple[list[dict], dict]:
    if len(periods) > 12:
        raise ValueError("UN Comtrade catalog selection is limited to 12 monthly periods")
    params = dict(period=",".join(periods),
                  reporterCode=",".join(str(int(COUNTRIES[r])) for r in reporters))
    url = CATALOG_ENDPOINT + "?" + urlencode(params)
    payload, blob, http_status = fetch_json(url)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(blob)
    records = payload.get("data")
    count = payload.get("count")
    expected_codes = {int(COUNTRIES[r]) for r in reporters}
    if (payload.get("error") or not isinstance(records, list) or
            not isinstance(count, int) or count != len(records)):
        status = "catalog_incomplete_or_error"
        records = records if isinstance(records, list) else []
    else:
        status = "catalog_received"
    entries = []
    for row in records:
        if not isinstance(row, dict):
            status = "catalog_incomplete_or_error"; continue
        try:
            reporter_code, period = int(row["reporterCode"]), str(row["period"])
        except (KeyError, TypeError, ValueError):
            status = "catalog_incomplete_or_error"; continue
        if (reporter_code not in expected_codes or period not in periods or
                row.get("typeCode") != "C" or row.get("freqCode") != "M"):
            status = "catalog_incomplete_or_error"; continue
        entries.append(dict(reporter_code=reporter_code, period=period,
            classification=row.get("classificationCode"),
            original_classification=row.get("isOriginalClassification"),
            dataset_checksum=row.get("datasetChecksum"),
            total_records=row.get("totalRecords"),
            first_released=row.get("firstReleased"),
            last_released=row.get("lastReleased")))
    record = dict(status=status, http_status=http_status, captured_at_utc=captured_at,
        source_url=url, raw_path=str(raw.relative_to(ROOT)).replace("\\", "/"),
        raw_sha256=hashlib.sha256(blob).hexdigest(), raw_bytes=len(blob),
        api_count=count, rows_received=len(records), entries=entries,
        limitation="Current latest-dataset metadata; release fields do not recover historical value vintages")
    write_json(manifest, record)
    return entries, record


def previous_changes(entries: list[dict], current_manifest: Path) -> dict:
    current = json.loads(current_manifest.read_text(encoding="utf-8"))
    previous = sorted((BASE / "manifests").glob("*/availability.manifest.json"))
    previous = [path for path in previous if path != current_manifest and
                json.loads(path.read_text(encoding="utf-8")).get("source_url") == current.get("source_url")]
    if not previous:
        return dict(previous_snapshot=None, comparison_scope="same_catalog_query_only",
                    added=len(entries), changed=0, removed=0)
    prior_path = previous[-1]
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    fields = ("dataset_checksum", "total_records", "first_released", "last_released")
    identity = lambda entry: (entry["reporter_code"], entry["period"], entry["classification"])
    old = {identity(entry): entry for entry in prior.get("entries", [])}
    new = {identity(entry): entry for entry in entries}
    return dict(previous_snapshot=str(prior_path.relative_to(ROOT)).replace("\\", "/"),
        comparison_scope="same_catalog_query_only",
        added=len(new.keys() - old.keys()), removed=len(old.keys() - new.keys()),
        changed=sum(any(old[key].get(field) != new[key].get(field) for field in fields)
                    for key in new.keys() & old.keys()))


def bulk_download(reporter_code: int, period: str, key: str, raw: Path,
                  max_bytes: int, captured_at: str) -> dict:
    # Bulk is premium-only.  The availability reply gives a file URL, then the
    # entire compressed country/period dataset is downloaded to an opaque file.
    public_url = BULK_ENDPOINT + "?" + urlencode(dict(reporterCode=reporter_code, period=period))
    result = dict(reporter_code=reporter_code, period=period, classification="HS",
                  source_url=public_url, captured_at_utc=captured_at,
                  status="bulk_request_failed", full_content_audited=False)
    try:
        payload, blob, http_status = fetch_json(public_url, key=key)
        result.update(http_status=http_status, availability_reply_sha256=hashlib.sha256(blob).hexdigest())
        candidate = payload.get("data")
        if isinstance(candidate, list):
            candidate = next((row for row in candidate if isinstance(row, dict) and row.get("fileUrl")), None)
        if not isinstance(candidate, dict):
            candidate = payload
        file_url = candidate.get("fileUrl")
        if not file_url or payload.get("error"):
            result["status"] = "bulk_file_unavailable"; return result
        clean = safe_url(str(file_url))
        parts = urlsplit(clean)
        if parts.scheme != "https" or parts.hostname != "comtradeapi.un.org":
            result.update(status="bulk_file_url_rejected",
                          rejection_reason="Download host is outside the official documented comtradeapi.un.org/bulk/v1/file example; never send a subscription key to an unverified host")
            return result
        result["file_url"] = manifest_file_url(clean)
        raw.parent.mkdir(parents=True, exist_ok=True)
        temporary = raw.with_name(raw.name + ".part")
        size = 0
        digest = hashlib.sha256()
        try:
            with urlopen(Request(keyed_url(clean, key), headers={"User-Agent": USER_AGENT}),
                         timeout=180) as response, temporary.open("wb") as stream:
                result["file_http_status"] = response.status
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        result["status"] = "bulk_file_exceeded_byte_bound"
                        return result
                    stream.write(chunk); digest.update(chunk)
            temporary.replace(raw)
        finally:
            if temporary.exists():
                temporary.unlink()
        result.update(status="bulk_file_downloaded", compressed_bytes=size,
                      raw_path=str(raw.relative_to(ROOT)).replace("\\", "/"),
                      raw_sha256=digest.hexdigest(),
                      completeness_scope="Official prepackaged full country/month HS original-classification file; content dimensions and row totals still require independent parsing")
    except HTTPError as exc:
        result.update(status="bulk_http_error", http_status=exc.code)
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError):
        result["status"] = "bulk_network_or_format_error"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reporter", action="append", type=parse_reporter)
    parser.add_argument("--period", action="append", type=parse_period)
    parser.add_argument("--bulk", action="store_true", help="Try bounded premium full-file downloads")
    parser.add_argument("--limit", type=int, default=1, help="Maximum bulk country/month files")
    parser.add_argument("--max-bytes", type=int, default=500_000_000,
                        help="Maximum compressed bytes per bulk file")
    args = parser.parse_args()
    if not 1 <= args.limit <= 12 or args.max_bytes < 1_000_000:
        parser.error("limit must be 1..12 and max-bytes at least 1000000")
    reporters = tuple(dict.fromkeys(args.reporter or ROUTE_SUPPLIERS))
    periods = tuple(dict.fromkeys(args.period or months("202510", "202609")))
    if len(periods) > 12:
        parser.error("at most 12 periods per catalog snapshot")
    now = dt.datetime.now(dt.timezone.utc)
    captured_at = now.isoformat()
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    raw = BASE / "raw" / run_id
    manifests = BASE / "manifests" / run_id
    if raw.exists() or manifests.exists():
        raise RuntimeError("UTC snapshot ID already exists")
    entries, catalog = dataset_catalog(reporters, periods, raw / "availability.json",
                                       manifests / "availability.manifest.json", captured_at)
    key = os.environ.get("COMTRADE_API_KEY") or os.environ.get("UN_COMTRADE_API_KEY")
    bulk = []
    if args.bulk and key and catalog["status"] == "catalog_received":
        native = sorted({(entry["reporter_code"], entry["period"])
                         for entry in entries if entry["original_classification"] is True
                         and entry["classification"] in ("H5", "H6")})
        for reporter_code, period in native[:args.limit]:
            bulk.append(bulk_download(reporter_code, period, key,
                raw / f"bulk-{reporter_code}-{period}.bin", args.max_bytes, captured_at))
    report = dict(run_id=run_id, captured_at_utc=captured_at,
        reporters=reporters, periods=periods, catalog_status=catalog["status"],
        catalog_rows=len(entries), catalog_change=previous_changes(entries,
            manifests / "availability.manifest.json"),
        subscription_key_configured=bool(key), bulk_requested=bool(args.bulk),
        bulk_attempted=len(bulk), bulk_results=bulk,
        bulk_skip_reason="key_not_configured" if args.bulk and not key else
                         "catalog_incomplete" if args.bulk and catalog["status"] != "catalog_received" else None,
        complete_monthly_source_verified=False,
        historical_asof_values_verified=False,
        vintage_rule="Only captures made at or before a cutoff may supply that cutoff's values; today's catalog and earlier-period files must never be backdated")
    write_json(manifests / "capture-summary.json", report)
    print(json.dumps({key: report[key] for key in ("run_id", "catalog_status",
        "catalog_rows", "subscription_key_configured", "bulk_attempted",
        "bulk_skip_reason")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
