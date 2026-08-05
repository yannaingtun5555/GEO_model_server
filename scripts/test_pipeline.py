#!/usr/bin/env python3
"""
scripts/test_pipeline.py — Helper CLI to test the model server pipeline with custom datasets (Async & Columnar Matrix defaults).
"""

import argparse
import sys
import time
import requests
import json


def main():
    parser = argparse.ArgumentParser(
        description="Submit a CSV dataset to the model server and save the output predictions as a compact JSON file."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the input CSV dataset file (e.g. data/raw/yangon/yangon.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pipeline_output.json",
        help="Path where the JSON prediction response will be saved (default: pipeline_output.json)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8001",
        help="Model server base URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max number of data rows to slice and send. Use -1 to send the entire file. (default: 5)",
    )
    parser.add_argument(
        "--sync",
        dest="sync_mode",
        action="store_true",
        help="Use synchronous HTTP request instead of the default async background worker pipeline.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="columnar",
        choices=["columnar", "rows"],
        help="Output JSON format: 'columnar' (matrix format default for ~90% size reduction) or 'rows'",
    )
    parser.add_argument(
        "--parquet",
        dest="download_parquet",
        action="store_true",
        help="Download the ultra-compact 3MB Parquet binary file directly upon completion.",
    )

    args = parser.parse_args()

    # Read and slice CSV if necessary
    try:
        with open(args.csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading input CSV file: {e}")
        sys.exit(1)

    if not lines or len(lines) <= 1:
        print("❌ Error: CSV file has no headers or data rows.")
        sys.exit(1)

    # Slice header + data rows
    if args.limit > 0:
        header = lines[0]
        data_rows = [line for line in lines[1:] if line.strip()]
        sliced_rows = data_rows[:args.limit]
        csv_payload = header + "".join(sliced_rows)
        filename = f"sliced_{args.limit}_{args.csv.split('/')[-1]}"
        print(f"📄 Sliced first {len(sliced_rows)} rows of '{args.csv}' to keep execution fast.")
    else:
        csv_payload = "".join(lines)
        filename = args.csv.split("/")[-1]
        print(f"📄 Preparing to upload full dataset '{args.csv}' ({len(lines)-1} rows).")

    # Files dict payload
    files = {"file": (filename, csv_payload, "text/csv")}
    params = {"format": args.format}

    if not args.sync_mode:
        # Default Async Ingestion Pipeline
        url_submit = f"{args.server}/api/v1/pipeline/run-async"
        print(f"🚀 Submitting ASYNC job to {url_submit} (format={args.format})...")
        start_time = time.perf_counter()
        try:
            res = requests.post(url_submit, files=files, params=params)
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection error: Could not reach server at {url_submit}.")
            print("   Make sure the server container is running (docker compose up).")
            sys.exit(1)

        if res.status_code != 200:
            print(f"❌ Server returned error status {res.status_code}:")
            print(res.text)
            sys.exit(1)

        submit_data = res.json()
        job_id = submit_data.get("job_id")
        print(f"📥 Job submitted successfully! Job ID: {job_id}")

        # Polling loop
        url_status = f"{args.server}/api/v1/pipeline/status/{job_id}"
        print(f"⏳ Polling status from {url_status}...")
        
        while True:
            try:
                status_res = requests.get(url_status)
            except Exception as e:
                print(f"\n⚠️ Error connecting to check status: {e}")
                time.sleep(1)
                continue

            if status_res.status_code != 200:
                print(f"\n❌ Status check returned error code {status_res.status_code}: {status_res.text}")
                sys.exit(1)

            job_data = status_res.json()
            status = job_data.get("status")
            progress = job_data.get("progress_pct", 0.0)

            if status == "completed":
                print(f"\n✅ Job completed!")
                response_data = job_data.get("result", {})
                download_url = job_data.get("download_parquet_url")
                if download_url:
                    parquet_full_url = f"{args.server}{download_url}"
                    print(f"📦 Ultra-compact Parquet binary available at: {parquet_full_url}")
                    if getattr(args, "download_parquet", False):
                        p_output = args.output.replace(".json", ".parquet")
                        print(f"⬇️ Downloading Parquet file to '{p_output}'...")
                        p_res = requests.get(parquet_full_url)
                        if p_res.status_code == 200:
                            with open(p_output, "wb") as pf:
                                pf.write(p_res.content)
                            print(f"💾 Parquet file successfully saved to '{p_output}' ({len(p_res.content)/1024/1024:.2f} MB)")
                break
            elif status == "failed":
                print(f"\n❌ Job failed with error: {job_data.get('error')}")
                sys.exit(1)
            else:
                sys.stdout.write(f"\rStatus: {status.upper()} | Progress: {progress}%")
                sys.stdout.flush()
                time.sleep(1)

        latency_sec = time.perf_counter() - start_time
        print(f"🕒 Total async execution time: {latency_sec:.3f} seconds")

    else:
        # Synchronous execution
        url_submit = f"{args.server}/api/v1/pipeline/run"
        print(f"🚀 Sending SYNC request to {url_submit} (format={args.format})...")
        start_time = time.perf_counter()
        try:
            res = requests.post(url_submit, files=files, params=params)
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection error: Could not reach server at {url_submit}.")
            print("   Make sure the server container is running (docker compose up).")
            sys.exit(1)

        latency_sec = time.perf_counter() - start_time
        print(f"🕒 Round-trip latency: {latency_sec:.3f} seconds")

        if res.status_code != 200:
            print(f"❌ Server returned error status {res.status_code}:")
            print(res.text)
            sys.exit(1)

        try:
            response_data = res.json()
        except json.JSONDecodeError:
            print("❌ Server returned non-JSON response.")
            sys.exit(1)

    # Save output to JSON file
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=2)
        print(f"💾 Success! Response saved to '{args.output}'")
    except Exception as e:
        print(f"❌ Error saving JSON response: {e}")
        sys.exit(1)

    # Display stats
    print(f"📊 Summary:")
    print(f"   - Status: success")
    print(f"   - Total rows predicted: {response_data.get('total_rows')}")
    print(f"   - Format: {response_data.get('format')}")
    print(f"   - Metadata: {response_data.get('pipeline_metadata')}")


if __name__ == "__main__":
    main()
