"""
run_agent.py — Headless CLI runner with APScheduler for production scheduling
Usage:
    python run_agent.py --topic "AI Regulation" --interval 60
    python run_agent.py --once   # single run
"""

import argparse
import time
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

load_dotenv()

from src.collector import collect_and_report


def main():
    parser = argparse.ArgumentParser(description="GenAI Report Agent — CLI runner")
    parser.add_argument("--topic",    default="AI Regulation",            help="Monitoring topic")
    parser.add_argument("--sources",  default="BBC Technology,Google News", help="Comma-separated sources")
    parser.add_argument("--interval", type=int, default=60,               help="Interval in minutes (default 60)")
    parser.add_argument("--once",     action="store_true",                help="Run once then exit")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",")]

    def job():
        print(f"\n{'='*60}")
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Starting collection cycle")
        report = collect_and_report(topic=args.topic, sources=sources)
        print(f"\nREPORT SUMMARY — {report['topic']}")
        print(f"   Articles: {report['article_count']}")
        print(f"\nSummary:\n{report['summary']}")
        print(f"\nTakeaways:")
        for i, tk in enumerate(report["takeaways"], 1):
            print(f"   {i}. {tk}")
        print(f"\nEntities: {', '.join(report['entities'])}")
        print(f"Topics:   {', '.join(report['key_topics'])}")
        print(f"{'='*60}")

    if args.once:
        job()
        return

    print(f"GenAI Report Agent started")
    print(f"   Topic: {args.topic} | Sources: {sources} | Interval: {args.interval} min")
    print(f"   Press Ctrl+C to stop\n")

    # Run immediately, then schedule
    job()

    scheduler = BlockingScheduler()
    scheduler.add_job(job, "interval", minutes=args.interval, id="report_job")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\nAgent stopped.")


if __name__ == "__main__":
    main()
