import argparse
import logging
import sys
from pathlib import Path

from src.config import Settings
from src.llm import create_llm_client
from src.pipeline.indexing_pipeline import run_indexing
from src.pipeline.query_pipeline import run_query


def configure_logging(settings: Settings) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if settings.logging.log_file:
        log_path = Path(settings.logging.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path)))

    logging.basicConfig(
        level=getattr(logging, settings.logging.level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=handlers,
        force=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redrob Intelligent Candidate Discovery & Ranking Engine"
    )
    parser.add_argument("--config", type=str, default="config/settings.yaml",
                        help="Path to settings.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # index
    index_parser = subparsers.add_parser("index", help="Build candidate index")
    index_parser.add_argument("--data", type=str, default=None,
                              help="Path to candidates.jsonl")
    index_parser.add_argument("--processed", type=str, default=None,
                              help="Directory for index artifacts")

    # query
    query_parser = subparsers.add_parser("query", help="Run query pipeline")
    query_parser.add_argument("--jd", type=str, default=None,
                              help="Path to JD text file")
    query_parser.add_argument("--processed", type=str, default=None,
                              help="Directory for index artifacts")
    query_parser.add_argument("--output", type=str, default=None,
                              help="Directory for ranked output files")

    # run
    run_parser = subparsers.add_parser("run", help="Run index + query")
    run_parser.add_argument("--data", type=str, default=None,
                            help="Path to candidates.jsonl")
    run_parser.add_argument("--jd", type=str, default=None,
                            help="Path to JD text file")
    run_parser.add_argument("--processed", type=str, default=None,
                            help="Directory for index artifacts")
    run_parser.add_argument("--output", type=str, default=None,
                            help="Directory for output files")

    args = parser.parse_args()

    # Load settings
    settings = Settings.from_yaml(Path(args.config))
    configure_logging(settings)
    logger = logging.getLogger(__name__)

    # Override paths from CLI args
    if args.command in ("index", "run"):
        if args.data:
            settings.paths.raw_data = args.data
        if getattr(args, "processed", None):
            settings.paths.processed_dir = args.processed

    if args.command == "query":
        if args.jd:
            settings.paths.raw_data = ""
        if args.processed:
            settings.paths.processed_dir = args.processed
        if args.output:
            settings.paths.output_dir = args.output

    if args.command == "run":
        if args.output:
            settings.paths.output_dir = args.output

    processed_dir = Path(settings.paths.processed_dir)
    output_dir = Path(settings.paths.output_dir)

    if args.command == "index":
        jsonl_path = Path(settings.paths.raw_data)
        if not jsonl_path.exists():
            logger.error("Data file not found: %s", jsonl_path)
            sys.exit(1)
        run_indexing(jsonl_path, processed_dir, settings)

    elif args.command in ("query", "run"):
        llm_client = create_llm_client(settings)
        if args.command == "query":
            jd_path = Path(args.jd) if args.jd else Path("data/jd.txt")
            if not jd_path.exists():
                logger.error("JD file not found: %s", jd_path)
                sys.exit(1)
            jd_text = jd_path.read_text(encoding="utf-8")
            run_query(jd_text, processed_dir, output_dir, settings, llm_client)
        else:
            jsonl_path = Path(settings.paths.raw_data)
            jd_path = Path(args.jd) if args.jd else Path("data/jd.txt")
            if not jsonl_path.exists():
                logger.error("Data file not found: %s", jsonl_path)
                sys.exit(1)
            if not jd_path.exists():
                logger.error("JD file not found: %s", jd_path)
                sys.exit(1)
            run_indexing(jsonl_path, processed_dir, settings)
            jd_text = jd_path.read_text(encoding="utf-8")
            run_query(jd_text, processed_dir, output_dir, settings, llm_client)


if __name__ == "__main__":
    main()
