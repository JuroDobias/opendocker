import argparse

from .pipeline import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="opendocker")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Run docking from a YAML config")
    run_cmd.add_argument("-c", "--config", required=True, help="Path to YAML config")

    args = parser.parse_args()

    if args.command == "run":
        run_from_config(args.config)
