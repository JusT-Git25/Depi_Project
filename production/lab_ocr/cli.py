import argparse
import json
import sys
from pathlib import Path

from .exceptions import LabOCRError
from .service import LabOCRService


def _print_results(report) -> None:
    print(f"\nExtracted {len(report.results)} result(s) from {report.source_filename}:")
    for i, r in enumerate(report.results, 1):
        op = r.value_operator or ""
        print(f"  {i:>2}. {r.test_name or '?':<28} {op}{r.value!s:<10} {r.unit or '':<10} [{r.validation_status}]")
    for w in report.warnings:
        print(f"  ! {w}")


def _analyze(args) -> int:
    try:
        report = LabOCRService().process_file(args.file)
    except (LabOCRError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    _print_results(report)
    if args.json_out:
        Path(args.json_out).write_text(report.model_dump_json(indent=2, exclude={"pages": {"__all__": {"words"}}}), encoding="utf-8")
        print(f"\nSaved extraction to {args.json_out}")
    if not args.model_spec:
        return 0
    if not args.yes:
        answer = input("\nHave you checked these values against the original report? [y/N] ").strip().lower()
        if answer != "y":
            print("Prediction skipped. Review the values first.")
            return 0
    try:
        from .prediction import RiskPredictor
        result = RiskPredictor.from_spec_file(args.model_spec).predict_report(report, allow_unreviewed=True)
    except LabOCRError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0 if result.status == "ok" else 2


def _serve(args) -> int:
    import os
    import uvicorn
    if args.model_spec:
        os.environ["LAB_OCR_MODEL_SPEC"] = args.model_spec
    uvicorn.run("lab_ocr.api:app", host=args.host, port=args.port)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lab_ocr", description="Lab report OCR and risk prediction")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="Start the web app (FastAPI)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--model-spec", help="Path to model spec JSON (or set LAB_OCR_MODEL_SPEC)")
    s.set_defaults(func=_serve)

    a = sub.add_parser("analyze", help="OCR a report from the command line, optionally predict")
    a.add_argument("file")
    a.add_argument("--model-spec", help="Path to model spec JSON")
    a.add_argument("--json-out", help="Write the extraction JSON here")
    a.add_argument("--yes", action="store_true", help="Skip the confirmation prompt (you vouch the values are correct)")
    a.set_defaults(func=_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
