"""CLI script for Phase 2: Dataset provenance verification and comprehensive data audit."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from liver_disease.config import REPORTS_DIR, ensure_project_directories
from liver_disease.data.loader import (
    ensure_raw_dataset,
    get_dataset_provenance,
    load_raw_dataset,
)
from liver_disease.data.validation import run_comprehensive_data_audit
from liver_disease.utils.io import save_json
from liver_disease.utils.logging_config import setup_logging


def main() -> None:
    logger = setup_logging("INFO")
    ensure_project_directories()

    raw_path = ensure_raw_dataset()
    provenance = get_dataset_provenance(raw_path)
    df = load_raw_dataset(raw_path)
    audit = run_comprehensive_data_audit(df)

    report = {
        "provenance": provenance,
        "audit": audit,
    }
    output_path = REPORTS_DIR / "data_audit_report.json"
    save_json(report, output_path)
    logger.info("Saved comprehensive data audit report to %s", output_path)

    # Print key findings summary
    print("\n=== ILPD DATA AUDIT SUMMARY ===")
    print(f"Shape: {audit['shape']['rows']} rows x {audit['shape']['columns']} columns")
    print(f"SHA-256: {provenance['sha256']}")
    print(f"Target (1=Disease): {audit['target_distribution']['liver_disease_count']} "
          f"({audit['target_distribution']['liver_disease_pct']}%)")
    print(f"Target (2=No Disease): {audit['target_distribution']['no_liver_disease_count']} "
          f"({audit['target_distribution']['no_liver_disease_pct']}%)")
    print(f"Missing cells: {audit['missing_values']['total_missing_cells']} "
          f"-> {audit['missing_values']['by_column']}")
    print(f"Exact duplicate extra rows: {audit['duplicates']['exact_duplicate_extra_rows']} "
          f"({audit['duplicates']['exact_duplicate_groups_count']} groups)")
    print(f"Feature-only duplicate extra rows: {audit['duplicates']['feature_only_duplicate_extra_rows']}")
    print(f"Target conflicts among feature duplicates: {len(audit['duplicates']['feature_duplicate_target_conflicts'])}")
    print(f"Near-duplicate pairs (std L2 < 0.25): {len(audit['duplicates']['near_duplicate_pairs_z_lt_0_25'])}")
    print(f"Domain consistency checks: {audit['domain_consistency_checks']}")


if __name__ == "__main__":
    main()
