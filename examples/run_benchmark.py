"""Run benchmark analysis using the evidence bundle pipeline.

Analyzes benchmark clusters from OPS, DepMap, and Proteomics datasets using the
bundle-based pipeline (accession lookup → evidence bundles → LLM Phase 1).

Usage:
    python run_benchmark.py --dataset ops --model claude-sonnet-4-6
    python run_benchmark.py --dataset ops --model claude-sonnet-4-6 --cot
    python run_benchmark.py --dataset all --model claude-sonnet-4-6 --cot
    python run_benchmark.py --dataset all --cot --mcp                      # unified (default)
    python run_benchmark.py --dataset all --cot --mcp --two-phase          # separate Phase 2 call
    python run_benchmark.py --dataset ops --clusters 21 37 --model claude-sonnet-4-6
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from mozzarellm.clients.llm_api_clients import create_client
from mozzarellm.pipeline.bundle_builder import (
    build_evidence_bundles,
    get_or_append_stable_accession,
)
from mozzarellm.pipeline.literature_mcp import (
    get_available_mcp_servers,
    validate_and_amend_with_mcp,
)
from mozzarellm.utils.cluster_utils import build_cluster_id_to_bundle_path
from mozzarellm.utils.llm_analysis_utils import process_cluster_response
from mozzarellm.utils.prompt_factory import (
    make_cluster_analysis_system_prompt,
    make_single_cluster_analysis_user_prompt,
)

load_dotenv()

EXAMPLES_DIR = Path(__file__).parent

# ─── Ground truth ────────────────────────────────────────────────────────────

VALIDATION_DATA = {
    "ops": {
        "21": {"function": "ribosome biogenesis", "genes": ["C1orf131"]},
        "37": {
            "function": "mTOR signaling/ER-Golgi transport/Integrator complex",
            "genes": ["C7orf26"],
        },
        "99": {
            "function": "No coherent biological pathway",
            "confidence": "Low",
            "genes": [],
        },
        "121": {"function": "Myc regulation/transcription", "genes": ["SETD2"]},
        "149": {"function": "mitochondrial homeostasis", "genes": ["KRAS", "BRAF"]},
        "167": {"function": "proteasome function", "genes": ["AKIRIN2"]},
        "197": {"function": "m6A mRNA modification", "genes": ["HNRNPD"]},
    },
    "depmap": {
        "2067": {"function": "clathrin-mediated endocytosis", "genes": ["C15orf57"]},
        "2213": {"function": "ether lipid synthesis", "genes": ["TMEM189"]},
    },
    "proteomics": {
        "C5255": {"function": "RNase P/mitochondrial RNA processing", "genes": ["C18orf21"]},
        "C5415": {"function": "interferon response regulation", "genes": ["DPP9"]},
    },
}

# ─── Dataset configurations ───────────────────────────────────────────────────

DATASET_CONFIGS = {
    "ops": {
        "csv": EXAMPLES_DIR / "ops" / "funk_2022.csv",
        "screen_context": EXAMPLES_DIR / "ops" / "screen_context.json",
        "gene_col": "gene_symbol",
        "cluster_col": "cluster",
        "feature_columns": ["up_features", "down_features"],
        "check_confidence": True,
    },
    "depmap": {
        "csv": EXAMPLES_DIR / "depmap" / "wainberg_2021.csv",
        "screen_context": EXAMPLES_DIR / "depmap" / "screen_context.json",
        "gene_col": "gene_symbol",
        "cluster_col": "cluster",
        "feature_columns": [],
        "check_confidence": False,
    },
    "proteomics": {
        "csv": EXAMPLES_DIR / "proteomics" / "schaffer_2025.csv",
        "screen_context": EXAMPLES_DIR / "proteomics" / "screen_context.json",
        "gene_col": "gene_symbol",
        "cluster_col": "cluster",
        "feature_columns": [],
        "check_confidence": False,
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _categorize_gene(gene: str, parsed: dict) -> str:
    if gene in parsed.get("established_genes", []):
        return "established"
    if any(g["gene"] == gene for g in parsed.get("novel_role_genes", [])):
        return "novel_role"
    if any(g["gene"] == gene for g in parsed.get("uncharacterized_genes", [])):
        return "uncharacterized"
    return "not_classified"


def _validate_cluster(cluster_id: str, parsed: dict, expected: dict, check_confidence: bool) -> dict:
    predicted_func = parsed.get("dominant_process", "")
    # Word-level match: any key word (>4 chars) from expected appears in prediction
    key_words = [w for w in expected["function"].lower().replace("/", " ").split() if len(w) > 4]
    function_match = any(w in predicted_func.lower() for w in key_words)

    confidence_match = True
    if check_confidence and "confidence" in expected:
        confidence_match = parsed.get("pathway_confidence") == expected["confidence"]

    gene_results = []
    for gene in expected.get("genes", []):
        category = _categorize_gene(gene, parsed)
        gene_results.append({
            "gene": gene,
            "actual_category": category,
            "validated": category in ("novel_role", "uncharacterized"),
        })

    return {
        "cluster_id": cluster_id,
        "expected_function": expected["function"],
        "predicted_function": predicted_func,
        "function_match": function_match,
        "confidence_match": confidence_match,
        "predicted_confidence": parsed.get("pathway_confidence"),
        "gene_results": gene_results,
    }


def run_dataset(
    dataset_name: str,
    model: str,
    cot: bool,
    mcp: bool,
    unified: bool,
    clusters: list[str] | None,
    run_dir: Path,
    bundle_cache_dir: Path,
    rebuild_bundles: bool,
) -> dict:
    """Run Phase 1 benchmark for one dataset. Returns validation summary dict."""
    cfg = DATASET_CONFIGS[dataset_name]
    validation = VALIDATION_DATA[dataset_name]
    screen_name = f"benchmark_{dataset_name}"

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}  |  Model: {model}  |  CoT: {cot}")
    print(f"{'='*60}")

    # Load gene-wise data and filter to benchmark clusters
    cluster_df = pd.read_csv(cfg["csv"])
    target_clusters = clusters or list(validation.keys())
    cluster_df = cluster_df[
        cluster_df[cfg["cluster_col"]].astype(str).isin(target_clusters)
    ].copy()
    print(f"Loaded {len(cluster_df)} genes across {cluster_df[cfg['cluster_col']].nunique()} clusters")

    bundle_dir = bundle_cache_dir / f"{screen_name}_analysis" / f"{screen_name}_evidence_bundles"

    if not rebuild_bundles and bundle_dir.exists():
        print(f"Using cached bundles at: {bundle_dir}")
    else:
        print("Looking up UniProt accessions...")
        acc_cluster_df = get_or_append_stable_accession(
            screen_name=screen_name,
            cluster_df=cluster_df,
            gene_column=cfg["gene_col"],
            organism_id=9606,
            warn_on_fallback=False,
            output_dir=bundle_cache_dir,
        )
        print("Building evidence bundles...")
        build_evidence_bundles(
            screen_name=screen_name,
            acc_cluster_df=acc_cluster_df,
            gene_column=cfg["gene_col"],
            cluster_id_column=cfg["cluster_col"],
            stable_accession_col="accession",
            feature_columns=cfg["feature_columns"] or None,
            output_dir=bundle_cache_dir,
        )

    cluster_to_bundle = build_cluster_id_to_bundle_path(bundle_dir, screen_name=screen_name)
    print(f"Bundles ready for: {sorted(cluster_to_bundle.keys())}")

    # Build system prompt — for unified mode, inject the literature validation step
    system_prompt_kwargs = {
        "screen_name": screen_name,
        "screen_context_path": cfg["screen_context"],
        "CoT_mode": cot,
        "output_dir": run_dir / "prompts_used",
    }
    if unified:
        from mozzarellm.prompt_components import COT_STEPS_UNIFIED_MCP
        system_prompt_kwargs["override_CoT_steps"] = COT_STEPS_UNIFIED_MCP
    system_prompt = make_cluster_analysis_system_prompt(**system_prompt_kwargs)

    # Init LLM client (only used for non-unified path)
    client = create_client(model=model, api_key=os.getenv("ANTHROPIC_API_KEY")) if not unified else None

    # Phase 1: query each benchmark cluster
    results = {}
    cluster_metrics = {}

    for cluster_id in target_clusters:
        if str(cluster_id) not in cluster_to_bundle:
            print(f"  [SKIP] Cluster {cluster_id}: no bundle found")
            continue

        label = "Querying+validating" if unified else "Querying"
        print(f"  {label} cluster {cluster_id}...", end=" ", flush=True)
        t0 = time.time()

        user_prompt = make_single_cluster_analysis_user_prompt(
            cluster_id, screen_name, cluster_to_bundle
        )

        if unified:
            from mozzarellm.pipeline.literature_mcp import analyze_and_validate_unified
            try:
                parsed = analyze_and_validate_unified(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                )
                elapsed = time.time() - t0
                meta = parsed.get("_validation_metadata", {})
                if "error" in meta and "input_tokens" not in meta:
                    print(f"ERROR ({elapsed:.1f}s): {meta.get('error')}")
                    results[str(cluster_id)] = {"error": meta.get("error")}
                    cluster_metrics[str(cluster_id)] = {"elapsed_s": round(elapsed, 1), "error": meta.get("error")}
                    continue
                results[str(cluster_id)] = parsed
                in_tok = meta.get("input_tokens", 0)
                out_tok = meta.get("output_tokens", 0)
                cost = meta.get("cost_usd", 0)
                tool_calls = meta.get("tool_calls", 0)
                pathway_rev = parsed.get("literature_informed_pathway_revision", {})
                cluster_metrics[str(cluster_id)] = {
                    "elapsed_s": round(elapsed, 1),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": cost,
                    "tool_calls": tool_calls,
                    "pathway_changed": pathway_rev.get("pathway_changed", False),
                    "pre_literature_pathway": pathway_rev.get("pre_literature_pathway"),
                    "post_literature_pathway": pathway_rev.get("post_literature_pathway"),
                }
                reclassified = parsed.get("literature_informed_reclassifications", [])
                print(f"OK ({elapsed:.1f}s | {tool_calls} tool calls | {in_tok}in/{out_tok}out | ${cost:.4f} | {len(reclassified)} lit-reclassifications)")
                for r in reclassified:
                    print(f"    {r.get('gene', '?')}: {r.get('initial_category', '?')} → {r.get('final_category', '?')} (PMIDs: {','.join(r.get('driving_pmids', []))})")
                if pathway_rev.get("pathway_changed"):
                    print(f"    PATHWAY REVISED: {pathway_rev.get('pre_literature_pathway', '?')} → {pathway_rev.get('post_literature_pathway', '?')}")
                    print(f"      Reason: {pathway_rev.get('rationale', '?')}")
                print(
                    f"    {parsed.get('dominant_process', '?')} "
                    f"[{parsed.get('pathway_confidence', '?')}]  "
                    f"est={len(parsed.get('established_genes', []))} "
                    f"nov={len(parsed.get('novel_role_genes', []))} "
                    f"unc={len(parsed.get('uncharacterized_genes', []))}"
                )
            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR ({elapsed:.1f}s): {e}")
                results[str(cluster_id)] = {"error": str(e)}
                cluster_metrics[str(cluster_id)] = {"elapsed_s": round(elapsed, 1), "error": str(e)}
            continue

        response_text, error = client.query(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=3,
        )

        elapsed = time.time() - t0
        usage = getattr(client, "last_usage", {})

        if error:
            print(f"ERROR ({elapsed:.1f}s)")
            print(f"    {error}")
            results[str(cluster_id)] = {"error": error}
            cluster_metrics[str(cluster_id)] = {"elapsed_s": round(elapsed, 1), "error": error}
            continue

        try:
            parsed = process_cluster_response(response_text)
            results[str(cluster_id)] = parsed

            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            # sonnet-4-6 pricing: $3/M in, $15/M out
            cost = (in_tok * 3 + out_tok * 15) / 1_000_000

            cluster_metrics[str(cluster_id)] = {
                "elapsed_s": round(elapsed, 1),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": round(cost, 4),
            }
            print(
                f"OK ({elapsed:.1f}s | {in_tok}in/{out_tok}out | ${cost:.4f})"
            )
            print(
                f"    {parsed.get('dominant_process', '?')} "
                f"[{parsed.get('pathway_confidence', '?')}]  "
                f"est={len(parsed.get('established_genes', []))} "
                f"nov={len(parsed.get('novel_role_genes', []))} "
                f"unc={len(parsed.get('uncharacterized_genes', []))}"
            )
        except Exception as e:
            print(f"PARSE ERROR ({elapsed:.1f}s)")
            print(f"    {e}")
            results[str(cluster_id)] = {"error": str(e), "raw": response_text}
            cluster_metrics[str(cluster_id)] = {"elapsed_s": round(elapsed, 1), "parse_error": str(e)}

    # Phase 2: MCP literature validation
    if mcp:
        print("\n  Running Phase 2 (MCP literature validation)...")
        for cluster_id in list(results.keys()):
            parsed = results[cluster_id]
            if "error" in parsed:
                continue
            print(f"  Validating cluster {cluster_id} via MCP...", end=" ", flush=True)
            t0 = time.time()
            try:
                amended = validate_and_amend_with_mcp(parsed, model=model)
                elapsed = time.time() - t0
                reclassified = amended.get("reclassifications", [])
                mcp_meta = amended.get("_validation_metadata", {})
                tool_calls = mcp_meta.get("tool_calls", 0)
                print(f"OK ({elapsed:.1f}s | {tool_calls} tool calls | ${mcp_meta.get('cost_usd', 0):.4f} | {len(reclassified)} reclassifications)")
                if reclassified:
                    for r in reclassified:
                        print(f"    {r['gene']} → {r['new_category']}")
                results[cluster_id] = amended
                cluster_metrics[cluster_id]["mcp_elapsed_s"] = round(elapsed, 1)
                cluster_metrics[cluster_id]["mcp_reclassifications"] = len(reclassified)
                mcp_meta = amended.get("_validation_metadata", {})
                if "input_tokens" in mcp_meta:
                    cluster_metrics[cluster_id]["mcp_input_tokens"] = mcp_meta["input_tokens"]
                    cluster_metrics[cluster_id]["mcp_output_tokens"] = mcp_meta["output_tokens"]
                    cluster_metrics[cluster_id]["mcp_cost_usd"] = mcp_meta["cost_usd"]
            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAILED ({elapsed:.1f}s): {e}")
                cluster_metrics[cluster_id]["mcp_error"] = str(e)

    # Validate
    validation_rows = []
    for cluster_id in target_clusters:
        expected = validation.get(str(cluster_id))
        parsed = results.get(str(cluster_id))
        if not expected or not parsed or "error" in parsed:
            continue
        row = _validate_cluster(str(cluster_id), parsed, expected, cfg["check_confidence"])
        row["dataset"] = dataset_name
        row["model"] = model
        row["cot"] = cot
        row["mcp"] = mcp
        row["metrics"] = cluster_metrics.get(str(cluster_id), {})
        validation_rows.append(row)

    # Totals
    total_cost = sum(m.get("cost_usd", 0) for m in cluster_metrics.values())
    total_time = sum(m.get("elapsed_s", 0) for m in cluster_metrics.values())
    total_in = sum(m.get("input_tokens", 0) for m in cluster_metrics.values())
    total_out = sum(m.get("output_tokens", 0) for m in cluster_metrics.values())
    total_mcp_cost = sum(m.get("mcp_cost_usd", 0) for m in cluster_metrics.values())
    total_mcp_time = sum(m.get("mcp_elapsed_s", 0) for m in cluster_metrics.values())
    cost_line = f"${total_cost:.4f}"
    if total_mcp_cost > 0:
        cost_line += f" + ${total_mcp_cost:.4f} MCP = ${total_cost + total_mcp_cost:.4f} total"
    print(
        f"\n  Totals: {total_time:.1f}s | {total_in}in/{total_out}out | {cost_line}"
    )
    if total_mcp_time > 0:
        print(f"  MCP:    {total_mcp_time:.1f}s")

    # Save per-dataset results
    dataset_out = run_dir / f"{dataset_name}_results.json"
    dataset_out.write_text(json.dumps({
        "dataset": dataset_name,
        "model": model,
        "cot": cot,
        "mcp": mcp,
        "cluster_metrics": cluster_metrics,
        "results": results,
    }, indent=2))
    print(f"  Results: {dataset_out}")

    return {
        "dataset": dataset_name,
        "model": model,
        "cot": cot,
        "validation": validation_rows,
        "totals": {
            "elapsed_s": round(total_time, 1),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cost_usd": round(total_cost, 4),
            "mcp_cost_usd": round(total_mcp_cost, 4),
            "mcp_elapsed_s": round(total_mcp_time, 1),
        },
    }


def print_summary(all_summaries: list[dict], run_dir: Path):
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    all_rows = []
    for summary in all_summaries:
        for row in summary["validation"]:
            all_rows.append(row)

    if not all_rows:
        print("No results to validate.")
        return

    total = len(all_rows)
    func_matches = sum(r["function_match"] for r in all_rows)
    conf_checks = [
        r for r in all_rows
        if "confidence" in VALIDATION_DATA.get(r["dataset"], {}).get(r["cluster_id"], {})
    ]
    conf_matches = sum(r["confidence_match"] for r in conf_checks)
    gene_total = sum(len(r["gene_results"]) for r in all_rows)
    gene_validated = sum(sum(g["validated"] for g in r["gene_results"]) for r in all_rows)

    pathway_revisions = sum(1 for r in all_rows if r["metrics"].get("pathway_changed"))

    print(f"Function matches:    {func_matches}/{total} ({100*func_matches/total:.0f}%)")
    if conf_checks:
        print(f"Confidence matches:  {conf_matches}/{len(conf_checks)} ({100*conf_matches/len(conf_checks):.0f}%)")
    if gene_total:
        print(f"Gene validation:     {gene_validated}/{gene_total} ({100*gene_validated/gene_total:.0f}%)")
    print(f"Pathway revisions:   {pathway_revisions}/{total}")

    # Overall cost/time
    grand_cost = sum(s["totals"]["cost_usd"] for s in all_summaries)
    grand_mcp_cost = sum(s["totals"].get("mcp_cost_usd", 0) for s in all_summaries)
    grand_time = sum(s["totals"]["elapsed_s"] for s in all_summaries)
    grand_mcp_time = sum(s["totals"].get("mcp_elapsed_s", 0) for s in all_summaries)
    print(f"Phase 1 cost:        ${grand_cost:.4f}")
    if grand_mcp_cost > 0:
        print(f"Phase 2 (MCP) cost:  ${grand_mcp_cost:.4f}")
    print(f"Total cost:          ${grand_cost + grand_mcp_cost:.4f}")
    print(f"Phase 1 time:        {grand_time:.1f}s")
    if grand_mcp_time > 0:
        print(f"Phase 2 (MCP) time:  {grand_mcp_time:.1f}s")

    print(f"\n{'Cluster':<12} {'Dataset':<12} {'Func':>5} {'Conf':>5} {'PthRev':>6} {'Time(s)':>8} {'Cost($)':>8}  Predicted function")
    print("-" * 95)
    for row in all_rows:
        func_mark = "✓" if row["function_match"] else "✗"
        conf_mark = "✓" if row["confidence_match"] else "✗"
        pth_mark = "✓" if row["metrics"].get("pathway_changed") else "·"
        predicted = row["predicted_function"][:38]
        t = row["metrics"].get("elapsed_s", "-")
        c = row["metrics"].get("cost_usd", "-")
        print(f"{row['cluster_id']:<12} {row['dataset']:<12} {func_mark:>5} {conf_mark:>5} {pth_mark:>6} {t:>8} {c:>8}  {predicted}")
        if row["metrics"].get("pathway_changed"):
            print(f"{'':>12} {'':>12} {'':>5} {'':>5} {'':>6} {'':>8} {'':>8}  was: {row['metrics'].get('pre_literature_pathway', '?')}")

    # Save summary JSON
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "run_dir": str(run_dir),
        "function_matches": f"{func_matches}/{total}",
        "confidence_matches": f"{conf_matches}/{len(conf_checks)}" if conf_checks else "N/A",
        "gene_validation": f"{gene_validated}/{gene_total}" if gene_total else "N/A",
        "pathway_revisions": f"{pathway_revisions}/{total}",
        "phase1_cost_usd": round(grand_cost, 4),
        "mcp_cost_usd": round(grand_mcp_cost, 4),
        "total_cost_usd": round(grand_cost + grand_mcp_cost, 4),
        "phase1_time_s": round(grand_time, 1),
        "mcp_time_s": round(grand_mcp_time, 1),
        "per_dataset": all_summaries,
    }, indent=2))
    print(f"\nSummary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark using the evidence bundle pipeline")
    parser.add_argument("--dataset", choices=["ops", "depmap", "proteomics", "all"], default="ops")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--cot", action="store_true", help="Enable chain-of-thought mode")
    parser.add_argument("--mcp", action="store_true", help="Enable MCP literature validation (unified single-call by default, requires --cot)")
    parser.add_argument("--two-phase", action="store_true", dest="two_phase", help="Use two-phase MCP: separate LLM call after Phase 1 instead of unified single-call (requires --mcp)")
    parser.add_argument("--clusters", nargs="+", help="Specific cluster IDs to test")
    parser.add_argument(
        "--bundle-cache-dir",
        type=Path,
        default=EXAMPLES_DIR / "benchmark_results" / "bundle_cache",
        help="Stable directory for cached evidence bundles (reused across runs)",
    )
    parser.add_argument("--rebuild-bundles", action="store_true", help="Force rebuild evidence bundles")
    args = parser.parse_args()

    # Mode validation
    if args.mcp and not args.cot:
        print("ERROR: --mcp requires --cot.")
        raise SystemExit(1)
    if args.two_phase and not args.mcp:
        print("ERROR: --two-phase requires --mcp.")
        raise SystemExit(1)

    # Derive unified flag: --mcp without --two-phase means unified
    unified = args.mcp and not args.two_phase

    # MCP preflight check
    if args.mcp:
        available = get_available_mcp_servers()
        if "pubmed" not in available:
            print("ERROR: PubMed MCP server is unavailable. Run without --mcp or try later.")
            raise SystemExit(1)
        print(f"MCP preflight: PubMed available ✓ ({'unified' if unified else 'two-phase'} mode)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cot_tag = "_cot" if args.cot else ""
    mcp_tag = "_mcp_twophase" if (args.mcp and args.two_phase) else ("_mcp" if args.mcp else "")
    run_dir = EXAMPLES_DIR / "benchmark_results" / f"run_{timestamp}_{args.dataset}_{args.model.replace('/', '_')}{cot_tag}{mcp_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]

    all_summaries = []
    for ds in datasets:
        summary = run_dataset(
            dataset_name=ds,
            model=args.model,
            cot=args.cot,
            mcp=args.mcp and args.two_phase,
            unified=unified,
            clusters=[str(c) for c in args.clusters] if args.clusters else None,
            run_dir=run_dir,
            bundle_cache_dir=args.bundle_cache_dir,
            rebuild_bundles=args.rebuild_bundles,
        )
        all_summaries.append(summary)

    print_summary(all_summaries, run_dir)


if __name__ == "__main__":
    main()
