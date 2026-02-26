#!/usr/bin/env python3
"""
Analyse historical clone data for FYP Section 4.
Extracts pre-LLM (2019-2021) vs post-LLM (2022-2025) statistics.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics

# LLM adoption boundary: GitHub Copilot GA June 2022
LLM_BOUNDARY = datetime(2022, 6, 1)

def load_history(json_path: Path) -> dict:
    """Load history JSON file."""
    with open(json_path) as f:
        return json.load(f)

def categorise_snapshot(date_str: str) -> str:
    """Categorise snapshot as pre-LLM or post-LLM."""
    date = datetime.strptime(date_str, '%Y-%m-%d')
    return 'post_llm' if date >= LLM_BOUNDARY else 'pre_llm'

def analyse_repository(history: dict) -> dict:
    """Analyse a single repository's history."""
    repo_name = history.get('repo_name', 'Unknown')
    snapshots = history.get('snapshots', [])
    
    if not snapshots:
        return None
    
    # Group snapshots by period
    periods = {'pre_llm': [], 'post_llm': []}
    
    for snap in snapshots:
        date = snap.get('date')
        metrics = snap.get('metrics', {})
        period = categorise_snapshot(date)
        
        periods[period].append({
            'date': date,
            'duplication_ratio': metrics.get('duplication_ratio', 0),
            'duplicate_pairs': metrics.get('duplicate_pairs', 0),
            'total_blocks': metrics.get('total_blocks', 0),
            'clone_types': metrics.get('clone_types', {})
        })
    
    # Calculate statistics for each period
    result = {
        'repo_name': repo_name,
        'total_snapshots': len(snapshots),
        'date_range': f"{snapshots[0]['date']} to {snapshots[-1]['date']}",
        'pre_llm_snapshots': len(periods['pre_llm']),
        'post_llm_snapshots': len(periods['post_llm']),
    }
    
    for period_name, period_data in periods.items():
        if period_data:
            ratios = [d['duplication_ratio'] for d in period_data]
            pairs = [d['duplicate_pairs'] for d in period_data]
            blocks = [d['total_blocks'] for d in period_data]
            
            result[f'{period_name}_mean_ratio'] = statistics.mean(ratios)
            result[f'{period_name}_median_ratio'] = statistics.median(ratios)
            result[f'{period_name}_std_ratio'] = statistics.stdev(ratios) if len(ratios) > 1 else 0
            result[f'{period_name}_mean_pairs'] = statistics.mean(pairs)
            result[f'{period_name}_mean_blocks'] = statistics.mean(blocks)
            result[f'{period_name}_start_ratio'] = ratios[0]
            result[f'{period_name}_end_ratio'] = ratios[-1]
        else:
            result[f'{period_name}_mean_ratio'] = None
            result[f'{period_name}_median_ratio'] = None
    
    # Calculate change between periods
    if result.get('pre_llm_mean_ratio') and result.get('post_llm_mean_ratio'):
        pre = result['pre_llm_mean_ratio']
        post = result['post_llm_mean_ratio']
        result['ratio_change'] = post - pre
        result['ratio_change_pct'] = ((post - pre) / pre) * 100 if pre > 0 else 0
        result['direction'] = 'increased' if post > pre else 'decreased'
    
    return result

def generate_summary_table(results: list) -> str:
    """Generate a summary table for the report."""
    lines = []
    lines.append("\n" + "="*100)
    lines.append("DATASET SUMMARY")
    lines.append("="*100)
    
    total_snapshots = sum(r['total_snapshots'] for r in results)
    lines.append(f"Total repositories analysed: {len(results)}")
    lines.append(f"Total snapshots analysed: {total_snapshots}")
    
    lines.append("\n" + "-"*100)
    lines.append(f"{'Repository':<20} {'Pre-LLM Ratio':>15} {'Post-LLM Ratio':>15} {'Change':>12} {'Direction':>12}")
    lines.append("-"*100)
    
    increased = 0
    decreased = 0
    
    for r in results:
        pre = r.get('pre_llm_mean_ratio')
        post = r.get('post_llm_mean_ratio')
        change = r.get('ratio_change_pct')
        direction = r.get('direction', 'N/A')
        
        pre_str = f"{pre*100:.1f}%" if pre else "N/A"
        post_str = f"{post*100:.1f}%" if post else "N/A"
        change_str = f"{change:+.1f}%" if change else "N/A"
        
        if direction == 'increased':
            increased += 1
        elif direction == 'decreased':
            decreased += 1
        
        lines.append(f"{r['repo_name']:<20} {pre_str:>15} {post_str:>15} {change_str:>12} {direction:>12}")
    
    lines.append("-"*100)
    lines.append(f"\nRepositories with INCREASED duplication post-LLM: {increased}/{len(results)} ({increased/len(results)*100:.0f}%)")
    lines.append(f"Repositories with DECREASED duplication post-LLM: {decreased}/{len(results)} ({decreased/len(results)*100:.0f}%)")
    
    return "\n".join(lines)

def generate_detailed_stats(results: list) -> str:
    """Generate detailed statistics for each repository."""
    lines = []
    lines.append("\n" + "="*100)
    lines.append("DETAILED STATISTICS BY REPOSITORY")
    lines.append("="*100)
    
    for r in results:
        lines.append(f"\n{r['repo_name']}")
        lines.append("-" * len(r['repo_name']))
        lines.append(f"  Date range: {r['date_range']}")
        lines.append(f"  Total snapshots: {r['total_snapshots']}")
        lines.append(f"  Pre-LLM snapshots: {r['pre_llm_snapshots']}")
        lines.append(f"  Post-LLM snapshots: {r['post_llm_snapshots']}")
        
        if r.get('pre_llm_mean_ratio'):
            lines.append(f"  Pre-LLM period (2019-2021):")
            lines.append(f"    Mean duplication ratio: {r['pre_llm_mean_ratio']*100:.2f}%")
            lines.append(f"    Median duplication ratio: {r['pre_llm_median_ratio']*100:.2f}%")
            lines.append(f"    Std deviation: {r['pre_llm_std_ratio']*100:.2f}%")
            lines.append(f"    Mean clone pairs: {r['pre_llm_mean_pairs']:.0f}")
            lines.append(f"    Mean code blocks: {r['pre_llm_mean_blocks']:.0f}")
        
        if r.get('post_llm_mean_ratio'):
            lines.append(f"  Post-LLM period (2022-2025):")
            lines.append(f"    Mean duplication ratio: {r['post_llm_mean_ratio']*100:.2f}%")
            lines.append(f"    Median duplication ratio: {r['post_llm_median_ratio']*100:.2f}%")
            lines.append(f"    Std deviation: {r['post_llm_std_ratio']*100:.2f}%")
            lines.append(f"    Mean clone pairs: {r['post_llm_mean_pairs']:.0f}")
            lines.append(f"    Mean code blocks: {r['post_llm_mean_blocks']:.0f}")
        
        if r.get('ratio_change_pct'):
            lines.append(f"  Change: {r['ratio_change_pct']:+.2f}% ({r['direction']})")
    
    return "\n".join(lines)

def export_csv(results: list, output_path: str):
    """Export results to CSV for further analysis."""
    fieldnames = [
        'repo_name', 'date_range', 'total_snapshots',
        'pre_llm_snapshots', 'post_llm_snapshots',
        'pre_llm_mean_ratio', 'pre_llm_median_ratio', 'pre_llm_std_ratio',
        'post_llm_mean_ratio', 'post_llm_median_ratio', 'post_llm_std_ratio',
        'ratio_change', 'ratio_change_pct', 'direction',
        'pre_llm_mean_pairs', 'post_llm_mean_pairs',
        'pre_llm_mean_blocks', 'post_llm_mean_blocks'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    print(f"CSV exported to {output_path}")

def main():
    results_dir = Path('results')
    
    # Look for JSON history files
    history_files = list(results_dir.glob('*_history.json'))
    
    if not history_files:
        print("ERROR: No *_history.json files found in results/")
        print("\nTo generate the JSON data, run:")
        print("  python -m cluain batch repositories.csv --output-dir ./results --years 6")
        print("\nOr for faster analysis (incremental mode):")
        print("  python -m cluain batch repositories.csv --output-dir ./results --years 6 --fast")
        return
    
    print(f"Found {len(history_files)} history files")
    
    results = []
    for json_path in sorted(history_files):
        print(f"Processing {json_path.name}...")
        history = load_history(json_path)
        analysis = analyse_repository(history)
        if analysis:
            results.append(analysis)
    
    # Generate outputs
    print(generate_summary_table(results))
    print(generate_detailed_stats(results))
    
    # Export CSV
    export_csv(results, 'results/analysis_summary.csv')
    
    # Save full analysis as JSON
    with open('results/analysis_full.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nFull analysis saved to results/analysis_full.json")

if __name__ == '__main__':
    main()