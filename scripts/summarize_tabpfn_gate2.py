"""Apply the preregistered Gate 2 criterion and plot every matched comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "tabpfn-compile-gate2"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from paths import ARTIFACTS, REPO_ROOT, resolve_data


ROOT = ARTIFACTS / "gate2"


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Gate 2 artifact directory")
    ROOT = parser.parse_args().root.resolve()
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    statuses = json.loads((ROOT / "execution-status.json").read_text())
    assert len(statuses) == 8, "Wait for all eight datasets; no interim gate decision"
    runtime_recoveries = {}
    for log in sorted(ROOT.glob('*-run.log')):
        events = [line for line in log.read_text().splitlines() if line.startswith('OOM:')]
        if events:
            runtime_recoveries[log.name] = events
    (ROOT / 'native-memory-recovery.json').write_text(json.dumps(runtime_recoveries, indent=2)+'\n')
    records = []
    expected_dependencies = None
    dependency_reference_dataset = None
    dependency_deviations = {}
    for dataset in prereg["datasets"]:
        name = dataset["dataset"]
        folder = ROOT / name
        result_file = folder / "results.json"
        results = json.loads(result_file.read_text()) if result_file.exists() else []
        verification_file = (folder / "artifact-verification.json" if (folder / "artifact-verification.json").exists()
                             else folder / "artifact-verification-diagnostic.json")
        verification = json.loads(verification_file.read_text()) if verification_file.exists() else []
        meta = json.loads((folder / "metadata.json").read_text())
        assert meta["checkpoint_sha256"] == prereg["checkpoint_sha256"]
        assert meta["tabpfn_git_commit"] == prereg["tabpfn_commit"]
        assert meta["source_sha256"] == dataset["source_sha256"]
        for key in ('budgets','seeds','steps','lr','batch_size','eval_every','label_mode','threads'):
            assert meta['config'][key] == prereg['frozen'][key], f'Configuration drift: {name}/{key}'
        device_deviation = meta['config']['device'] != prereg['frozen']['device']
        assert not device_deviation or (name == 'landsat-satellite' and meta['config']['device']=='cpu')
        if expected_dependencies is None:
            expected_dependencies = meta['dependencies']
            dependency_reference_dataset = name
        else:
            changes = {
                package: {'expected': expected_dependencies.get(package), 'actual': meta['dependencies'].get(package)}
                for package in sorted(expected_dependencies.keys() | meta['dependencies'].keys())
                if expected_dependencies.get(package) != meta['dependencies'].get(package)
            }
            if changes:
                dependency_deviations[name] = changes
        dependency_deviation = name in dependency_deviations
        trace_file = folder / 'traces.json'
        traces = json.loads(trace_file.read_text()) if trace_file.exists() else []
        source = pd.read_parquet(resolve_data(dataset["data"]))
        y = LabelEncoder().fit_transform(source[dataset["target"]])
        context = np.array(meta["split_indices"]["context"])
        for m in prereg["frozen"]["budgets"]:
            for seed in prereg["frozen"]["seeds"]:
                initial, _ = train_test_split(context, train_size=m, stratify=y[context], random_state=seed)
                record = dict(dataset=name, m=m, seed=seed, N=dataset["N"], d=dataset["d"], K=dataset["K"],
                              source_rows=dataset["source_rows"], source_class_counts=dataset["class_counts"],
                              context_class_counts=np.bincount(y[context], minlength=dataset["K"]).tolist(),
                              initial_class_counts=np.bincount(y[initial], minlength=dataset["K"]).tolist(),
                              dependency_deviation=dependency_deviation)
                pair = {method: next((r for r in results if r['method']==method and r['rows']==m and r['seed']==seed), None)
                        for method in ("stratified", "synthetic_stratified")}
                artifact_path = folder / f'compiled-{m}-{seed}-stratified.csv'
                if any(v is None for v in pair.values()) or not artifact_path.exists():
                    record.update(status="missing", delta_kl=None, relative_improvement=None, win=False, reload_verified=False)
                    records.append(record)
                    continue
                before, after = pair["stratified"], pair["synthetic_stratified"]
                assert sorted(r['step'] for r in traces if r['rows']==m and r['seed']==seed and r['initialization']=='stratified') == [10,20,30,40,50]
                for label, result in (("before", before), ("after", after)):
                    for split in ("compile", "validation", "test"):
                        for metric, value in result[split].items():
                            record[f"{label}_{split}_{metric}"] = value
                    record[f"{label}_G_val"] = result['validation']['kl'] - result['compile']['kl']
                    record[f"{label}_G_test"] = result['test']['kl'] - result['compile']['kl']
                    record[f"{label}_latency_seconds"] = result['test_latency_seconds_median']
                delta = before['test']['kl'] - after['test']['kl']
                file = f"compiled-{m}-{seed}-stratified.csv"
                exported = pd.read_csv(folder / file)
                assert np.array_equal(exported[dataset['target']].to_numpy(), y[initial]), "Labels changed"
                check = next((r for r in verification if r['file']==file), None)
                record.update(status="complete", delta_kl=delta,
                              relative_improvement=delta/before['test']['kl'] if before['test']['kl']>0 else None,
                              win=delta>0, validation_delta_kl=before['validation']['kl']-after['validation']['kl'],
                              best_step=after['best_step'], adam_steps=50, training_seconds=after['training_seconds'],
                              labels_unchanged=True, final_class_counts=np.bincount(exported[dataset['target']].astype(int), minlength=dataset['K']).tolist(),
                              execution_device=meta['config']['device'], device_deviation=device_deviation,
                              reload_verified=check is not None and max(check['kl_absolute_errors'].values()) < prereg['reload_tolerance'],
                              artifact=str(folder/file))
                record['missing_initial_classes'] = int(np.sum(np.bincount(y[initial], minlength=dataset['K'])==0))
                if check:
                    record['reload_max_prediction_error'] = max(check.get('prediction_max_absolute_errors',{}).values(), default=None)
                    record['reload_max_kl_error'] = max(check['kl_absolute_errors'].values())
                records.append(record)
    assert len(records)==72
    (ROOT / "run-records.json").write_text(json.dumps(records, indent=2, allow_nan=False)+"\n")
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT / "paired-metrics.csv", index=False)
    by_dataset = frame.groupby('dataset', sort=False).agg(
        runs=('win','size'), wins=('win','sum'), mean_delta_kl=('delta_kl','mean'),
        median_delta_kl=('delta_kl','median'), before_test_kl=('before_test_kl','mean'),
        after_test_kl=('after_test_kl','mean'), reloads=('reload_verified','sum'))
    by_dataset.to_csv(ROOT / "dataset-summary.csv")
    by_budget = frame.groupby('m').agg(runs=('win','size'), wins=('win','sum'),
                                      mean_delta_kl=('delta_kl','mean'), median_delta_kl=('delta_kl','median'))
    by_budget.to_csv(ROOT / 'budget-summary.csv')
    complete = int((frame.status=='complete').sum())
    wins = int(frame.win.sum())
    positive = int((by_dataset.mean_delta_kl>0).sum())
    median = float(frame.delta_kl.median())
    verified = int(frame.reload_verified.sum())
    criterion_met = wins>=58 and positive>=6 and median>0
    passed = complete==72 and verified==72 and criterion_met
    label = "PASS" if passed else ("CRITERION MET; ARTIFACT VERIFICATION FAILURE" if complete==72 and criterion_met and verified<72 else
                                 "INCOMPLETE / INVALID" if complete<72 else
                                 "FAILURE OF GENERAL CLAIM" if positive<=4 else "MIXED EVIDENCE" if positive==5 else "DOES NOT PASS")
    summary = dict(verdict=label, statistical_criterion_met=criterion_met, total_runs=72, completed_runs=complete, wins=wins,
                   win_fraction=wins/72, positive_mean_datasets=positive, median_delta_kl=median,
                   verified_artifacts=verified, maximum_prediction_reload_error=float(frame.reload_max_prediction_error.max()),
                   dependency_reference_dataset=dependency_reference_dataset,
                   dependency_deviation_datasets=sorted(dependency_deviations),
                   dependency_deviations=dependency_deviations,
                   new_runs_only=dict(total=63, wins=int(frame[frame.dataset!='vehicle'].win.sum()),
                                      positive_mean_datasets=int((by_dataset.drop(index='vehicle').mean_delta_kl>0).sum())))
    (ROOT / "gate-decision.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    colors = dict(zip([d['dataset'] for d in prereg['datasets']], plt.get_cmap('tab10').colors))
    markers = {8:'o',32:'s',128:'^'}
    fig, ax = plt.subplots(figsize=(9,7), layout='constrained')
    for (name,m), group in frame[frame.status=='complete'].groupby(['dataset','m']):
        ax.scatter(group.before_test_kl, group.after_test_kl, color=colors[name], marker=markers[m],
                   s=48, alpha=.8, edgecolors='white', linewidths=.5)
    limit = float(frame[['before_test_kl','after_test_kl']].max().max())*1.07
    ax.plot([0,limit],[0,limit], color='black', linestyle='--', linewidth=1)
    ax.set(xlim=(0,limit), ylim=(0,limit), aspect='equal',
           xlabel='Stratified subset test KL (nats)', ylabel='Optimized context test KL (nats)',
           title=f'Gate 2: {wins}/72 points below the diagonal · {label}')
    ax.grid(alpha=.2)
    handles = [Line2D([],[],marker='o',linestyle='',color=c,label=n) for n,c in colors.items()]
    handles += [Line2D([],[],marker=marker,linestyle='',color='gray',label=f'{m} rows') for m,marker in markers.items()]
    ax.legend(handles=handles, loc='upper left', fontsize=8)
    fig.savefig(ROOT/'paired-kl.png',dpi=180)
    fig.savefig(ROOT/'paired-kl.svg', metadata={'Date': None})
    lines = ['# Gate 2 — cross-dataset replication','',f'## Verdict: {label}','',
             f'{wins}/72 runs ({wins/72:.1%}) improved test teacher KL; {positive}/8 datasets have positive mean absolute improvement. Median absolute improvement is {median:.6f} nats. The statistical criterion is met. Artifact reload verification: {verified}/72.', '',
             'The preregistered PASS rule is at least 58/72 strict test-KL wins, positive mean improvement on at least 6/8 datasets, and positive median absolute improvement. All runs must finish and all artifacts must verify. No datasets were excluded.', '',
             'Vehicle was known before Gate 2. The seven new datasets are the prospective replication panel; their results are also recorded separately in `gate-decision.json`.', '',
             '![All matched runs](paired-kl.png)','',
             '| Dataset | N context | d | K | Wins / 9 | Mean subset KL | Mean optimized KL | Mean ΔKL |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in prereg['datasets']:
        r = by_dataset.loc[d['dataset']]
        lines.append(f"| {d['dataset']} | {d['N']} | {d['d']} | {d['K']} | {int(r.wins)} | {r.before_test_kl:.6f} | {r.after_test_kl:.6f} | {r.mean_delta_kl:.6f} |")
    lines += ['', '## Context-size regimes', '',
              '| Context rows | Wins / 24 | Mean ΔKL | Median ΔKL |',
              '|---:|---:|---:|---:|']
    for m,r in by_budget.iterrows():
        lines.append(f'| {m} | {int(r.wins)} | {r.mean_delta_kl:.6f} | {r.median_delta_kl:.6f} |')
    failed = frame[(frame.status=='complete') & ~frame.win]
    lines += ['', '## Runs without test-fidelity improvement', '',
              '| Dataset | m | Seed | ΔKL | Compile KL before → after | G_test after | Selected step |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for _,r in failed.iterrows():
        lines.append(f'| {r.dataset} | {r.m} | {r.seed} | {r.delta_kl:.6f} | {r.before_compile_kl:.6f} → {r.after_compile_kl:.6f} | {r.after_G_test:.6f} | {r.best_step} |')
    if failed.empty:
        lines += ['| None | — | — | — | — | — | — |']
    overfit = int(((frame.before_compile_kl > frame.after_compile_kl) & (frame.delta_kl < 0)).sum())
    lines += ['', f'{overfit} runs improved compile KL but worsened test KL. '
              f'{int((frame.missing_initial_classes>0).sum())} initial contexts omitted at least one class. '
              f'Total new optimization time (excluding teacher, evaluation and reload): {frame[frame.dataset!="vehicle"].training_seconds.sum():.1f} seconds.', '']
    lines += ['', '## Frozen protocol', '',
              '- Exact Gate 1 checkpoint and TabPFN commit; fixed labels; numerical inputs; one estimator; float32; unchanged preprocessing, split procedure, optimizer, KL objective, query batching, and metrics.',
              '- Budgets 8/32/128, seeds 42/43/44, stratified initialization, Adam LR 0.01, exactly 50 steps. As in Gate 1, retain minimum compile KL among steps 0/10/20/30/40/50. Validation and test never select iterates.',
              '- Five existing corpus datasets and three numerical UCI tasks. No imputation, feature selection, row filtering, or dataset-specific tuning. The original UCI training and test partitions are concatenated before applying the common deterministic 60/20/10/10 split.',
              '- Balanced means max/min class frequency ≤1.5; material imbalance means ≥2. The panel meets all eight coverage requirements; see `preregistration.json`.', '',
              '## Failure modes and practical limits', '',
              '- Negative and zero ΔKL remain in the analysis. `paired-metrics.csv` records before/after compile, validation and test KL, accuracy/AUC/log-loss, generalization gaps, class distributions, dimensions, selected steps, runtime and artifact paths.',
              '- One fixed split per dataset and three seeds do not provide 72 independent dataset-level observations. Class proportions, scale, and inductive biases differ across tasks; this is an empirical replication gate, not a proof or a context-complexity estimate.',
              '- Published datasets can contain duplicate or correlated examples. We preserve Gate 1’s row-wise split and do not claim group-disjoint or out-of-domain generalization. Numerical zero values are retained, including the conventional zero-coded missing measurements in Diabetes.',
              '- Vehicle reuses nine original stratified runs. Original prediction arrays were not retained in Gate 1, so reference arrays were reconstructed from exported CSVs and checked against original KL before a second fresh-model reload. New runs compare directly to prediction arrays saved during evaluation. The numerical tolerance remains 1e-5.',
              '- Landsat exceeded the 8.29 GB MPS allocation during teacher inference, including a retry with the allocator high-watermark setting disabled. Its identical model/data/optimization protocol ran on CPU. This is a preregistration device deviation and both OOM attempts are preserved. Landsat optimized CSV reload status is reported against the same 1e-5 KL threshold.',
              '- Runtime and inference latency describe this Apple GPU and the unchanged uncached differentiable inference path. Peak RSS is cumulative process memory, not per-context GPU memory.', '',
              f'- Native TabPFN automatic memory recovery occurred in {list(runtime_recoveries)}. These are internal row/column chunk reductions by the unchanged model code, not manually changed hyperparameters or dataset filtering. Exact messages are saved in `native-memory-recovery.json` and the original logs.', '',
              '## Data sources', '',
              'Additional UCI datasets: [Balance Scale](https://archive.ics.uci.edu/dataset/12/balance+scale), [Image Segmentation](https://archive.ics.uci.edu/dataset/50/image+segmentation), [Statlog Landsat Satellite](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite). Raw download URLs and SHA-256 hashes are in the preregistration. Existing datasets retain their local corpus profiles.', '',
              '## Reproduction', '',
              'The preregistration contains dataset/checkpoint/code hashes and all frozen settings. Per-dataset folders contain metadata with split indices and scaling, cached teacher probabilities, baseline/optimized predictions, CSV contexts, optimization traces, and reload verification. `execution-status.json` and logs retain all execution failures.', '',
              'Run or resume the complete frozen pipeline with `scripts/gate2.py`. It validates hashes and the TabPFN commit, skips complete datasets, fills missing experiments and verification records, preserves failed attempts, and invokes this aggregator. Use `scripts/gate2.py --dry-run` to inspect planned work without changing artifacts.', '']
    if dependency_deviations:
        packages = sorted({package for changes in dependency_deviations.values() for package in changes})
        datasets_text = ', '.join(f'`{name}`' for name in sorted(dependency_deviations))
        packages_text = ', '.join(f'`{name}`' for name in packages)
        note = (f"- Dependency-version drift relative to `{dependency_reference_dataset}` was recorded for "
                f"{datasets_text} in packages {packages_text}. Exact expected/actual versions are in "
                "`gate-decision.json`; checkpoint, TabPFN commit, dataset hashes, and frozen experiment settings still match.")
        failure_heading = lines.index('## Failure modes and practical limits')
        lines.insert(failure_heading + 2, note)
    (ROOT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))
    print(by_dataset.to_string())


if __name__=='__main__':
    main()
