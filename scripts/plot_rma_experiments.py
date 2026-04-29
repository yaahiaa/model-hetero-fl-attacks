import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ['base', 'committee', 'ldp', 'ddp']
METHOD_COLORS = {
    'base': '#1f77b4',
    'committee': '#d62728',
    'ldp': '#2ca02c',
    'ddp': '#ff7f0e',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Plot aggregated RMA experiment results')
    parser.add_argument('--results_dir', default='results', type=str)
    parser.add_argument('--out_dir', default='results/plots', type=str)
    return parser.parse_args()


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def read_csv_or_empty(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def ordered_methods(values):
    values = [str(v).lower() for v in values if pd.notna(v)]
    present = []
    for method in METHOD_ORDER:
        if method in values and method not in present:
            present.append(method)
    for method in sorted(set(values)):
        if method not in present:
            present.append(method)
    return present


def coerce_numeric(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def ci95(series):
    clean = pd.to_numeric(series, errors='coerce').dropna()
    if len(clean) <= 1:
        return 0.0
    return float(1.96 * clean.std(ddof=1) / math.sqrt(len(clean)))


def mean_or_zero(series):
    clean = pd.to_numeric(series, errors='coerce').dropna()
    if len(clean) == 0:
        return 0.0
    return float(clean.mean())


def std_or_zero(series):
    clean = pd.to_numeric(series, errors='coerce').dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1))


def save_figure(fig, out_dir, basename):
    ensure_dir(out_dir)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{basename}.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{basename}.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_no_data(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.2)


def add_noise_value(df):
    if df.empty:
        return df
    if 'attack_noise_amount' not in df.columns and 'noise_scale' not in df.columns:
        df['attack_noise_value'] = pd.Series(dtype=float)
        return df
    attack_noise = pd.to_numeric(df.get('attack_noise_amount'), errors='coerce')
    noise_scale = pd.to_numeric(df.get('noise_scale'), errors='coerce')
    if isinstance(attack_noise, pd.Series) and isinstance(noise_scale, pd.Series):
        df['attack_noise_value'] = attack_noise.fillna(noise_scale)
    elif isinstance(attack_noise, pd.Series):
        df['attack_noise_value'] = attack_noise
    else:
        df['attack_noise_value'] = noise_scale
    return df


def aggregate_dataset_size(leakage_df):
    if leakage_df.empty:
        return pd.DataFrame(columns=[
            'experiment_method', 'local_train_size', 'best_pearson_max',
            'avg_pearson_mean', 'avg_pearson_ci95', 'num_runs',
        ])
    leakage_df = coerce_numeric(leakage_df.copy(), ['local_train_size', 'best_pearson', 'avg_pearson'])
    grouped = leakage_df.dropna(subset=['local_train_size']).groupby(
        ['experiment_method', 'local_train_size'], dropna=False
    )
    return grouped.agg(
        best_pearson_max=('best_pearson', 'max'),
        avg_pearson_mean=('avg_pearson', 'mean'),
        num_runs=('avg_pearson', 'count'),
    ).reset_index().assign(
        avg_pearson_ci95=lambda df: [
            ci95(grouped.get_group((row['experiment_method'], row['local_train_size']))['avg_pearson'])
            for _, row in df.iterrows()
        ]
    )


def aggregate_noise(leakage_df):
    if leakage_df.empty:
        return pd.DataFrame(columns=[
            'experiment_method', 'attack_noise_value', 'best_pearson_max',
            'avg_pearson_mean', 'avg_pearson_ci95', 'num_recovered_mean',
            'num_recovered_ci95', 'num_runs',
        ])
    leakage_df = add_noise_value(leakage_df.copy())
    leakage_df = coerce_numeric(
        leakage_df,
        ['attack_noise_value', 'best_pearson', 'avg_pearson', 'num_recovered'],
    )
    grouped = leakage_df.dropna(subset=['attack_noise_value']).groupby(
        ['experiment_method', 'attack_noise_value'], dropna=False
    )
    return grouped.agg(
        best_pearson_max=('best_pearson', 'max'),
        avg_pearson_mean=('avg_pearson', 'mean'),
        num_recovered_mean=('num_recovered', 'mean'),
        num_runs=('avg_pearson', 'count'),
    ).reset_index().assign(
        avg_pearson_ci95=lambda df: [
            ci95(grouped.get_group((row['experiment_method'], row['attack_noise_value']))['avg_pearson'])
            for _, row in df.iterrows()
        ],
        num_recovered_ci95=lambda df: [
            ci95(grouped.get_group((row['experiment_method'], row['attack_noise_value']))['num_recovered'])
            for _, row in df.iterrows()
        ],
    )


def aggregate_convergence(epoch_df):
    if epoch_df.empty:
        return pd.DataFrame(columns=[
            'experiment_method', 'epoch', 'global_accuracy_mean',
            'global_accuracy_ci95', 'global_accuracy_std', 'num_runs',
        ])
    epoch_df = coerce_numeric(epoch_df.copy(), ['epoch', 'global_accuracy'])
    grouped = epoch_df.dropna(subset=['epoch']).groupby(['experiment_method', 'epoch'], dropna=False)
    return grouped.agg(
        global_accuracy_mean=('global_accuracy', 'mean'),
        num_runs=('global_accuracy', 'count'),
    ).reset_index().assign(
        global_accuracy_ci95=lambda df: [
            ci95(grouped.get_group((row['experiment_method'], row['epoch']))['global_accuracy'])
            for _, row in df.iterrows()
        ],
        global_accuracy_std=lambda df: [
            std_or_zero(grouped.get_group((row['experiment_method'], row['epoch']))['global_accuracy'])
            for _, row in df.iterrows()
        ],
    )


def aggregate_overhead(overhead_df):
    if overhead_df.empty:
        return pd.DataFrame(columns=[
            'experiment_method', 'mean_epoch_time_sec', 'mean_train_time_sec',
            'mean_aggregation_time_sec', 'mean_committee_time_sec',
            'mean_dp_time_sec', 'mean_test_time_sec', 'num_runs',
        ])
    overhead_df = coerce_numeric(
        overhead_df.copy(),
        [
            'mean_epoch_time_sec', 'mean_train_time_sec', 'mean_aggregation_time_sec',
            'mean_committee_time_sec', 'mean_dp_time_sec', 'mean_test_time_sec',
        ],
    )
    return overhead_df.groupby('experiment_method', dropna=False).agg(
        mean_epoch_time_sec=('mean_epoch_time_sec', 'mean'),
        mean_train_time_sec=('mean_train_time_sec', 'mean'),
        mean_aggregation_time_sec=('mean_aggregation_time_sec', 'mean'),
        mean_committee_time_sec=('mean_committee_time_sec', 'mean'),
        mean_dp_time_sec=('mean_dp_time_sec', 'mean'),
        mean_test_time_sec=('mean_test_time_sec', 'mean'),
        num_runs=('mean_epoch_time_sec', 'count'),
    ).reset_index()


def plot_best_pearson_vs_n(dataset_size_agg, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if dataset_size_agg.empty:
        plot_no_data(ax, 'Best Pearson vs Local Train Size', 'local_train_size', 'best_pearson')
        save_figure(fig, out_dir, 'fig1_best_pearson_vs_N')
        return
    methods = ordered_methods(dataset_size_agg['experiment_method'].tolist())
    for method in methods:
        method_df = dataset_size_agg[dataset_size_agg['experiment_method'] == method].sort_values('local_train_size')
        ax.plot(
            method_df['local_train_size'],
            method_df['best_pearson_max'],
            marker='o',
            label=method,
            color=METHOD_COLORS.get(method),
        )
    ax.set_title('Best Pearson vs Local Train Size')
    ax.set_xlabel('local_train_size')
    ax.set_ylabel('best_pearson')
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir, 'fig1_best_pearson_vs_N')


def plot_avg_pearson_vs_n(dataset_size_agg, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if dataset_size_agg.empty:
        plot_no_data(ax, 'Average Pearson vs Local Train Size', 'local_train_size', 'avg_pearson')
        save_figure(fig, out_dir, 'fig2_avg_pearson_vs_N')
        return
    methods = ordered_methods(dataset_size_agg['experiment_method'].tolist())
    for method in methods:
        method_df = dataset_size_agg[dataset_size_agg['experiment_method'] == method].sort_values('local_train_size')
        x = method_df['local_train_size']
        y = method_df['avg_pearson_mean']
        ci = method_df['avg_pearson_ci95']
        ax.plot(x, y, marker='o', label=method, color=METHOD_COLORS.get(method))
        ax.fill_between(x, y - ci, y + ci, alpha=0.18, color=METHOD_COLORS.get(method))
    ax.set_title('Average Pearson vs Local Train Size')
    ax.set_xlabel('local_train_size')
    ax.set_ylabel('avg_pearson')
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir, 'fig2_avg_pearson_vs_N')


def plot_recovered_vs_noise(noise_agg, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if noise_agg.empty:
        plot_no_data(ax, 'Recovered Images vs Attack Noise', 'attack_noise', 'num_recovered')
        save_figure(fig, out_dir, 'fig3_recovered_vs_attack_noise')
        return
    methods = ordered_methods(noise_agg['experiment_method'].tolist())
    for method in methods:
        method_df = noise_agg[noise_agg['experiment_method'] == method].sort_values('attack_noise_value')
        x = method_df['attack_noise_value']
        y = method_df['num_recovered_mean']
        ci = method_df['num_recovered_ci95']
        ax.plot(x, y, marker='o', label=method, color=METHOD_COLORS.get(method))
        ax.fill_between(x, y - ci, y + ci, alpha=0.18, color=METHOD_COLORS.get(method))
    ax.set_title('Recovered Images vs Attack Noise')
    ax.set_xlabel('attack_noise')
    ax.set_ylabel('num_recovered')
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir, 'fig3_recovered_vs_attack_noise')


def plot_pearson_vs_noise(noise_agg, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if noise_agg.empty:
        plot_no_data(axes[0], 'Best Pearson vs Attack Noise', 'attack_noise', 'best_pearson')
        plot_no_data(axes[1], 'Average Pearson vs Attack Noise', 'attack_noise', 'avg_pearson')
        save_figure(fig, out_dir, 'fig4_pearson_vs_attack_noise')
        return
    methods = ordered_methods(noise_agg['experiment_method'].tolist())
    for method in methods:
        method_df = noise_agg[noise_agg['experiment_method'] == method].sort_values('attack_noise_value')
        x = method_df['attack_noise_value']
        best_y = method_df['best_pearson_max']
        avg_y = method_df['avg_pearson_mean']
        avg_ci = method_df['avg_pearson_ci95']
        axes[0].plot(x, best_y, marker='o', label=method, color=METHOD_COLORS.get(method))
        axes[1].plot(x, avg_y, marker='o', label=method, color=METHOD_COLORS.get(method))
        axes[1].fill_between(x, avg_y - avg_ci, avg_y + avg_ci, alpha=0.18, color=METHOD_COLORS.get(method))
    axes[0].set_title('Best Pearson vs Attack Noise')
    axes[1].set_title('Average Pearson vs Attack Noise')
    for ax, ylabel in zip(axes, ['best_pearson', 'avg_pearson']):
        ax.set_xlabel('attack_noise')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    save_figure(fig, out_dir, 'fig4_pearson_vs_attack_noise')


def plot_convergence(convergence_agg, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if convergence_agg.empty:
        plot_no_data(ax, 'Convergence Accuracy vs Epoch', 'epoch', 'global_accuracy')
        save_figure(fig, out_dir, 'fig5_convergence_accuracy_vs_epoch')
        return
    methods = ordered_methods(convergence_agg['experiment_method'].tolist())
    for method in methods:
        method_df = convergence_agg[convergence_agg['experiment_method'] == method].sort_values('epoch')
        x = method_df['epoch']
        y = method_df['global_accuracy_mean']
        spread = method_df['global_accuracy_ci95']
        ax.plot(x, y, marker='o', label=method, color=METHOD_COLORS.get(method))
        ax.fill_between(x, y - spread, y + spread, alpha=0.18, color=METHOD_COLORS.get(method))
    ax.set_title('Convergence Accuracy vs Epoch')
    ax.set_xlabel('epoch')
    ax.set_ylabel('global_accuracy')
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir, 'fig5_convergence_accuracy_vs_epoch')


def plot_overhead(overhead_agg, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if overhead_agg.empty:
        plot_no_data(ax, 'Overhead Per Epoch', 'experiment_method', 'seconds')
        save_figure(fig, out_dir, 'fig6_overhead_seconds_per_epoch')
        return
    methods = ordered_methods(overhead_agg['experiment_method'].tolist())
    plot_df = overhead_agg.set_index('experiment_method').reindex(methods).fillna(0.0)
    x = list(range(len(plot_df.index)))
    labels = list(plot_df.index)
    component_columns = [
        ('mean_train_time_sec', 'train'),
        ('mean_aggregation_time_sec', 'aggregation'),
        ('mean_committee_time_sec', 'committee'),
        ('mean_dp_time_sec', 'dp'),
        ('mean_test_time_sec', 'test'),
    ]
    bottom = [0.0] * len(labels)
    plotted_any_component = False
    for column, label in component_columns:
        if column not in plot_df.columns:
            continue
        values = plot_df[column].tolist()
        if max(values) <= 0:
            continue
        ax.bar(x, values, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, values)]
        plotted_any_component = True
    if not plotted_any_component:
        totals = plot_df['mean_epoch_time_sec'].tolist()
        ax.bar(x, totals, color='#4c78a8', label='epoch')
    else:
        totals = plot_df['mean_epoch_time_sec'].tolist()
        for idx, total in enumerate(totals):
            ax.text(idx, total, f'{total:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_title('Mean Overhead Per Epoch')
    ax.set_xlabel('experiment_method')
    ax.set_ylabel('seconds')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir, 'fig6_overhead_seconds_per_epoch')


def main():
    args = parse_args()
    ensure_dir(args.results_dir)
    ensure_dir(args.out_dir)

    leakage_df = read_csv_or_empty(os.path.join(args.results_dir, 'leakage_raw.csv'))
    epoch_df = read_csv_or_empty(os.path.join(args.results_dir, 'epoch_raw.csv'))
    overhead_df = read_csv_or_empty(os.path.join(args.results_dir, 'overhead_raw.csv'))

    dataset_size_agg = aggregate_dataset_size(leakage_df)
    noise_agg = aggregate_noise(leakage_df)
    convergence_agg = aggregate_convergence(epoch_df)
    overhead_agg = aggregate_overhead(overhead_df)

    dataset_size_agg.to_csv(os.path.join(args.results_dir, 'dataset_size_agg.csv'), index=False)
    noise_agg.to_csv(os.path.join(args.results_dir, 'noise_agg.csv'), index=False)
    convergence_agg.to_csv(os.path.join(args.results_dir, 'convergence_agg.csv'), index=False)
    overhead_agg.to_csv(os.path.join(args.results_dir, 'overhead_agg.csv'), index=False)

    plot_best_pearson_vs_n(dataset_size_agg, args.out_dir)
    plot_avg_pearson_vs_n(dataset_size_agg, args.out_dir)
    plot_recovered_vs_noise(noise_agg, args.out_dir)
    plot_pearson_vs_noise(noise_agg, args.out_dir)
    plot_convergence(convergence_agg, args.out_dir)
    plot_overhead(overhead_agg, args.out_dir)


if __name__ == '__main__':
    main()
