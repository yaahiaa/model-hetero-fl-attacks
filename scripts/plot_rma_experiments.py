import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHOD_ORDER = ['base', 'committee', 'ldp', 'ddp']


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def load_csv_or_empty(path, columns=None):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns or [])


def method_sort_key(method):
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method)
    return len(METHOD_ORDER)


def ordered_methods(methods):
    return sorted([m for m in methods if pd.notna(m)], key=method_sort_key)


def save_figure(fig, out_dir, stem):
    ensure_dir(out_dir)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '{}.png'.format(stem)), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, '{}.pdf'.format(stem)), bbox_inches='tight')
    plt.close(fig)


def ci95(series):
    clean = pd.to_numeric(series, errors='coerce').dropna()
    count = int(clean.count())
    if count <= 1:
        return 0.0
    return float(1.96 * clean.std(ddof=1) / (count ** 0.5))


def plot_empty(ax, title):
    ax.set_title(title)
    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    ax.set_axis_off()


def prepare_leakage(leakage_df):
    df = leakage_df.copy()
    if df.empty:
        return df
    for col in ['local_train_size', 'attack_noise_amount', 'noise_scale', 'best_pearson', 'avg_pearson', 'best_psnr', 'avg_psnr', 'num_recovered']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'attack_noise_amount' not in df.columns:
        df['attack_noise_amount'] = pd.NA
    if 'noise_scale' not in df.columns:
        df['noise_scale'] = pd.NA
    df['attack_noise'] = df['attack_noise_amount'].fillna(df['noise_scale'])
    return df


def plot_best_pearson_vs_n(leakage_df, out_dir):
    cols = ['experiment_method', 'local_train_size', 'seed', 'best_pearson']
    data = leakage_df.reindex(columns=cols).dropna(subset=['experiment_method', 'local_train_size', 'best_pearson']) if not leakage_df.empty else pd.DataFrame(columns=cols)
    if data.empty:
        agg = pd.DataFrame(columns=['experiment_method', 'local_train_size', 'best_pearson'])
    else:
        per_seed = data.groupby(['experiment_method', 'local_train_size', 'seed'], as_index=False)['best_pearson'].max()
        agg = per_seed.groupby(['experiment_method', 'local_train_size'], as_index=False)['best_pearson'].max()
        agg = agg.sort_values(['experiment_method', 'local_train_size'])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if agg.empty:
        plot_empty(ax, 'Best Pearson vs Local Train Size')
    else:
        for method in ordered_methods(agg['experiment_method'].unique()):
            subset = agg[agg['experiment_method'] == method]
            ax.plot(subset['local_train_size'], subset['best_pearson'], marker='o', label=method)
        ax.set_title('Best Pearson vs Local Train Size')
        ax.set_xlabel('local_train_size')
        ax.set_ylabel('best_pearson')
        ax.legend()
        ax.grid(True, alpha=0.3)
    save_figure(fig, out_dir, 'fig1_best_pearson_vs_N')
    return agg


def plot_avg_pearson_vs_n(leakage_df, out_dir):
    cols = ['experiment_method', 'local_train_size', 'seed', 'avg_pearson']
    data = leakage_df.reindex(columns=cols).dropna(subset=['experiment_method', 'local_train_size', 'avg_pearson']) if not leakage_df.empty else pd.DataFrame(columns=cols)
    if data.empty:
        agg = pd.DataFrame(columns=['experiment_method', 'local_train_size', 'avg_pearson_mean', 'avg_pearson_ci95', 'count'])
    else:
        per_seed = data.groupby(['experiment_method', 'local_train_size', 'seed'], as_index=False)['avg_pearson'].mean()
        agg = per_seed.groupby(['experiment_method', 'local_train_size']).agg(
            avg_pearson_mean=('avg_pearson', 'mean'),
            avg_pearson_ci95=('avg_pearson', ci95),
            count=('avg_pearson', 'count'),
        ).reset_index()
        agg = agg.sort_values(['experiment_method', 'local_train_size'])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if agg.empty:
        plot_empty(ax, 'Average Pearson vs Local Train Size')
    else:
        for method in ordered_methods(agg['experiment_method'].unique()):
            subset = agg[agg['experiment_method'] == method]
            ax.plot(subset['local_train_size'], subset['avg_pearson_mean'], marker='o', label=method)
            ax.fill_between(
                subset['local_train_size'],
                subset['avg_pearson_mean'] - subset['avg_pearson_ci95'],
                subset['avg_pearson_mean'] + subset['avg_pearson_ci95'],
                alpha=0.2,
            )
        ax.set_title('Average Pearson vs Local Train Size')
        ax.set_xlabel('local_train_size')
        ax.set_ylabel('avg_pearson')
        ax.legend()
        ax.grid(True, alpha=0.3)
    save_figure(fig, out_dir, 'fig2_avg_pearson_vs_N')
    return agg


def plot_noise_curves(leakage_df, out_dir):
    cols = ['experiment_method', 'attack_noise', 'seed', 'num_recovered', 'best_pearson', 'avg_pearson']
    data = leakage_df.reindex(columns=cols).dropna(subset=['experiment_method', 'attack_noise']) if not leakage_df.empty else pd.DataFrame(columns=cols)
    if data.empty:
        agg = pd.DataFrame(columns=[
            'experiment_method', 'attack_noise', 'num_recovered_mean', 'num_recovered_ci95',
            'best_pearson_max', 'avg_pearson_mean', 'avg_pearson_ci95',
        ])
    else:
        per_seed = data.groupby(['experiment_method', 'attack_noise', 'seed'], as_index=False).agg(
            num_recovered=('num_recovered', 'mean'),
            best_pearson=('best_pearson', 'max'),
            avg_pearson=('avg_pearson', 'mean'),
        )
        agg = per_seed.groupby(['experiment_method', 'attack_noise']).agg(
            num_recovered_mean=('num_recovered', 'mean'),
            num_recovered_ci95=('num_recovered', ci95),
            best_pearson_max=('best_pearson', 'max'),
            avg_pearson_mean=('avg_pearson', 'mean'),
            avg_pearson_ci95=('avg_pearson', ci95),
        ).reset_index()
        agg = agg.sort_values(['experiment_method', 'attack_noise'])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if agg.empty:
        plot_empty(ax, 'Recovered Images vs Attack Noise')
    else:
        for method in ordered_methods(agg['experiment_method'].unique()):
            subset = agg[agg['experiment_method'] == method]
            ax.plot(subset['attack_noise'], subset['num_recovered_mean'], marker='o', label=method)
            ax.fill_between(
                subset['attack_noise'],
                subset['num_recovered_mean'] - subset['num_recovered_ci95'],
                subset['num_recovered_mean'] + subset['num_recovered_ci95'],
                alpha=0.2,
            )
        ax.set_title('Recovered Images vs Attack Noise')
        ax.set_xlabel('attack_noise_amount')
        ax.set_ylabel('num_recovered')
        ax.legend()
        ax.grid(True, alpha=0.3)
    save_figure(fig, out_dir, 'fig3_recovered_vs_attack_noise')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if agg.empty:
        plot_empty(axes[0], 'Best Pearson vs Attack Noise')
        plot_empty(axes[1], 'Average Pearson vs Attack Noise')
    else:
        for method in ordered_methods(agg['experiment_method'].unique()):
            subset = agg[agg['experiment_method'] == method]
            axes[0].plot(subset['attack_noise'], subset['best_pearson_max'], marker='o', label=method)
            axes[1].plot(subset['attack_noise'], subset['avg_pearson_mean'], marker='o', label=method)
            axes[1].fill_between(
                subset['attack_noise'],
                subset['avg_pearson_mean'] - subset['avg_pearson_ci95'],
                subset['avg_pearson_mean'] + subset['avg_pearson_ci95'],
                alpha=0.2,
            )
        axes[0].set_title('Best Pearson vs Attack Noise')
        axes[1].set_title('Average Pearson vs Attack Noise')
        for ax in axes:
            ax.set_xlabel('attack_noise_amount')
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel('best_pearson')
        axes[1].set_ylabel('avg_pearson')
        axes[0].legend()
    save_figure(fig, out_dir, 'fig4_pearson_vs_attack_noise')
    return agg


def plot_convergence(epoch_df, out_dir):
    cols = ['experiment_method', 'epoch', 'seed', 'global_accuracy']
    data = epoch_df.reindex(columns=cols).dropna(subset=['experiment_method', 'epoch', 'global_accuracy']) if not epoch_df.empty else pd.DataFrame(columns=cols)
    if data.empty:
        agg = pd.DataFrame(columns=['experiment_method', 'epoch', 'global_accuracy_mean', 'global_accuracy_ci95', 'count'])
    else:
        per_seed = data.groupby(['experiment_method', 'epoch', 'seed'], as_index=False)['global_accuracy'].mean()
        agg = per_seed.groupby(['experiment_method', 'epoch']).agg(
            global_accuracy_mean=('global_accuracy', 'mean'),
            global_accuracy_ci95=('global_accuracy', ci95),
            count=('global_accuracy', 'count'),
        ).reset_index()
        agg = agg.sort_values(['experiment_method', 'epoch'])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if agg.empty:
        plot_empty(ax, 'Global Accuracy vs Epoch')
    else:
        for method in ordered_methods(agg['experiment_method'].unique()):
            subset = agg[agg['experiment_method'] == method]
            ax.plot(subset['epoch'], subset['global_accuracy_mean'], marker='o', label=method)
            ax.fill_between(
                subset['epoch'],
                subset['global_accuracy_mean'] - subset['global_accuracy_ci95'],
                subset['global_accuracy_mean'] + subset['global_accuracy_ci95'],
                alpha=0.2,
            )
        ax.set_title('Global Accuracy vs Epoch')
        ax.set_xlabel('epoch')
        ax.set_ylabel('global_accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    save_figure(fig, out_dir, 'fig5_convergence_accuracy_vs_epoch')
    return agg


def plot_overhead(overhead_df, out_dir):
    cols = [
        'experiment_method', 'mean_epoch_time_sec', 'mean_train_time_sec',
        'mean_aggregation_time_sec', 'mean_committee_time_sec',
        'mean_dp_time_sec', 'mean_test_time_sec',
    ]
    data = overhead_df.reindex(columns=cols).dropna(subset=['experiment_method']) if not overhead_df.empty else pd.DataFrame(columns=cols)
    if data.empty:
        agg = pd.DataFrame(columns=cols)
    else:
        for col in cols[1:]:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0.0)
        agg = data.groupby('experiment_method', as_index=False).agg(
            mean_epoch_time_sec=('mean_epoch_time_sec', 'mean'),
            mean_train_time_sec=('mean_train_time_sec', 'mean'),
            mean_aggregation_time_sec=('mean_aggregation_time_sec', 'mean'),
            mean_committee_time_sec=('mean_committee_time_sec', 'mean'),
            mean_dp_time_sec=('mean_dp_time_sec', 'mean'),
            mean_test_time_sec=('mean_test_time_sec', 'mean'),
        )
        agg = agg.sort_values('experiment_method', key=lambda s: s.map(method_sort_key))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if agg.empty:
        plot_empty(ax, 'Seconds per Epoch by Method')
    else:
        methods = list(agg['experiment_method'])
        components = [
            ('mean_train_time_sec', 'train'),
            ('mean_aggregation_time_sec', 'aggregation'),
            ('mean_committee_time_sec', 'committee'),
            ('mean_dp_time_sec', 'dp'),
            ('mean_test_time_sec', 'test'),
        ]
        bottom = pd.Series([0.0] * len(agg))
        plotted_component = False
        for col, label in components:
            values = agg[col].fillna(0.0)
            if float(values.sum()) > 0.0:
                ax.bar(methods, values, bottom=bottom, label=label)
                bottom = bottom + values
                plotted_component = True
        if not plotted_component:
            ax.bar(methods, agg['mean_epoch_time_sec'], label='epoch')
        ax.set_title('Seconds per Epoch by Method')
        ax.set_xlabel('experiment_method')
        ax.set_ylabel('seconds')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()
    save_figure(fig, out_dir, 'fig6_overhead_seconds_per_epoch')
    return agg


def main():
    parser = argparse.ArgumentParser(description='Plot RMA experiment summaries')
    parser.add_argument('--results_dir', default='results', type=str)
    parser.add_argument('--out_dir', default=os.path.join('results', 'plots'), type=str)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    leakage_path = os.path.join(args.results_dir, 'leakage_raw.csv')
    epoch_path = os.path.join(args.results_dir, 'epoch_raw.csv')
    overhead_path = os.path.join(args.results_dir, 'overhead_raw.csv')

    leakage_df = prepare_leakage(load_csv_or_empty(leakage_path))
    epoch_df = load_csv_or_empty(epoch_path)
    overhead_df = load_csv_or_empty(overhead_path)

    if not epoch_df.empty:
        for col in ['epoch', 'global_accuracy']:
            if col in epoch_df.columns:
                epoch_df[col] = pd.to_numeric(epoch_df[col], errors='coerce')

    dataset_size_agg = plot_best_pearson_vs_n(leakage_df, args.out_dir)
    avg_dataset_size_agg = plot_avg_pearson_vs_n(leakage_df, args.out_dir)
    noise_agg = plot_noise_curves(leakage_df, args.out_dir)
    convergence_agg = plot_convergence(epoch_df, args.out_dir)
    overhead_agg = plot_overhead(overhead_df, args.out_dir)

    if not dataset_size_agg.empty or not avg_dataset_size_agg.empty:
        merged = pd.merge(
            dataset_size_agg,
            avg_dataset_size_agg,
            on=['experiment_method', 'local_train_size'],
            how='outer',
        )
    else:
        merged = pd.DataFrame()

    merged.to_csv(os.path.join(args.out_dir, 'dataset_size_agg.csv'), index=False)
    noise_agg.to_csv(os.path.join(args.out_dir, 'noise_agg.csv'), index=False)
    convergence_agg.to_csv(os.path.join(args.out_dir, 'convergence_agg.csv'), index=False)
    overhead_agg.to_csv(os.path.join(args.out_dir, 'overhead_agg.csv'), index=False)


if __name__ == '__main__':
    main()
