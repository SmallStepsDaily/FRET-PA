import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ========== 全局绘图风格 ==========
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300


# ========== 模块化函数 ==========
def calculate_pt_score_value(T: float, P: float, C: float) -> float:
    """计算单个样本的 PT_score"""
    return P * T


def calculate_pt_score(p_file, t_file, c_file, output_dir=None):
    # ---------- 读取 P 值表 ----------
    try:
        p_df = pd.read_csv(p_file)
        print(f"成功读取P值表: {p_file}, shape={p_df.shape}")
    except Exception as e:
        print(f"读取P值表失败: {e}")
        return

    # ---------- 读取 T 值表 ----------
    try:
        t_df = pd.read_csv(t_file)
        print(f"成功读取T值表: {t_file}, shape={t_df.shape}")
    except Exception as e:
        print(f"读取T值表失败: {e}")
        return

    # ---------- 读取 C 值表 ----------
    try:
        c_df = pd.read_csv(c_file)
        print(f"成功读取C值表: {c_file}")
        c_df.set_index('Metadata_treatment', inplace=True)
        c_dict = c_df['mean_diff'].to_dict()
    except Exception as e:
        print(f"读取C值表失败: {e}")
        return

    # ---------- 合并 P 和 T ----------
    key_cols = [
        'Metadata_hour', 'Metadata_concentration', 'Metadata_treatment',
        'Metadata_dish', 'Metadata_site', 'ObjectNumber'
    ]
    df = pd.merge(p_df, t_df, on=key_cols, how='inner')
    print(f"合并P与T后数据: {df.shape}")

    # ---------- 设置 P 和 T ----------
    df['P_score'] = df['P']
    df['T'] = df['T'].astype(float)

    # ---------- 添加 C_value ----------
    missing_groups = [g for g in df['Metadata_treatment'].unique() if g not in c_dict]
    if missing_groups:
        print(f"C值缺失: {', '.join(missing_groups)}，无法继续")
        return

    df['C_value'] = df['Metadata_treatment'].map(c_dict)

    # ---------- 计算 PT_score ----------
    df['PT_score'] = [
        calculate_pt_score_value(T, P, C)
        for T, P, C in zip(df['T'], df['P_score'], df['C_value'])
    ]

    # ---------- 排序分组 ----------
    all_groups = df['Metadata_treatment'].unique().tolist()
    groups_ordered = ['control'] + [g for g in all_groups if g != 'control']
    df['Metadata_treatment'] = pd.Categorical(df['Metadata_treatment'], categories=groups_ordered, ordered=True)

    # ---------- 输出目录 ----------
    if output_dir is None:
        output_dir = f"results"
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    df.to_csv(os.path.join(output_dir, f"PT_score.csv"), index=False)

    # 绘图
    plot_bar_scatter(df, groups_ordered, output_dir)
    return df


def plot_bar_scatter(df, groups_ordered, output_dir):
    """
    绘制 Nature 风格: 平均值柱状图 + 散点分布 + SEM 误差棒 + 显著性标记
    """
    # 分组统计
    group_stats = df.groupby('Metadata_treatment')['PT_score'].agg(['mean', 'sem']).reindex(groups_ordered)
    print(group_stats)
    # 画布
    fig, ax = plt.subplots(figsize=(6, 6))

    # 颜色设置
    colors = sns.color_palette("Set2", n_colors=len(groups_ordered))

    # 绘制柱形图 (均值)
    bars = ax.bar(
        group_stats.index,
        group_stats['mean'],
        yerr=group_stats['sem'],
        capsize=5,
        color=colors,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.7
    )

    # 绘制散点 (单个样本)
    for i, group in enumerate(groups_ordered):
        y = df.loc[df['Metadata_treatment'] == group, 'PT_score'].values
        x = np.random.normal(i, 0.08, size=len(y))  # 添加抖动避免重叠
        ax.scatter(x, y, color=colors[i], s=40, edgecolor='black', alpha=0.9, zorder=3)

    # y 轴标签
    ax.set_ylabel('Normalized $PT_{score}$', fontweight='bold')

    # 格式
    ax.set_xticks(range(len(groups_ordered)))
    ax.set_xticklabels(groups_ordered, rotation=45, ha='right', fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # ---------- 显著性检验 ----------
    if len(groups_ordered) == 2:  # 只做两组比较
        g1 = df.loc[df['Metadata_treatment'] == groups_ordered[0], 'PT_score']
        g2 = df.loc[df['Metadata_treatment'] == groups_ordered[1], 'PT_score']
        stat, p = stats.ttest_ind(g1, g2, equal_var=False)

        # 设置显著性标记
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        # 添加标记
        y_max = max(group_stats['mean'] + group_stats['sem']) * 1.2
        ax.plot([0, 1], [y_max, y_max], color='black', linewidth=1.5)
        ax.text(0.5, y_max * 1.05, sig, ha='center', va='bottom', fontsize=16, fontweight='bold')

    plt.tight_layout()
    out_file = os.path.join(output_dir, "PT_score_bar_scatter.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"bar+scatter 图已保存至: {out_file}")


# ========== 主程序入口 ==========
if __name__ == "__main__":
    p_file = "data/P_m.csv"
    t_file = "data/T.csv"
    c_file = "data/C.csv"
    result_df = calculate_pt_score(p_file, t_file, c_file)
