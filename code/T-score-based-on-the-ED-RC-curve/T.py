import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取数据
df = pd.read_csv("FRET.csv")

# 2. 定义 Michaelis-Menten 拟合函数
def michaelis_menten(x, ed_max, kd):
    return ed_max * x / (kd + x)

# 3. 固定拟合参数（你已经拟合好了）
ed_max_fit, kd_fit = 0.317, 1  # 根据你的拟合结果来

# 4. 计算每个细胞的残差：实际ED - 拟合ED
fit_ed = michaelis_menten(df['Ed_region_rc_mean'], ed_max_fit, kd_fit)
df['residual'] = df['Ed_region_mean'] - fit_ed
# 残差 > 0: 点在曲线上方；残差 < 0: 点在曲线下方

# 5. 只对“曲线下方”的点（residual < 0）计算连续 T 值
#    思路：用负残差的绝对值表示偏离程度，并做一个0-1归一化
neg_mask = df['residual'] < 0
neg_residual = -df.loc[neg_mask, 'residual']  # 变成正数，偏离越大数值越大

# 选一个缩放因子（用 95 分位数避免极端值把尺度拉爆）
if len(neg_residual) > 0:
    scale = np.percentile(neg_residual, 95)
    # 防止 scale 为 0
    if scale == 0:
        scale = neg_residual.max() if neg_residual.max() > 0 else 1.0
else:
    scale = 1.0  # 没有曲线下的点时避免报错

# 初始化 T = 0
df['T'] = 0.0
# 对曲线下方的点，根据偏离程度线性归一到 [0,1]，越偏离 T 越接近 1
df.loc[neg_mask, 'T'] = (neg_residual / scale).clip(0, 1)

# 6. 可视化：RC vs residual
plt.figure(figsize=(8, 6))
sc = plt.scatter(df['Ed_region_rc_mean'], df['residual'], c=df['T'], cmap='viridis', s=10)
plt.axhline(0, color='black', linestyle='--', label='Residual = 0')
plt.xlabel('RC (Ed_region_rc_mean)')
plt.ylabel('Residual (ED - Fit)')
plt.legend()
cbar = plt.colorbar(sc)
cbar.set_label('T (continuous engagement score)')
plt.title('Residuals vs RC with continuous T')
plt.show()

# 7. 选择需要的列并输出到新的 DataFrame
metadata_columns = [col for col in df.columns if col.startswith('Metadata_')]
columns_to_output = metadata_columns + [
    'ObjectNumber',
    'Ed_region_mean',
    'Ed_region_rc_mean',
    'residual',
    'T'
]

output_df = df[columns_to_output]

# 8. 保存结果到新的 CSV 文件
output_df.to_csv("FRET_with_T_scores.csv", index=False)

print(output_df.head())
