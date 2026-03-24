import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. 读取数据
df = pd.read_csv("FRET.csv")

# 2. 去除缺失值
df = df.dropna(subset=['Ed_region_mean', 'Ed_region_rc_mean'])

df = df[(df['Ed_region_rc_mean'] <= 5) & (df['Ed_region_rc_mean'] >= 0)]

# 3. 定义拟合函数（Michaelis-Menten 型）
def michaelis_menten(x, ed_max, kd):
    return ed_max * x / (kd + x)



# 4. 将 RC 值按 0.1 为间隔划分 bin
bins = np.arange(0, 5.0, 0.2)
df['RC_bin'] = np.digitize(df['Ed_region_rc_mean'], bins)

# 5. 对每个 bin 计算 RC 和 ED 的平均值
binned_data = df.groupby('RC_bin').agg({
    'Ed_region_mean': 'mean',
    'Ed_region_rc_mean': 'mean'
}).reset_index()

# 6. 使用 curve_fit 拟合 Michaelis-Menten 函数
popt, pcov = curve_fit(
    michaelis_menten,
    binned_data['Ed_region_rc_mean'],
    binned_data['Ed_region_mean'],
    bounds=(0, 1)
)

# 拟合参数
ed_max_fit, kd_fit = popt
print(f"拟合出的 ED_max: {ed_max_fit:.3f}, Kd: {kd_fit:.3f}")

# 7. 计算拟合曲线
y_true = binned_data['Ed_region_mean'].values
y_pred = michaelis_menten(binned_data['Ed_region_rc_mean'], *popt)

# 8. 计算 R²
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"R² = {r2:.4f}")

# 9. 绘制拟合曲线和散点分布
plt.figure(figsize=(8, 6))
plt.scatter(binned_data['Ed_region_rc_mean'], binned_data['Ed_region_mean'],
            color='blue', label='Binned Data')
plt.plot(binned_data['Ed_region_rc_mean'], y_pred,
         color='red', label=f'Fit: ED_max={ed_max_fit:.2f}, Kd={kd_fit:.2f}, R²={r2:.2f}')
plt.xlabel('RC (Ed_region_rc_mean)')
plt.ylabel('ED (Ed_region_mean)')
plt.legend()
plt.title('Fitting ED vs RC using Michaelis-Menten')
plt.show()

print("拟合的 ED_max 和 Kd:", popt)
print("R²:", r2)
