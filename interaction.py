# ===================== 1. 导入依赖库 =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 设置 matplotlib 中文显示，避免画图乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 2. 构建模拟数据集 =====================
# 固定随机种子，保证每次运行结果一致
np.random.seed(42)
n_samples = 12000  # 样本总数

# 生成8项材料成分特征（特征顺序固定，后续预测必须保持一致）
features = pd.DataFrame({
    '平均原子序数': np.random.uniform(10, 80, n_samples),
    '平均原子量': np.random.uniform(20, 200, n_samples),
    '平均电负性': np.random.uniform(1.0, 3.5, n_samples),
    '电负性差': np.random.uniform(0.2, 2.0, n_samples),
    '平均价电子数': np.random.uniform(2, 8, n_samples),
    '价电子浓度': np.random.uniform(0.1, 1.5, n_samples),
    '平均原子半径': np.random.uniform(0.5, 2.0, n_samples),
    '原子半径差': np.random.uniform(0.05, 0.8, n_samples)
})

# 生成对应带隙值（模拟物理规律：电负性差越大、价电子数适中，带隙越大）
band_gap = (
        0.8 * features['电负性差']
        + 0.5 * features['平均价电子数']
        - 0.3 * features['平均原子序数'] / 50
        + np.random.normal(0, 0.4, n_samples)
)
band_gap = np.clip(band_gap, 0, 6)  # 带隙范围限制在 0~6 eV

# ===================== 3. 划分训练集与测试集 =====================
X = features.values  # 输入特征
y = band_gap.values  # 目标值（带隙）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== 4. 训练三种预测模型 =====================
print("正在训练模型，请稍候...\n")

# 模型1：线性回归（基线对比）
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 模型2：支持向量回归
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)

# 模型3：随机森林回归（核心模型）
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# ===================== 5. 模型性能评估 =====================
def evaluate(y_true, y_pred, model_name):
    """计算并打印模型评估指标"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"===== {model_name} =====")
    print(f"决定系数 R² : {r2:.3f}")
    print(f"平均绝对误差 MAE : {mae:.3f} eV")
    print(f"均方根误差 RMSE : {rmse:.3f} eV\n")
    return r2, mae, rmse


# 打印三个模型的评估结果
evaluate(y_test, y_pred_lr, "线性回归")
evaluate(y_test, y_pred_svr, "支持向量回归")
evaluate(y_test, y_pred_rf, "随机森林")

# ===================== 6. 可视化结果 =====================
# 图1：随机森林 真实值 vs 预测值 散点图
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.3, s=10, color='#1f77b4')
plt.plot([0, 6], [0, 6], 'r--', label='理想拟合线 y=x')
plt.xlabel('真实带隙 (eV)')
plt.ylabel('预测带隙 (eV)')
plt.title('随机森林模型带隙预测结果')
plt.legend()
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.tight_layout()
plt.show()

# 图2：特征重要性柱状图
feature_names = features.columns
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]  # 从大到小排序

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[indices], color='#ff7f0e')
plt.xticks(range(len(importances)), feature_names[indices], rotation=30)
plt.ylabel('特征重要性得分')
plt.title('随机森林模型特征重要性排序')
plt.tight_layout()
plt.show()

# ===================== 7. 交互式单材料带隙预测 =====================
print("=" * 50)
print("【交互式带隙预测】")
print("请依次输入8个特征数值，用空格隔开")
print("特征顺序：")
print("1.平均原子序数  2.平均原子量  3.平均电负性  4.电负性差")
print("5.平均价电子数  6.价电子浓度  7.平均原子半径  8.原子半径差")
print("参考示例输入：22 48 2.5 1.2 5 0.8 1 0.3\n")

while True:
    try:
        user_input = input("请输入特征值（输入 q 退出）：")
        if user_input.strip().lower() == 'q':
            print("预测结束。")
            break

        user_values = [float(x) for x in user_input.split()]
        if len(user_values) != 8:
            print("输入错误：必须输入8个数值，请重新输入！\n")
            continue

        predicted_bandgap = rf.predict([user_values])[0]
        print(f"预测带隙值：{predicted_bandgap:.3f} eV\n")

    except ValueError:
        print("输入格式错误，请输入有效的数字！\n")