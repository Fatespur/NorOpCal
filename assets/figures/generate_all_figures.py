"""NorOpCal Figure Generation Suite.

Generates 4 publication-grade diagnostic and conceptual figures (Fig 2–5):
(Note: Fig1_系统框架图.png is the authentic author static framework diagram)
- Fig2_帕累托风险增益前沿.png (Authentic continuous-scale ablation trade-off on DINOv2)
- Fig3_样本效率与骨干迁移性.png (Conceptual SE decay & detector feasibility classification)
- Fig4_安全门控消融与危害率压制.png (Controlled mechanism attribution between Bounding and Gating)
- Fig5_决策边界与工况漂移对比.png (Strict mathematically consistent synthetic schematic)

Note on Fig 6: Radar chart has been deprecated and removed to prevent misleading impressions,
as multi-category performance is not supported by empirical radar evaluation.
"""

import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Strict check: Treat any missing glyph warnings as errors to prevent unreadable box characters
warnings.filterwarnings("error", message=r".*Glyph.*missing from font.*")

# Global style and font setup with prioritized CJK fonts across Linux and Windows
plt.style.use('default')
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei',
    'SimHei', 'Microsoft YaHei', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_fig2():
    """Generates Fig 2 using authentic continuous coordinates (no artificial truncation)."""
    fig, ax = plt.subplots(figsize=(11.5, 7.0), facecolor='white', dpi=300)
    ax.set_facecolor('#FFFFFF')
    # Full un-truncated natural scale covering authentic Harm Rate up to 42%
    ax.set_xlim(-1.5, 42.0)
    ax.set_ylim(-0.5, 4.8)
    ax.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1')

    # Background low-risk reference band (Harm Rate <= 5%)
    safe_zone = patches.Rectangle((-1.5, -0.5), 6.5, 5.3, fc='#ECFDF5', ec='none', alpha=0.6, zorder=1)
    ax.add_patch(safe_zone)
    ax.text(1.75, 4.5, "低决策危害参考区间 (Harm Rate ≤ 5%)", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color='#059669', zorder=2)

    # Authentic frozen ablation points on DINOv2 (K=8, 10 TEST units: fabric + walnuts)
    methods = [
        {"name": "不校准基线 (NoAdapt, 冻结阈值)\n[ΔBA=0.00, Harm=0.00%]", "x": 0.0, "y": 0.0, "color": "#64748B", "marker": "s", "size": 110, "text_xy": (1.2, 0.4), "ha": "left", "arrow": True},
        {"name": "仅有界收缩 (λ=0.25, 无门控)\n[ΔBA=+3.64, Harm=3.67%, 覆盖率=100%]", "x": 3.67, "y": 3.64, "color": "#0284C7", "marker": "^", "size": 130, "text_xy": (4.5, 3.9), "ha": "left"},
        {"name": "仅门控机制 (λ=1.0, 激进更新)\n[运作危害率高达 37.67%, 风险过大]", "x": 37.67, "y": 4.25, "color": "#D97706", "marker": "d", "size": 130, "text_xy": (36.5, 3.8), "ha": "right"},
        {"name": "NorOpCal 全策略\n[Q50门控 + λ=0.25有界收缩]\n[ΔBA=+3.23, Harm=3.67%, 覆盖率=40.3%]", "x": 3.67, "y": 3.23, "color": "#059669", "marker": "o", "size": 170, "text_xy": (4.5, 2.7), "ha": "left", "highlight": True}
    ]

    for m in methods:
        ax.scatter(m["x"], m["y"], color=m["color"], s=m["size"], marker=m["marker"],
                   edgecolors='#1E293B', linewidths=1.5, zorder=5)
        
        if m.get("highlight"):
            ax.annotate(m["name"], xy=(m["x"], m["y"]), xytext=m["text_xy"],
                        fontsize=9.5, fontweight='bold', color='#065F46', ha=m["ha"],
                        bbox=dict(boxstyle="round,pad=0.45", fc="#D1FAE5", ec="#10B981", lw=1.5),
                        arrowprops=dict(arrowstyle="->", color='#059669', lw=1.8), zorder=6)
        elif m.get("arrow"):
            ax.annotate(m["name"], xy=(m["x"], m["y"]), xytext=m["text_xy"],
                        fontsize=9.0, fontweight='bold', color='#1E293B', ha=m["ha"],
                        bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#94A3B8", lw=1.0),
                        arrowprops=dict(arrowstyle="->", color='#64748B', lw=1.3), zorder=6)
        elif m["x"] > 20:
            ax.annotate(m["name"], xy=(m["x"], m["y"]), xytext=m["text_xy"],
                        fontsize=9.2, fontweight='bold', color='#B45309', ha=m["ha"],
                        bbox=dict(boxstyle="round,pad=0.35", fc="#FEF3C7", ec="#F59E0B", lw=1.2),
                        arrowprops=dict(arrowstyle="->", color='#D97706', lw=1.5), zorder=6)
        else:
            ax.annotate(m["name"], xy=(m["x"], m["y"]), xytext=m["text_xy"],
                        fontsize=9.0, fontweight='bold', color='#1E293B', ha=m["ha"], zorder=6)

    ax.set_xlabel("决策危害率 (Harm Rate, %，真实连续尺度，越低越好)", fontsize=11, fontweight='bold', color='#0F172A')
    ax.set_ylabel("平衡准确率净增益 (Δ Balanced Accuracy, pp，越高越好)", fontsize=11, fontweight='bold', color='#0F172A')
    ax.set_title("策略消融权衡分析：真实决策危害率 vs. 净增益\n(DINOv2 主干, K=8, 10 个 TEST 单元冻结实测真实坐标，无坐标截断)", 
                 fontsize=12.0, fontweight='bold', pad=15, color='#0F172A')

    out = os.path.join(OUTPUT_DIR, "Fig2_帕累托风险增益前沿.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 2 generated:", out)

def generate_fig3():
    """Generates Fig 3 with rigorous statistical terminology and non-numerical mechanism cards."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='white', dpi=300)
    
    # Left: Asymptotic Standard Error decay (O(1/sqrt(K)))
    K_theory = np.array([2, 4, 8, 16, 32])
    # For sample median, asymptotic variance is O(1/K), standard error is O(1/sqrt(K))
    se_decay = 1.0 / np.sqrt(K_theory)
    
    ax1.plot(K_theory, se_decay, marker='s', color='#0284C7', lw=2.2, label='中位数估计量渐近标准误 ~ 1/√K\n(渐近方差尺度为 O(1/K))')
    ax1.axvline(8, color='#D97706', linestyle='--', lw=1.8, label='预注册选定规模 (K=8，工程折中)')
    ax1.scatter([8], [1.0 / np.sqrt(8)], color='#F59E0B', s=130, zorder=5, edgecolors='#78350F', lw=1.8)
    ax1.annotate("工程选定规模 (K=8)\n在样本获取代价与估计标准误衰减间折中\n(非数学全局最优推导证明)", 
                 xy=(8, 1.0 / np.sqrt(8)), xytext=(11, 0.42),
                 arrowprops=dict(arrowstyle="->", color='#D97706', lw=1.8),
                 fontsize=9.2, fontweight='bold', color='#B45309',
                 bbox=dict(boxstyle="round,pad=0.35", fc="#FEF3C7", ec="#F59E0B", lw=1.2))
    ax1.set_xlabel("目标域正常样本数 (K)", fontsize=10.5, fontweight='bold')
    ax1.set_ylabel("渐近估计标准误 (相对尺度，非方差)", fontsize=10.5, fontweight='bold')
    ax1.set_title("【理论示意】小样本中位数估计的渐近标准误衰减\n(理论渐近规律：估计标准误随 K 增加而收敛)", fontsize=11.5, fontweight='bold', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.legend(loc='upper right', fontsize=8.8)

    # Right: Structural mechanism diagnostic cards (strictly NO arbitrary numerical bar lengths)
    ax2.set_facecolor('#FFFFFF')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title("【机制判定】主干检测器评测结论与适用边界\n(分类判定状态卡片，无数值大小柱长)", fontsize=11.5, fontweight='bold', pad=12)

    cards = [
        {
            "detector": "DINOv2 (dinov2_vits14)",
            "role": "核心主干 (Principal)",
            "verdict": "USEFUL_SAFE_SELECTABLE",
            "bg": "#ECFDF5", "border": "#10B981", "badge_bg": "#059669",
            "desc": "实测现象：在 10 个测试单元测得 +3.23 pp 显著净增益。\n机理假设：深层自监督特征空间在工况漂移下呈现相对刚性平移，标量中位数对齐有效。"
        },
        {
            "detector": "PatchCore (wide_resnet50_2)",
            "role": "机制对照 (Mechanism Control)",
            "verdict": "ACTUATOR_MISMATCH",
            "bg": "#FFF1F2", "border": "#F43F5E", "badge_bg": "#E11D48",
            "desc": "实测现象：阈值平移导致漏检或误报剧烈波动，判定动作器失配。\n机理假设：基于记忆库局部邻域距离的异常判别对非各向同性形变敏感，标量平移不适用。"
        },
        {
            "detector": "EfficientAD (70,000 steps)",
            "role": "机制对照 (Mechanism Control)",
            "verdict": "NO_HEADROOM",
            "bg": "#F8FAFC", "border": "#94A3B8", "badge_bg": "#64748B",
            "desc": "实测现象：目标域下基线模型无可调增益空间。\n机理假设：学生-教师结构在工况漂移下的预测残差分布具有特有收敛特性，阈值调节无余量。"
        }
    ]

    y_tops = [9.5, 6.2, 2.9]
    card_h = 2.8
    for c, y_top in zip(cards, y_tops):
        rect = patches.FancyBboxPatch((0.2, y_top - card_h), 9.6, card_h,
                                      boxstyle="round,pad=0.15", ec=c["border"], fc=c["bg"], lw=1.5)
        ax2.add_patch(rect)
        # Separate detector name and role vertically so they never collide with the badge
        ax2.text(0.5, y_top - 0.52, c['detector'], fontsize=9.8, fontweight='bold', color="#0F172A")
        ax2.text(0.5, y_top - 0.90, f"角色定位：{c['role']}", fontsize=8.4, color="#475569", fontweight='bold')
        badge = patches.FancyBboxPatch((6.6, y_top - 0.72), 3.0, 0.52,
                                       boxstyle="round,pad=0.08", ec="none", fc=c["badge_bg"])
        ax2.add_patch(badge)
        ax2.text(8.1, y_top - 0.46, c["verdict"], fontsize=8.2, fontweight='bold', color="#FFFFFF", ha="center", va="center")
        ax2.text(0.5, y_top - 1.38, c["desc"], fontsize=8.3, color="#334155", va="top", linespacing=1.35)

    out = os.path.join(OUTPUT_DIR, "Fig3_样本效率与骨干迁移性.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 3 generated:", out)

def generate_fig4():
    """Generates Fig 4 with controlled mechanism attribution:
    1. Bounding effect: Compare 'Gate-only λ=1.0' vs 'Full policy λ=0.25' (same gate Q50).
    2. Gating effect: Compare 'Bounded-only λ=0.25' vs 'Full policy λ=0.25' (same bounding λ=0.25).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.8), facecolor='white', dpi=300)
    
    stages = [
        '① 不校准基线\n(NoAdapt)',
        '② 仅门控机制\n(λ=1.0, 激进更新)',
        '③ 仅有界收缩\n(λ=0.25, 无门控)',
        '④ 完整 NorOpCal\n(门控 + 有界收缩)'
    ]
    ba_gains = [0.0, 4.25, 3.64, 3.23]
    harm_rates = [0.0, 37.67, 3.67, 3.67]
    coverages = [0.0, 40.33, 100.0, 40.33]
    
    # Subplot 1: Gain
    colors1 = ['#64748B', '#D97706', '#0284C7', '#059669']
    bars1 = ax1.bar(stages, ba_gains, color=colors1, width=0.55)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 0.12, f"+{h:.2f} pp",
                 ha='center', va='bottom', fontsize=9.2, fontweight='bold', color='#1E293B')
    ax1.set_ylabel("平衡准确率净增益 (ΔBA, pp)", fontsize=10.5, fontweight='bold')
    ax1.set_title("策略消融：检出增益分析\n(DINOv2, K=8, 10 TEST 单元冻结实测)", fontsize=11.5, fontweight='bold', pad=12)
    ax1.set_ylim(0, 5.2)
    ax1.grid(True, linestyle='--', alpha=0.35, axis='y')

    # Subplot 2: Controlled Mechanism Attribution
    colors2 = ['#64748B', '#DC2626', '#0284C7', '#059669']
    bars2 = ax2.bar(stages, harm_rates, color=colors2, width=0.55)
    for b, cov in zip(bars2, coverages):
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + 1.1, f"{h:.2f}%\n(覆盖率{cov:.1f}%)",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E293B')

    # Explicit attribution annotations
    # 1. Bounding effect: ② vs ④ (control: same Q50 gate)
    ax2.annotate("有界收缩控危害 (② vs ④，同门控):\nλ 从 1.0 降为 0.25 -> 危害率由 37.67% 骤降至 3.67%",
                 xy=(0.72, 30.0), xytext=(0.38, 45.0), ha='center',
                 arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.5),
                 fontsize=8.5, fontweight='bold', color='#991B1B',
                 bbox=dict(boxstyle="round,pad=0.35", fc="#FEF2F2", ec="#FECDD3", lw=1.0))
    
    # 2. Gating effect: ③ vs ④ (control: same λ=0.25 bounding)
    ax2.annotate("门控机制施选择 (③ vs ④，同收缩):\n收缩均为 λ=0.25，Harm 均为 3.67%\n覆盖率从 100% 降至 40.33%，实现选择性校准",
                 xy=(2.5, 6.0), xytext=(2.5, 20.0), ha='center',
                 arrowprops=dict(arrowstyle="->", color='#059669', lw=1.5),
                 fontsize=8.5, fontweight='bold', color='#065F46',
                 bbox=dict(boxstyle="round,pad=0.35", fc="#ECFDF5", ec="#A7F3D0", lw=1.0))

    ax2.set_ylabel("决策危害率 (Harm Rate, %)", fontsize=10.5, fontweight='bold')
    ax2.set_title("机制归因：严格单一变量对照分析\n[收缩作用(② vs ④): 压降危害率；门控作用(③ vs ④): 赋予选择性]", fontsize=11.0, fontweight='bold', pad=12)
    ax2.set_ylim(0, 52)
    ax2.grid(True, linestyle='--', alpha=0.35, axis='y')

    out = os.path.join(OUTPUT_DIR, "Fig4_安全门控消融与危害率压制.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 4 generated:", out)

def generate_fig5():
    """Generates Fig 5 with strict mathematical consistency from a single synthetic dataset."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    # 1. Deterministic synthetic data generation
    np.random.seed(42)
    src_normals = np.random.normal(loc=0.25, scale=0.06, size=500)
    tgt_k8 = np.random.normal(loc=0.42, scale=0.06, size=8)

    # 2. Strict Mathematical Calculations according to NorOpCal equations
    m_s = float(np.median(src_normals))
    tau_s = float(np.percentile(src_normals, 95.0))
    m_t = float(np.median(tgt_k8))
    delta_m = float(m_t - m_s)
    cutoff_q50 = 0.015890181064605713
    lam = 0.25
    gate_pass = bool(delta_m >= cutoff_q50)
    tau_t = float(tau_s + lam * delta_m) if gate_pass else float(tau_s)

    x = np.linspace(0, 1.0, 600)
    src_pdf = 1.0 / (0.06 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0.25) / 0.06)**2)
    tgt_pdf = 1.0 / (0.06 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0.42) / 0.06)**2)
    defect_pdf = 1.0 / (0.08 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0.72) / 0.08)**2)

    # Subplot 1: Before
    ax1.set_facecolor('#FFFFFF')
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 8.8)
    ax1.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1')

    ax1.plot(x, src_pdf, color='#94A3B8', linestyle='--', lw=2.0, label=f"源域正常得分分布 (标定中位数 m_s={m_s:.4f})")
    ax1.plot(x, tgt_pdf, color='#EA580C', lw=2.5, label="目标域正常得分分布 (漂移工况)")
    ax1.plot(x, defect_pdf, color='#DC2626', lw=2.2, label="缺陷/异常样本得分分布")

    ax1.vlines(x=tau_s, ymin=0, ymax=6.6, color='#0F172A', lw=2.2, linestyle='-', zorder=4)
    ax1.text(tau_s, 7.3, f"源域静态阈值 τ_s = {tau_s:.4f}\n(未自适应，无法感知采集工况漂移)", 
             fontsize=9.2, fontweight='bold', color='#0F172A', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.4", fc="#F8FAFC", ec="#0F172A", lw=1.2), zorder=5)

    mask_fa = (x >= tau_s) & (x <= 0.65)
    ax1.fill_between(x[mask_fa], 0, tgt_pdf[mask_fa], color='#EA580C', alpha=0.32, hatch='//')
    ax1.annotate("【工况漂移诱发的误报区域】\n正常得分整体右移，固定静态阈值导致\n部分良品得分超过 τ_s 产生假阳性虚警", 
                 xy=(0.42, 2.8), xytext=(0.58, 4.5),
                 fontsize=9.0, fontweight='bold', color='#9A3412',
                 bbox=dict(boxstyle="round,pad=0.4", fc="#FFF7ED", ec="#F97316", lw=1.2),
                 arrowprops=dict(arrowstyle="->", color='#EA580C', lw=1.8, connectionstyle="arc3,rad=-0.15"), zorder=6)

    ax1.set_ylabel("概率密度 (概念尺度)", fontsize=10.5, fontweight='bold', color='#1E293B')
    ax1.set_title("【概念示意图·非真实实验数据】工况漂移下的静态阈值困境：源域静态阈值诱发目标域良品误报增加", 
                  fontsize=11.8, fontweight='bold', color='#991B1B', pad=12)
    ax1.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=8.8)

    # Subplot 2: After
    ax2.set_facecolor('#FFFFFF')
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 8.8)
    ax2.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1')

    ax2.plot(x, tgt_pdf, color='#059669', lw=2.5, label="目标域正常样本分布")
    ax2.fill_between(x, 0, tgt_pdf, color='#ECFDF5', alpha=0.45)
    ax2.plot(x, defect_pdf, color='#DC2626', lw=2.2, label="缺陷/异常样本分布")
    ax2.fill_between(x, 0, defect_pdf, color='#FEF2F2', alpha=0.35)

    ax2.scatter(tgt_k8, np.full_like(tgt_k8, 0.25), color='#F59E0B', s=70, marker='o', 
                edgecolor='#78350F', lw=1.4, zorder=6, label=f"目标域正常参考样本 (K=8, 样本中位数 m_t={m_t:.4f})")

    ax2.vlines(x=tau_s, ymin=0, ymax=4.8, color='#94A3B8', lw=1.8, linestyle=':', zorder=4)
    ax2.text(tau_s - 0.03, 4.8, f"原静态阈值 τ_s = {tau_s:.4f}", fontsize=8.5, color='#475569', ha='right', va='center',
             bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="#94A3B8", lw=0.9, alpha=0.95), zorder=5)

    ax2.vlines(x=tau_t, ymin=0, ymax=6.6, color='#059669', lw=2.6, linestyle='-', zorder=4)
    ax2.text(tau_t, 7.3, f"NorOpCal 自适应校准阈值 τ_t = {tau_t:.4f}\n[Δm={delta_m:.4f} ≥ Q50({cutoff_q50:.4f}) -> 触发校准]\n[严格数学生成: τ_t = τ_s + 0.25·Δm]", 
             fontsize=9.2, fontweight='bold', color='#065F46', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.4", fc="#D1FAE5", ec="#10B981", lw=1.3), zorder=5)

    # Shift indicator bracket and annotation in clean whitespace to the left
    ax2.annotate("", xy=(tau_t, 3.5), xytext=(tau_s, 3.5),
                 arrowprops=dict(arrowstyle="<->", color='#059669', lw=2.0), zorder=5)
    ax2.annotate(f"有界平移量: λ·Δm = {lam * delta_m:.4f}\n(阈值适度后移，抑制误报)",
                 xy=((tau_s + tau_t)/2, 3.5), xytext=(0.18, 3.5),
                 ha='right', va='center',
                 arrowprops=dict(arrowstyle="->", color='#059669', lw=1.5),
                 fontsize=8.5, fontweight='bold', color='#065F46',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#A7F3D0", lw=1.0), zorder=6)

    ax2.text(0.18, 1.8, "【工况自适应区】\n感知中位数正向漂移适度后移阈值\n显著抑制目标域良品误判", 
             fontsize=8.8, fontweight='bold', color='#065F46', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.35", fc="#ECFDF5", ec="#A7F3D0", lw=1.0), zorder=5)

    ax2.text(0.85, 1.8, "【异常判别区】\n有界收缩约束阈值更新幅度\n维持总体平均 FNR 预算约束", 
             fontsize=8.8, fontweight='bold', color='#991B1B', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.35", fc="#FFF1F2", ec="#FECDD3", lw=1.0), zorder=5)

    ax2.set_xlabel("样本异常判别得分 (Anomaly Score, 合成概念尺度)", fontsize=10.5, fontweight='bold', color='#1E293B')
    ax2.set_ylabel("概率密度 (概念尺度)", fontsize=10.5, fontweight='bold', color='#1E293B')
    ax2.set_title("【概念示意图·非真实实验数据】NorOpCal 校准机制：基于合成数据动态自洽计算之阈值重标定过程", 
                  fontsize=11.8, fontweight='bold', color='#065F46', pad=12)
    ax2.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=8.8)

    out = os.path.join(OUTPUT_DIR, "Fig5_决策边界与工况漂移对比.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 5 generated:", out)

if __name__ == "__main__":
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    print("All authentic and diagnostic figures generated successfully without glyph errors!")
