"""
Generate a clean academic-style flow diagram inspired by traditional
queueing figures (monochrome look with dashed separators).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["Times New Roman", "SimHei", "serif"]
plt.rcParams["axes.unicode_minus"] = False


def draw_section_divider(ax, x, label):
    ax.plot([x, x], [0.8, 8.7], linestyle="--", color="black", linewidth=1)
    ax.text(
        x + 0.1,
        8.9,
        label,
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )


def add_block(ax, xy, width, height, title, lines, fontsize=12):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.08",
        linewidth=1.4,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height - 0.3,
        title,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
    )
    for idx, line in enumerate(lines):
        ax.text(
            x + 0.2,
            y + height - 0.8 - idx * 0.5,
            line,
            ha="left",
            va="top",
            fontsize=fontsize - 1,
        )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        linewidth=2,
        color="#1F77B4",
        mutation_scale=16,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    ax.text(
        8,
        9.1,
        "Cache-Queue-Migration Cooperation",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    draw_section_divider(ax, 3.3, "Source")
    draw_section_divider(ax, 7.1, "Processing")
    draw_section_divider(ax, 12.0, "Space Release")

    add_block(
        ax,
        (0.9, 5.2),
        2.2,
        2.6,
        "车辆/应用请求",
        [
            "content_id, 数据量",
            "QoS / priority / 时限",
            "命中缓存 → 返回",
        ],
    )

    add_block(
        ax,
        (3.6, 5.2),
        3.0,
        2.6,
        "协作缓存 (CollaborativeCacheManager)",
        [
            "热度+预测+邻居同步",
            "LRU/LFU/FIFO/Hybrid & 背包替换",
            "cache miss → Task",
        ],
    )

    add_block(
        ax,
        (7.4, 5.2),
        3.0,
        2.6,
        "队列管理 (PriorityQueueManager)",
        [
            "(lifetime, priority) 二维槽",
            "M/M/1 + 瞬时等待预测",
            "输出 queue_stats / load_factor",
        ],
    )

    add_block(
        ax,
        (10.8, 5.2),
        3.0,
        2.6,
        "迁移控制 (TaskMigrationManager)",
        [
            "多维评分选目标 + 迁移计划",
            "Keep-Before-Break 执行 + 重试",
            "更新 node_states / migration_stats",
        ],
    )

    add_block(
        ax,
        (14.2, 5.2),
        1.8,
        2.6,
        "结果",
        [
            "完成任务",
            "时限违约或继续迁移",
        ],
    )

    add_block(
        ax,
        (5.1, 1.4),
        5.6,
        2.4,
        "监控 / 策略 / 学习",
        [
            "消费命中率、队列、迁移指标",
            "调节缓存/队列/迁移策略阈值",
            "驱动 RL 或自适应控制",
        ],
        fontsize=11,
    )

    add_arrow(ax, (3.1, 6.5), (3.6, 6.5))
    add_arrow(ax, (6.6, 6.5), (7.4, 6.5))
    add_arrow(ax, (10.4, 6.5), (10.8, 6.5))
    add_arrow(ax, (13.8, 6.5), (14.2, 6.5))

    add_arrow(ax, (6.1, 5.2), (7.9, 3.8))
    add_arrow(ax, (8.9, 5.2), (8.1, 3.8))
    add_arrow(ax, (12.3, 5.2), (10.7, 3.8))

    add_arrow(ax, (8.5, 3.8), (7.7, 5.2))
    add_arrow(ax, (10.6, 3.8), (11.6, 5.2))
    add_arrow(ax, (6.2, 3.8), (5.0, 5.2))

    ax.text(
        8,
        0.7,
        "Figure: Sequential cooperation from cache-centric admission to queueing and migration (vector-friendly).",
        ha="center",
        va="center",
        fontsize=10,
    )

    output_path = Path("docs") / "cache_queue_migration_architecture.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved diagram to {output_path}")


if __name__ == "__main__":
    main()
