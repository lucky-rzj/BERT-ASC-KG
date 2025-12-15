# -------------------------- 检查生成的种子词 -------------------------------

import pickle as pk
import argparse
import os
import json


def check_seeds(dataset, seed_type="enhanced", custom_path=None):

    # ------------------- 选择文件路径 -------------------
    if custom_path:
        seeds_path = custom_path
        print(f"[INFO] 使用自定义路径: {seeds_path}")
    else:
        if seed_type == "baseline":
            seeds_path = f'../../datasets/{dataset}/categories_seeds.pk'
        elif seed_type == "enhanced":
            seeds_path = f'../../datasets/{dataset}/categories_seeds_lc.pk'
        else:
            raise ValueError("seed_type 必须为 baseline 或 enhanced")
    
    json_path = seeds_path.replace('.pk', '.json')

    seeds = None

    # ------------------- 加载 pk 或 json -------------------
    try:
        with open(seeds_path, 'rb') as f:
            seeds = pk.load(f)
        print(f"[INFO] 成功加载 PK 文件: {seeds_path}")
    except FileNotFoundError:
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                seeds = json.load(f)
            print(f"[INFO] 已从 JSON 文件加载: {json_path}")
        else:
            print(f"[ERROR] 未找到文件: {seeds_path}")
            return
    except Exception as e:
        print(f"[ERROR] 加载失败：{str(e)}")
        return

    # ------------------- 打印统计信息 -------------------
    print(f"\n===== 数据集 {dataset} —— {seed_type} seeds 检查结果 =====\n")
    print(f"共包含 {len(seeds)} 个类别")

    total_words = sum(len(v) for v in seeds.values())
    avg_words = total_words / len(seeds)
    longest = max(seeds.items(), key=lambda x: len(x[1]))

    print(f"总词数: {total_words}")
    print(f"平均每类种子数: {avg_words:.2f}")
    print(f"最大类别: {longest[0]} ({len(longest[1])} 个词)\n")

    print("=" * 80)

    # ------------------- 打印全部类别 + 全部种子词 -------------------
    for category, words in seeds.items():
        print(f"类别：{category}")
        print(f"种子词数量：{len(words)}")
        print("种子词列表：")
        print(", ".join(words))  # 打印整行，方便阅读
        print("-" * 80)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="检查 baseline / enhanced 的种子词")
    parser.add_argument('--dataset', default='semeval', help='数据集名称')
    parser.add_argument('--type', default='enhanced',
                        choices=['baseline', 'enhanced'],
                        help='检查 baseline 或 enhanced 种子词')
    parser.add_argument('--path', type=str, default=None,
                        help='自定义 pk / json 文件路径')
    
    args = parser.parse_args()

    check_seeds(args.dataset, seed_type=args.type, custom_path=args.path)
