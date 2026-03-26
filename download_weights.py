#!/usr/bin/env python3
"""
下载OpenUni所需的所有模型权重
Download all necessary model weights for OpenUni
"""

import os
from huggingface_hub import snapshot_download, hf_hub_download

# 目标目录
TARGET_DIR = "/gemini/user/private/LLM-CKPT"

def download_model(repo_id, local_name):
    """下载单个模型"""
    local_dir = os.path.join(TARGET_DIR, local_name)

    print(f"\n{'='*60}")
    print(f"正在下载: {repo_id}")
    print(f"保存到: {local_dir}")
    print(f"{'='*60}\n")

    # 检查是否已存在
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"✓ 已存在，跳过: {local_dir}\n")
        return True

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ 下载完成: {repo_id}\n")
        return True
    except Exception as e:
        print(f"✗ 下载失败: {str(e)}\n")
        return False

def download_openuni_checkpoints():
    """下载OpenUni的checkpoint文件"""
    repo_id = "wusize/openuni"
    local_dir = os.path.join(TARGET_DIR, "openuni-checkpoints")

    print(f"\n{'='*60}")
    print(f"正在下载OpenUni checkpoints")
    print(f"保存到: {local_dir}")
    print(f"{'='*60}\n")

    os.makedirs(local_dir, exist_ok=True)

    # OpenUni checkpoint文件列表
    checkpoint_files = [
        "openuni_b_internvl3_1b_sana_0_6b_512_hf_blip3o60k.pth",
        "openuni_b_internvl3_1b_sana_0_6b_512_hf_text2image23m.pth",
        "openuni_l_internvl3_2b_sana_1_6b_512_hf_blip3o60k.pth",
        "openuni_l_internvl3_2b_sana_1_6b_512_hf_text2image23m.pth",
        "openuni_l_internvl3_2b_sana_1_6b_1024_hf_blip3o60k.pth",
        "openuni_l_internvl3_2b_sana_1_6b_1024_hf_text2image23m.pth",
    ]

    success_count = 0
    for filename in checkpoint_files:
        filepath = os.path.join(local_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ 已存在: {filename}")
            success_count += 1
            continue

        try:
            print(f"正在下载: {filename}")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                resume_download=True,
            )
            print(f"✓ 下载完成: {filename}\n")
            success_count += 1
        except Exception as e:
            print(f"✗ 下载失败 {filename}: {str(e)}\n")

    return success_count == len(checkpoint_files)

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("OpenUni 模型权重下载脚本")
    print("OpenUni Model Weights Download Script")
    print("="*60 + "\n")

    # 需要下载的基础模型
    base_models = [
        # InternVL3 模型
        ("OpenGVLab/InternVL3-1B", "InternVL3-1B"),
        ("OpenGVLab/InternVL3-2B", "InternVL3-2B"),

        # SANA Diffusion 模型
        ("Efficient-Large-Model/Sana_600M_512px_diffusers", "Sana_600M_512px_diffusers"),
        ("Efficient-Large-Model/Sana_1600M_512px_diffusers", "Sana_1600M_512px_diffusers"),
        ("Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers", "SANA1.5_1.6B_1024px_diffusers"),
    ]

    successful = 0
    failed = []

    # 下载基础模型
    for repo_id, local_name in base_models:
        if download_model(repo_id, local_name):
            successful += 1
        else:
            failed.append(repo_id)

    # 下载OpenUni checkpoints
    print("\n" + "="*60)
    print("下载OpenUni Checkpoints")
    print("="*60)
    if download_openuni_checkpoints():
        successful += 1
        print("✓ OpenUni checkpoints下载完成\n")
    else:
        failed.append("wusize/openuni (checkpoints)")

    # 打印总结
    print("\n" + "="*60)
    print("下载总结 / Download Summary")
    print("="*60)
    print(f"基础模型 / Base Models: {len(base_models)}")
    print(f"OpenUni Checkpoints: 6个文件")
    print(f"成功 / Successful: {successful}")
    print(f"失败 / Failed: {len(failed)}")

    if failed:
        print("\n失败的模型 / Failed models:")
        for model in failed:
            print(f"  - {model}")

    print("\n" + "="*60)
    print("下载完成! / Download completed!")
    print("="*60 + "\n")

    print("模型保存位置 / Models saved at:")
    for _, local_name in base_models:
        path = os.path.join(TARGET_DIR, local_name)
        if os.path.exists(path):
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} (未下载)")

    checkpoint_path = os.path.join(TARGET_DIR, "openuni-checkpoints")
    if os.path.exists(checkpoint_path):
        print(f"  ✓ {checkpoint_path}")

    print("\n接下来的步骤 / Next steps:")
    print("1. 创建虚拟环境并安装依赖")
    print("2. 运行推理脚本测试模型")
    print()

if __name__ == "__main__":
    main()
