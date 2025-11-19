#!/usr/bin/env python3
"""生成Kaggle笔记本的脚本"""
import json
import os

# Kaggle笔记本结构
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": []
}

# 单元格1: 标题说明
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🚀 VEC系统 - RSU计算资源对比实验 (Kaggle)\n",
        "\n",
        "## 📋 实验配置\n",
        "- **实验类型**: RSU计算资源敏感性分析\n",
        "- **训练轮次**: 1500 episodes\n",
        "- **随机种子**: 42\n",
        "- **预计时长**: 2-3小时 (P100 GPU)\n",
        "\n",
        "## ⚙️ 使用前准备\n",
        "1. 右侧设置面板选择 **GPU P100** 或 **GPU T4**\n",
        "2. 开启 **Internet** 连接\n",
        "3. 按顺序运行下面的单元格"
    ]
})

# 单元格2: 加载项目代码
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 📦 步骤1：加载项目代码"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 方法1：从GitHub克隆（推荐）\n",
        "import os\n",
        "import subprocess\n",
        "\n",
        "# 📌 修改这里：填入你的Git仓库地址\n",
        "GIT_REPO_URL = 'https://github.com/WeiY11/VEC_mig_caching.git'  # ← 修改为你的仓库\n",
        "\n",
        "# 切换到工作目录\n",
        "os.chdir('/kaggle/working')\n",
        "\n",
        "# 删除旧目录（如果存在）\n",
        "!rm -rf VEC_mig_caching\n",
        "\n",
        "print(f'📦 正在克隆: {GIT_REPO_URL}')\n",
        "result = subprocess.run(['git', 'clone', GIT_REPO_URL, 'VEC_mig_caching'], \n",
        "                       capture_output=True, text=True)\n",
        "\n",
        "if result.returncode != 0:\n",
        "    print('❌ 克隆失败！')\n",
        "    print('错误信息:', result.stderr)\n",
        "    print('\\n💡 可能的原因:')\n",
        "    print('1. 仓库是私有的 → 需要使用Token（见下方备用代码）')\n",
        "    print('2. 仓库地址错误 → 检查GIT_REPO_URL')\n",
        "    print('3. 网络问题 → 检查Kaggle的Internet设置是否开启')\n",
        "else:\n",
        "    os.chdir('VEC_mig_caching')\n",
        "    print(f'✅ 项目目录: {os.getcwd()}')\n",
        "    !ls -la | head -15"
    ]
})

# 备用：私有仓库克隆
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 方法2：克隆私有仓库（需要Token）\n",
        "# 如果上面的克隆失败，取消下面的注释\n",
        "\n",
        "# from getpass import getpass\n",
        "# import os\n",
        "# import subprocess\n",
        "# \n",
        "# os.chdir('/kaggle/working')\n",
        "# !rm -rf VEC_mig_caching\n",
        "# \n",
        "# GITHUB_USERNAME = 'WeiY11'\n",
        "# REPO_NAME = 'VEC_mig_caching'\n",
        "# \n",
        "# print('🔑 请输入GitHub Token:')\n",
        "# print('   获取地址: https://github.com/settings/tokens')\n",
        "# print('   需要权限: repo (Full control of private repositories)')\n",
        "# TOKEN = getpass('Token: ')\n",
        "# \n",
        "# repo_url = f'https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git'\n",
        "# result = subprocess.run(['git', 'clone', repo_url, 'VEC_mig_caching'],\n",
        "#                        capture_output=True, text=True)\n",
        "# \n",
        "# if result.returncode == 0:\n",
        "#     os.chdir('VEC_mig_caching')\n",
        "#     print(f'✅ 项目目录: {os.getcwd()}')\n",
        "#     !ls -la | head -15\n",
        "# else:\n",
        "#     print('❌ 克隆失败:', result.stderr)"
    ]
})

# 备用：Dataset方式
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 方法2：从Kaggle Dataset加载（如果你上传了Dataset）\n",
        "# 取消下面的注释来使用\n",
        "\n",
        "# import shutil\n",
        "# import os\n",
        "# \n",
        "# dataset_path = '/kaggle/input/vec-migration-caching'  # ← 修改为Dataset名称\n",
        "# work_path = '/kaggle/working/VEC_mig_caching'\n",
        "# \n",
        "# if os.path.exists(work_path):\n",
        "#     shutil.rmtree(work_path)\n",
        "# shutil.copytree(dataset_path, work_path)\n",
        "# os.chdir(work_path)\n",
        "# print(f'✅ 项目加载完成: {os.getcwd()}')"
    ]
})

# 单元格3：安装依赖
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 🔧 步骤2：安装依赖"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 安装依赖\n",
        "!pip install flask-socketio pyyaml -q\n",
        "\n",
        "# 创建目录\n",
        "!mkdir -p results/td3_strategy_suite academic_figures\n",
        "\n",
        "print('✅ 依赖安装完成')"
    ]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 验证GPU\n",
        "import torch\n",
        "print(f'PyTorch: {torch.__version__}')\n",
        "print(f'CUDA: {torch.cuda.is_available()}')\n",
        "if torch.cuda.is_available():\n",
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
    ]
})

# 单元格4：运行实验
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 🎯 步骤3：运行RSU计算资源实验"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 🚀 运行完整实验（1500轮）\n",
        "!python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py \\\n",
        "    --experiment-types rsu_compute \\\n",
        "    --rsu-compute-levels default \\\n",
        "    --episodes 1500 \\\n",
        "    --seed 42"
    ]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 💡 快速验证模式（500轮，仅用于测试）\n",
        "# !python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py \\\n",
        "#     --experiment-types rsu_compute \\\n",
        "#     --rsu-compute-levels default \\\n",
        "#     --episodes 500 \\\n",
        "#     --seed 42"
    ]
})

# 单元格5：查看结果
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 📊 步骤4：查看结果"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 查看结果文件\n",
        "print('📁 实验结果:')\n",
        "!ls -lh results/td3_strategy_suite/ | grep rsu_compute\n",
        "\n",
        "print('\\n📊 生成图表:')\n",
        "!ls -lh academic_figures/ | tail -10"
    ]
})

# 单元格6：保存结果
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 💾 步骤5：保存结果"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 打包结果\n",
        "from datetime import datetime\n",
        "import shutil\n",
        "\n",
        "timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n",
        "result_zip = f'rsu_results_{timestamp}'\n",
        "figure_zip = f'rsu_figures_{timestamp}'\n",
        "\n",
        "shutil.make_archive(result_zip, 'zip', 'results/td3_strategy_suite')\n",
        "shutil.make_archive(figure_zip, 'zip', 'academic_figures')\n",
        "\n",
        "print(f'✅ 结果已打包：')\n",
        "print(f'   {result_zip}.zip')\n",
        "print(f'   {figure_zip}.zip')\n",
        "print('\\n📂 可在Kaggle Output中下载')"
    ]
})

# 保存笔记本
output_dir = os.path.dirname(__file__)
kaggle_output = os.path.join(os.path.dirname(output_dir), 'kaggle', 'VEC_RSU_Compute_Kaggle.ipynb')
with open(kaggle_output, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print(f"✅ Kaggle笔记本已生成: {kaggle_output}")

# Colab笔记本结构
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": []
}

# 单元格1: 标题说明
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {"id": "header"},
    "source": [
        "# 🚀 VEC系统 - RSU计算资源对比实验\n",
        "\n",
        "## 📋 实验配置\n",
        "- **实验类型**: RSU计算资源敏感性分析\n",
        "- **训练轮次**: 1500 episodes\n",
        "- **随机种子**: 42\n",
        "- **预计时长**: 2-3小时（T4 GPU）\n",
        "\n",
        "## ⚙️ 使用前准备\n",
        "1. 菜单栏: **代码执行程序** → **更改运行时类型** → 选择 **T4 GPU**\n",
        "2. 依次运行下面的单元格"
    ]
})

# 单元格2: 检查GPU并挂载Drive
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "setup_gpu"},
    "outputs": [],
    "source": [
        "# 检查GPU\n",
        "import torch\n",
        "print(f'PyTorch: {torch.__version__}')\n",
        "print(f'CUDA: {torch.cuda.is_available()}')\n",
        "if torch.cuda.is_available():\n",
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
        "else:\n",
        "    print('⚠️ 请在菜单栏选择GPU运行时！')\n",
        "\n",
        "# 挂载Google Drive\n",
        "from google.colab import drive\n",
        "import os\n",
        "drive.mount('/content/drive')\n",
        "save_dir = '/content/drive/MyDrive/VEC_results'\n",
        "os.makedirs(save_dir, exist_ok=True)\n",
        "print(f'✅ 结果保存目录: {save_dir}')"
    ]
})

# 单元格3: 克隆Git仓库
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "clone_project"},
    "outputs": [],
    "source": [
        "# 方法1：从GitHub克隆项目（推荐）\n",
        "import os\n",
        "\n",
        "# 📌 修改这里：填入你的Git仓库地址\n",
        "GIT_REPO_URL = 'https://github.com/YOUR_USERNAME/VEC_mig_caching.git'  # ← 修改为你的仓库地址\n",
        "\n",
        "print(f'📦 正在克隆仓库: {GIT_REPO_URL}')\n",
        "!git clone {GIT_REPO_URL} /content/VEC_mig_caching\n",
        "\n",
        "# 进入项目目录\n",
        "os.chdir('/content/VEC_mig_caching')\n",
        "print(f'✅ 项目目录: {os.getcwd()}')\n",
        "\n",
        "# 查看目录结构\n",
        "!ls -la | head -15"
    ]
})

# 单元格3备用: 上传ZIP（备选方案）
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "upload_project_alternative"},
    "outputs": [],
    "source": [
        "# 方法2：上传ZIP文件（如果没有Git仓库，使用此方法）\n",
        "# 取消下面的注释来使用\n",
        "\n",
        "# from google.colab import files\n",
        "# import zipfile\n",
        "# import shutil\n",
        "# \n",
        "# print('📤 请选择VEC_mig_caching.zip文件上传...')\n",
        "# uploaded = files.upload()\n",
        "# \n",
        "# zip_name = list(uploaded.keys())[0]\n",
        "# with zipfile.ZipFile(zip_name, 'r') as z:\n",
        "#     z.extractall('/content')\n",
        "# \n",
        "# project_dir = '/content/VEC_mig_caching'\n",
        "# if not os.path.exists(project_dir):\n",
        "#     for item in os.listdir('/content'):\n",
        "#         if 'VEC' in item and os.path.isdir(f'/content/{item}'):\n",
        "#             shutil.move(f'/content/{item}', project_dir)\n",
        "#             break\n",
        "# \n",
        "# os.chdir(project_dir)\n",
        "# print(f'✅ 项目目录: {os.getcwd()}')\n",
        "# !ls -la | head -15"
    ]
})

# 单元格4: 安装依赖
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "install_deps"},
    "outputs": [],
    "source": [
        "# 安装依赖\n",
        "!pip install flask-socketio pyyaml -q\n",
        "\n",
        "# 创建目录\n",
        "!mkdir -p results/td3_strategy_suite academic_figures logs\n",
        "\n",
        "# 验证文件\n",
        "print('✅ 依赖安装完成\\n')\n",
        "print('📂 关键文件检查:')\n",
        "files_check = [\n",
        "    'experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py',\n",
        "    'requirements.txt',\n",
        "    'config/system_config.py'\n",
        "]\n",
        "for f in files_check:\n",
        "    print(f\"{'✅' if os.path.exists(f) else '❌'} {f}\")"
    ]
})

# 单元格5: 运行实验
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "run_experiment"},
    "outputs": [],
    "source": [
        "# 🚀 运行RSU计算资源对比实验（1500轮）\n",
        "!python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py \\\n",
        "    --experiment-types rsu_compute \\\n",
        "    --rsu-compute-levels default \\\n",
        "    --episodes 1500 \\\n",
        "    --seed 42"
    ]
})

# 单元格6: 快速验证模式（注释掉）
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "fast_mode"},
    "outputs": [],
    "source": [
        "# 💡 快速验证模式（500轮，仅用于测试）\n",
        "# !python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py \\\n",
        "#     --experiment-types rsu_compute \\\n",
        "#     --rsu-compute-levels default \\\n",
        "#     --episodes 500 \\\n",
        "#     --seed 42"
    ]
})

# 单元格7: 查看结果
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "view_results"},
    "outputs": [],
    "source": [
        "# 查看结果文件\n",
        "print('📁 实验结果:')\n",
        "!ls -lh results/td3_strategy_suite/ | grep rsu_compute\n",
        "\n",
        "print('\\n📊 生成图表:')\n",
        "!ls -lh academic_figures/ | tail -10\n",
        "\n",
        "# 显示图表\n",
        "from IPython.display import Image, display\n",
        "import glob\n",
        "\n",
        "figures = sorted(glob.glob('academic_figures/*rsu_compute*.png'))\n",
        "if figures:\n",
        "    print(f'\\n找到 {len(figures)} 张图表')\n",
        "    for fig in figures[-3:]:\n",
        "        print(f'\\n📈 {os.path.basename(fig)}')\n",
        "        display(Image(filename=fig))\n",
        "else:\n",
        "    print('⚠️ 未找到图表')"
    ]
})

# 单元格8: 显示指标摘要
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "show_metrics"},
    "outputs": [],
    "source": [
        "# 显示关键指标\n",
        "import json\n",
        "import pandas as pd\n",
        "import glob\n",
        "\n",
        "summaries = sorted(glob.glob('results/td3_strategy_suite/*summary*.json'))\n",
        "if summaries:\n",
        "    with open(summaries[-1], 'r', encoding='utf-8') as f:\n",
        "        data = json.load(f)\n",
        "    \n",
        "    if 'rsu_compute' in data:\n",
        "        print('🎯 RSU计算资源对比结果\\n' + '='*80)\n",
        "        for level, strategies in data['rsu_compute'].items():\n",
        "            print(f'\\n📌 {level}\\n' + '-'*80)\n",
        "            rows = []\n",
        "            for strat, metrics in strategies.items():\n",
        "                rows.append({\n",
        "                    '策略': strat,\n",
        "                    '时延(s)': f\"{metrics.get('avg_delay', 0):.4f}\",\n",
        "                    '能耗(J)': f\"{metrics.get('avg_energy_consumption', 0):.2f}\",\n",
        "                    '吞吐量': f\"{metrics.get('avg_throughput_mbps', 0):.2f}\",\n",
        "                    'RSU利用率': f\"{metrics.get('avg_rsu_utilization', 0):.2%}\"\n",
        "                })\n",
        "            print(pd.DataFrame(rows).to_string(index=False))\n",
        "    else:\n",
        "        print('⚠️ 未找到RSU实验数据')\n",
        "else:\n",
        "    print('⚠️ 未找到摘要文件')"
    ]
})

# 单元格9: 保存结果
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "save_results"},
    "outputs": [],
    "source": [
        "# 打包并保存结果\n",
        "from datetime import datetime\n",
        "from google.colab import files\n",
        "\n",
        "timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n",
        "result_zip = f'rsu_results_{timestamp}'\n",
        "figure_zip = f'rsu_figures_{timestamp}'\n",
        "\n",
        "!zip -r {result_zip}.zip results/td3_strategy_suite/\n",
        "!zip -r {figure_zip}.zip academic_figures/\n",
        "\n",
        "print(f'✅ 已打包:')\n",
        "print(f'   {result_zip}.zip')\n",
        "print(f'   {figure_zip}.zip')\n",
        "\n",
        "# 保存到Drive\n",
        "if os.path.exists('/content/drive/MyDrive/VEC_results'):\n",
        "    !cp {result_zip}.zip /content/drive/MyDrive/VEC_results/\n",
        "    !cp {figure_zip}.zip /content/drive/MyDrive/VEC_results/\n",
        "    print('\\n📤 已保存到Google Drive')\n",
        "\n",
        "# 下载到本地\n",
        "print('\\n📥 开始下载到本地...')\n",
        "files.download(f'{result_zip}.zip')\n",
        "files.download(f'{figure_zip}.zip')"
    ]
})

# 保存笔记本
output_path = os.path.join(os.path.dirname(__file__), 'VEC_RSU_Compute_Experiment.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print(f"✅ Colab笔记本已生成: {output_path}")
