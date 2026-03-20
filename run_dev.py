#!/usr/bin/env python3
"""
启动脚本 - 同时启动Flask后端和前端开发服务器
"""

import subprocess
import sys
import os
import time

def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.abspath(__file__))

def get_venv_python():
    """获取虚拟环境中的Python解释器路径"""
    project_root = get_project_root()
    if os.name == 'nt':  # Windows
        venv_python = os.path.join(project_root, "venv_web", "Scripts", "python.exe")
    else:  # Unix/Linux/Mac
        venv_python = os.path.join(project_root, "venv_web", "bin", "python")
    # 如果虚拟环境不存在，返回系统Python
    return venv_python if os.path.exists(venv_python) else sys.executable

def is_port_in_use(port):
    """检查端口是否被占用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def create_venv_if_needed():
    """创建虚拟环境（如果不存在）"""
    project_root = get_project_root()
    venv_path = os.path.join(project_root, "venv_web")

    if not os.path.exists(venv_path):
        print("创建虚拟环境...")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        print("[OK] 虚拟环境创建完成")

def install_python_deps():
    """安装Python依赖（如果需要）"""
    print("检查Python依赖...")
    print("=" * 50)

    # 确保虚拟环境存在
    create_venv_if_needed()

    requirements_file = os.path.join(get_project_root(), "requirements_web.txt")

    if not os.path.exists(requirements_file):
        print("警告: requirements_web.txt 不存在，跳过依赖检查")
        return True

    venv_python = get_venv_python()
    print(f"使用Python: {venv_python}")

    try:
        # 先升级pip
        subprocess.run(
            [venv_python, "-m", "pip", "install", "--upgrade", "pip", "-q"],
            capture_output=True,
            timeout=120
        )
        # 安装依赖
        result = subprocess.run(
            [venv_python, "-m", "pip", "install", "-r", requirements_file, "-q"],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )

        if result.returncode == 0:
            print("[OK] Python依赖安装完成")
            return True
        else:
            print(f"[FAIL] 依赖安装失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[FAIL] 依赖安装超时（超过5分钟）")
        return False
    except Exception as e:
        print(f"[FAIL] 依赖安装出错: {e}")
        return False

def start_flask():
    """启动Flask服务"""
    print("正在启动Flask后端服务...")
    print("=" * 50)

    venv_python = get_venv_python()

    env = os.environ.copy()
    project_root = get_project_root()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root};{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root

    process = subprocess.Popen(
        [venv_python, "app.py"],
        cwd=get_project_root(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 等待服务启动
    print("等待Flask服务启动...")
    for _ in range(60):  # 最多等30秒
        if is_port_in_use(5000):
            print("[OK] Flask服务已启动 (http://localhost:5000)")
            break
        time.sleep(0.5)
    else:
        print("[FAIL] Flask服务启动失败")
        process.terminate()
        return None

    return process

def start_frontend():
    """启动前端开发服务器"""
    print("\n正在启动前端开发服务器...")
    print("=" * 50)

    frontend_dir = os.path.join(get_project_root(), "frontend")

    # 安装依赖（如果需要）
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("首次运行，正在安装前端依赖...")
        install_result = subprocess.run(
            ["npm", "install"],
            cwd=frontend_dir,
            capture_output=True,
            text=True
        )
        if install_result.returncode != 0:
            print(f"[FAIL] 依赖安装失败: {install_result.stderr}")
            return None
        print("[OK] 依赖安装完成")

    # 启动开发服务器
    process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 等待服务启动
    print("等待前端服务启动...")
    for _ in range(60):
        if is_port_in_use(5173):
            print("[OK] 前端服务已启动 (http://localhost:5173)")
            break
        time.sleep(0.5)
    else:
        print("[FAIL] 前端服务启动失败")
        process.terminate()
        return None

    return process

def main():
    """主函数"""
    print("AIGC检测器 - 开发环境启动")
    print("=" * 50)
    print(f"项目目录: {get_project_root()}")
    print()

    processes = []

    try:
        # 安装Python依赖
        if not install_python_deps():
            print("[FAIL] 无法安装Python依赖，退出")
            sys.exit(1)

        # 启动Flask
        flask_process = start_flask()
        if flask_process:
            processes.append(flask_process)
        else:
            print("[FAIL] 无法启动Flask服务")
            sys.exit(1)

        # 启动前端
        frontend_process = start_frontend()
        if frontend_process:
            processes.append(frontend_process)
        else:
            print("[FAIL] 无法启动前端服务")
            sys.exit(1)

        print("\n" + "=" * 50)
        print("[OK] 所有服务已启动!")
        print("=" * 50)
        print("访问地址:")
        print("  - 前端: http://localhost:5173")
        print("  - 后端: http://localhost:5000")
        print("\n按 Ctrl+C 停止服务")

        # 保持运行
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n正在停止服务...")

    finally:
        for process in processes:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        print("[OK] 服务已停止")

if __name__ == "__main__":
    main()
