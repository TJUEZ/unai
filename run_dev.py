#!/usr/bin/env python3
"""
启动脚本 - 同时启动Flask后端和前端开发服务器
"""

import subprocess
import sys
import os
import time
import signal
import platform

def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.abspath(__file__))

def is_port_in_use(port):
    """检查端口是否被占用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_flask():
    """启动Flask服务"""
    print("正在启动Flask后端服务...")
    print("=" * 50)

    env = os.environ.copy()
    # 设置PYTHONPATH
    project_root = get_project_root()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root};{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root

    process = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=get_project_root(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 等待服务启动
    print("等待Flask服务启动...")
    for _ in range(30):
        if is_port_in_use(5000):
            print("✓ Flask服务已启动 (http://localhost:5000)")
            break
        time.sleep(0.5)
    else:
        print("✗ Flask服务启动失败")
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
            print(f"✗ 依赖安装失败: {install_result.stderr}")
            return None
        print("✓ 依赖安装完成")

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
    for _ in range(30):
        if is_port_in_use(5173):
            print("✓ 前端服务已启动 (http://localhost:5173)")
            break
        time.sleep(0.5)
    else:
        print("✗ 前端服务启动失败")
        process.terminate()
        return None

    return process

def main():
    """主函数"""
    print("AIGC检测器 - 开发环境启动")
    print("=" * 50)

    processes = []

    try:
        # 启动Flask
        flask_process = start_flask()
        if flask_process:
            processes.append(flask_process)
        else:
            print("无法启动Flask服务")
            sys.exit(1)

        # 启动前端
        frontend_process = start_frontend()
        if frontend_process:
            processes.append(frontend_process)
        else:
            print("无法启动前端服务")
            sys.exit(1)

        print("\n" + "=" * 50)
        print("✓ 所有服务已启动!")
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

        print("✓ 服务已停止")

if __name__ == "__main__":
    main()
