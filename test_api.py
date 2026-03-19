#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 用于测试AIGC检测API
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json

BASE_URL = "http://localhost:5000"

# 测试文本
TEST_TEXTS = {
    "ai": "人工智能技术的快速发展正在深刻改变我们的生活方式和工作模式。从智能手机到自动驾驶汽车，从智能家居到医疗诊断，AI技术已经渗透到各个领域。未来，随着算法的不断优化和计算能力的提升，人工智能将会在更多场景中发挥重要作用，为人类创造更美好的生活。",
    "human": "今天天气真好，我和爸爸妈妈一起去公园玩。公园里有许多花在开放，有红的、黄的、紫的，非常漂亮。小鸟在树上唱歌，蝴蝶在花丛中飞舞。我们在草地上野餐，还放了风筝。这是我最开心的一天。",
    "mixed": "人工智能（Artificial Intelligence），英文缩写为AI。它是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。AI领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。"
}

def test_detect(text, name):
    """测试单个文本检测"""
    print(f"\n{'='*50}")
    print(f"测试: {name}")
    print(f"{'='*50}")
    print(f"文本: {text[:50]}...")

    try:
        response = requests.post(
            f"{BASE_URL}/api/detect-full",
            json={"text": text, "mode": "original", "chunk_size": "original"},
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"\n[OK] 检测成功!")
                print(f"  总体AI概率: {data['overall_probability']:.2%}")
                print(f"  文本长度: {data['text_length']} 字符")
                print(f"  分块数量: {len(data['chunks'])}")
                print(f"\n  各段落检测结果:")
                for chunk in data["chunks"]:
                    print(f"    段落{chunk['index']}: {chunk['probability']:.2%} (长度: {chunk['text_length']})")
                return True
            else:
                print(f"[X] 检测失败: {data.get('error')}")
                return False
        else:
            print(f"[X] HTTP错误: {response.status_code}")
            return False
    except Exception as e:
        print(f"[X] 请求失败: {e}")
        return False

def test_upload():
    """测试文件上传"""
    print(f"\n{'='*50}")
    print(f"测试: 文件上传")
    print(f"{'='*50}")

    # 创建测试文件
    test_content = "这是一个测试文档。\n\n人工智能正在改变世界。"

    try:
        files = {'file': ('test.txt', test_content, 'text/plain')}
        response = requests.post(f"{BASE_URL}/api/upload", files=files)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"[OK] 文件上传成功!")
                print(f"  文件名: {data.get('filename')}")
                print(f"  内容长度: {len(data.get('text', ''))} 字符")
                return True
            else:
                print(f"[X] 上传失败: {data.get('error')}")
                return False
        else:
            print(f"[X] HTTP错误: {response.status_code}")
            return False
    except Exception as e:
        print(f"[X] 请求失败: {e}")
        return False

def main():
    """主函数"""
    print("AIGC检测API测试")
    print("=" * 50)
    print(f"API地址: {BASE_URL}")

    # 检查服务是否可用
    try:
        response = requests.get(BASE_URL)
        print(f"[OK] 服务可访问 (状态码: {response.status_code})")
    except Exception as e:
        print(f"[X] 无法连接到服务: {e}")
        print("请确保Flask服务正在运行: python app.py")
        sys.exit(1)

    # 运行测试
    results = []

    # 测试各类文本
    for name, text in TEST_TEXTS.items():
        results.append(test_detect(text, name))

    # 测试文件上传
    results.append(test_upload())

    # 总结
    print(f"\n{'='*50}")
    print("测试总结")
    print(f"{'='*50}")
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("[OK] 所有测试通过!")
    else:
        print("[X] 部分测试失败")

    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
