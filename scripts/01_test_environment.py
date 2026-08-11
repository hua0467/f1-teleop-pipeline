"""F1 遥操管线 — 环境检查。跑一遍确认所有依赖都在。"""
import sys

print("=" * 55)
print("  F1 VR 遥操管线 — 环境检查")
print("=" * 55)

ok, fail = 0, 0

def check(name, import_name=None, required=True):
    global ok, fail
    if import_name is None:
        import_name = name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        tag = "[OK]"
        if required:
            ok += 1
        else:
            tag = "[可选]"
        print(f"  {name:12s} {str(ver):12s} {tag}")
    except ImportError:
        if required:
            print(f"  {name:12s} {'未安装':12s} [缺少!]")
            fail += 1
        else:
            print(f"  {name:12s} {'未安装':12s} [可选]")

print(f"\nPython {sys.version.split()[0]}\n")
print("--- 必需 ---")
check("numpy")
check("scipy")
check("h5py")
check("pandas")
check("pyarrow")
check("fastapi")
check("uvicorn")

print("\n--- 可选（不影响录制，训练/画图才用）---")
check("matplotlib", required=False)
check("torch", required=False)
check("opencv", "cv2", required=False)

print("\n" + "=" * 55)
if fail:
    print(f"  {fail} 个必需包缺失，请先装好再开始。")
    print("  安装命令: python -m pip install numpy scipy h5py pandas pyarrow fastapi uvicorn --proxy \"\"")
else:
    print(f"  环境就绪，{ok} 个必需包全部到位。可以开始录数据了。")
print("=" * 55)
