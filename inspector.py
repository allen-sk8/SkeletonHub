import numpy as np
import pickle
import sys
import os

#example: python inspector.py data/smpl_joints/samples_52j/walking_01_poses.npy

def inspect_file(file_path):
    """
    通用動作數據探針：支援 .npy 與 .pkl 檔案。
    印出維度、數值範圍與首幀資訊，幫助理解座標系與單位。
    """
    if not os.path.exists(file_path):
        print(f"❌ 錯誤：找不到檔案 {file_path}")
        return

    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.npy':
            data = np.load(file_path, allow_pickle=True)
        elif ext == '.pkl' or ext == '.npz':
            # 處理 .pkl 或 .npz 字典
            if ext == '.npz':
                data = np.load(file_path, allow_pickle=True)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
        else:
            print(f"❌ 錯誤：不支援的副檔名 {ext}")
            return
    except Exception as e:
        print(f"❌ 讀取時發生錯誤: {e}")
        return

    print("\n" + "="*60)
    print(f"🔍 探針報告: {os.path.basename(file_path)}")
    print("="*60)

    # 1. 如果是字典結構 (常見於 SMPL .pkl 或 AMASS .npz)
    if isinstance(data, (dict, np.lib.npyio.NpzFile)):
        print(f"【類型】: 字典 (Dictionary / Mapping)")
        keys = list(data.keys())
        print(f"【所有 Keys】: {keys}")
        print("-" * 30)
        for k in keys:
            val = data[k]
            if hasattr(val, 'shape'):
                print(f"  - 🔑 '{k}': Shape {val.shape}, Dtype {val.dtype}")
            elif isinstance(val, (list, tuple)):
                print(f"  - 🔑 '{k}': Length {len(val)}")
            else:
                print(f"  - 🔑 '{k}': {type(val)}")

    # 2. 如果是 Numpy 陣列 (常見於轉換後的 Joint 或 Feature)
    elif isinstance(data, np.ndarray):
        print(f"【類型】: Numpy Array")
        print(f"【維度 (Shape)】: {data.shape}")
        print(f"【型態 (Dtype)】: {data.dtype}")
        
        # 數值統計 (判斷單位公尺/毫米，或是否存在異常值)
        print("-" * 30)
        print(f"【數值統計】:")
        print(f"  - 最小值 (Min): {data.min():.4f}")
        print(f"  - 最大值 (Max): {data.max():.4f}")
        print(f"  - 平均值 (Mean): {data.mean():.4f}")
        
        # 內容首覽
        print("-" * 30)
        if len(data.shape) == 3: # (T, J, C)
            print(f"【首幀預覽 (First 3 Joints)】:")
            print(data[0, :3, :])
            print(f"\n💡 提示：若數值在 1.0 左右，單位可能為公尺(m)；若在 1000 左右則為毫米(mm)。")
        elif len(data.shape) == 2: # (T, D)
            print(f"【首幀特徵預覽 (First 10 dims)】:")
            print(data[0, :10])
            if data.shape[-1] == 263:
                print(f"\n💡 偵測到 263D 特徵：這很可能是 HumanML3D 格式。")
        elif len(data.shape) == 1:
            print(f"【內容預覽】: {data[:10]} ...")

    else:
        print(f"【類型】: {type(data)}")
        print(f"【內容】: {data}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("💡 使用方式: python utils/inspector.py <檔案路徑>")
    else:
        inspect_file(sys.argv[1])
