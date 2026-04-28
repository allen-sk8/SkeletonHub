import numpy as np
import os
import pickle
import argparse
import sys

# 將專案路徑加入以匯入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def convert_amass_to_smpl(src_path, save_path, target_fps=20):
    """
    將 AMASS 的 .npz 檔案轉換為 SkeletonHub 標準的 SMPL .pkl 格式
    並進行下採樣至 target_fps (預設 20 FPS)
    """
    if not os.path.exists(src_path):
        print(f"❌ 找不到檔案: {src_path}")
        return

    # AMASS 資料通常較大，使用 allow_pickle=True
    try:
        bdata = np.load(src_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ 讀取 .npz 失敗: {e}")
        return

    # 取得原始影格率
    try:
        fps = float(bdata['mocap_framerate'])
    except:
        # 有些檔案可能沒有這個欄位，嘗試從其他地方獲取或報錯
        print(f"⚠️ 警告: 無法從 {src_path} 讀取 FPS，嘗試預設 120 FPS")
        fps = 120.0

    # 計算下採樣倍率
    down_sample = max(1, int(fps / target_fps))
    
    # 提取核心 SMPL 參數
    # 注意：AMASS 的 poses 通常包含 156 維 (SMPL+H)
    try:
        data = {
            'poses': bdata['poses'][::down_sample, ...].astype(np.float32),
            'trans': bdata['trans'][::down_sample, ...].astype(np.float32),
            'betas': bdata['betas'][:16].astype(np.float32), # AMASS betas 通常是 16 維
            'gender': str(bdata['gender']),
            'mocap_framerate': fps,
            'target_fps': target_fps,
            'source_path': os.path.abspath(src_path)
        }
    except KeyError as e:
        print(f"❌ 檔案格式不符，缺少必要欄位: {e}")
        return

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 以 Pickle 格式儲存
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ 轉換成功: {os.path.basename(src_path)}")
    print(f"   - 原始影格數: {len(bdata['poses'])} (@{fps} FPS)")
    print(f"   - 轉換後影格數: {len(data['poses'])} (@{target_fps} FPS)")
    print(f"   - 儲存路徑: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMASS .npz to SMPL .pkl 轉換器")
    parser.add_argument("input", help="輸入的 AMASS .npz 路徑")
    parser.add_argument("--output", help="輸出的 .pkl 路徑 (預設存入 data/amass/)")
    parser.add_argument("--fps", type=int, default=20, help="目標影格率 (預設 20)")
    
    args = parser.parse_args()
    
    output = args.output
    if not output:
        # 預設路徑處理：存入 data/smpl/smplh/
        default_dir = os.path.join(os.getcwd(), "data", "smpl", "smplh")
        os.makedirs(default_dir, exist_ok=True)
        base_name = os.path.basename(args.input).replace(".npz", ".pkl")
        output = os.path.join(default_dir, base_name)
    
    convert_amass_to_smpl(args.input, output, target_fps=args.fps)
