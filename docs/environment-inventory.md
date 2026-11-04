# 実装開始時の環境記録

確認日：2026-11-04（Asia/Tokyo）。

| 項目 | 確認内容 |
| --- | --- |
| ホスト・ユーザー | abert / taku（ローカルコマンドで確認） |
| リポジトリ | /home/taku/he-pose-mtg |
| OS | Ubuntu 22.04.5 LTS / x86_64 |
| Python | 3.11.11（既存のpyenv環境） |
| uv | 0.11.15 |
| FFmpeg | 4.4.2-0ubuntu0.22.04.1 |
| GPU | GPU 0 / NVIDIA GeForce RTX 3090 / VRAM 24576 MiB |
| GPU UUID | GPU-25e750aa-24f4-8a0d-edb2-f2f948526eec（/procで確認） |
| GPU使用許可 | ユーザーがabertのGPU使用を明示。GPU 0を対象とする |
| ドライバ | 580.178.04 |
| nvidia-smiのCUDA表示 | 13.0。Python側のCUDAランタイム導入済みを意味しない |
| ストレージ | 調査時の空き約279 GiB |

## 確認元の区別

GPU型番・UUIDとカーネルドライバは、このチャットのコマンドから/proc・PCI情報で確認した。ユーザーが提示した2026-11-04 10:59:37の端末出力では、nvidia-smiが正常動作し、GPU 0の型番・VRAM・ドライバを確認できる。提示時点でGPU使用率は0%、VRAM使用量は205 MiBだったが、常時の空き状況を保証する値ではない。

このチャットのコマンド実行環境では/dev/nvidia*が見えず、nvidia-smiのドライバ通信が失敗した。ホスト自体のGPU故障や未接続とは扱わない。GPU推論はモデル環境を用意した後に確認する。

## 未実施

PyTorch・各モデル環境・重み取得・GPU推論・MacからのSSHライブ・Macカメラ・性能測定は未実施。開始時のPythonにはpytest・ruff・OpenCV・NumPy・PyTorchが入っていなかった。ユーザーの表示用プロセス等は停止しない。
