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

このチャットのコマンド実行環境では/dev/nvidia*が見えず、nvidia-smiのドライバ通信が失敗した。ホスト自体のGPU故障や未接続とは扱わない。その後、既存のホスト鍵確認と認証を利用したSSH接続から、同じabert上でPyTorchのCUDA利用と実推論を確認した。Macからのライブ経路の確認とは異なる。

## 調査開始時点

開始時のPythonにはpytest・ruff・OpenCV・NumPy・PyTorchが入っていなかった。共通基盤とYOLO専用環境をプロジェクト内へ準備した。ユーザーの表示用プロセス等は停止していない。


## 共通基盤用の仮想環境

プロジェクト内の.venvへvideo・dev依存を導入した。採用版はNumPy 2.4.6、opencv-python-headless 4.14.0.94、pytest 9.2.0、ruff 0.17.3。ビルド用setuptoolsは84.2.1。Python・uvは上表の版を使用した。配布物のハッシュと他の対応Python／プラットフォーム向けの解決結果はuv.lockに記録する。

この環境にPyTorch・Ultralytics等のモデル依存は導入していない。現在のモデル不要テストは63件＋11 subtests。Macのカメラ・表示・SSHライブの受入とは区別する。

## YOLO26専用環境と実GPU検証

| 項目 | 実際に確認した値 |
| --- | --- |
| 環境 | /home/taku/he-pose-mtg/.venv-yolo26 |
| Python | 3.11.11 |
| Ultralytics | 8.4.202 |
| PyTorch / torchvision | 2.14.1+cu126 / 0.29.1+cu126 |
| NumPy / OpenCV | 2.4.6 / opencv-python 4.14.0.94 |
| PyTorch CUDAランタイム | 12.6。nvidia-smiの13.0はドライバの対応上限表示 |
| 実使用GPU | cuda:0、RTX 3090、UUIDは上表と一致 |
| 推論条件 | YOLO26n-Pose、PyTorch、FP32、batch=1、640×640、rect=false、nms=false |
| 導入固定情報 | environments/yolo26/pyproject.tomlとuv.lock |
| 上流・重み・ライセンス | docs/model-manifests/yolo26.json、パッチなし |
| 実動画の証拠 | outputs/yolo26-motion-gpu/video/environment.jsonとsummary.json |

公式画像で4人、公開運動動画375フレームの処理・出力を確認した。GPUピークメモリはこのプロセスのPyTorch allocatorの観測値であり、nvidia-smiの全使用量ではない。GPUを変えず、他のプロセスを停止せず実行した。

## DETRPose専用環境と実GPU検証

.venv-detrposeへ公式inference_onlyのda12e0cを導入した。torch 2.14.1+cu126、torchvision 0.29.1+cu126、safetensors 0.8.0、huggingface-hub 2.4.0。PyTorch／NumPy／OpenCVはYOLOと同じ版だが仮想環境は分離する。固定依存とbuild制約はenvironments/detrpose/pyproject.tomlとuv.lock。

同じGPU 0・UUIDでFP32／640×640／batch=1の実推論が成功。公式画像APIと共通アダプタの座標一致と、375/375フレームの録画動画・6出力・再復号を確認した。証拠はoutputs/detrpose-motion-gpu、docs/experiments/detrpose-2026-11-04.json。コード・重み・パッチ・ライセンスはdocs/model-manifests/detrpose.json。

## TAR専用環境と実GPU検証

.venv-tarへPython 3.11.11、torch 2.0.1+cu118、torchvision 0.15.2+cu118、NumPy 1.26.4、OpenCV 4.9.0.80、MMCV 2.1.0、MMEngine 0.10.4、MMPose 1.3.1、MMPretrain 1.2.0を準備。MMCVは公式cu118／torch2.0のcp311 wheelを使用した。pkg_resourcesを必要とするMMEngine向けにsetuptools 75.8.0を固定した。完全な依存とハッシュはenvironments/tar-vitpose/uv.lock。

旧PyTorchではプロパティからGPU UUIDを得られないため、CUDA runtimeのPCI IDとドライバUUIDで同じプロセス内デバイスを確認した。実際にGPU 0の割当UUIDと一致した。CUDA 11.8でRTX 3090のTAR BaseとYOLO11xを推論でき、3モデルのdoctorもready=true。

最終実行のソース73ee1be、dirty=false。共通375枚（371成功＋境界4枚）、公式CLI371枚を保存・再復号した。人数・クロップと4,740関節の座標を照合し、当該実行の最大座標差は約0.0000204ピクセルだった。先行実行で同一入力の鼻に1セルのピーク差も観測したため、数値完全再現は保証しない。cuDNN決定性、PyTorch全体の決定性、cuBLAS設定、TF32の実際の値をenvironment.jsonに記録する。

MMPose初期重みの未使用backbone.cls_token、PyTorchのTypedStorage非推奨とnvrtc.soのcuDNN回避処理の警告はログに保持した。最終TAR状態はstrict=Trueで全キー一致。ソースパッチはなく、設定の不足キーとローカルパスだけを補った。詳細はdocs/pr-notes/tar-vitpose.mdとdocs/model-manifests/tar-vitpose.json。

Macカメラ・遠隔ライブ、正式benchmark、3人分の実機受入は未実施。TARブランチのLinux／macOS CIは73ee1beで成功した。
