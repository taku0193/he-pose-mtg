# TAR-ViTPose Base：公式再現と共通動画推論

2026-11-04。abertの許可済みGPU 0（RTX 3090）で公式TAR Baseと人物検出器YOLO11xを動かし、共通CLIへ接続した。遠隔ライブ・モデル比較benchmark・最終3素材・MTGガイド・3人分の受入は未完了。

## 変更と再現条件

src/hepose/tar_vitpose.py：3チェックポイントのSHA256、上流コミット、設定、依存版、GPU UUIDを確認する。連続5枚の中心を推定し、公式のクロップ前処理・モデル・ヒートマップの最大位置を使用して元画像座標へ戻す。nativeのhead_bottom／head_topを保持し、意味の異なるCOCOの目と、未使用の耳・取得不能なconfidence／visibilityをnullにする。

adapters.py／doctor.py／config.pyへ接続し、pretrainedとdetector_weightsのローカルパスを追加。scripts/prepare_tar_vitpose.pyが明示的に固定環境・コード・全重みと推論設定を準備する。通常のdoctor／runから取得しない。モデル不要の単体テストとintegration_tests/tar_vitpose_smoke.pyを分離した。

公式コードは42164c2c789a5297544dd836846e509459b0f1f3で、ソースパッチはない。READMEと実在設定のファイル名の違いを避け、configs/posetrack17/configTARViTPoseVitB.yamlから推論用設定を生成した。USE_MASK=trueは公式defaults、USE_FEATURE_FUSION=falseはローダーが要求するがモデルでは未使用のフィールド。ViTPose設定・初期重みのパスは準備済みローカルへ置き換えた。学習用の未使用フィールドは推論設定へ持ち込まない。

全チェックポイント・取得元・SHA256・設定差分・ライセンスと依存版は[モデル記録](../model-manifests/tar-vitpose.json)。TAR本体と初期重みは公式Drive、YOLO11xは公式Ultralytics v8.3.0配布を使用。Driveツールの256MiB上限を超えるため、公式公開ダウンロードから取得してサイズ・SHA256を確認した。重みはGit管理外。

Python 3.11.11、PyTorch 2.0.1+cu118／torchvision 0.15.2+cu118、MMPose 1.3.1、MMCV 2.1.0、MMEngine 0.10.4、NumPy 1.26.4を独立した.venv-tarへ導入。environments/tar-vitpose/uv.lockに全依存と配布物ハッシュを保存した。MMCVは公式cu118／torch2.0用cp311 wheel。MMEngineが必要とするpkg_resourcesを保持するためsetuptools 75.8.0を固定した。

## 公式実装と揃える点

入力は288×384、5フレーム・間隔1・未来2枚、人物のbboxを1.25倍に拡大して全5枚を同じ領域で切り出す。元フレームID・時刻・参照番号を保つ。冒頭・末尾2枚ずつは未推定として保存・描画し、境界を複製・補間しない。

YOLO11xのConv／BN融合は公式と同じCPU上で行い、初回predict時に指定GPUへ移す。FP32 bboxの加減算と整数クロップの丸めも公式と揃えた。先にGPUへ移して融合すると、わずかな数値差が整数境界を越えて1ピクセルのクロップ差になったため、公式の初期化順序を保つ。TARのforwardは公式と同じtorch.no_gradを使う。初回応答には検出器の遅延初期化・転送・内部warmupも含まれる。

旧PyTorchがdevice UUIDを公開しないため、runtime.pyでCUDAのプロセス内番号からPCI IDを取得し、CUDAドライバで同じPCIデバイスのUUIDを照合する。nvidia-smiの番号をプロセス内番号と同一視しない。別GPU／CPUへの自動切替はない。GPU実測でも許可済みUUIDとの一致を確認した。

MMPoseによる初期重みロードは未使用backbone.cls_tokenを報告する。最終TAR状態はstrict=Trueで全キー一致を必須とし、一致を確認した。PyTorchのTypedStorage非推奨と、nvrtc.soに関するcuDNN回避処理の警告も元ログに保持した。回避処理下で動画は正常終了したが、今回の記録は正式benchmarkではない。

## 計時と出力の範囲

inference_msは検出した各人物のTAR forwardをCUDAイベントで囲み、終了を待って合計した時間。人物検出・クロップ・描画・I/Oは含めず、adapter_msと全体時間に含める。人物が0人でTARを呼ばない場合のモデル時間はnull。起動・モデルロード・初回応答を分け、GPUメモリはこのプロセスのPyTorch allocatorのピークと明記する。

共通出力は6種類。公式CLIの短い動画・生JSONは別に保存し、元動画の出力通番＋2が対象フレームであることを記録する。公式の生JSONはクロップ内座標と固定visibilityのままなので、共通形式や関節confidenceとして扱わない。

## 再現コマンド

READMEの準備とhepose.local.tomlの設定後、abertの通常シェルでリポジトリ直下から実行する。既存結果と異なる出力先を選ぶ。

```bash
python3 scripts/prepare_tar_vitpose.py
.venv/bin/python -m hepose.cli doctor --model all --json
.venv-tar/bin/python integration_tests/tar_vitpose_smoke.py --output outputs/tar-motion-recheck
```


## 再実行差の扱い

同じヒートマップに対する公式extract_kpsと共通座標復元は厳密に照合する。別々のGPU実行の予測値は観測として記録する。初回の全件照合では3件のクロップ差が見つかり、CPU上での検出器融合とFP32の丸めを公式に揃えて解消した。その後もフレーム306の鼻に1ヒートマップセルの差が残ったため、同じモデル・同じ入力の再実行を確認した。再実行でも鼻のピークが1セル変わり、同じヒートマップのデコーダー間の差は約0.00001ピクセルだった。

公式のcuDNN決定性設定は維持し、PyTorch全体の決定性・cuBLAS設定・TF32の実際の値をenvironment.jsonへ追加する。独立したGPU実行の関節座標を完全一致とは扱わず、クロップの一致と座標変換の正しさ、予測値の再実行差を分ける。詳細原因とGTを使う精度は未評価。再現性の制限を除去したと主張しない。

## TDDと実機の証拠

| 記録 | 実際の結果 |
| --- | --- |
| work/tdd/16-tar-contracts-red.log | 新設モジュールがなくModuleNotFoundError |
| work/tdd/16-tar-contracts-green.log | 関節・座標・窓・3重みパスの初期8件成功 |
| work/tdd/17-tar-old-cuda-*.log | UUIDが得られないRedから新旧GPU確認の22件成功。実GPUの一致は17-tar-gpu-uuid-green.log |
| work/tdd/18-tar-official-crop-*.log | 関数不足のRedから、公式FP32丸めの境界を含む11件のGreen |
| work/tdd/19-tar-official-replay-*.log | 初回動画のフレーム218でクロップ不一致のRed。初期化・丸めを揃えた最終動画で全クロップ一致とI/OのGreen |

最終GPUソースは73ee1be347634192565c8fec6bd51c457a39d0a0、dirty=false。入力は15秒・640×480・25fps・375枚、SHA256=67d48a46e86fe66e8d096761f0c23098097855b2d04a94a8d88bf14d00d16642。共通出力は375枚を再復号し、371成功・失敗0・未推定4（元番号0・1・373・374）。316フレームで人物を検出し、残る55件は正常な0人検出。各成功フレームの前後2枚の参照番号を確認した。

公式CLIは371枚を再復号し、生JSONも保存した。最終実行の371中心フレームで人数・クロップが一致し、316人・4,740関節は原点を加えた公式座標と0.001ピクセル以内（最大約0.0000204）で一致した。この実行の一致を一般的な数値完全再現の保証にはしない。先行の別実行とフレーム306の再投入で観測した1セルのピーク差も保存した。

画像・動画・公式CLI中のPython socket.connect（IPv4／IPv6）の接続試行は0件。準備時の取得とSSHは範囲外で、OS全体の通信遮断試験ではない。全モデルdoctorはready=true。YOLOとDETRも公式画像と反復動画3フレームを同じGPUで回帰確認した。

モデル不要の単体テスト86件＋11 subtests、ruff、全CLI help、lock整合を確認した。[GitHub Actions](https://github.com/taku0193/he-pose-mtg/actions/runs/41822458294)ではソース73ee1beのLinux／macOSテスト・ruff・CLI helpが成功した。

原本はoutputs/tar-final-gpu/commonとupstream、機械可読の転記は[実験記録](../experiments/tar-vitpose-2026-11-04.json)。初回と途中の結果も上書きせず保持した。出典・加工・利用条件は[素材記録](../validation-assets/jumping-jacks.json)を参照する。最終3素材や3人分の受入は未完了。次はfeat/remote-liveでMacカメラ、SSHワーカー、同じフレームへの返送・描画、切断処理を実装する。
