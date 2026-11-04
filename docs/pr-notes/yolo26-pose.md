# YOLO26n-Pose：画像・動画の実GPU検証

2026-11-04。公式のYOLO26n-Poseを共通runへ接続し、abertの許可済みGPU 0で画像と録画動画の実推論を確認した。画像・動画工程の検証であり、遠隔ライブや3人分の最終受入は未完了。

## 変更と再現条件

- src/hepose/yolo26.py：ローカル重みのSHA256照合、固定依存版、指定デバイスとGPU UUID確認、実推論、COCO17の元画像座標への変換。
- adapters.py／doctor.py／cli.py／config.py：遅延ロード、専用Python環境の振り分け、重み・依存・GPUの確認、input_size設定。
- pipeline.py：初回アダプタ応答時間と、処理後・解放前のGPUメモリ情報を保存。
- environments/yolo26：独立した固定依存環境と全配布物のハッシュ。
- scripts/prepare_yolo26.py：環境・公式重み・公式画像の明示的な準備。通常のrun／doctorでは取得しない。
- scripts/prepare_validation_video.pyとassets/sample/jumping-jacks-ATTRIBUTION.md：利用条件を確認した実動画の準備、変換、出典・識別情報。
- integration_tests/yolo26_smoke.py：明示的に起動するGPU試験。通常のpytest収集から分離。

Ultralytics 8.4.202、公式コミットae725d5cb20c2d849b864cd34fc15f5c55f877a5、PyTorch 2.14.1+cu126、torchvision 0.29.1+cu126。公式yolo26n-pose.ptのSHA256はeb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9。重み取得元・サイズ・AGPL-3.0ライセンスは[モデル記録](../model-manifests/yolo26.json)に保存した。上流パッチはない。

推論はRTX 3090、cuda:0、FP32、batch=1、640×640、rect=false、nms=false。GPU UUIDは環境記録と照合した。person confidence閾値は0.25、関節confidenceが0.5未満の座標はnullとし、元スコアは保持する。confidenceからvisibilityを作らず、追跡IDはnullとする。

## TDDと自動確認

| 記録 | Red | Green |
| --- | --- | --- |
| work/tdd/11-yolo26-*.log | 新設モジュールがなく収集時ModuleNotFoundError | 変換・null・人数・サイズ・重み・依存・GPU契約の12件成功 |
| work/tdd/12-yolo26-uuid-*.log | PyTorchのUUIDにGPU-接頭辞がなく割当と不一致 | 接頭辞を正規化し、全63件＋11 subtests成功 |
| work/tdd/13-first-response-*.log | first_adapter_call_msがなくKeyError | 初回応答を保存し、全63件＋11 subtests成功 |

現行単体テストは63件＋11 subtests、ruff check／formatと全CLI helpを確認する。tests/conftest.pyが各試験の作業場所を一時ディレクトリへ分離し、開発者のhepose.local.tomlから実モデル環境が起動しないようにした。core環境にはtorch／ultralyticsを導入していない。

コード2753c06の[GitHub Actions](https://github.com/taku0193/he-pose-mtg/actions/runs/41793664651)でLinux／macOSのモデル不要テスト、ruff、CLI helpが成功。最初のRedコミット0dee599のCI失敗は想定どおりモジュール不足であり、実装後に成功した。

## 実動作の証拠

| 入力 | 実際の結果 | 保存先 |
| --- | --- | --- |
| 公式bus.jpg | 4人検出、骨格画像・共通形式JSONを保存 | outputs/yolo26-motion-gpu/image-* |
| 同画像を反復した動画30フレーム | 30/30処理、6出力・再復号成功。運動動画ではない | outputs/yolo26-first-gpu |
| FP32指定修正後の反復動画3フレーム | 3/3処理、実パラメータdtype=float32を確認 | outputs/yolo26-fp32-gpu |
| 実際の運動動画15秒・25fps・375フレーム | 375/375処理、元番号0～374、出力動画375枚を再復号。358枚で人物検出、17枚は正常な0人検出 | outputs/yolo26-motion-gpu/video |

運動動画実行時のソースは2753c066b14104bde1acd2f90d927e2307d4ca68、dirty=false。入力SHA256は67d48a46e86fe66e8d096761f0c23098097855b2d04a94a8d88bf14d00d16642。実行日時・入力・版・計時と判定を[実験記録](../experiments/yolo26-2026-11-04.json)へ転記し、元のenvironment.json／summary.jsonも保持した。動画フレーム50の骨格位置を目視確認した。GTはなく、精度評価はN/A。

素材はTaco fleur氏の[Jumping jacks and burpees](https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm)、CC BY-SA 4.0。元動画の冒頭15秒を25fpsのH.264 MP4へ変換し、音声を除去。推論結果は骨格・フレーム表示を重ねた。元・変換後SHA256と加工内容は[素材記録](../validation-assets/jumping-jacks.json)を参照する。最終basic／fast／multiの3本が揃ったことにはしない。

GPU実行は既存の認証とホスト鍵確認を有効にしたSSH経由で同じabert上から行った。チャット側のサンドボックスで/dev/nvidia*が見えない状態と、ホスト側のGPU利用を区別した。Macのカメラ・通信・表示を確認した実績ではない。他のGPU・表示プロセスには触れていない。

## 再現手順

READMEの環境・重み・動画の準備を済ませ、hepose.local.tomlに自分の割当を設定する。以下はabertの通常シェルでリポジトリ直下から実行する。既存出力と異なるディレクトリ名を使う。

```bash
.venv/bin/python -m hepose.cli doctor --model yolo26 --device cuda:0 --json
.venv-yolo26/bin/python integration_tests/yolo26_smoke.py --video assets/sample/jumping-jacks-validation.mp4 --output outputs/yolo26-motion-recheck
```

スモーク中はPythonのsocket.socket.connect（IPv4／IPv6）を禁止して試行を記録する。画像・動画推論中の接続試行は0件だった。インターネット不要の根拠の範囲はこのプロセスであり、OS全体の通信遮断試験ではない。準備時の取得とSSH接続はこの範囲外。

Ultralyticsの非推奨half引数をquantize=32へ変更し、実際のパラメータdtypeを記録した。モデル時間は固定した公式実装の、forward前後にCUDA同期を行うProfile値。CPU変換等を含むadapter_msとは別であり、推論時間へ代用しない。ロードと初回応答は別記録。GPUメモリはこのプロセスのPyTorch allocatorのピークであり、デバイス全使用量ではない。

今回は通常runで全入力を保存し、公式predict内部の初回warmupだけを使用した。仕様の比較benchmarkに必要な20件warmup＋100件以上の本計測や、他モデルとの比較は未実施。今回の処理FPSから優劣・保証性能を主張しない。

## 残作業

YOLO工程をmainへ統合後、feat/detrposeで公式Nモデルの画像・動画対応へ進む。TAR、Mac遠隔ライブ、benchmark／compare、最終3素材、MTGガイドと3人それぞれの受入確認を残す。全体を完了扱いにしない。
