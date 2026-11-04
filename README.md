# HE Pose MTG

HE研究室の3人・90分MTG向け姿勢推定デモプロジェクト。

Macのカメラ映像をSSH先LinuxのRTX 3090で推論し、結果をMacに表示します。参加者は各人専用のGPUを使用します。

| モデル | 必須範囲 |
| --- | --- |
| YOLO26n-Pose | 全員が録画動画と遠隔ライブを実行 |
| DETRPose-N | 全員が同一の録画動画を実行 |
| TAR-ViTPose | 動作可能な代表環境でオフライン動画推論を実演 |

## 現在の状態

2026-11-04：確定仕様はv1.4。ユーザーの許可によりGitHubは公開で管理します。共通CLI・動画I/O・結果保存とYOLO26n-Pose／DETRPose-Nを実装し、abertのGPU 0（RTX 3090）で公式画像と同じ15秒・375フレームの運動動画を処理しました。必須6出力とフレーム対応を確認済みです。TAR-ViTPose、遠隔ライブ、ベンチマーク、比較動画、MTGガイドは未完了です。3人全員の実機受入確認も残っています。

MTG資料はGoogleドキュメントのような章立ての文書とし、Markdown原稿を管理します。作業先はLinuxホストabertの`/home/taku/he-pose-mtg`です。YOLOの再現条件・実行記録・制限は[検証記録](docs/pr-notes/yolo26-pose.md)を参照してください。

確定仕様は[docs/implementation-spec.md](docs/implementation-spec.md)です。[v1.2のレビュー記録](docs/pr-notes/spec-v1.2.md)と[MTG資料形式の変更記録](docs/pr-notes/mtg-document-format.md)に修正点と確認範囲をまとめています。実装開始時は[AGENTS.md](AGENTS.md)と[HANDOFF.md](HANDOFF.md)、[CODEX_START_PROMPT.md](CODEX_START_PROMPT.md)を読んでください。

## 開発方針

`chore/bootstrap` → モデル別ブランチ → `feat/remote-live` → `feat/benchmark` → `docs/mtg-materials`の順に、TDDで実装します。小さな単位でcommitし、GitHubへpushします。

重み・大容量動画・認証情報はGit管理対象外です。公式ベンチマークと自前の実測値を区別します。

最新のチャットの依頼範囲を優先します。共通基盤の次は、モデル別ブランチで公式モデルへの対応を進めます。通常実行はローカルに準備済みの重みを使い、不足時に自動取得しない方針です。

## 開発環境の準備

Python 3.11以上とuvを使います。以下はリポジトリ直下で実行します。パッケージ取得はこの準備工程で行い、通常のCLI実行では自動取得しません。

```bash
# Linux：GUI不要の動画I/Oと開発ツール
uv sync --frozen --extra video --extra dev

# Mac：表示用OpenCVを使う場合（videoとclientは同時に選択しない）
# uv sync --frozen --extra client --extra dev
```

採用版と配布ファイルのハッシュはuv.lock、ビルド用setuptoolsの版はpyproject.tomlに固定しています。モデルの依存パッケージと重みはこの環境に混在させず、モデル別環境へ準備します。

## 共通基盤の確認

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts integration_tests
.venv/bin/ruff format --check src tests scripts integration_tests
.venv/bin/python -m hepose.cli --help
.venv/bin/python -m hepose.cli doctor --role client --json
.venv/bin/python -m hepose.cli doctor --model all --device cuda:0 --json
```

既定のテストはtests/だけを収集し、学習済みモデル・重み・GPU・ネットワークを使いません。テスト用の短い映像とアダプタで動画I/Oを検証しており、実モデルの推論結果や性能ではありません。外部モデルが必要な試験はintegration_tests/へ分けます。

doctorはカメラを開かず、パッケージや重みを取得しません。不足や未実装があれば終了コード1です。clientの確認ではGUIのないOpenCVを表示可能とは扱いません。実際のカメラ許可・描画・SSHライブの成功は、後続工程で別途確認します。

## ローカル設定とモデル環境

hepose.example.tomlをhepose.local.tomlへコピーし、実際に準備したPython・重み・設定ファイルの場所へ編集します。既に設定がある場合は必要な項目を追記してください。各モデルは準備工程を実行するまで未取得です。hepose.local.tomlはGit管理外です。

各コマンドの--configで別のTOMLを指定でき、CLIの明示値を設定より優先します。相対パスは設定ファイルのある場所を基準に解釈し、モデル別Pythonは仮想環境のリンクを保持して起動します。doctor --model allは各環境を順番に確認します。準備したモデル環境にも同じ版のheposeを導入してください。

```bash
.venv/bin/python -m hepose.cli doctor --config hepose.local.toml --model yolo26 --json
.venv/bin/python -m hepose.cli run --config hepose.local.toml --model yolo26 \
  --input assets/sample/jumping-jacks-validation.mp4 --output outputs/yolo26 --device cuda:0
```

YOLO26／DETRPoseのrunは実装済みです。doctor --model allは未実装のTARも確認するため、現段階では終了コード1です。live・benchmark・compareはhelpのみの段階で、実行すると未実装を明示して終了します。

## YOLO26n-Poseの準備とGPU検証

Linux上でPython 3.11、uv、FFmpegと使用許可のあるGPUを用意します。準備時はインターネット接続が必要です。

```bash
# 固定版の専用環境、公式重み、公式画像を明示的に取得
python3 scripts/prepare_yolo26.py

# 出典・利用条件を記録した検証用動画を取得してCFRへ変換
python3 scripts/prepare_validation_video.py

# hepose.local.tomlのyolo26節を編集してから実行
.venv/bin/python -m hepose.cli doctor --model yolo26 --device cuda:0 --json
.venv-yolo26/bin/python integration_tests/yolo26_smoke.py \
  --video assets/sample/jumping-jacks-validation.mp4 --output outputs/yolo26-motion-check
```

専用環境の固定版はenvironments/yolo26/uv.lock、公式コミット・重みSHA256・ライセンスは[モデル記録](docs/model-manifests/yolo26.json)にあります。Ultralytics 8.4.202、PyTorch 2.14.1+cu126、torchvision 0.29.1+cu126で確認しました。入力は640×640、batch=1、FP32です。通常実行では公式重みのSHA256を照合し、不足・不一致や指定GPUの利用不可は理由付きの失敗にします。割当GPUのUUIDを設定すると、別GPUの誤使用も検出できます。

検証用動画はTaco fleur氏の[Jumping jacks and burpees](https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm)（CC BY-SA 4.0）の冒頭15秒です。[出典・加工内容](assets/sample/jumping-jacks-ATTRIBUTION.md)と[元・変換後の識別情報](docs/validation-assets/jumping-jacks.json)を残しています。骨格を重ねた動画を持ち出すときも出典・ライセンス・加工内容を添えてください。仕様のbasic／fast／multiという最終3素材はまだ揃っていません。

GPUスモークは画像の人物検出、動画の最後までの処理、元フレーム対応、出力動画の再復号を確認します。Pythonのインターネット接続呼出しも禁止・記録します。既定の30フレームは公式静止画の反復動画で、運動動画の検証には--videoを明示します。正解アノテーションがないため精度は未評価です。通常runの時間記録は残りますが、ウォームアップ20件＋本計測100件以上のモデル比較benchmarkは未実装です。

## DETRPose-Nの準備とGPU検証

```bash
python3 scripts/prepare_detrpose.py
# hepose.local.tomlにdetrpose節を設定してから実行
.venv/bin/python -m hepose.cli doctor --model detrpose --device cuda:0 --json
.venv/bin/python -m hepose.cli run --model detrpose \
  --input assets/sample/jumping-jacks-validation.mp4 --output outputs/detrpose --device cuda:0
# 公式画像APIとの座標照合と動画検証（通常pytestから分離）
.venv-detrpose/bin/python integration_tests/detrpose_smoke.py \
  --video assets/sample/jumping-jacks-validation.mp4 --output outputs/detrpose-motion-check
```

公式inference_onlyのコミット、Hugging Face上の重みの版・SHA256、Apache-2.0ライセンスは[モデル記録](docs/model-manifests/detrpose.json)に固定しています。専用環境の依存はenvironments/detrpose/uv.lockに記録します。公式配布のsafetensorsをローカルで厳密に読み込むため、読み込み箇所のみ[パッチ](patches/detrpose/local-safetensors.patch)を適用します。モデル計算・前処理・後処理は公式実装を使います。

DETRPose-Nでは人物の矩形、関節confidence、visibilityが得られないため共通JSONはnullです。人物scoreで0.5を超えるものを採用し、そのscoreを関節confidenceへ転用しません。入力サイズは固定設定に合わせ640×640です。公式画像APIと共通アダプタの座標の一致、運動動画375/375フレームの出力を確認しました。詳細は[検証記録](docs/pr-notes/detrpose.md)にあります。

## 動画と結果の契約

当初の動画入力は固定FPS、MP4出力の幅・高さは正の偶数を対象にします。奇数サイズを黙って切り落としません。読めない動画、不正なFPS、既知の総フレーム数に満たない復号終了、出力の復号検証失敗はエラーにします。可変FPSの正しい同期や、総フレーム数が不明な素材の全ての破損検出は保証しません。

正常終了時の出力はannotated.mp4、poses.jsonl、timings.csv、summary.json、environment.json、run.logです。元動画のフレーム順・FPSを保ち、時間窓の境界もNOT ESTIMATED（未推定）として記録・描画します。推定可能なフレームが一つもない入力は成功にしません。出力先は新しいディレクトリを指定し、既存の結果を上書きしません。

共通形式はschema_version=1.0。フレーム番号は0始まり、元時刻はframe_index/source_fps、時間の単位はmsです。関節は元画像上の名前付き座標で、対応しない点・取得できない矩形・不明な測定値はJSONでnull、CSVで空欄です。信頼度からvisibilityを推測しません。正常な0人検出・処理失敗・未推定を区別します。

environment.jsonには入力SHA256、ソースのGitコミットと未コミット変更の有無、依存版、要求デバイスとアダプタが報告した実使用デバイスを分けて保存します。モデル時間が取れないときはnullとし、アダプタ全体のホスト時間で代用しません。処理全体の計測には動画・結果の書込みと出力動画の復号検証を含み、ロード時間は別記録です。

環境と確認元は[環境記録](docs/environment-inventory.md)、Red/Greenと検証範囲は[共通基盤の記録](docs/pr-notes/bootstrap.md)にあります。GitHub ActionsではLinuxとmacOSで同じモデル不要のテスト、lint、CLI helpを確認します。各参加者の実機検証は別の受入条件です。
