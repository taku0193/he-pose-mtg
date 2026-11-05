# 環境セットアップと検証

[README](../README.md)から各モデルの準備、録画・ライブの記録と開発時のテストへ進めます。以下のコマンドはリポジトリ直下で実行します。

## 共通環境とローカル設定

`ssh abert` で接続済みのタブでは `cd /home/taku/he-pose-mtg` へ移動します。Macでの操作はMacの別タブから行います。Python 3.11とuvを用意します。Linuxは `video`、MacのGUIクライアントは `client` を選びます。開発時には `--extra dev` を追加できます。モデルの依存と重みはそれぞれの専用環境へ用意します。共通CLIの固定版と配布ファイルのハッシュは [uv.lock](../uv.lock)、ビルド用setuptoolsの版は [pyproject.toml](../pyproject.toml) を参照してください。

```bash
# Linux
uv sync --frozen --extra video

# Mac
uv sync --frozen --extra client
```

Linuxで [hepose.example.toml](../hepose.example.toml) を `hepose.local.toml` にコピーして、用意したモデルの `python`、`device`、`gpu_uuid`、重みと設定ファイルの場所を記入します。既存の設定がある場合は必要な項目を追記します。GPU UUIDは割当先を確認してから設定します。ライブとベンチマークにはGPU UUIDが必要です。

各コマンドの `--config` で別のTOMLを指定でき、CLIの明示値を設定より優先します。相対パスは設定ファイルのある場所を基準に解釈します。モデル別Pythonは仮想環境のリンクを保持して起動します。準備したモデル環境にもMacと同じコミットのheposeを導入してください。通常の実行にはローカルの重みを使い、不足時はエラーを表示します。

## YOLO26n-Poseの準備とGPU検証

LinuxでPython 3.11・uv・FFmpeg・割当GPUを用意します。初回の準備はインターネット接続が必要です。

```bash
python3 scripts/prepare_yolo26.py
# 検証素材がまだない場合に取得・変換
test -f assets/sample/jumping-jacks-validation.mp4 || python3 scripts/prepare_validation_video.py
# hepose.local.toml の [models.yolo26] を記入後
.venv/bin/python -m hepose.cli doctor --model yolo26 --json
.venv-yolo26/bin/python integration_tests/yolo26_smoke.py \
  --video assets/sample/jumping-jacks-validation.mp4 --output outputs/yolo26-motion-check
```

[モデル記録](model-manifests/yolo26.json)と [モデル環境のlock](../environments/yolo26/uv.lock) に、公式コミット・重みSHA256・ライセンス・採用版を保存しています。Ultralytics 8.4.202、PyTorch 2.14.1+cu126、torchvision 0.29.1+cu126、入力640×640、batch=1、FP32で検証しました。

GPUスモークでは人物検出、動画の全編処理、元フレーム対応、出力の再復号を確認します。既定の短い入力は公式静止画の反復で、運動動画を使う場合は `--video` を指定します。検証用素材の出典・加工・SHA256は [クレジット](../assets/sample/jumping-jacks-ATTRIBUTION.md) と [素材記録](validation-assets/jumping-jacks.json)、結果と制限は [YOLO26検証記録](pr-notes/yolo26-pose.md) を参照してください。

## DETRPose-Nの準備とGPU検証

```bash
python3 scripts/prepare_detrpose.py
# hepose.local.toml の [models.detrpose] を記入後
.venv/bin/python -m hepose.cli doctor --model detrpose --json
.venv/bin/python -m hepose.cli run --model detrpose \
  --input assets/sample/jumping-jacks-validation.mp4 --output outputs/detrpose-check
# 公式画像APIとの座標照合を含む試験
.venv-detrpose/bin/python integration_tests/detrpose_smoke.py \
  --video assets/sample/jumping-jacks-validation.mp4 --output outputs/detrpose-motion-check
```

公式推論コミット、重みの版・SHA256、Apache-2.0ライセンスは [モデル記録](model-manifests/detrpose.json) に固定しています。依存は [モデル環境のlock](../environments/detrpose/uv.lock) を使います。[ローカルsafetensors用パッチ](../patches/detrpose/local-safetensors.patch) は重みの読み込み箇所に適用します。

入力は640×640、人物scoreの閾値は0.5です。人物の矩形・関節confidence・visibilityは共通JSONでnullです。公式画像APIと座標の一致、運動動画375枚の出力を確認しました。[DETRPose検証記録](pr-notes/detrpose.md)

## TAR-ViTPoseの準備とGPU検証

TARはLinux x86_64・Python 3.11の独立環境を使います。公式Base重み、ViTPose初期重み、YOLO11x検出器を合わせたチェックポイントは約2.1GBです。

```bash
python3 scripts/prepare_tar_vitpose.py
# hepose.local.toml の [models.tar-vitpose] を記入後
.venv/bin/python -m hepose.cli doctor --model tar-vitpose --json
.venv/bin/python -m hepose.cli run --model tar-vitpose \
  --input assets/sample/jumping-jacks-validation.mp4 --output outputs/tar-check
# 実際の5枚窓・共通動画・公式CLIを検証
.venv-tar/bin/python integration_tests/tar_vitpose_smoke.py --output outputs/tar-motion-check
```

[公式コード・重み・SHA256・設定差分](model-manifests/tar-vitpose.json) と [モデル環境のlock](../environments/tar-vitpose/uv.lock) に採用版を保存しました。PyTorch 2.0.1+cu118、MMPose 1.3.1、MMCV 2.1.0を使います。公式設定の不足キーとローカルパスを補った設定を使い、`USE_MASK=true` を保持します。最終チェックポイントは全キーの一致を必須にします。

モデルに渡す人物クロップは288×384。中心と前後2枚を同じ領域で切り出します。共通出力は元画像座標を使い、head_bottom・head_topを保持します。共通COCOの目・耳と、取得できないconfidence・visibilityはnullです。

375枚中371枚に対して推定し、前後各2枚は `NOT ESTIMATED` として保存します。公式CLIの生JSONはクロップ内座標を使います。警告や上流との再現確認は [TAR検証記録](pr-notes/tar-vitpose.md) を参照してください。

## ライブ設定と記録

| 項目 | 既定 |
| --- | --- |
| カメラと要求サイズ | 0、640×480・25 FPS |
| 圧縮 | JPEG品質80 |
| 期限 | SSH接続5秒、起動45秒、応答15秒、セッション最大30分 |

Macでカメラへのアクセスを求められたら、起動に使うアプリへ許可を与えます。取得サイズと実際の取得FPSは `environment.json` で確認できます。設定の変更は [hepose.example.toml](../hepose.example.toml) の `[live]` を参照してください。

既定の出力は `outputs/live-<ID>/` 以下の `summary.json`、`environment.json`、`timings.csv`、`run.log` です。接続に失敗した場合は `run.log` を見ます。`--show-fps` で表示FPS、直前のアプリ内遅延、推論時間、待機画像の置き換え数を表示します。`--record` では描画した結果を固定FPSのMP4に保存します。実際の取得間隔はCSVで確認できます。

カメラの代わりに `--video INPUT.mp4 --headless` を指定すると、SSH通信と描画処理を検証できます。このモードの表示FPSと表示遅延は未測定です。[遠隔ライブ検証記録](pr-notes/remote-live.md)

## 比較と持ち帰り

`benchmark` は全モデルが推定可能な元フレームを揃え、20件をウォームアップして続く100件を計測します。計測件数が不足した場合は `insufficient_samples` と記録します。モデル別の全編出力、`benchmark.csv`、`report.md`、`environment.json` を保存します。TARは人物検出と前後の画像を使うため、レポートではオフライン処理として別枠で集計します。

`compare` には2種類以上のモデルの保存結果が必要です。比較対象ごとに1実行を指定します。元動画を移動した場合は `--input` で同じSHA256のファイルを指定できます。元の枚数・FPSを保ち、未推定や処理失敗を画面に表示します。

abertのタブで表示された出力先を控え、**Macの別タブ**へ切り替えます。`run_name` に、出力先の先頭の `outputs/` を除いたフォルダ名を入れます。

```bash
run_name=benchmark-YYYYMMDD-HHMMSS
mkdir -p outputs &&
scp -r "abert:/home/taku/he-pose-mtg/outputs/$run_name" outputs/ &&
open "outputs/$run_name/comparison.mp4"
```

## 動画と結果の契約

入力は固定FPSの動画、MP4出力の幅・高さは正の偶数を使います。読めない動画、不正なFPS、既知の総フレーム数に満たない復号終了、出力の復号検証失敗はエラーになります。出力先には新しいディレクトリを指定します。

正常終了時は `annotated.mp4`、`poses.jsonl`、`timings.csv`、`summary.json`、`environment.json`、`run.log` の6ファイルを保存します。フレームは0始まりで、元時刻は `frame_index / source_fps`、時間の単位はmsです。正常な0人検出・処理失敗・未推定を区別します。取得できない点や時間はJSONでnull、CSVで空欄です。環境記録は入力SHA256、ソースコミット、依存版、要求と実使用のGPUを含みます。[確定仕様](implementation-spec.md)

## テスト

```bash
uv sync --frozen --extra video --extra dev
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts integration_tests
.venv/bin/ruff format --check src tests scripts integration_tests
.venv/bin/python -m hepose.cli --help
```

既定の `tests/` はモデル・重み・GPU・ネットワークを使いません。実モデルの試験は [integration_tests/](../integration_tests/) に分けています。環境と確認元は [環境記録](environment-inventory.md)、共通基盤のRed/Greenと検証範囲は [共通基盤の記録](pr-notes/bootstrap.md) にあります。Mac実機のカメラと表示は [受入表](mtg/acceptance.md) で別途確認します。
