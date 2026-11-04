# 仕様v1.2の確定レビュー

日付：2026-11-04。ブランチ：docs/spec-v1.2。

## 目的と対象

ユーザーの「それらを全部確認して，仕様書を確定させましょう」に基づく文書レビュー。対象はAGENTS.md、README.md、CODEX_START_PROMPT.md、HANDOFF.md、docs/implementation-spec.mdの5文書。背景原稿v1.0は現リポジトリにないため未照合。

## 変更

- 既存の3人・3モデル・Macカメラ／Linux推論／Mac表示・90分・必須資料を維持し、仕様をv1.2へ更新した。
- push許可の文書内の食い違いを修正し、開始プロンプトと最新の依頼範囲の関係を明確にした。
- モデル別環境と共通CLI、doctorのclient／worker、通常実行での自動ダウンロード抑止を明記した。
- 出力先の保護、失敗・中断、座標・時刻・欠測、TARの境界、比較対象の識別、benchmarkの測定区間を具体化した。
- 必須機能の合格条件と残る環境・運営項目を決める時点を一覧にした。

## 一次情報の再確認

以下は閲覧だけの静的確認であり、インストール・重み取得・GPU推論の実行ではない。

- [Ultralytics公式pose資料](https://docs.ultralytics.com/tasks/pose/)：YOLO26n-poseとCOCO17、公式性能値の測定条件。
- [DETRPose推論版README](https://github.com/SebastianJanampa/DETRPose/blob/da12e0cd29864b1d3a289c562c34d423371a4157/README.md)：Nモデル、公式値のTensorRT／V100条件、ローカル重み指定と既定の自動取得。
- TARの[README](https://github.com/zgspose/TARViTPose/blob/42164c2c789a5297544dd836846e509459b0f1f3/README.md)、[推論コード](https://github.com/zgspose/TARViTPose/blob/42164c2c789a5297544dd836846e509459b0f1f3/tools/inference.py)、[設定](https://github.com/zgspose/TARViTPose/blob/42164c2c789a5297544dd836846e509459b0f1f3/configs/posetrack17/configTARViTPoseVitB.yaml)、[関節定義](https://github.com/zgspose/TARViTPose/blob/42164c2c789a5297544dd836846e509459b0f1f3/datasets/zoo/posetrack/pose_skeleton.py)：5フレーム窓、クロップ内JSON座標、固定可視性、頭部関節の意味、設定パスと参照キーの不整合。
- [TAR論文](https://arxiv.org/abs/2603.05929)：時間情報を用いた設計とPoseTrack2017の論文結果。
- [NVIDIA CUDA 10.2リリースノート](https://docs.nvidia.com/cuda/archive/10.2/cuda-toolkit-release-notes/)：macOSサポート終了に関する既存記述。

## 確認結果と制限

- 開始時のgit status、branch、remote、直近5件以内の履歴を確認。main・ab821b9、作業ツリーに変更なし。originは専用he-pose-mtgリポジトリ。
- ローカルのhostname・ユーザー・作業パスを確認。MacからのSSH接続やGPU割当の確認とは区別した。
- 6文書のv1.2表記、内部リンク5件の参照先、Markdownコードブロックの閉じ忘れ、差分の空白エラーを確認し、すべて問題なし。文書間の要件も照合した。
- 文書だけの変更のためコードの単体テスト・ruff・CLI helpは対象外。Red／Greenや実動作の実績はない。
- 重み・サンプル動画・実測・PPTXは未取得または未作成。仕様の確定と実装の完了を区別する。

## 次の作業

実装を依頼された際にchore/bootstrapへ切り替え、環境調査を記録して共通基盤をTDDで作る。GPU割当が不明な場合も、GPUを使用しない基盤実装とテストは進められる。
