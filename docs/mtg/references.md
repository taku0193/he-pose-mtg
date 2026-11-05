# 参考文献・環境・重み・ライセンス

実行コマンドと環境構築はREADMEを起点にします。ローカル設定には各人の割当を記入します。 [README](../../README.md) · [環境記録](../environment-inventory.md)

| 使用モデル | 準備する重み | ライセンスの記録 |
| --- | --- | --- |
| YOLO26n-Pose | `yolo26n-pose.pt` | Ultralytics AGPL-3.0 |
| DETRPose-N | COCO用 `model.safetensors` | 公式推論リポジトリと配布モデルに記載されたApache-2.0 |
| TAR-ViTPose Base | `tarvitpose_b_17.pt`、`pretrain_model_b.pth`、`yolo11x.pt` | TARコード Apache-2.0、補助検出器 AGPL-3.0。TAR本体・初期重みには公式Driveで独立した利用条件の記載を確認できていない |

採用した環境、上流コミット、重みの取得元とSHA256、依存のライセンス情報はモデル別に記録しています。 [YOLO26の採用記録](../model-manifests/yolo26.json) · [DETRPoseの採用記録](../model-manifests/detrpose.json) · [TARの採用記録](../model-manifests/tar-vitpose.json)

- YOLO26の姿勢推定モデル、関節定義と公式計測条件。 [Ultralytics Pose](https://docs.ultralytics.com/tasks/pose/)
- DETRPoseの設計と公式推論コード。 [DETRPose論文](https://arxiv.org/abs/2506.13027) · [公式推論版](https://github.com/SebastianJanampa/DETRPose/tree/da12e0cd29864b1d3a289c562c34d423371a4157)
- TARの時間文脈、公式実装とモデル設定。 [TAR論文](https://arxiv.org/abs/2603.05929) · [公式コード](https://github.com/zgspose/TARViTPose/tree/42164c2c789a5297544dd836846e509459b0f1f3)
- 自前計測の条件と実ファイルの場所。 [性能計測](../pr-notes/benchmark.md) · [SSH録画再送](../pr-notes/remote-live.md) · [本番素材の実行記録](../experiments/mtg-abert-2026-11-05.json)
- 動画3本と、それを加工した比較動画の出典・条件。 [素材のクレジット](sample-sources.md)
- 仕様と設計変更の経緯。 [確定仕様](../implementation-spec.md) · [資料形式の変更](../pr-notes/mtg-document-format.md)
