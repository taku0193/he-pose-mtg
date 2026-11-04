# HE Pose MTG

HE研究室の3人・90分MTG向け姿勢推定デモプロジェクト。

Macのカメラ映像をSSH先LinuxのRTX 3090で推論し、結果をMacに表示します。参加者は各人専用のGPUを使用します。

| モデル | 必須範囲 |
| --- | --- |
| YOLO26n-Pose | 全員が録画動画と遠隔ライブを実行 |
| DETRPose-N | 全員が同一の録画動画を実行 |
| TAR-ViTPose | 動作可能な代表環境でオフライン動画推論を実演 |

## 現在の状態

2026-11-04：5文書を再確認し、実装仕様書v1.2を確定しました。現在の作業先はLinuxホストabertの`/home/taku/he-pose-mtg`です。GPU割当・MacからのSSH・カメラ動作は未確認で、実装コード・テスト・モデル環境・推論結果はまだありません。

確定仕様は[docs/implementation-spec.md](docs/implementation-spec.md)です。[今回のレビュー記録](docs/pr-notes/spec-v1.2.md)に修正点と確認範囲をまとめています。実装開始時は[AGENTS.md](AGENTS.md)と[HANDOFF.md](HANDOFF.md)、[CODEX_START_PROMPT.md](CODEX_START_PROMPT.md)を読んでください。

## 開発方針

`chore/bootstrap` → モデル別ブランチ → `feat/remote-live` → `feat/benchmark` → `docs/mtg-materials`の順に、TDDで実装します。小さな単位でcommitし、GitHubへpushします。

重み・大容量動画・認証情報はGit管理対象外です。公式ベンチマークと自前の実測値を区別します。

最新のチャットの依頼範囲を優先します。今回の仕様確認を終え、次の実装工程は環境調査と`chore/bootstrap`の共通基盤です。通常実行はローカルに準備済みの重みを使い、不足時に自動取得しない方針です。
