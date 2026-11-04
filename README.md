# HE Pose MTG

HE研究室の3人・90分MTG向け姿勢推定デモプロジェクト。

Macのカメラ映像をSSH先LinuxのRTX 3090で推論し、結果をMacに表示します。参加者は各人専用のGPUを使用します。

| モデル | 必須範囲 |
| --- | --- |
| YOLO26n-Pose | 全員が録画動画と遠隔ライブを実行 |
| DETRPose-N | 全員が同一の録画動画を実行 |
| TAR-ViTPose | 動作可能な代表環境でオフライン動画推論を実演 |

## 現在の状態

2026-11-04：仕様確定。新しいチャットでの実装開始に備えた引き継ぎ段階です。実装コード・テスト・モデル環境・推論結果はまだありません。

確定仕様は[docs/implementation-spec.md](docs/implementation-spec.md)です。実装開始時は[AGENTS.md](AGENTS.md)と[HANDOFF.md](HANDOFF.md)、[CODEX_START_PROMPT.md](CODEX_START_PROMPT.md)を読んでください。

## 開発方針

`chore/bootstrap` → モデル別ブランチ → `feat/remote-live` → `feat/benchmark` → `docs/mtg-materials`の順に、TDDで実装します。小さな単位でcommitし、GitHubへpushします。

重み・大容量動画・認証情報はGit管理対象外です。公式ベンチマークと自前の実測値を区別します。

