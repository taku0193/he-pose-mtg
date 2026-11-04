# HE Pose MTG

HE研究室の3人・90分MTG向け姿勢推定デモプロジェクト。

Macのカメラ映像をSSH先LinuxのRTX 3090で推論し、結果をMacに表示します。参加者は各人専用のGPUを使用します。

| モデル | 必須範囲 |
| --- | --- |
| YOLO26n-Pose | 全員が録画動画と遠隔ライブを実行 |
| DETRPose-N | 全員が同一の録画動画を実行 |
| TAR-ViTPose | 動作可能な代表環境でオフライン動画推論を実演 |

## 現在の状態

2026-11-04：実装仕様書v1.3。MTG資料はGoogleドキュメントのような章立ての文書とし、編集可能なMarkdown原稿をリポジトリで管理します。現在の作業先はLinuxホストabertの`/home/taku/he-pose-mtg`です。GPU 0（RTX 3090）の使用許可と環境情報を確認し、共通CLI・結果形式・結果保存の初期実装を進めています。モデル環境・動画I/O・実モデル推論・遠隔ライブは未実装または未検証です。

確定仕様は[docs/implementation-spec.md](docs/implementation-spec.md)です。[v1.2のレビュー記録](docs/pr-notes/spec-v1.2.md)と[MTG資料形式の変更記録](docs/pr-notes/mtg-document-format.md)に修正点と確認範囲をまとめています。実装開始時は[AGENTS.md](AGENTS.md)と[HANDOFF.md](HANDOFF.md)、[CODEX_START_PROMPT.md](CODEX_START_PROMPT.md)を読んでください。

## 開発方針

`chore/bootstrap` → モデル別ブランチ → `feat/remote-live` → `feat/benchmark` → `docs/mtg-materials`の順に、TDDで実装します。小さな単位でcommitし、GitHubへpushします。

重み・大容量動画・認証情報はGit管理対象外です。公式ベンチマークと自前の実測値を区別します。

最新のチャットの依頼範囲を優先します。今回の仕様確認を終え、次の実装工程は環境調査と`chore/bootstrap`の共通基盤です。通常実行はローカルに準備済みの重みを使い、不足時に自動取得しない方針です。

## 共通基盤の初期実装

Python 3.11以上で、モデルや追加パッケージを入れずに確認できます。以下はリポジトリ直下で実行します。

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m hepose.cli --help
PYTHONPATH=src python3 -m hepose.cli doctor --role client --json
PYTHONPATH=src python3 -m hepose.cli doctor --model all --device cuda:0 --json
```

doctorはカメラを開かず、パッケージや重みを取得しません。不足や未実装があれば終了コード1です。run・live・benchmark・compareはhelpを確認できる段階で、実行は「未実装」として終了します。推論成功を示す機能ではありません。

共通形式はschema_version=1.0で、元フレーム番号、欠測値null、関節名による対応、TAR等の時間窓を扱います。結果保存は新しい出力先を使い、既存ディレクトリを上書きしません。環境と確認元は[環境記録](docs/environment-inventory.md)、実行したRed/Greenと残作業は[共通基盤の記録](docs/pr-notes/bootstrap.md)にあります。

動画I/O・骨格描画・CLIへの接続、ruff、依存版の固定、CIが次の作業です。共通基盤全体は未完了であり、mainへのコード統合条件を満たしたという報告ではありません。
