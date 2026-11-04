# GitHub公開管理の許可と保留解除

2026-11-04。ユーザーが「こまめにcommitしていいですよ．公開で大丈夫です」と明示したため、GitHubの公開管理を許可する仕様v1.4へ更新した。GitHub APIでもtaku0193/he-pose-mtgがpublicであることを確認した。リポジトリ設定を変更する必要はなかった。

公開範囲確認に伴って保留していた通常pushを再開する。YOLO26とDETRPoseの実装・GPU検証記録の既存commitを保ち、モデル別ブランチとmainを同期する。モデル重み・録画素材・認証情報はGit管理外のままとし、force pushは行わない。

変更対象はAGENTS.md、仕様書、README、HANDOFF.md、CODEX_START_PROMPT.md、公開範囲とCI保留に関する記録。機能要件・受入条件・実装コードの変更はない。文書の版・相互参照・公開方針を確認し、形式的な単体テストは追加しない。

保留中のコードを含むmainをpushする際は、既存のモデル不要テスト、ruff、CLI helpを再確認する。GitHub ActionsのLinux／macOS結果は実行後に追記する。TAR・遠隔ライブ・比較測定・最終素材と資料・3人分の受入は引き続き未完了。

## 同期と確認の結果

- 公開方針のcommit：6cc9ff024acb16d8c5bcd6b266797a41bfd87476。main、feat/yolo26-pose、feat/detrpose、docs/public-repositoryを通常pushで同期し、GitHub APIのmainのSHAとローカルを照合した。
- commitの履歴は保持した。モデル重み・動画・ローカル設定はpushしていない。
- work/tdd/public-sync-*.log：単体テスト75件＋11 subtests、ruff check／formatが成功。全CLI help、文書の版・相互参照・差分も確認した。
- サンドボックスからのGitHub認証が一時失敗したため、既存の認証とホスト鍵確認を使うtaku@abert経由でpushした。SSH認証の成功も確認した。秘密鍵や認証設定は変更していない。
- push後、接続ツールとGitHubの公開APIでActionsを照会した。ワークフローはactiveだが、照会時点で6cc9ff0のrun／check suiteは存在しなかった。過去のmain（97e31a2）のCI成功を今回の75件のCI成功として扱わない。最新の実行状況は[Actions](https://github.com/taku0193/he-pose-mtg/actions)で確認する。
