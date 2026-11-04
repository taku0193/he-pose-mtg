# GitHub公開管理の許可と保留解除

2026-11-04。ユーザーが「こまめにcommitしていいですよ．公開で大丈夫です」と明示したため、GitHubの公開管理を許可する仕様v1.4へ更新した。GitHub APIでもtaku0193/he-pose-mtgがpublicであることを確認した。リポジトリ設定を変更する必要はなかった。

公開範囲確認に伴って保留していた通常pushを再開する。YOLO26とDETRPoseの実装・GPU検証記録の既存commitを保ち、モデル別ブランチとmainを同期する。モデル重み・録画素材・認証情報はGit管理外のままとし、force pushは行わない。

変更対象はAGENTS.md、仕様書、README、HANDOFF.md、CODEX_START_PROMPT.md、公開範囲とCI保留に関する記録。機能要件・受入条件・実装コードの変更はない。文書の版・相互参照・公開方針を確認し、形式的な単体テストは追加しない。

保留中のコードを含むmainをpushする際は、既存のモデル不要テスト、ruff、CLI helpを再確認する。GitHub ActionsのLinux／macOS結果は実行後に追記する。TAR・遠隔ライブ・比較測定・最終素材と資料・3人分の受入は引き続き未完了。
