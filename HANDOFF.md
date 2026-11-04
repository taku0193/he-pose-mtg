# 実装チャットへの引き継ぎ

更新日：2026-11-04。確定仕様はdocs/implementation-spec.md v1.4。最新のユーザー指示とAGENTS.mdを優先する。

## 現在地

ユーザーは実装・こまめなcommit・本専用リポジトリへの通常のpushを許可済み。中断後に続行を指示し、abertのGPU使用も明示した。手元の動画はないとの回答を受け、利用条件を確認した公開素材で実動画の検証を進めた。

| 項目 | 状態 |
| --- | --- |
| 作業先 | abert / taku / /home/taku/he-pose-mtg |
| 共通基盤・モデル | YOLOとDETRの画像・動画工程をローカルmainへ統合。公開管理の許可を受け、リモートへの同期を進める |
| YOLO26n-Pose | 公式画像・反復画像動画・実際の運動動画をGPU検証済み。DETR工程で共通GPU確認処理を分離した後もGPU回帰確認 |
| DETRPose-N | feat/detrposeで画像・動画・共通CLIを実装し、公式画像APIとの座標一致と375フレーム動画をGPU検証済み |
| 最新のDETR実動画検証ソース | a92b2c047fb8ff3b5b3fef255c57be8b72ae96e3、dirty=false |
| 単体テスト | モデル不要75件＋11 subtests。ruff／CLI helpも確認。DETR工程のGitHub CIはpush後に確認する |
| モデル環境 | .venv-yolo26、固定版とハッシュはenvironments/yolo26/uv.lock |
| 重み | models/yolo26/yolo26n-pose.pt、公式SHA256照合済み。Git管理外 |
| 実動画 | CC BY-SA 4.0素材を15秒・25fps・375フレームへ変換。出典はassets/sample/jumping-jacks-ATTRIBUTION.md |
| YOLOの動画結果 | outputs/yolo26-motion-gpu/video、375/375処理、358フレームで人物を検出 |
| DETRの環境・結果 | .venv-detrpose、models/detrpose/model.safetensors、third_party/DETRPose、outputs/detrpose-motion-gpu/video |
| 残りのモデル | TAR-ViTPoseは未実装、依存環境・重み未準備 |
| 未完了の必須機能 | Mac遠隔ライブ、benchmark、compare、最終3素材、MTGガイド、3人分の受入 |
| GitHub公開範囲 | 2026-11-04にユーザーが公開管理を明示許可。APIのpublic設定と一致し、push保留は解除済み |

各モデルの実動作・TDD・制限はdocs/pr-notes/yolo26-pose.mdとdetrpose.md、機械可読の証拠はdocs/experiments/に記録する。全プロジェクトの完了ではない。DETRの矩形・関節confidence・visibilityは得られずnullとして保存する。公式safetensorsをローカルで読むパッチだけを加え、重み・上流・パッチのハッシュを照合する。

## GPUと実行経路

GPU 0はNVIDIA GeForce RTX 3090、UUIDはGPU-25e750aa-24f4-8a0d-edb2-f2f948526eec。ユーザーがこのGPUの使用を許可済み。ドライバ580.178.04、PyTorchのCUDAランタイム12.6で実際に推論した。表示用プロセス等は停止していない。

チャットのサンドボックスでは/dev/nvidia*が見えず直接のnvidia-smiは失敗するが、ホストのGPU故障や未接続ではない。既存の認証・ホスト鍵確認を使用した次のSSHから、同じabert上の環境とファイルへアクセスしてGPU推論できる。

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ConnectTimeout=5 taku@abert 'cd /home/taku/he-pose-mtg && .venv-yolo26/bin/python -m hepose.cli doctor --local --config hepose.local.toml --model yolo26 --device cuda:0 --json'
```

Macからの接続確認ではなく、現在の実行環境からabertへの接続である。Macの機種・OS・カメラ許可・SSHライブは別途確認する。/devのファイルやディレクトリへの書込権限を要求して解決しようとしない。

## 再開時の確認

1. AGENTS.md、本書、確定仕様、Git状態を読み、未コミット変更と既存の結果を保護する。
2. ローカルmainとモデル別ブランチ、リモートとの差、最新CIを確認する。mainへ直接実装しない。公開管理と通常pushは許可済みで、改めて承認を求めない。
3. hepose.local.tomlと準備済み環境を確認する。既存の設定を例で上書きしない。
4. GPU工程は許可済みのGPU 0で行う。利用不可なら理由を記録し、別GPUやCPUへ自動切替しない。
5. 小さな機能単位でRed→Greenを実行し、work/tddとdocs/pr-notesへ記録する。単体試験はモデル・重み・GPU・ネットワークなしで実行する。

既存のGPU結果ディレクトリは上書きしない。新たに試す場合はREADMEの手順と新しい出力先を使う。通常runは入力フレームを削らない。性能比較のウォームアップ20件＋本計測100件以上は後続benchmarkの条件である。

## 残りの順序

| ブランチ | 内容 |
| --- | --- |
| feat/tar-vitpose | 公式再現、設定整合と明示パッチ、関節・座標・時間窓、動画検証 |
| feat/remote-live | Macカメラ、SSHトンネル、Linuxワーカー、同一フレームへの返送・描画、切断・終了 |
| feat/benchmark | 同一GPU・同一素材で順次測定、比較動画、表・レポート |
| docs/mtg-materials | GoogleドキュメントのようなMTGガイドのMarkdown原稿、進行表・台本・チートシート・まとめ |

各人のMac→専用Linux GPU→Mac表示という構成を維持する。YOLO26とDETRPoseは3人全員、TARは代表環境で実証する。開催日と準備期限は未定。

## 上流調査の注意点

DETRPoseの静的確認コミットはinference_onlyのda12e0cd29864b1d3a289c562c34d423371a4157。TARは42164c2c789a5297544dd836846e509459b0f1f3。採用時は再確認し、公式コード・全重み・依存版・必要パッチ・ライセンスを記録する。

TARはREADMEの設定パスと実在するconfigs/posetrack17/configTARViTPoseVitB.yamlが異なり、USE_FEATURE_FUSION／USE_MASKの整合確認が必要。TAR本体のほかViTPose初期化用重みと公式人物検出器yolo11x.ptが必要。既定の5フレーム窓は前後2枚を参照し、境界は未推定のまま元時間軸へ戻す。head_bottom／head_topをCOCOの目へ置き換えず、クロップ座標を復元し、固定visibilityを推定confidenceに流用しない。

遠隔ライブでは同じフレームIDの画像へ結果を重ねる。未処理画像は最大1枚として最新を優先する。Mac内の同じ単調時計で取得から表示APIまでを測り、Linuxの時計を直接引かない。異常・切断時はセッション所有資源を解放し、過去映像をライブとして表示し続けない。

失敗記録を提出したことと機能完了を区別する。実在する結果だけを資料に使用し、未測定・未実施と3人分の未確認を明示する。重み・録画素材・認証情報をcommitしない。
