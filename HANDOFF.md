# 実装チャットへの引き継ぎ

更新日：2026-11-05。確定仕様はdocs/implementation-spec.md v1.4。最新のユーザー指示とAGENTS.mdを優先する。

## 現在地

ユーザーは実装・こまめなcommit・本専用リポジトリへの通常のpushを許可済み。中断後に続行を指示し、abertのGPU使用も明示した。手元の動画はないとの回答を受け、利用条件を確認した公開素材で実動画の検証を進めた。

| 項目 | 状態 |
| --- | --- |
| 作業先 | abert / taku / /home/taku/he-pose-mtg |
| 共通基盤・モデル | 3モデルの録画工程をmainに統合し、同じ運動動画でGPU検証済み |
| YOLO26n-Pose | 公式画像・反復画像動画・実際の運動動画をGPU検証済み。DETR工程で共通GPU確認処理を分離した後もGPU回帰確認 |
| DETRPose-N | feat/detrposeで画像・動画・共通CLIを実装し、公式画像APIとの座標一致と375フレーム動画をGPU検証済み |
| 最新のDETR実動画検証ソース | a92b2c047fb8ff3b5b3fef255c57be8b72ae96e3、dirty=false |
| 単体テスト | モデル不要99件＋11 subtests、ruff、CLI help。最新のLinux／macOS CIはActionsを参照 |
| モデル環境 | .venv-yolo26、固定版とハッシュはenvironments/yolo26/uv.lock |
| 重み | models/yolo26/yolo26n-pose.pt、公式SHA256照合済み。Git管理外 |
| 実動画 | CC BY-SA 4.0素材を15秒・25fps・375フレームへ変換。出典はassets/sample/jumping-jacks-ATTRIBUTION.md |
| YOLOの動画結果 | outputs/yolo26-motion-gpu/video、375/375処理、358フレームで人物を検出 |
| DETRの環境・結果 | .venv-detrpose、models/detrpose/model.safetensors、third_party/DETRPose、outputs/detrpose-motion-gpu/video |
| TAR-ViTPose | .venv-tar、公式Base・ViTPose初期重み・YOLO11x、公式コード無修正。設定の不足キーを補いローカルパスを使用 |
| TARのGPU検証 | ソース73ee1be347634192565c8fec6bd51c457a39d0a0、dirty=false。共通375枚（371成功＋境界4枚）、公式CLI371枚。outputs/tar-final-gpu |
| 全モデルのdoctor | abertの割当GPU 0で3モデルともready=true。work/doctor-all-gpu.json |
| 遠隔ライブ | Linuxクライアント→SSH→abertのRTX 3090を実証。25 FPS録画再送は375枚中369枚を描画、6枚を置き換え。切断・Ctrl+C・期限後のワーカー終了を確認 |
| 最新の遠隔ライブ検証 | コミット0af7c37、outputs/live-ssh-final。再現手順・負荷試験・制限はdocs/pr-notes/remote-live.md |
| 未完了の必須機能 | Mac実機のカメラ・表示、benchmark、compare、最終3素材、MTGガイド、3人分の受入 |
| GitHub公開範囲 | 2026-11-04にユーザーが公開管理を明示許可。APIのpublic設定と一致し、push保留は解除済み |

各モデルの実動作・TDD・制限はdocs/pr-notes/、機械可読の証拠はdocs/experiments/に記録する。全プロジェクトの完了ではない。DETRの取得不能な矩形・関節confidence・visibilityはnull。TARはhead_bottom／head_topをnative関節へ保持し、COCOの目・耳はnullとする。TARの境界4枚も元時間軸で保存する。

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
| feat/benchmark | 同一GPU・同一素材で順次測定、比較動画、表・レポート |
| docs/mtg-materials | GoogleドキュメントのようなMTGガイドのMarkdown原稿、進行表・台本・チートシート・まとめ |

各人のMac→専用Linux GPU→Mac表示はREADMEの手順で別途受け入れる。YOLO26とDETRPoseは3人全員、TARは代表環境で実証する。開催日と準備期限は未定。

## 上流調査の注意点

DETRPoseの静的確認コミットはinference_onlyのda12e0cd29864b1d3a289c562c34d423371a4157。TARは42164c2c789a5297544dd836846e509459b0f1f3。採用時は再確認し、公式コード・全重み・依存版・必要パッチ・ライセンスを記録する。

TARは実在するBase設定を使い、USE_MASK=true（公式defaults）と、モデルでは未使用のUSE_FEATURE_FUSION=falseを推論用設定へ補った。公式Drive配布のTAR本体とViTPose初期重み、yolo11x.ptを準備済み。MMPose初期化時に未使用backbone.cls_tokenの警告が出るが、最終TAR状態はstrict=Trueで全キー一致。旧PyTorchのcuDNN回避処理の警告も記録済み。ソースパッチはない。詳細はdocs/pr-notes/tar-vitpose.md。

遠隔ライブでは同じフレームIDの画像へ結果を重ねる。未処理画像は最大1枚として最新を優先する。Mac内の同じ単調時計で取得から表示APIまでを測り、Linuxの時計を直接引かない。異常・切断時はセッション所有資源を解放し、過去映像をライブとして表示し続けない。

失敗記録を提出したことと機能完了を区別する。実在する結果だけを資料に使用し、未測定・未実施と3人分の未確認を明示する。重み・録画素材・認証情報をcommitしない。
