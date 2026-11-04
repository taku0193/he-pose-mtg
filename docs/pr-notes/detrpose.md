# DETRPose-N：公式画像APIと共通動画推論

2026-11-04。公式inference_onlyのCOCO用Nモデルを共通CLIへ接続し、abertの許可済みGPU 0で画像・動画推論を確認した。3人全員の受入やプロジェクト全体の完了ではない。

## 実装と固定情報

src/hepose/detrpose.pyにローカル重み・上流コミット・パッチ照合、実推論、共通変換を実装。adapters.py／doctor.pyへ接続した。GPU割当確認をruntime.pyへ移し、YOLOでも同じ処理を利用する。DETRの矩形が得られないためschema.pyとrender.pyでbbox=nullを扱い、推定された関節だけを描画する。

- 公式コード：[inference_onlyのda12e0c](https://github.com/SebastianJanampa/DETRPose/tree/da12e0cd29864b1d3a289c562c34d423371a4157)。設定はdetrpose_hgnetv2_n.py。
- 公式重み：[DETRPose_N_COCO](https://huggingface.co/SebasJanampa/DETRPose_N_COCO/tree/8e11f255908c7869d9a257b2669a9012957b6d0a)、版8e11f255908c7869d9a257b2669a9012957b6d0a。model.safetensors、16,677,992 bytes、SHA256=b618ccec659b42e92372290645c3c4712813571ecda914bf3ffa52d1173e20fc。
- パッチ：patches/detrpose/local-safetensors.patch。公式のローカルresumeがpthだけをロードするため、safetensors.torch.load_model(strict=True)を追加。モデル計算の修正や重みの変換はない。パッチSHA256=20dbd6a0e13e4cd496cee5a7c7f90428848db376bcc07eece0f43091adef32eb。
- 専用環境：Python 3.11.11、torch 2.14.1+cu126、torchvision 0.29.1+cu126、detrpose 0.0.1、safetensors 0.8.0、huggingface-hub 2.4.0。全依存・配布物ハッシュはenvironments/detrpose/uv.lock。
- コード・重みはApache-2.0。取得元、設定JSON、パッチと変更後ファイルのハッシュ、導入パッケージのライセンス表記は[モデル記録](../model-manifests/detrpose.json)。

scripts/prepare_detrpose.pyが固定版コードを取得してパッチを適用し、専用環境と公式重みを準備する。既存の別版・未知の変更を上書きしない。doctor／runでは取得せず、上流コミット・差分ハッシュ・重みSHA256・GPU割当を照合する。

## 結果の意味と計時

固定N設定の入力は640×640、batch=1、PyTorch、FP32。前処理は公式と同じBGR→RGB PIL、Resize、ToTensor。公式forwardとpostprocessorを使用し、公式と同じ人物score > 0.5を採用する。

このモデルは人物矩形や関節ごとのconfidence、visibilityを出力しない。共通JSONではbbox、関節confidence、visibilityをnullとし、人物scoreを関節confidenceへ流用しない。COCO17座標は公式postprocessorが元画像の幅・高さで復元したピクセル座標。追跡IDはnull。

モデル時間は指定GPUのstream上のCUDAイベントでforwardを囲み、終了イベントの完了を待って計測する。前処理・後処理・描画・I/Oは含めず、アダプタのホスト時間と区別する。ロードと初回応答も別記録。ピークGPUメモリはこのプロセスのPyTorch allocatorの値で、デバイス全体の使用量ではない。

今回の通常runは全入力を保持し、比較用warmupを設けていない。正式benchmarkのwarmup 20件＋本計測100件以上、同条件の順次比較、GTを使う精度評価は未実施。

## TDDと動作確認

| 記録 | 実行結果 |
| --- | --- |
| work/tdd/14-detrpose-red.log | hepose.detrposeが存在せずModuleNotFoundError。期待したRed |
| work/tdd/15-null-bbox-red.log | bbox=Noneを拒否してTypeError。取得不能な矩形を扱う契約のRed |
| work/tdd/15-null-bbox-green.log | 動画・描画の9件成功 |
| work/tdd/14-detrpose-green.log | 全75件＋11 subtests成功。モデル・GPU・ネットワークなし |

単体試験は人数・閾値・元座標・null・関節数・不正な結果・欠損重み・異なるモデル設定・矩形なし描画等を確認した。ruff、CLI help、core／モデル環境のlock整合も確認した。初回記録時は公開範囲の確認によりpushとCIを保留していたが、その後ユーザーが公開管理を許可した。push後のCI結果はdocs/pr-notes/public-repository.mdへ追記する。

GPU検証ソースはa92b2c047fb8ff3b5b3fef255c57be8b72ae96e3、dirty=false。既存の認証・ホスト鍵確認を有効にしたSSHでtaku@abertへ接続し、同じリポジトリの専用環境から実行した。RTX 3090のUUIDは環境記録と一致。別GPUや他人のプロセスを使用・停止していない。

1. doctorが依存・公式ファイル・パッチ・指定GPUを確認しready=true。
2. 公式bus.jpgで4人を検出。パッチ適用した公式DETR APIの出力と共通アダプタを照合し、全関節がrtol=1e-5、atol=1e-3ピクセルの範囲で一致。公式API結果・画像と共通画像・JSONを保存した。
3. YOLOと同じ実際の運動動画15秒・25fps・375枚を最後まで処理。元番号0～374と出力動画375枚の再復号を確認し、6出力を保存。368フレームで人物を検出し、7フレームは正常な0人検出だった。
4. Pythonのsocket.socket.connect（IPv4／IPv6）を禁止・記録した状態で、画像・動画推論中の接続試行は0件。OS全体の通信遮断試験ではない。
5. 共通GPU確認処理を分離した後、YOLOの画像・反復動画3フレームも同じGPUで成功。

検証結果はoutputs/detrpose-motion-gpu、YOLO回帰はoutputs/yolo26-after-detrpose-gpu。元のenvironment.json／summary.jsonを保持し、[実験記録](../experiments/detrpose-2026-11-04.json)へ転記した。実動画フレーム50を目視確認した。正解データがないため精度はN/Aで、人物検出のフレーム数から他モデルとの精度の優劣を判断しない。

動画の出典・加工内容・利用条件は[CC BY-SA 4.0素材の記録](../validation-assets/jumping-jacks.json)。最終basic／fast／multiの3素材が揃ったことにはしない。

## 再現コマンドと残作業

READMEの明示的な準備とhepose.local.tomlの設定を済ませ、abertのリポジトリ直下で実行する。既存結果と異なる出力先を選ぶ。

```bash
.venv/bin/python -m hepose.cli doctor --model detrpose --device cuda:0 --json
.venv-detrpose/bin/python integration_tests/detrpose_smoke.py --video assets/sample/jumping-jacks-validation.mp4 --output outputs/detrpose-motion-recheck
```

TAR、Macカメラ→SSH先GPU→Mac表示、benchmark／compare、最終3素材・MTGガイド、3人それぞれの受入確認が残る。2026-11-04にユーザーが公開管理を許可し、追加pushの保留は解除した。
