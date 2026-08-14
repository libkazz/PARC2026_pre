# SmolVLA 提出サーバー

`examples/smolvla_libero_spatial_lora.ipynb` で作成したマージ済みモデルを、PARC2026
Track 1 のオフライン提出 ZIP に変換する実装です。

## 1. 学習成果物を用意する

Colab で Notebook を上から順に実行し、最後に作られる
`smolvla_libero_plus_spatial_lora_merged.zip` をダウンロードして展開します。更新済み
Notebook はモデル本体だけでなく、オフライン推論に必要な VLM の config と tokenizer を
`vlm_assets/` に保存します。

## 2. 提出 ZIP を作る

モデルの展開先を `results/smolvla_merged/` とした例です。

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  parc2026 \
  python examples/smolvla_policy_server/build_submission.py \
    --model-dir results/smolvla_merged \
    --output results/smolvla_submission.zip
```

ビルド時に PyPI 公式の LeRobot 0.6.0 wheel を取得し、SHA-256 を検証してから
`vendor/` へ同梱します。採点時には外部通信を使いません。wheel を事前取得済みなら
`--lerobot-wheel` で指定できます。

## 3. 検証する

まず CPU 用 Docker イメージで構造と依存解決を確認します。

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace parc2026 \
  python validate_submission.py results/smolvla_submission.zip --static
```

モデルの起動と推論は CUDA が必要です。本番相当の GPU イメージを README の手順で
構築したうえで、次を実行します。

```bash
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace parc2026-gpu \
  python validate_submission.py results/smolvla_submission.zip --install
```

`POLICY_DEVICE` は既定で `cuda`、モデル位置は既定で
`model_weights/smolvla` です。調査時だけ変更する場合は、それぞれ環境変数で上書きできます。
