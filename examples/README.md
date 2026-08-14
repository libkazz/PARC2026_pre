# 参考例（examples）

| ファイル | 内容 |
|---|---|
| [smolvla_libero_spatial_lora_mac.ipynb](smolvla_libero_spatial_lora_mac.ipynb) | Apple Silicon Mac の MPS で SmolVLA を LIBERO-plus Spatial に LoRA 追加学習し、推論評価する Notebook |

## smolvla_libero_spatial_lora_mac.ipynb

`lerobot/smolvla_libero_plus` を初期重みとし、LIBERO-plus Spatial の 10 タスクを
LoRA で追加学習する。学習後は LoRA を元の重みへマージし、追加学習の前後を
同一条件で推論して比較する。Google Colab は使用しない。

### 動作条件

- Apple Silicon（M1 以降）の Mac
- macOS 14 以降
- ユニファイドメモリ 16 GB 以上を推奨
- 空きディスク 30 GB 以上を推奨
- Homebrew

8 GB の Mac では学習中にメモリ不足になる可能性がある。Intel Mac は MPS を
利用できないため対象外である。

### 初回セットアップ

Python 3.12 または 3.13 の専用 venv を作る。Homebrew の最新 `python3` が
3.14 以降の場合は、必ずバージョン付きの実行ファイルを使う。

```bash
brew install python@3.13 ffmpeg imagemagick glfw

/opt/homebrew/bin/python3.13 -m venv .venv-smolvla
source .venv-smolvla/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.7,<2.12" "jupyter>=1,<2"
```

`torch.backends.mps.is_available()` が `True` になることを確認する。

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

### 学習と推論評価

リポジトリのルートで Jupyter を起動し、Notebook のセルを上から順に
実行する。

```bash
source .venv-smolvla/bin/activate
jupyter notebook examples/smolvla_libero_spatial_lora_mac.ipynb
```

Notebook は次を自動的に行う。

1. LeRobot `v0.6.0` と必要パッケージを venv に導入する
2. モデルとデータセットをローカルキャッシュに取得する
3. MPS で LoRA 追加学習する
4. LoRA をマージし、MPS 推論で追加学習前後の成功率を比較する
5. モデル ZIP と比較 CSV をローカルに保存する

キャッシュは `~/Library/Caches/PARC2026/smolvla/`、実行ごとの成果物は
`~/PARC2026_outputs/<YYYYMMDD_HHMMSS>/` に保存される。新しい実行で過去の学習結果を
削除しない。

既定の学習条件は 10 タスク × 各 5 エピソード（計 50 エピソード）、
3,000 steps、バッチサイズ 1 である。action と state が同じ系列を同一の
行動軌跡としてまとめ、異なる軌跡から学習エピソードを選ぶ。タスク 3・6・7
については、開く→閉じる→開くの順になっていないグリッパー操作を候補から
除外する。選択根拠は実行ディレクトリの
`training_episode_selection.json` に保存される。

まず動作を確認する場合は Notebook の `STEPS` と
`EVAL_EPISODES_PER_TASK` を小さくする。Mac の構成によっては学習と推論評価に
長時間かかる。

### 提出物にするまでの作業

出力されるのは LeRobot 形式のモデル重みであり、これ単体では提出できない。
[submission_template/](../submission_template/) の `MyPolicy` にモデルを組み込み、
ポリシーサーバーの形にする。観測と action の仕様は
[submission_template/policy_server.py](../submission_template/policy_server.py) の docstring にある。

推論は 1 リクエストあたり 10 秒以内に収める必要がある
（[ルートの README](../README.md#タイムアウト仕様)）。Mac での MPS 実行は学習・比較用であり、
本番の CUDA 採点環境におけるレイテンシ検証の代わりにはならない。

### Notebook 内の評価と本番の採点の違い

Notebook 内の評価は学習の効果を手早く確認するためのもので、採点とは条件が異なる。
出てくる成功率は本番スコアの目安にはならない。

| 項目 | Notebook | 本番の採点 |
|---|---|---|
| 評価タスク | LIBERO-plus Spatial の 10 タスク | Track 1（`compe/t1/` のタスクセット） |
| 実行方法 | LeRobot の `lerobot-eval`（MPS） | `python -m pipeline` + 提出したポリシーサーバー（CUDA） |
| 観測の解像度 | 256×256 | 128×128 |
| 1 タスクあたりの試行数 | 3（`EVAL_EPISODES_PER_TASK` で変更可） | 非公開（配布キットの既定は 20） |

試行数が 3 のままだと 1 エピソードの成否で成功率が約 33 ポイント動くため、
追加学習の前後を比べる場合は `EVAL_EPISODES_PER_TASK` を増やすこと。

Notebook の環境は学習・推論評価用であり、ルートの [setup.sh](../setup.sh) とは
独立している。評価と提出前チェックはリポジトリ側の Docker 環境で行うこと。

Notebook が利用する第三者ソフトウェア・モデル・データセットのライセンスは、
各配布元の表記を参照すること。
