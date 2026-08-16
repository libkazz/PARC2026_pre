# 参考例（examples）

| ファイル | 内容 |
|---|---|
| [smolvla_libero_spatial_lora.ipynb](smolvla_libero_spatial_lora.ipynb) | SmolVLA を LIBERO-plus Spatial で LoRA 追加学習する Google Colab／ローカル Jupyter 対応ノートブック |

## smolvla_libero_spatial_lora.ipynb

`lerobot/smolvla_libero_plus` を初期重みとし、LIBERO-plus Spatial の 10 タスクを
LoRA で追加学習する。学習後は LoRA を元の重みへマージし、追加学習の前後を
同一条件で比較する。

### 使い方

Google Colab では次の手順で実行する。

1. ノートブックを開き、ランタイムのタイプを GPU（T4 で足りる）に変更する
2. 上から順に実行する。所要時間は T4 で数時間程度である
3. マージ済みモデル一式（zip）と、追加学習前後の成功率の比較（CSV）が自動でダウンロードされる

ローカルでは Python 3.12 以上と `git`、`ffmpeg`、`unzip` が必要である。
Linux では MuJoCo 用の共有ライブラリも導入する。

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg git unzip libgl1 libglib2.0-0 libsm6 libxext6 \
  libexpat1 libfontconfig1-dev libmagickwand-dev

# macOS
brew install ffmpeg imagemagick
```

リポジトリの評価環境（Python 3.10）とは分けて、ノートブック専用の仮想環境を作る。
PyTorch は使用する CUDA または Apple Silicon MPS に合うものを導入する。

```bash
python3.12 -m venv .venv-smolvla
source .venv-smolvla/bin/activate
python -m pip install --upgrade pip jupyterlab torch
jupyter lab examples/smolvla_libero_spatial_lora.ipynb
```

ノートブックは Colab／ローカルを自動判定する。ローカルのダウンロード、キャッシュ、
学習結果は既定で `~/.cache/parc2026_smolvla` に保存される。保存先を変更する場合は
Jupyter の起動前に `PARC2026_SMOLVLA_WORKDIR` を設定する。

CUDA と Apple Silicon MPS を自動選択し、どちらもない場合は CPU を使用する。
CPU でもセルは実行できるが、学習と評価には非常に長い時間がかかる。
ローカルに既存の `~/.libero/config.yaml` がある場合は、初回実行時に
`~/.libero/config.yaml.bak` へ退避してから Notebook 用の設定へ更新する。

学習条件は 10 タスク × 各 20 エピソード（計 200 エピソード）、3,000 steps、
バッチサイズ 1 である。同じ action・state 系列を持つエピソードを同一の行動軌跡として
まとめ、異なる軌跡から代表エピソードを選ぶ。全 10 タスクでさらに、グリッパーが
「開く→閉じる→開く」の順で各 3 フレーム以上操作されているエピソードだけを候補にする。
選定件数、除外理由、episode index、軌跡 fingerprint は作業ディレクトリの
`training_episode_selection.json` へ保存される。

### 提出物にするまでの作業

Notebook の最終セルは、学習済みモデルにポリシーサーバーとオフライン実行環境を同梱した
`PARC2026_track1_submission.zip` を作成する。この ZIP はルート直下に必須の
`policy_server.py` と `requirements.txt` を含み、静的検査に合格した場合だけダウンロードされる。
採点環境には、モデル重みだけをまとめた ZIP ではなく、この提出 ZIP をアップロードする。

Notebook とは別に提出 ZIP を作り直す場合は、
[smolvla_policy_server/README.md](smolvla_policy_server/README.md) の手順を使用する。

推論は 1 リクエストあたり 10 秒以内に収める必要がある
（[ルートの README](../README.md#タイムアウト仕様)）。

### ノートブック内の評価と、本番の採点の違い

ノートブック内の評価は学習の効果を手早く確認するためのもので、採点とは条件が異なる。
出てくる成功率は本番スコアの目安にはならない。

| 項目 | ノートブック | 本番の採点 |
|---|---|---|
| 評価タスク | LIBERO-plus Spatial の 10 タスク | Track 1（`compe/t1/` のタスクセット） |
| 実行方法 | LeRobot の `lerobot-eval` | `python -m pipeline` + 提出したポリシーサーバー |
| 観測の解像度 | 256×256 | 128×128 |
| 1 タスクあたりの試行数 | 3（`EVAL_EPISODES_PER_TASK` で変更可） | 非公開（配布キットの既定は 20） |

試行数が 3 のままだと 1 エピソードの成否で成功率が約 33 ポイント動くため、
追加学習の前後を比べる場合は `EVAL_EPISODES_PER_TASK` を増やすこと。

### 実行環境

ノートブックの環境構築は Colab／ローカル Jupyter 向けで、[setup.sh](../setup.sh) とは独立している。
依存パッケージのバージョンが一致しない箇所があるため、評価と提出前チェックは
リポジトリ側の環境（`setup.sh` + `env.sh`）で行うこと。

ノートブックが利用する第三者製ソフトウェア・モデル・データセットのライセンスは、
各配布元の表記を参照すること。
