# ADR-0002: 学習 Notebook を Colab とローカル Jupyter で共用する

- ステータス: Accepted
- 決定日: 2026-08-14

## 背景

`examples/smolvla_libero_spatial_lora.ipynb` は Google Colab の `/content`、`apt-get`、
`google.colab.files`、CUDA デバイスを直接参照していた。そのため、同じ Notebook を
ローカル Jupyter で実行すると、パス、権限、モジュール、デバイスの違いによって失敗していた。

一方、学習 Notebook が必要とする LeRobot v0.6.0 は Python 3.12 以上を前提とするが、
評価キットは本番採点環境に合わせて Python 3.10 を使用する。両者の依存を一つの環境へ
統合すると、本番相当の評価環境を損なうおそれがある。

## 決定

学習 Notebook は起動時に Google Colab かローカル Jupyter かを自動判定し、同じセル順で
実行できるようにする。

- Colab では従来どおり `/content` を作業先にし、必要なシステムパッケージを `apt-get` で導入する。
- ローカルでは既定で `~/.cache/parc2026_smolvla` を作業先にする。
  `PARC2026_SMOLVLA_WORKDIR` で任意の保存先へ変更できる。
- ローカルのシステムパッケージは Notebook から変更せず、必要なコマンドが存在することだけを検査する。
- PyTorch デバイスは CUDA、Apple Silicon MPS、CPU の順に選択し、学習と評価へ同じ値を渡す。
- MuJoCo のレンダリングは Linux で EGL、それ以外で GLFW を既定とし、
  `PARC2026_MUJOCO_GL` で上書きできる。
- 成果物は Colab の場合だけブラウザへダウンロードし、ローカルでは作業先へ保存して絶対パスを表示する。
- 学習環境は Python 3.12 以上の専用 venv とし、Python 3.10 の評価環境（`setup.sh`）とは分離する。
- ローカルの既存 `~/.libero/config.yaml` は、Notebook 用設定を書き込む前に一度だけ
  `config.yaml.bak` へ退避する。

## 影響

### 利点

- Colab とローカルで Notebook を分岐・複製せず、同じ学習手順を維持できる。
- ローカル実行時に `/content` や `google.colab` へ依存しない。
- ローカルの Python パッケージ変更を専用 venv に限定できる。
- GPU がない環境でも処理自体は開始できる。

### 注意点

- CPU での学習と評価は非常に低速であり、実用上は CUDA または MPS を推奨する。
- ローカルでは OS ごとのシステムパッケージを利用者が事前に導入する必要がある。
- Notebook が使用する依存バージョンは評価環境と一致しないため、提出前の検証は引き続き
  `setup.sh` または Docker の評価環境で行う。
- Notebook は LeRobot と LIBERO-plus のソースへ互換性パッチを適用するため、専用作業先を使用する。

## 検証結果

- Notebook JSON の妥当性を確認した。
- 全コードセルを Python の AST として構文解析した。
- Colab 専用の `apt-get` と `google.colab.files` が環境判定の内側にあることをテストした。
- `/content` 固定パスが Colab の既定作業先以外に残っていないことをテストした。
- ローカル Python 3.12 環境で、環境・デバイス・作業先の判定とシステムコマンド確認を実行した。
- 既存 Docker 依存環境でテスト一式を実行し、54 passed、1 skipped となった。
- 学習・評価の全実行はモデルとデータの大容量取得および数時間の GPU 実行を伴うため、実施していない。
