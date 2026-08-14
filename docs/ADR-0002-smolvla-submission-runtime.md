# ADR-0002: SmolVLA 提出ランタイムをオフライン同梱する

- Status: Accepted
- Date: 2026-08-14

## Context

学習用 Notebook は LeRobot 0.6.0 で SmolVLA の LoRA 学習とマージを行う。一方、
PARC2026 の採点環境は Python 3.10 と CUDA 13 対応の `torch 2.11.0+cu130` を使用する。
LeRobot 0.6.0 の配布メタデータは Python 3.12 以上を要求し、ソースには Python 3.12 の
型構文が4ファイルにある。また採点時の外部通信は禁止されるため、LeRobot や Hugging Face Hub
上の VLM 設定・tokenizer を起動時に取得できない。

LeRobot の旧版を通常の依存として導入すると、学習済み checkpoint との互換性が保証できず、
旧版が要求する `torch` によって採点イメージの CUDA 構成を崩す可能性もある。

## Decision

1. 学習と推論で LeRobot 0.6.0 を一致させる。
2. PyPI 公式 wheel の URL と SHA-256 を固定し、提出 ZIP の `vendor/` に同梱する。
3. wheel 展開後、Python 3.12 の型パラメータ・`type` 文だけを等価な Python 3.10 構文へ
   機械的に変換し、Python 3.10 にない `typing.Self` / `typing.Unpack` は
   `typing_extensions` から読む。対象件数が想定と
   違う場合はビルドを失敗させる。
4. `lerobot` と `torch` は `requirements.txt` に書かず、採点イメージの CUDA 13 対応
   `torch` を維持する。SmolVLA の実行に必要な周辺依存だけを明示する。
   SmolVLA が使わない HIL processor の再エクスポートを外し、CPU/GPU wheel が衝突する
   `torchvision` も導入しない。
5. Notebook で VLM の config と tokenizer を `vlm_assets/` に保存する。LeRobot が
   `AutoProcessor` から使う値は tokenizer の画像トークン ID だけなので、提出 runtime では
   `AutoTokenizer` に限定して `torchvision` 依存を避ける。
   推論時は Hugging Face の offline mode を有効にし、同梱したローカル資産だけを読む。
6. LIBERO の画像180度回転、quaternion から axis-angle への変換、action chunk の reset は
   LeRobot の評価経路と同じ意味になるよう提出アダプターで維持する。
7. vendored package の `configs` / `policies` 公開 API は SmolVLA 推論に必要な対象へ限定し、
   学習・全ポリシー・HIL 用モジュールを起動時に import しない。

## Alternatives considered

### LeRobot 0.4.3 を PyPI から通常インストールする

Python 3.10 には対応するが、Notebook の 0.6.0 checkpoint・processor 形式との互換性が
保証されない。また依存解決によって採点イメージの `torch` を置き換える可能性があるため
採用しない。

### LeRobot 0.6.0 を requirements.txt に記載する

Python 3.10 の requires-python 判定でインストールできないため採用しない。

### 採点時に GitHub / Hugging Face Hub から取得する

外部通信禁止の提出条件を満たさず、再現性も低下するため採用しない。

## Consequences

- 学習版と推論版のモデル実装を一致させられる。
- 提出 ZIP はネットワークなしで起動でき、wheel の供給元と内容を検証できる。
- LeRobot の版を更新する場合、SHA-256、互換パッチ、依存、GPU スモークテストを再検証する
  必要がある。
- CPU 用 Docker では構造・依存・前処理までを確認し、実モデル推論は CUDA 13 の GPU
  イメージで別途確認する必要がある。
