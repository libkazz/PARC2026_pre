# Repository Guidelines

## プロジェクト構成とモジュール

- `pipeline/`: 評価処理本体。`pipeline/__main__.py` がコマンドラインの入口です。
- `compe/t1/`: Track 1 のタスク登録とメタデータです。
- `submission_template/`: 参加者向けポリシーサーバーのひな形です。
- `examples/`: 学習用 Notebook などの参考実装です。
- `evaluate.py`: 提出 ZIP を一括採点します。
- `validate_submission.py`: 提出物の構造と動作を検査します。
- `tests/`: `pytest` のテストを対象機能ごとに配置します。

## ビルド・テスト・開発コマンド

- 開発・テストはホストの Python 環境ではなく Docker 内で行います。
- `docker build -t parc2026 .`: 依存関係とアセットを含む CPU 用の開発環境を構築します。
- `docker run -it --rm parc2026`: 開発用コンテナのシェルを開きます。
- `docker run --rm parc2026 pytest tests`: テスト一式を実行します。
- `docker run --rm parc2026 python validate_submission.py submission_template/`: テンプレートの必須ファイル、API、応答形式、レイテンシを検査します。
- `docker run --rm -v "$PWD/my_submission.zip:/sub.zip:ro" parc2026 python evaluate.py /sub.zip --n-episodes 2`: 提出 ZIP を短時間で評価します。

## コーディング規約と命名

- Python 3.10 を対象とし、インデントは4スペースにします。
- モジュール・関数・変数は `snake_case`、クラスは `PascalCase`、定数は `UPPER_CASE` とします。
- 公開関数や複雑なデータ構造には型ヒントを付けます。
- `submission_template/policy_server.py` の HTTP・シリアライズ仕様との互換性を維持します。
- formatter・linter 設定は未配置のため、周辺コードの PEP 8 風スタイルに合わせます。

## テスト方針

- コード変更後はイメージを再ビルドし、`docker run --rm parc2026 pytest tests` を実行します。
- テストファイルは `test_<対象>.py`、関数は `test_<振る舞い>()` と命名します。
- 不具合修正には回帰テストを追加し、検証エラー、採点、タスク絞り込み、安全な ZIP 処理を確認します。
- ファイル操作には `tmp_path`、入力差分には `pytest.mark.parametrize` を優先します。
- 動的スモークテストを含め、skip の有無を Docker のテスト結果で確認します。

## コミットとプルリクエスト

- 日本語の Conventional Commits を使います。例: `fix: 不正な提出ZIPの検証を強化`。
- 1コミットを1つの論理的意図に限定します。
- PR には動機、変更内容、Docker での確認コマンドと結果、関連 Issue を記載します。
- Notebook や文書の表示変更では、必要に応じてスクリーンショットを添付します。
- Draft で作成し、CI とレビュー指摘への対応後に Ready for review へ変更します。

## セキュリティと設定上の注意

- 取得済み LIBERO ディレクトリ、モデル重み、提出物、評価結果はコミットしません。
- ZIP 展開、外部依存元、タイムアウト、パス処理はセキュリティ上重要です。
- `validate_submission.py` の安全性検査を維持してください。
