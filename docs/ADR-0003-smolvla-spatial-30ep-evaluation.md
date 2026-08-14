# ADR-0003: Spatial 30エピソード学習の比較条件と成果物を分離する

- Status: Accepted
- Date: 2026-08-14

## Context

学習 Notebook の既定条件は、LIBERO-Spatial 10タスクから各5エピソード、合計50エピソードを
選択して3,000ステップ学習するものだった。30エピソード/taskへ拡張した結果を既定条件と比較する
には、学習データ数以外のハイパーパラメータと評価条件を固定し、既存の5エピソード成果物を
上書きしない必要がある。

また、Notebook のエピソード選択セルには合計50エピソードを前提とする固定チェックがあり、
設定値だけを30へ変更すると300エピソードの選択後に停止することが判明した。

## Decision

1. LIBERO-Spatial の10タスクから各30エピソードを等間隔に選択し、合計300エピソードを使う。
2. 学習データ数の影響を比較しやすくするため、学習ステップ数3,000、batch size 1、LoRA
   `r=16` / `alpha=16`、学習率、seedなどの既定ハイパーパラメータを維持する。
3. 評価は追加学習前後で同じ `seed=2026` を使用し、各タスク3エピソード、合計60ロールアウト
   とする。これは短時間の比較用であり、正式な評価では各タスク10エピソード以上を使用する。
4. エピソード件数の検証は固定値50ではなく、
   `len(SPATIAL_TASK_NAMES) * TRAIN_EPISODES_PER_TASK` から算出する。
5. 5エピソード成果物を上書きしないよう、モデル、評価結果、CSV、ZIPは `30ep` を含む専用パスへ
   保存する。
6. Colab Pro のA100ランタイムを使用し、A100が対応するBF16で学習する。
7. 実行条件、loss、タスク別成功率、実測時間、成果物を自己完結したHTMLレポートに保存する。

## Consequences

- 学習データの多様性を50から300エピソードへ増やしつつ、学習計算量は3,000ステップに固定できる。
- 1エピソードあたりの反復回数は5エピソード条件より減るため、データ数以外にサンプル反復密度も
  変化する。
- 5エピソードと30エピソードの成果物を同じColabランタイム内に共存させられる。
- 各タスク3エピソードでは1回の成否が33.3ポイントに相当し、ベースラインも実行間で変動したため、
  小さな差を学習データ数の効果として断定できない。
- `/content` 配下は一時ストレージなので、ランタイム切断前に必要な成果物を永続化する必要がある。

## Validation result

- GPU: NVIDIA A100、mixed precision: BF16
- Training data: 10 tasks × 30 episodes = 300 episodes
- Training: 3,000 steps、約12.1分、最終loss 0.143
- Evaluation: 2 models × 10 tasks × 3 episodes = 60 rollouts、約54.4分
- Baseline overall success: 70.0%
- 30ep Spatial LoRA overall success: 86.7%
- Difference: +16.7 percentage points
- タスク別: 改善4、同率6、悪化0
- ZIP: `/content/smolvla_libero_plus_spatial_lora_30ep_merged.zip`（0.85 GiB）
- CSV: `/content/libero_spatial_comparison_30ep.csv`
- HTML: `docs/reports/smolvla-libero-spatial-lora-30ep.html`
