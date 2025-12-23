# Data

本项目默认不把原始数据提交到 GitHub（见仓库根目录 `.gitignore`）。

## Expected files

请把以下文件放到：`playground-series-s5e12/data/raw/`

- `train.csv`
- `test.csv`
- （可选）`sample_submission.csv`

## Download (Kaggle CLI)

你需要先在本机配置 Kaggle API（`~/.kaggle/kaggle.json`）。

```bash
mkdir -p playground-series-s5e12/data/raw
cd playground-series-s5e12/data/raw
kaggle competitions download -c playground-series-s5e12
unzip -o playground-series-s5e12.zip
```

如果你不想用 CLI，也可以直接在 Kaggle 比赛页面下载 zip 后手动解压到该目录。
