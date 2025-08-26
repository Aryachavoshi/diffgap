# diffgap

diffgap/
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ src/
│  └─ diffgap/
│     ├─ __init__.py
│     ├─ samplers.py          # DiffusionSampler (your sampler, tidied)
│     ├─ data.py              # InpaintingDataset + masks (incl. FBM cloud masks)
│     ├─ baselines.py         # OpenCV inpaint + 2D linear interpolation
│     ├─ postprocess.py       # Batch denoising helpers (OpenCV)
│     ├─ metrics.py           # PSNR, SSIM, RMSE + compare() utility
│     └─ utils.py             # small helpers (device, scaling)
├─ examples/
│  └─ colab_demo.ipynb        # (optional) minimal demo
└─ tests/
   └─ test_metrics.py         # (optional) quick sanity tests
