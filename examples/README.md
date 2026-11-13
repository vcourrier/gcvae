# Examples

Each example uses the installed `gcvae` package (`pip install -e ".[examples]"` from the repository root) plus, if needed, its own small utils module in the example folder for application-specific code (data loading, preprocessing, plots). Examples never import from `train.py`, `search.py` or `evaluate.py`.

**Private data.** Put data that must not be published in `data/private/` (ignored by git), never in an example folder.

**Notebooks.** Notebook outputs are never committed: they can contain rows of private data. Run once, after cloning:

```bash
pip install nbstripout
nbstripout --install
```

The filter is then applied to every `.ipynb` file at `git add` (the outputs stay in your local copy). Check with `nbstripout --status`.
