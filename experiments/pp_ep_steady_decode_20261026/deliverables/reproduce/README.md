# Reproduction

The frozen runner/adapters, private launch configs and large original traces remain in the original run directories. Use launch_manifest.json and data/source_hashes.json to verify provenance. Do not rerun a GPU experiment without Experiment Guard.

CPU analysis: python analyze.py --run ORIGINAL_RUN_DIRECTORY

Deck build originally used Python 3.11, python-pptx 1.0.2, matplotlib 3.11.0 and Pillow 13.0.0. Install python-pptx, matplotlib, numpy and Pillow into an isolated environment. Obtain NotoSansCJKsc-Regular.otf and NotoSansCJKsc-Bold.otf from the official notofonts/noto-cjk GitHub repository (Sans/OTF/SimplifiedChinese), license included. Set deck_engine.FONT_ROOT to that directory. Run build_deck.py from the original experiment directory; it reads verified source artifacts at the paths in sources.json. The PDF uses the same deterministic layout, not Office rendering.

Archive data/ contains portable CSV/JSON inputs for independently redrawing every quantitative comparison; PPT text/shapes are editable. The separate figure PDFs are vector charts.
