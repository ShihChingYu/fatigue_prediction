# tasks/doc.just

# run doc tasks
[group('doc')]
doc: doc-build

# build documentation (static HTML files)
[group('doc')]
doc-build format="google" output="docs": clean-docs
    uv run pdoc --docformat={{format}} --output-directory={{output}} src/fatigue

# serve documentation (live reload)
[group('doc')]
doc-serve format="google" port="8088":
    uv run pdoc --docformat={{format}} --port={{port}} src/fatigue
