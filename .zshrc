export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export GUNICORN_CMD_ARGS="--workers=1 --threads=4"
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


. "$HOME/.local/bin/env"
alias mlflow-fix='uv run python -c "import os; os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"]="YES"; from mlflow.server import app; app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False)"'
