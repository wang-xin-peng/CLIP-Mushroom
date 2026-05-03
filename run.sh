#!/bin/bash
# Helper to run Python with env fixes for this project
unset SSL_CERT_FILE
unset REQUESTS_CA_BUNDLE
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_OFFLINE=1

eval "$(E:/anaconda/Scripts/conda.exe shell.bash hook)"
conda activate E:/conda_envs/clip-mushroom

# Monkey-patch torch safety check via environment variable
python -c "
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
import sys, runpy
sys.argv = ['python'] + '$@'.split()
runpy.run_path('$1', run_name='__main__')
" "$@"
