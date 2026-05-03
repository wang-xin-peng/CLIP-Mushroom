@echo off
set "SSL_CERT_FILE="
set "REQUESTS_CA_BUNDLE="
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
call E:\anaconda\Scripts\conda.exe shell.bat hook
call conda activate E:\conda_envs\clip-mushroom
python -c "import transformers.utils.import_utils; transformers.utils.import_utils.check_torch_load_is_safe = lambda: None; import transformers.modeling_utils; transformers.modeling_utils.check_torch_load_is_safe = lambda: None" >nul 2>&1
python %*
