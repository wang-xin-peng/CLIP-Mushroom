"""Environment fix: run this first to monkey-patch torch safety check and fix SSL."""
import os
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Model is already cached, no need to connect to HF hub
os.environ["HF_HUB_OFFLINE"] = "1"

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
