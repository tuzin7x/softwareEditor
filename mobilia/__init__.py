"""7xAi — programa para o mercado imobiliário.

* AI Studio: redesenho de ambientes por IA a partir de uma única foto, no computador do usuário
  (``mobilia.design``, ``mobilia.video``), com os modos legados de vídeo de montagem.
* Módulos da janela principal em ``mobilia.modules`` (ex.: o Lead Finder).

``__version__`` é a fonte única da versão (``docs/RELEASES.md``).
"""

import os as _os

# o transformers baixa em segundo plano (e às vezes espera) uma conversão .safetensors de ~900 MB do OneFormer a
# cada carga online: os pesos .bin do instalador bastam (o worker já define isto; a CLI direta também precisa)
_os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

__version__ = "0.9.0"
APP_NAME = "7xAi"
