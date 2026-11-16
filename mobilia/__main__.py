"""``python -m mobilia`` abre a interface; ``python -m mobilia --worker <args>`` roda a CLI."""
import sys

from mobilia.ui.app import main

sys.exit(main())
