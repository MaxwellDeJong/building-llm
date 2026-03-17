"""Root package for the building-llm project.

Installs the jaxtyping import hook so that Float/Int annotations are checked
at runtime via typeguard.
"""
from jaxtyping import install_import_hook

install_import_hook("src", "typeguard")
