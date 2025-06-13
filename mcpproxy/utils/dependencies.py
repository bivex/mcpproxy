"""Dependency checking utilities for optional features."""

import sys


def check_optional_dependency(
    import_name: str, package_name: str, feature_name: str, install_extra: str
) -> None:
    """Check if an optional dependency is available and provide helpful error if not.

    Args:
        import_name: The module name to import (e.g., 'sentence_transformers')
        package_name: The package name for pip install (e.g., 'sentence-transformers')
        feature_name: Human-readable feature name (e.g., 'HuggingFace embeddings')
        install_extra: The extra to install (e.g., 'huggingface')
    """
    try:
        __import__(import_name)
    except ImportError:
        print(
            f"\n❌ ERROR: {feature_name} requires {package_name} but it's not installed.",
            file=sys.stderr
        )
        print(f"   To use {feature_name}, install with:", file=sys.stderr)
        print(f"   pip install mcpproxy[{install_extra}]", file=sys.stderr)
        print(f"   or pip install {package_name}", file=sys.stderr)
        print(file=sys.stderr)
        sys.exit(1)


def optional_import(
    import_name: str, package_name: str, feature_name: str, install_extra: str
):
    """Import an optional dependency with helpful error message if missing.

    Returns:
        The imported module or a placeholder that raises error when used
    """
    try:
        return __import__(import_name)
    except ImportError:
        # Return a placeholder that raises an error when accessed
        class MissingDependency:
            def __getattr__(self, name):
                print(
                    f"\n❌ ERROR: {feature_name} requires {package_name} but it's not installed.",
                    file=sys.stderr
                )
                print(f"   To use {feature_name}, install with:", file=sys.stderr)
                print(f"   pip install mcpproxy[{install_extra}]", file=sys.stderr)
                print(f"   or pip install {package_name}", file=sys.stderr)
                print(file=sys.stderr)
                sys.exit(1)

            def __call__(self, *args, **kwargs):
                print(
                    f"\n❌ ERROR: {feature_name} requires {package_name} but it's not installed.",
                    file=sys.stderr
                )
                print(f"   To use {feature_name}, install with:", file=sys.stderr)
                print(f"   pip install mcpproxy[{install_extra}]", file=sys.stderr)
                print(f"   or pip install {package_name}", file=sys.stderr)
                print(file=sys.stderr)
                sys.exit(1)

        return MissingDependency()


def check_embedder_dependencies(embedder_type: str) -> None:
    """Check dependencies for specific embedder types."""
    if embedder_type == "HF":
        check_optional_dependency(
            "sentence_transformers",
            "sentence-transformers",
            "HuggingFace embeddings",
            "huggingface",
        )
        check_optional_dependency(
            "faiss",
            "faiss-cpu",
            "vector storage for HuggingFace embeddings",
            "huggingface",
        )
    elif embedder_type == "OPENAI":
        check_optional_dependency("openai", "openai", "OpenAI embeddings", "openai")
        check_optional_dependency(
            "faiss", "faiss-cpu", "vector storage for OpenAI embeddings", "openai"
        )
