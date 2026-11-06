"""
Paquete ``core``: modelo de datos, base de conocimiento y coordinador del comité.

IMPORTANTE (prevención de importaciones circulares): este ``__init__`` NO importa
``core.coordinator``, porque el coordinador depende de ``agents`` y los agentes
dependen de ``core.models`` / ``core.knowledge_base``. Importe el coordinador
explícitamente con ``from core.coordinator import ConsensusCoordinator``.
"""

__all__ = ["models", "knowledge_base", "coordinator"]
