"""
agents/base_agent.py
====================

Clase abstracta ``BaseAgent``: contrato común de todos los agentes del comité.

Cada agente:
  * posee un nombre y un rol dentro del comité clínico-bioético;
  * evalúa hechos de la base de hechos mediante reglas de producción SI-ENTONCES;
  * registra cada regla disparada como una ``TrazaInferencia`` (XAI);
  * entrega un ``DictamenAgente`` con su veredicto y todas sus trazas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from core.models import DictamenAgente, Etapa, Nivel, TrazaInferencia


class BaseAgent(ABC):
    """Agente simbólico basado en reglas con registro de trazas lógicas."""

    nombre: str = "BaseAgent"
    rol: str = "Agente abstracto"

    def __init__(self) -> None:
        self._trazas: List[TrazaInferencia] = []

    # ------------------------------------------------------------------ trazas
    def reiniciar(self) -> None:
        """Vacía la memoria de trabajo del agente antes de una nueva deliberación."""
        self._trazas = []

    def registrar(
        self,
        regla_id: str,
        condicion: str,
        conclusion: str,
        fundamento: str,
        paciente_id: Optional[str] = None,
        nivel: Nivel = Nivel.INFO,
        etapa: Etapa = Etapa.REGLA,
    ) -> TrazaInferencia:
        """Registra el disparo de una regla de producción como paso explicativo."""
        traza = TrazaInferencia(
            agente=self.nombre,
            regla_id=regla_id,
            condicion_evaluada=condicion,
            conclusion=conclusion,
            fundamento_normativo_o_medico=fundamento,
            paciente_id=paciente_id,
            nivel=nivel.value,
            etapa=etapa.value,
        )
        self._trazas.append(traza)
        return traza

    @property
    def trazas(self) -> List[TrazaInferencia]:
        return list(self._trazas)

    def construir_dictamen(self, veredicto: str, nivel: Nivel, **datos: Any) -> DictamenAgente:
        """Empaqueta el veredicto y una copia de las trazas registradas."""
        return DictamenAgente(
            agente=self.nombre,
            rol=self.rol,
            veredicto=veredicto,
            nivel=nivel.value,
            trazas=self.trazas,
            datos=dict(datos),
        )

    # --------------------------------------------------------------- contrato
    @abstractmethod
    def evaluar(self, *args: Any, **kwargs: Any) -> DictamenAgente:
        """Ejecuta el ciclo de inferencia del agente y devuelve su dictamen."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - utilidad de depuración
        return f"<{self.nombre} · {self.rol} · {len(self._trazas)} trazas>"
