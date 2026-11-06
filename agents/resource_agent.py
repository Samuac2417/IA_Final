"""
agents/resource_agent.py
========================

Agente 2 — ``ResourceManagerAgent`` (Gestor de Capacidad Hospitalaria).

Contrasta la demanda de pacientes en Prioridad 1 con el inventario físico de la
UCI (camas y ventiladores) y ejecuta la asignación material siguiendo el orden de
prioridad que le entrega el ``ConsensusCoordinator``.

Reglas de producción:
    REG-REC-00  Inventario y ocupación de la UCI
    REG-REC-01  Capacidad normal (R1): P1 ≤ camas disponibles
    REG-REC-02  Declaratoria de colapso / triaje de catástrofe (R2): P1 > camas
    REG-REC-03  Escasez de ventiladores mecánicos
    REG-REC-04  Asignación física de cama / ventilador según el orden del comité
    REG-REC-05  Ruteo de pacientes no-P1 (no consumen cama UCI)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from agents.base_agent import BaseAgent
from core import knowledge_base as kb
from core.models import (
    DictamenAgente, EstadoHospital, Etapa, EvaluacionClinica, Nivel, Paciente, RecursoHospitalario,
)


class ResourceManagerAgent(BaseAgent):
    nombre = "ResourceManagerAgent"
    rol = "Gestor de Capacidad Hospitalaria · Inventario UCI y detección de colapso"

    def __init__(self) -> None:
        super().__init__()
        self._datos: Dict[str, object] = {}
        self._veredicto: str = ""
        self._nivel: Nivel = Nivel.INFO

    # ------------------------------------------------------------- evaluación
    def evaluar(  # type: ignore[override]
        self,
        pacientes: List[Paciente],
        evaluaciones: Dict[str, EvaluacionClinica],
        estado: RecursoHospitalario,
    ) -> DictamenAgente:
        self.reiniciar()
        por_id = {p.id: p for p in pacientes}
        p1 = [pid for pid, e in evaluaciones.items() if e.prioridad == 1]
        p1_vent = [pid for pid in p1 if por_id[pid].requiere_ventilador]
        camas, vents = estado.camas_uci_disponibles, estado.ventiladores_disponibles

        # REG-REC-00 · Inventario
        self.registrar(
            "REG-REC-00",
            f"Camas UCI totales = {estado.camas_uci_totales} · disponibles = {camas} · ventiladores "
            f"disponibles = {vents}",
            f"Ocupación UCI actual: {estado.ocupacion_pct:.1f} %. Demanda P1 = {len(p1)} pacientes "
            f"({len(p1_vent)} requieren ventilador).",
            f"{kb.cita('CONST_ART_49')} (eficiencia en la gestión del servicio público de salud).",
            None, Nivel.INFO, Etapa.RECURSO,
        )

        escasez_camas = len(p1) > camas
        escasez_vent = len(p1_vent) > vents

        if not escasez_camas:
            self.registrar(
                "REG-REC-01", f"Pacientes P1 ({len(p1)}) ≤ Camas UCI disponibles ({camas})",
                "CAPACIDAD NORMAL: se aprueba la asignación directa de cama UCI a todos los pacientes P1 "
                "sin conflicto ético distributivo.",
                f"{kb.cita('LEY_1751_2015')} Art. 6 (oportunidad y continuidad).",
                None, Nivel.OK, Etapa.RECURSO,
            )
        else:
            self.registrar(
                "REG-REC-02", f"Pacientes P1 ({len(p1)}) > Camas UCI disponibles ({camas})",
                f"DECLARATORIA DE ESCASEZ CRÍTICA (triaje de catástrofe): déficit de {len(p1) - camas} "
                "cama(s). Se transfiere el expediente al BioethicsAuditAgent y al ConsensusCoordinator para "
                "ejecutar el protocolo de desempate según el marco ético activo.",
                f"{kb.cita('LEY_1523_2012')}; estándares de atención en crisis (Crisis Standards of Care).",
                None, Nivel.CRITICO, Etapa.RECURSO,
            )

        if escasez_vent:
            self.registrar(
                "REG-REC-03", f"Pacientes P1 con ventilador ({len(p1_vent)}) > Ventiladores disponibles ({vents})",
                f"ESCASEZ DE VENTILADORES: déficit de {len(p1_vent) - vents} equipo(s). Un paciente que requiere "
                "ventilación solo puede recibir cama UCI si además existe ventilador disponible.",
                "Gestión de tecnología biomédica en crisis; Crisis Standards of Care.",
                None, Nivel.CRITICO, Etapa.RECURSO,
            )

        # REG-REC-05 · ruteo de no-P1
        destinos = {
            2: ("Cuidados intermedios / monitoreo continuo con reevaluación seriada; no consume cama UCI en "
                "esta ronda.", Nivel.ALERTA),
            3: ("Hospitalización general; no consume recursos críticos.", Nivel.OK),
            4: ("Unidad de cuidado paliativo / habitación individual con acompañamiento familiar; no consume "
                "cama UCI ni ventilador.", Nivel.INFO),
        }
        for pid, e in evaluaciones.items():
            if e.prioridad in destinos:
                texto, nivel = destinos[e.prioridad]
                self.registrar("REG-REC-05", f"Prioridad clínica = P{e.prioridad}", texto,
                               "Ruteo asistencial por nivel de complejidad.", pid, nivel, Etapa.RECURSO)
            elif not escasez_camas and not escasez_vent:
                self.registrar("REG-REC-01", f"Paciente P1 · capacidad suficiente ({len(p1)} ≤ {camas})",
                               "Asignación directa aprobada sin necesidad de desempate.",
                               f"{kb.cita('LEY_1751_2015')} Art. 6.", pid, Nivel.OK, Etapa.RECURSO)
            else:
                self.registrar("REG-REC-02", f"Paciente P1 en competencia por {camas} cama(s) / {vents} ventilador(es)",
                               "Expediente remitido al protocolo de desempate del comité.",
                               f"{kb.cita('LEY_1523_2012')}.", pid, Nivel.ALERTA, Etapa.RECURSO)

        estado_h = EstadoHospital.ESCASEZ_CRITICA if (escasez_camas or escasez_vent) else EstadoHospital.NORMAL
        if estado_h == EstadoHospital.NORMAL:
            self._veredicto = f"Capacidad normal: {len(p1)} paciente(s) P1 para {camas} cama(s) y {vents} ventilador(es)."
            self._nivel = Nivel.OK
        else:
            self._veredicto = (f"ESCASEZ CRÍTICA: {len(p1)} paciente(s) P1 compiten por {camas} cama(s) y "
                               f"{vents} ventilador(es). Se activa protocolo de desempate.")
            self._nivel = Nivel.CRITICO
        self._datos = {
            "estado_hospital": estado_h.value, "demanda_p1": len(p1), "demanda_p1_ventilador": len(p1_vent),
            "camas_disponibles": camas, "ventiladores_disponibles": vents,
            "deficit_camas": max(0, len(p1) - camas), "deficit_ventiladores": max(0, len(p1_vent) - vents),
            "candidatos_p1": p1,
        }
        return self.construir_dictamen(self._veredicto, self._nivel, **self._datos)

    # ------------------------------------------------------------- asignación
    def asignar(
        self,
        orden: List[str],
        pacientes: List[Paciente],
        estado: RecursoHospitalario,
    ) -> Tuple[DictamenAgente, List[str], List[str], List[str]]:
        """Asigna recursos físicos respetando el orden de prioridad del comité.

        Devuelve (dictamen_final, asignados_uci, asignados_ventilador, no_asignados).
        """
        por_id = {p.id: p for p in pacientes}
        camas, vents = estado.camas_uci_disponibles, estado.ventiladores_disponibles
        asignados: List[str] = []
        con_vent: List[str] = []
        no_asignados: List[str] = []

        for posicion, pid in enumerate(orden, start=1):
            p = por_id[pid]
            necesita_vent = p.requiere_ventilador
            if camas > 0 and (not necesita_vent or vents > 0):
                camas -= 1
                if necesita_vent:
                    vents -= 1
                    con_vent.append(pid)
                asignados.append(pid)
                self.registrar(
                    "REG-REC-04",
                    f"Posición {posicion} en el orden del comité · cama disponible = Sí · ventilador "
                    f"{'requerido y disponible' if necesita_vent else 'no requerido'}",
                    f"CAMA UCI ASIGNADA{' + VENTILADOR MECÁNICO' if necesita_vent else ''}. Inventario restante: "
                    f"{camas} cama(s), {vents} ventilador(es).",
                    "Asignación física según orden de prioridad validado por el comité.",
                    pid, Nivel.OK, Etapa.RECURSO,
                )
            else:
                motivo = ("no quedan camas UCI disponibles" if camas <= 0
                          else "hay cama disponible pero no hay ventilador para un paciente que lo requiere")
                no_asignados.append(pid)
                self.registrar(
                    "REG-REC-04", f"Posición {posicion} en el orden del comité · {motivo}",
                    "SIN RECURSO UCI EN ESTA RONDA: se ingresa a lista de espera priorizada y se exige plan "
                    "alternativo obligatorio (ver REG-BIO-03).",
                    "Escasez física del recurso (no constituye negación administrativa del servicio).",
                    pid, Nivel.CRITICO, Etapa.RECURSO,
                )

        self._datos.update({"asignados": asignados, "asignados_ventilador": con_vent,
                            "no_asignados": no_asignados})
        dictamen = self.construir_dictamen(self._veredicto, self._nivel, **self._datos)
        return dictamen, asignados, con_vent, no_asignados
