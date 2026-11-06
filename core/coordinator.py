"""
core/coordinator.py
===================

``ConsensusCoordinator``: orquestador del Comité Clínico-Bioético Multiagente.

Ciclo de deliberación (determinista y auditable):

    1. ClinicalTriageAgent   → prioridad P1-P4, reversibilidad y sobrevida.
    2. ResourceManagerAgent  → contraste demanda P1 vs. inventario (R1 / R2).
    3. Motor de desempate    → si hay escasez, ordena a los candidatos P1 según el
                               marco ético activo (Utilitarista / Deontológico /
                               Consenso Constitucional).
    4. ResourceManagerAgent  → asignación física de camas y ventiladores.
    5. Prueba contrafáctica  → repite el desempate con la edad neutralizada.
    6. BioethicsAuditAgent   → auditoría de sesgos, garantías y marco legal.
    7. Consenso XAI          → recomendación final por paciente + acta.

Dependencias: core.models, core.knowledge_base y agents.* (los agentes nunca
importan este módulo, por lo que no existen importaciones circulares).
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from agents.bioethics_agent import BioethicsAuditAgent
from agents.clinical_agent import ClinicalTriageAgent
from agents.resource_agent import ResourceManagerAgent
from core import knowledge_base as kb
from core.models import (
    ContextoAuditoria, DecisionFinal, Destino, EstadoHospital, Etapa, EvaluacionClinica, ModoEtico,
    Nivel, Paciente, RecursoHospitalario, ResultadoComite, TrazaInferencia, ValidacionEtica,
)

NOMBRE_COORDINADOR = "ConsensusCoordinator"


class ConsensusCoordinator:
    """Orquesta el comité multiagente y produce el consenso explicable."""

    nombre = NOMBRE_COORDINADOR

    def __init__(self, umbral_empate_sofa: int = kb.UMBRAL_EMPATE_TECNICO_SOFA,
                 semilla_sorteo: int = kb.SEMILLA_SORTEO_AUDITABLE) -> None:
        if umbral_empate_sofa < 0:
            raise ValueError("El umbral de empate técnico no puede ser negativo.")
        self.umbral_empate_sofa = umbral_empate_sofa
        self.semilla_sorteo = semilla_sorteo
        self.agente_clinico = ClinicalTriageAgent()
        self.agente_recursos = ResourceManagerAgent()
        self.agente_bioetico = BioethicsAuditAgent()

    # ================================================================ público
    def deliberar(self, pacientes: List[Paciente], estado: RecursoHospitalario) -> ResultadoComite:
        """Ejecuta una sesión completa del comité y devuelve el acta."""
        ids = [p.id for p in pacientes]
        if len(ids) != len(set(ids)):
            raise ValueError("Existen identificadores de paciente duplicados.")
        modo = estado.modo
        por_id = {p.id: p for p in pacientes}
        trazas: List[TrazaInferencia] = []

        # 1. Evaluación clínica ------------------------------------------------
        d_clin = self.agente_clinico.evaluar(pacientes)
        evaluaciones: Dict[str, EvaluacionClinica] = d_clin.datos["evaluaciones"]

        # 2. Evaluación de capacidad -------------------------------------------
        d_rec = self.agente_recursos.evaluar(pacientes, evaluaciones, estado)
        estado_h = d_rec.datos["estado_hospital"]
        candidatos: List[str] = list(d_rec.datos["candidatos_p1"])
        escasez = estado_h == EstadoHospital.ESCASEZ_CRITICA.value

        marco = kb.MARCOS_ETICOS[modo.value]
        trazas.append(self._traza(
            "REG-COORD-01", f"Marco ético activo = {modo.value} · estado hospitalario = {estado_h}",
            f"Se convoca al comité bajo el marco '{marco['nombre']}'. "
            + ("Hay escasez: se ejecuta el motor de desempate." if escasez
               else "No hay escasez: el marco ético no necesita desempatar."),
            f"{marco['escuela']}.", None, Nivel.INFO, Etapa.REGLA,
        ))

        # 3. Motor de desempate ------------------------------------------------
        proteccion: List[str] = []
        empates: List[List[str]] = []
        if escasez and candidatos:
            orden, puntajes, t_motor, proteccion, empates, etiqueta = self._motor_desempate(
                candidatos, por_id, evaluaciones, modo, neutralizar_edad=False)
            trazas.extend(t_motor)
        else:
            orden = sorted(candidatos, key=lambda pid: (-evaluaciones[pid].puntaje_urgencia,
                                                        por_id[pid].hora_llegada))
            puntajes = {pid: evaluaciones[pid].puntaje_urgencia for pid in orden}
            etiqueta = "Puntaje de urgencia fisiológica (0-100)"
            for pos, pid in enumerate(orden, start=1):
                trazas.append(self._traza(
                    "REG-COORD-02", "Capacidad suficiente para todos los P1",
                    f"No se requiere desempate ético. Orden operativo de ingreso #{pos} por urgencia "
                    f"fisiológica ({puntajes[pid]:.1f}).",
                    "Asignación directa (REG-REC-01).", pid, Nivel.OK, Etapa.REGLA,
                ))

        # 4. Asignación física -------------------------------------------------
        d_rec, asignados, con_vent, no_asignados = self.agente_recursos.asignar(orden, pacientes, estado)

        # 5. Prueba contrafáctica de edad --------------------------------------
        contrafactual: Optional[List[str]] = None
        if escasez and candidatos:
            orden_cf, *_ = self._motor_desempate(candidatos, por_id, evaluaciones, modo, neutralizar_edad=True)
            contrafactual = ResourceManagerAgent().asignar(orden_cf, pacientes, estado)[1]

        # 6. Auditoría bioética ------------------------------------------------
        ctx = ContextoAuditoria(
            pacientes=pacientes, evaluaciones=evaluaciones, estado=estado, estado_hospital=estado_h,
            candidatos_p1=candidatos, orden_prioridad=orden, puntajes=puntajes, asignados=asignados,
            no_asignados=no_asignados, asignados_contrafactual_sin_edad=contrafactual,
            proteccion_reforzada_aplicada=proteccion, empates_detectados=empates,
        )
        d_bio = self.agente_bioetico.evaluar(ctx)
        validaciones: Dict[str, str] = d_bio.datos["validacion_por_paciente"]
        planes: Dict[str, str] = d_bio.datos["planes_alternativos"]

        # 7. Consenso XAI ------------------------------------------------------
        decisiones: Dict[str, DecisionFinal] = {}
        for p in pacientes:
            decision = self._decision_final(p, evaluaciones[p.id], orden, puntajes, asignados, con_vent,
                                            validaciones[p.id], planes.get(p.id, ""), marco["nombre"])
            decisiones[p.id] = decision
            nivel = {ValidacionEtica.VETO.value: Nivel.CRITICO, ValidacionEtica.OBJETADO.value: Nivel.OBJECION,
                     ValidacionEtica.CON_OBSERVACIONES.value: Nivel.ALERTA}.get(decision.validacion_etica, Nivel.OK)
            trazas.append(self._traza(
                "REG-COORD-04",
                f"Prioridad P{decision.prioridad} · asignado UCI = {'Sí' if decision.asignado_uci else 'No'} · "
                f"auditoría ética = {decision.validacion_etica}",
                f"RECOMENDACIÓN FINAL: {decision.destino}. {decision.justificacion}",
                f"Consenso del comité bajo '{marco['nombre']}' + {kb.cita('LEY_23_1981')} (decisión humana final).",
                p.id, nivel, Etapa.DECISION,
            ))

        validacion_global = d_bio.datos["validacion_global"]
        acta = self._redactar_acta(modo, estado_h, estado, pacientes, decisiones, d_bio.datos["objeciones"],
                                   validacion_global, contrafactual, asignados)

        return ResultadoComite(
            modo=modo.value, estado_hospital=estado_h, pacientes=list(pacientes), evaluaciones=evaluaciones,
            dictamenes={d_clin.agente: d_clin, d_rec.agente: d_rec, d_bio.agente: d_bio},
            decisiones=decisiones, orden_prioridad=orden, puntajes=puntajes, trazas_coordinador=trazas,
            acta_consenso=acta, validacion_global=validacion_global,
            asignados_contrafactual_sin_edad=contrafactual, etiqueta_puntaje=etiqueta,
        )

    def comparar_marcos(self, pacientes: List[Paciente],
                        estado: RecursoHospitalario) -> Dict[str, ResultadoComite]:
        """Ejecuta la deliberación bajo los tres marcos éticos (análisis de sensibilidad)."""
        return {m.value: self.deliberar(pacientes, replace(estado, modo_etico_activo=m.value))
                for m in ModoEtico}

    # ======================================================= motor desempate
    def puntaje_utilitarista(self, p: Paciente, e: EvaluacionClinica, neutralizar_edad: bool = False) -> float:
        """Puntaje de sobrevida neta (marco utilitarista). Mayor = más prioridad."""
        edad = kb.EDAD_REFERENCIA_CONTRAFACTUAL if neutralizar_edad else p.edad
        anos = min(1.0, max(0, kb.EXPECTATIVA_VIDA_COLOMBIA - edad) / kb.ANOS_VIDA_NORMALIZACION)
        w = kb.PESOS_UTILITARISTAS
        base = 100.0 * (w["sobrevida"] * e.prob_sobrevida + w["pafi"] * min(p.pafi, 300) / 300.0
                        + w["anos_vida"] * anos)
        penal = kb.PENALIZACION_POR_COMORBILIDAD * min(len(p.comorbilidades_graves), kb.MAX_COMORBILIDADES_PENALIZADAS)
        return round(base - penal, 2)

    def _sorteo(self, ids: List[str]) -> Tuple[List[str], str]:
        """Sorteo auditable y reproducible: semilla pública + IDs ordenados."""
        semilla = f"{self.semilla_sorteo}|" + "|".join(sorted(ids))
        resultado = sorted(ids)
        random.Random(semilla).shuffle(resultado)
        return resultado, semilla

    def _motor_desempate(
        self, candidatos: List[str], por_id: Dict[str, Paciente], ev: Dict[str, EvaluacionClinica],
        modo: ModoEtico, neutralizar_edad: bool,
    ) -> Tuple[List[str], Dict[str, float], List[TrazaInferencia], List[str], List[List[str]], str]:
        trazas: List[TrazaInferencia] = []
        proteccion: List[str] = []
        empates: List[List[str]] = []

        # ---------------------------------------------------- Utilitarismo
        if modo == ModoEtico.UTILITARISTA_CLINICO:
            puntajes = {pid: self.puntaje_utilitarista(por_id[pid], ev[pid], neutralizar_edad) for pid in candidatos}
            orden = self._ordenar_con_desempate(candidatos, lambda pid: -puntajes[pid], por_id, trazas, empates)
            for pos, pid in enumerate(orden, start=1):
                p, e = por_id[pid], ev[pid]
                anos = max(0, kb.EXPECTATIVA_VIDA_COLOMBIA - p.edad)
                trazas.append(self._traza(
                    "REG-COORD-UTIL",
                    f"Sobrevida SOFA = {e.prob_sobrevida:.0%} · PaFi = {p.pafi} · años de vida potenciales = "
                    f"{anos} (edad {p.edad}) · comorbilidades = {len(p.comorbilidades_graves)}",
                    f"Puntaje de sobrevida neta = {puntajes[pid]:.1f} → posición #{pos} en el orden utilitarista.",
                    "Utilitarismo clínico: maximizar vidas y años de vida salvados.",
                    pid, Nivel.INFO, Etapa.REGLA,
                ))
            return orden, puntajes, trazas, proteccion, empates, "Puntaje de sobrevida neta (mayor = prioridad)"

        # ---------------------------------------------------- Deontología
        if modo == ModoEtico.DEONTOLOGICO_IGUALITARIO:
            puntajes = {pid: float(por_id[pid].minutos_espera(kb.MINUTO_ACTUAL_TURNO)) for pid in candidatos}
            orden = self._ordenar_con_desempate(candidatos, lambda pid: por_id[pid].hora_llegada,
                                                por_id, trazas, empates)
            for pos, pid in enumerate(orden, start=1):
                p = por_id[pid]
                trazas.append(self._traza(
                    "REG-COORD-DEONT",
                    f"Todos los P1 poseen idéntica dignidad · llegada en minuto {p.hora_llegada} del turno "
                    f"(hace {p.minutos_espera(kb.MINUTO_ACTUAL_TURNO)} min)",
                    f"Criterio neutral FIFO → posición #{pos}. No se compara el 'valor' de las vidas.",
                    f"Imperativo categórico kantiano (no instrumentalización); {kb.cita('CONST_ART_1')}.",
                    pid, Nivel.INFO, Etapa.REGLA,
                ))
            return orden, puntajes, trazas, proteccion, empates, "Minutos de espera (FIFO)"

        # ---------------------------------------------------- Consenso constitucional
        puntajes = {pid: float(por_id[pid].escala_sofa) for pid in candidatos}
        restantes = list(candidatos)
        orden: List[str] = []
        while restantes:
            s_min = min(por_id[pid].escala_sofa for pid in restantes)
            grupo = sorted([pid for pid in restantes if por_id[pid].escala_sofa <= s_min + self.umbral_empate_sofa],
                           key=lambda pid: (por_id[pid].escala_sofa, pid))
            pos = len(orden) + 1
            if len(grupo) == 1:
                elegido = grupo[0]
                motivo = (f"Mejor reserva funcional entre los candidatos restantes (SOFA {s_min}); ningún otro "
                          f"candidato está dentro del umbral de empate técnico (±{self.umbral_empate_sofa}).")
                regla, fundamento = "REG-COORD-CONST", "Reversibilidad fisiológica a corto plazo (SOFA)."
            else:
                empates.append(list(grupo))
                vulnerables = [pid for pid in grupo if por_id[pid].vulnerabilidad_social]
                subgrupo = vulnerables if vulnerables else grupo
                primera = min(por_id[pid].hora_llegada for pid in subgrupo)
                finalistas = [pid for pid in subgrupo if por_id[pid].hora_llegada == primera]
                semilla = ""
                if len(finalistas) > 1:
                    finalistas, semilla = self._sorteo(finalistas)
                elegido = finalistas[0]
                nombres = ", ".join(f"{por_id[x].nombre} (SOFA {por_id[x].escala_sofa})" for x in grupo)
                if vulnerables and len(vulnerables) < len(grupo):
                    proteccion.append(elegido)
                    motivo = (f"EMPATE TÉCNICO (|ΔSOFA| ≤ {self.umbral_empate_sofa}) entre {nombres}. Se aplica "
                              "PROTECCIÓN REFORZADA al sujeto de especial protección constitucional.")
                else:
                    motivo = (f"EMPATE TÉCNICO (|ΔSOFA| ≤ {self.umbral_empate_sofa}) entre {nombres}, sin "
                              "diferencia en protección constitucional: se desempata por orden de llegada.")
                if semilla:
                    motivo += f" Persistió el empate: sorteo auditable con semilla pública '{semilla}'."
                regla = "REG-COORD-CONST-EMPATE"
                fundamento = (f"{kb.cita('CONST_ART_13')} inciso 3; {kb.cita('CONST_ART_43_46')}; "
                              f"{kb.cita('LEY_1751_2015')} Art. 11.")
            restantes.remove(elegido)
            orden.append(elegido)
            p = por_id[elegido]
            trazas.append(self._traza(
                regla,
                f"SOFA = {p.escala_sofa} · reversibilidad {ev[elegido].reversibilidad} · edad y condición "
                "socioeconómica EXCLUIDAS como penalizadores",
                f"Posición #{pos}. {motivo}", fundamento, elegido, Nivel.INFO, Etapa.REGLA,
            ))
        return orden, puntajes, trazas, proteccion, empates, "Escala SOFA (menor = mayor reversibilidad)"

    def _ordenar_con_desempate(self, candidatos: List[str], clave, por_id: Dict[str, Paciente],
                               trazas: List[TrazaInferencia], empates: List[List[str]]) -> List[str]:
        """Ordena por ``clave``; los empates exactos se resuelven por llegada y luego por sorteo auditable."""
        grupos: Dict[object, List[str]] = {}
        for pid in candidatos:
            grupos.setdefault(clave(pid), []).append(pid)
        orden: List[str] = []
        for valor in sorted(grupos):
            grupo = grupos[valor]
            if len(grupo) == 1:
                orden.extend(grupo)
                continue
            empates.append(list(grupo))
            por_llegada: Dict[int, List[str]] = {}
            for pid in grupo:
                por_llegada.setdefault(por_id[pid].hora_llegada, []).append(pid)
            for llegada in sorted(por_llegada):
                sub = por_llegada[llegada]
                if len(sub) > 1:
                    sub, semilla = self._sorteo(sub)
                    trazas.append(self._traza(
                        "REG-COORD-SORTEO", f"Empate exacto entre {sub} (mismo criterio y misma hora de llegada)",
                        f"Sorteo auditable con semilla pública '{semilla}' → orden {sub}.",
                        "Igualdad de oportunidades; procedimiento reproducible y verificable.",
                        None, Nivel.ALERTA, Etapa.REGLA,
                    ))
                orden.extend(sub)
        return orden

    # ============================================================ utilidades
    def _decision_final(self, p: Paciente, e: EvaluacionClinica, orden: List[str], puntajes: Dict[str, float],
                        asignados: List[str], con_vent: List[str], validacion: str, plan: str,
                        nombre_marco: str) -> DecisionFinal:
        posicion = orden.index(p.id) + 1 if p.id in orden else None
        asignado = p.id in asignados
        if e.prioridad == 1 and asignado:
            destino = Destino.UCI.value + (" + VENTILADOR MECÁNICO" if p.id in con_vent else "")
            just = f"Posición #{posicion} en el orden del comité ({nombre_marco})."
        elif e.prioridad == 1:
            destino = Destino.CRUE_EXPANSION.value
            just = (f"Posición #{posicion} en el orden del comité ({nombre_marco}); sin recurso UCI disponible. "
                    "Lista de espera priorizada.")
        elif e.prioridad == 2:
            destino, just = Destino.INTERMEDIOS.value, "Riesgo de deterioro sin indicación de intubación inmediata."
        elif e.prioridad == 3:
            destino, just = Destino.HOSPITALIZACION.value, "Paciente estable."
        else:
            destino = Destino.PALIATIVO.value
            just = "Futilidad terapéutica (REG-CLIN-03): adecuación del esfuerzo terapéutico, no abandono."
        return DecisionFinal(
            paciente_id=p.id, nombre=p.nombre, prioridad=e.prioridad, destino=destino, asignado_uci=asignado,
            asignado_ventilador=p.id in con_vent, posicion_ranking=posicion,
            puntaje_desempate=puntajes.get(p.id), validacion_etica=validacion, justificacion=just,
            plan_alternativo=plan,
        )

    @staticmethod
    def _redactar_acta(modo: ModoEtico, estado_h: str, estado: RecursoHospitalario, pacientes: List[Paciente],
                       decisiones: Dict[str, DecisionFinal], objeciones: List[str], validacion: str,
                       contrafactual: Optional[List[str]], asignados: List[str]) -> str:
        por_id = {p.id: p for p in pacientes}
        marco = kb.MARCOS_ETICOS[modo.value]
        ganadores = ", ".join(por_id[x].nombre for x in asignados) or "ninguno"
        lineas = [
            f"**Marco ético:** {marco['nombre']}  ",
            f"**Estado hospitalario:** {estado_h} · {estado.camas_uci_disponibles} cama(s) UCI y "
            f"{estado.ventiladores_disponibles} ventilador(es) disponibles  ",
            f"**Recurso UCI asignado a:** {ganadores}  ",
            f"**Validación bioético-legal:** {validacion}",
        ]
        if contrafactual is not None and set(contrafactual) != set(asignados):
            alt = ", ".join(por_id[x].nombre for x in contrafactual) or "ninguno"
            lineas.append(f"\n⚠️ **Prueba contrafáctica:** con la edad neutralizada, el recurso se asignaría a: {alt}.")
        if objeciones:
            lineas.append("\n**Objeciones registradas:**")
            lineas.extend(f"- {o}" for o in objeciones)
        lineas.append("\n> Recomendación técnica auditable. La decisión final corresponde al médico tratante y al "
                      "Comité de Ética Hospitalaria (Ley 23 de 1981 · Human-in-the-Loop).")
        return "\n".join(lineas)

    def _traza(self, regla: str, condicion: str, conclusion: str, fundamento: str, pid: Optional[str],
               nivel: Nivel, etapa: Etapa) -> TrazaInferencia:
        return TrazaInferencia(agente=self.nombre, regla_id=regla, condicion_evaluada=condicion,
                               conclusion=conclusion, fundamento_normativo_o_medico=fundamento,
                               paciente_id=pid, nivel=nivel.value, etapa=etapa.value)
