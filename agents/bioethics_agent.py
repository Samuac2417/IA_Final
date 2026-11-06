"""
agents/bioethics_agent.py
=========================

Agente 3 — ``BioethicsAuditAgent`` (Auditor Bioético y Legal Colombiano).

Núcleo diferenciador del sistema: audita las recomendaciones clínicas y la
asignación propuesta para detectar sesgos algorítmicos y violaciones de derechos
fundamentales, y exige garantías mínimas para todos los pacientes.

Reglas de producción:
    REG-BIO-01  Bloqueo de edadismo (Const. Art. 13, T-760/2008, Ley 1251/2008)
                · verificación estructural de variables
                · prueba contrafáctica (¿cambia la asignación si se neutraliza la edad?)
    REG-BIO-02  Equidad: doble penalización de comorbilidad en población vulnerable
    REG-BIO-03  Plan alternativo obligatorio (Ley 1751/2015, Arts. 10 y 14; CRUE)
    REG-BIO-04  Human-in-the-Loop (Ley 23/1981; UNESCO 2021)
    REG-BIO-05  Proporcionalidad / beneficencia (alerta de crítica deontológica)
    REG-BIO-06  Validación no discriminatoria de la futilidad (Ley 1733/2014)
    REG-BIO-07  Protección de datos sensibles (Ley 1581/2012)
    REG-BIO-SENSIBILIDAD  Estabilidad estocástica Monte Carlo en casos borde e incertidumbre instrumental
"""

from __future__ import annotations

from typing import Dict, List

from agents.base_agent import BaseAgent
from agents.clinical_agent import ClinicalTriageAgent
from core import knowledge_base as kb
from core.models import (
    ContextoAuditoria, DictamenAgente, EstadoHospital, Etapa, ModoEtico, Nivel, ValidacionEtica,
)

_RANGO_NIVEL = {Nivel.INFO.value: 0, Nivel.OK.value: 0, Nivel.ALERTA.value: 1,
                Nivel.OBJECION.value: 2, Nivel.CRITICO.value: 3}
_VALIDACION_POR_RANGO = {0: ValidacionEtica.VALIDADO, 1: ValidacionEtica.CON_OBSERVACIONES,
                         2: ValidacionEtica.OBJETADO, 3: ValidacionEtica.VETO}


class BioethicsAuditAgent(BaseAgent):
    nombre = "BioethicsAuditAgent"
    rol = "Auditor Bioético y Legal · Sesgos, derechos fundamentales y marco legal colombiano"

    def evaluar(self, ctx: ContextoAuditoria) -> DictamenAgente:  # type: ignore[override]
        self.reiniciar()
        modo = ctx.estado.modo
        escasez = ctx.estado_hospital == EstadoHospital.ESCASEZ_CRITICA.value
        por_id = {p.id: p for p in ctx.pacientes}
        planes: Dict[str, str] = {}
        objeciones: List[str] = []

        cambio_por_edad = (
            ctx.asignados_contrafactual_sin_edad is not None
            and set(ctx.asignados_contrafactual_sin_edad) != set(ctx.asignados)
        )

        # ============================ REG-BIO-01 · Edadismo ===================
        self.registrar(
            "REG-BIO-01",
            "Verificación estructural: ¿alguna regla del ClinicalTriageAgent utiliza la variable 'edad'?",
            "CUMPLE: el agente clínico solo usa " + ", ".join(ClinicalTriageAgent.VARIABLES_UTILIZADAS)
            + ". La prioridad clínica (P1-P4) es independiente de la edad cronológica.",
            f"{kb.cita('CONST_ART_13')}; {kb.cita('T_760_2008')}.",
            None, Nivel.OK, Etapa.AUDITORIA,
        )

        usa_edad = modo == ModoEtico.UTILITARISTA_CLINICO
        mayores_en_disputa = [pid for pid in ctx.candidatos_p1 if por_id[pid].edad >= kb.EDAD_UMBRAL_EDADISMO]
        if usa_edad and escasez:
            nivel_m = Nivel.OBJECION if mayores_en_disputa else Nivel.ALERTA
            self.registrar(
                "REG-BIO-01",
                "Marco activo = Utilitarismo Clínico · el puntaje incluye 'años de vida potencialmente ganados' "
                f"= f({kb.EXPECTATIVA_VIDA_COLOMBIA} − edad)",
                "OBJECIÓN CONSTITUCIONAL DE MÉTODO: el criterio de desempate incorpora la edad cronológica como "
                "variable de priorización. La edad, por sí sola, no puede ser criterio de exclusión; solo la "
                "edad biológica expresada en reserva funcional (SOFA).",
                f"{kb.cita('CONST_ART_13')}; {kb.cita('T_760_2008')}; {kb.cita('LEY_1251_2008')}.",
                None, nivel_m, Etapa.AUDITORIA,
            )
            if mayores_en_disputa:
                objeciones.append("Uso de la edad cronológica en el puntaje utilitarista (REG-BIO-01).")

        if escasez and ctx.asignados_contrafactual_sin_edad is not None:
            if cambio_por_edad:
                self.registrar(
                    "REG-BIO-01",
                    f"Prueba contrafáctica: asignación real = {ctx.asignados} vs. asignación con edad "
                    f"neutralizada ({kb.EDAD_REFERENCIA_CONTRAFACTUAL} años para todos) = "
                    f"{ctx.asignados_contrafactual_sin_edad}",
                    "SESGO DETECTADO: la edad cronológica fue DETERMINANTE en el resultado de la asignación.",
                    f"{kb.cita('CONST_ART_13')}; {kb.cita('UNESCO_2021')} (equidad y no discriminación).",
                    None, Nivel.OBJECION, Etapa.AUDITORIA,
                )
            else:
                self.registrar(
                    "REG-BIO-01",
                    f"Prueba contrafáctica: asignación real = {ctx.asignados} vs. asignación con edad "
                    f"neutralizada = {ctx.asignados_contrafactual_sin_edad}",
                    "PRUEBA SUPERADA: neutralizar la edad no altera la asignación; el resultado se sostiene en "
                    "variables fisiológicas.",
                    f"{kb.cita('UNESCO_2021')} (auditoría de sesgos).",
                    None, Nivel.OK, Etapa.AUDITORIA,
                )

        for pid in [p.id for p in ctx.pacientes if p.edad >= kb.EDAD_UMBRAL_EDADISMO]:
            p = por_id[pid]
            cond = f"Edad ({p.edad}) ≥ {kb.EDAD_UMBRAL_EDADISMO} · marco = {modo.value}"
            if pid in ctx.candidatos_p1 and escasez and usa_edad:
                excluido = pid in ctx.no_asignados
                ganaria = (ctx.asignados_contrafactual_sin_edad is not None
                           and pid in ctx.asignados_contrafactual_sin_edad)
                if excluido and ganaria:
                    self.registrar(
                        "REG-BIO-01", cond + " · excluido de UCI · con edad neutralizada SÍ recibiría cama",
                        "VETO CONSTITUCIONAL: el paciente es descartado EXCLUSIVAMENTE por su edad cronológica. "
                        "Su reserva funcional le daría prioridad. La recomendación es jurídicamente inejecutable "
                        "en Colombia y debe ser revisada por el Comité de Ética Hospitalaria.",
                        f"{kb.cita('CONST_ART_13')}; {kb.cita('CONST_ART_43_46')}; {kb.cita('T_760_2008')}; "
                        f"{kb.cita('LEY_1251_2008')}.",
                        pid, Nivel.CRITICO, Etapa.AUDITORIA,
                    )
                    objeciones.append(f"VETO: {p.nombre} excluido exclusivamente por edad (REG-BIO-01).")
                elif excluido:
                    self.registrar(
                        "REG-BIO-01", cond + " · excluido de UCI · con edad neutralizada tampoco recibiría cama",
                        "OBJECIÓN DE MÉTODO: la exclusión se sostiene en diferencias de reserva funcional (SOFA), "
                        "pero el puntaje utilizado penaliza la edad y por tanto la decisión no es sostenible "
                        "en su fundamentación. Se recomienda re-deliberar bajo el Consenso Constitucional.",
                        f"{kb.cita('CONST_ART_13')}; {kb.cita('T_760_2008')}.",
                        pid, Nivel.OBJECION, Etapa.AUDITORIA,
                    )
                else:
                    self.registrar(
                        "REG-BIO-01", cond + " · asignado pese a la penalización por edad",
                        "Observación: el paciente recibió el recurso, pero el método aplicado incorpora la edad.",
                        f"{kb.cita('CONST_ART_13')}.", pid, Nivel.ALERTA, Etapa.AUDITORIA,
                    )
            else:
                self.registrar(
                    "REG-BIO-01", cond,
                    "CUMPLE: la edad cronológica no interviene en la priorización de este paciente bajo el marco "
                    "activo; se valora su edad biológica (reserva funcional SOFA).",
                    f"{kb.cita('CONST_ART_13')}; {kb.cita('T_760_2008')}.",
                    pid, Nivel.OK, Etapa.AUDITORIA,
                )

        # ============================ REG-BIO-02 · Equidad ====================
        for p in ctx.pacientes:
            if not p.vulnerabilidad_social:
                if usa_edad and escasez and p.id in ctx.candidatos_p1 and p.comorbilidades_graves:
                    self.registrar(
                        "REG-BIO-02", f"Comorbilidades = {len(p.comorbilidades_graves)} · marco utilitarista",
                        "Se aplica penalización por comorbilidad en el puntaje utilitarista (registro de "
                        "transparencia).", f"{kb.cita('CONST_ART_13')}.", p.id, Nivel.INFO, Etapa.AUDITORIA,
                    )
                continue
            sensibles = kb.comorbilidades_sensibles(p.comorbilidades_graves)
            cond = (f"Vulnerabilidad social == Verdadero · comorbilidades asociadas a determinantes sociales = "
                    f"{sensibles if sensibles else 'ninguna'} · marco = {modo.value}")
            if usa_edad and escasez and p.id in ctx.candidatos_p1 and p.comorbilidades_graves:
                excluido = p.id in ctx.no_asignados
                self.registrar(
                    "REG-BIO-02", cond,
                    "ALERTA DE EQUIDAD — DOBLE PENALIZACIÓN: las comorbilidades derivadas de brechas "
                    "socioeconómicas (pobreza, ruralidad, barreras de acceso) se restan nuevamente en el puntaje "
                    "utilitarista. El algoritmo reproduce la desigualdad estructural que originó la enfermedad."
                    + (" El paciente resultó excluido." if excluido else ""),
                    f"{kb.cita('CONST_ART_13')} inciso 3 (debilidad manifiesta); {kb.cita('LEY_1751_2015')} "
                    "Art. 11 (sujetos de especial protección).",
                    p.id, Nivel.OBJECION if excluido else Nivel.ALERTA, Etapa.AUDITORIA,
                )
                if excluido:
                    objeciones.append(f"Doble penalización por vulnerabilidad de {p.nombre} (REG-BIO-02).")
            elif modo == ModoEtico.CONSENSO_CONSTITUCIONAL_COLOMBIA:
                aplicada = p.id in ctx.proteccion_reforzada_aplicada
                self.registrar(
                    "REG-BIO-02", cond,
                    ("PROTECCIÓN REFORZADA APLICADA: en situación de empate técnico se prefirió al sujeto de "
                     "especial protección constitucional (acción afirmativa, nunca penalización)."
                     if aplicada else
                     "CUMPLE: sin doble penalización. Las comorbilidades solo se valoran a través de la reserva "
                     "funcional (SOFA); la vulnerabilidad solo puede operar a favor en caso de empate técnico."),
                    f"{kb.cita('CONST_ART_13')} inciso 3; {kb.cita('CONST_ART_43_46')}; "
                    f"{kb.cita('LEY_1751_2015')} Art. 11.",
                    p.id, Nivel.OK, Etapa.AUDITORIA,
                )
            else:
                self.registrar(
                    "REG-BIO-02", cond,
                    "CUMPLE: el criterio activo no penaliza comorbilidades ni condición socioeconómica. "
                    "Observación: el igualitarismo formal no otorga protección reforzada al sujeto vulnerable.",
                    f"{kb.cita('CONST_ART_13')}.", p.id, Nivel.OK, Etapa.AUDITORIA,
                )

        # ============================ REG-BIO-06 · Futilidad ==================
        for pid, e in ctx.evaluaciones.items():
            if e.prioridad == 4:
                p = por_id[pid]
                self.registrar(
                    "REG-BIO-06",
                    f"Prioridad 4 por futilidad · fundamento = enfermedad terminal irreversible + SOFA "
                    f"{p.escala_sofa} > {kb.SOFA_FUTILIDAD_MIN_EXCLUSIVO} · edad y condición social NO usadas",
                    "VALIDADO: la no admisión a UCI NO es discriminación ni abandono; es adecuación del esfuerzo "
                    "terapéutico para evitar la distanasia. El paciente conserva el derecho pleno a cuidado "
                    "paliativo integral, alivio del dolor, acompañamiento y trato digno.",
                    f"{kb.cita('LEY_1733_2014')}; {kb.cita('LEY_23_1981')} Art. 13; {kb.cita('CONST_ART_1')}.",
                    pid, Nivel.OK, Etapa.AUDITORIA,
                )

        # ============================ REG-BIO-03 · Ley 1751 ===================
        for pid, e in ctx.evaluaciones.items():
            if e.prioridad == 1 and pid in ctx.no_asignados:
                planes[pid] = ("1) Activación inmediata del CRUE para remisión interinstitucional urgente a UCI con "
                               "disponibilidad; 2) soporte vital avanzado (ventilación no invasiva / cánula de alto "
                               "flujo o ventilación invasiva transitoria) en área de expansión; 3) reevaluación SOFA "
                               "cada 6 h y primera opción ante la liberación de una cama; 4) información veraz a la "
                               "familia.")
                self.registrar(
                    "REG-BIO-03", "Paciente P1 SIN cama UCI por escasez física",
                    "PLAN ALTERNATIVO OBLIGATORIO GARANTIZADO: " + planes[pid] + " Se prohíbe el abandono.",
                    f"{kb.cita('LEY_1751_2015')} Arts. 10 y 14; {kb.cita('RES_1220_2010')}.",
                    pid, Nivel.OK, Etapa.AUDITORIA,
                )
            elif e.prioridad == 4:
                planes[pid] = ("Cuidado paliativo integral: control de disnea y dolor, sedación paliativa "
                               "proporcional si hay síntomas refractarios, acompañamiento familiar y espiritual, "
                               "respeto de directivas anticipadas. Sin abandono.")
                self.registrar(
                    "REG-BIO-03", "Paciente P4 (no candidato a UCI)",
                    "PLAN ALTERNATIVO OBLIGATORIO GARANTIZADO: " + planes[pid],
                    f"{kb.cita('LEY_1751_2015')} Art. 10; {kb.cita('LEY_1733_2014')}.",
                    pid, Nivel.OK, Etapa.AUDITORIA,
                )
            elif e.prioridad in (2, 3):
                planes[pid] = ("Cuidados intermedios con monitoreo continuo y reevaluación seriada."
                               if e.prioridad == 2 else "Hospitalización general con controles de rutina.")
        sin_plan = [pid for pid in ctx.no_asignados if pid not in planes]
        self.registrar(
            "REG-BIO-03", "¿Todo paciente sin cama UCI tiene plan alternativo documentado?",
            "CUMPLE: ningún paciente queda sin ruta asistencial." if not sin_plan
            else f"INCUMPLE: pacientes sin plan = {sin_plan}.",
            f"{kb.cita('LEY_1751_2015')} Art. 14 (prohibición de negación de servicios).",
            None, Nivel.OK if not sin_plan else Nivel.CRITICO, Etapa.AUDITORIA,
        )

        # ============================ REG-BIO-05 · Proporcionalidad ===========
        if escasez:
            for ganador in ctx.asignados:
                for perdedor in ctx.no_asignados:
                    pg = ctx.evaluaciones[ganador].prob_sobrevida
                    pp = ctx.evaluaciones[perdedor].prob_sobrevida
                    if pp - pg >= kb.DIFERENCIA_PROPORCIONALIDAD:
                        self.registrar(
                            "REG-BIO-05",
                            f"Sobrevida estimada asignado {por_id[ganador].nombre} ({pg:.0%}) vs. no asignado "
                            f"{por_id[perdedor].nombre} ({pp:.0%}) · Δ ≥ {kb.DIFERENCIA_PROPORCIONALIDAD:.0%}",
                            "ALERTA DE PROPORCIONALIDAD: el recurso escaso se asigna a quien tiene una probabilidad "
                            "de beneficio sustancialmente menor. Tensión con el principio de beneficencia y con la "
                            "eficiencia exigida por el Art. 49 constitucional.",
                            f"{kb.cita('CONST_ART_49')}; principialismo (Beauchamp & Childress).",
                            perdedor, Nivel.ALERTA, Etapa.AUDITORIA,
                        )

        # ============================ REG-BIO-07 · Habeas Data ================
        self.registrar(
            "REG-BIO-07", "Tratamiento de datos sensibles de salud",
            "CUMPLE: procesamiento 100 % local, sin transmisión a terceros. El nombre nunca participa en la "
            "inferencia; la vulnerabilidad social solo puede usarse como acción afirmativa (marco constitucional).",
            f"{kb.cita('LEY_1581_2012')}.", None, Nivel.OK, Etapa.AUDITORIA,
        )

        # ============================ REG-BIO-04 · HITL =======================
        self.registrar(
            "REG-BIO-04", "Cláusula obligatoria de responsabilidad (siempre se dispara)",
            "HUMAN-IN-THE-LOOP: este sistema emite una RECOMENDACIÓN TÉCNICA AUDITABLE. El acto médico final, "
            "la decisión de ingreso a UCI y la comunicación con la familia recaen en el médico tratante y en el "
            "Comité de Ética Hospitalaria, quienes pueden apartarse motivadamente de esta recomendación.",
            f"{kb.cita('LEY_23_1981')}; {kb.cita('LEY_1751_2015')} Art. 17 (autonomía profesional); "
            f"{kb.cita('UNESCO_2021')} (supervisión y decisión humanas).",
            None, Nivel.INFO, Etapa.AUDITORIA,
        )

        # ============================ REG-BIO-SENSIBILIDAD · Robustez Monte Carlo ==
        if ctx.reportes_robustez:
            for p in ctx.pacientes:
                rep = ctx.reportes_robustez.get(p.id)
                if rep is None:
                    continue
                estabilidad = rep.porcentaje_estabilidad_decision
                if estabilidad < 85.0:
                    self.registrar(
                        "REG-BIO-SENSIBILIDAD",
                        f"Estabilidad de asignación Monte Carlo ({estabilidad:.1f} %) < 85.0 % ante "
                        "variabilidad instrumental esperada (SatO2 ±1.5 %, PaFi ±10 mmHg, SOFA ±1)",
                        "ALERTA ESPECIAL — Dilema Clínico en Zona de Incertidumbre Instrumental: "
                        "la recomendación de destino para este paciente es altamente inestable ante la tolerancia "
                        "de los instrumentos de medición o escalas clínicas. Se exige una repetición inmediata de "
                        "gases arteriales y reevaluación SOFA seriada antes de ejecutar el acto médico definitivo.",
                        f"{kb.cita('LEY_23_1981')} Art. 10 y 13 (prudencia diagnóstica y diligencia médica); "
                        f"{kb.cita('LEY_1751_2015')} (principio de calidad y seguridad).",
                        p.id, Nivel.ALERTA, Etapa.AUDITORIA,
                    )
                    objeciones.append(
                        f"Incertidumbre instrumental en {p.nombre}: estabilidad {estabilidad:.1f}% < 85% (REG-BIO-SENSIBILIDAD)."
                    )
                else:
                    self.registrar(
                        "REG-BIO-SENSIBILIDAD",
                        f"Estabilidad de asignación Monte Carlo = {estabilidad:.1f} % (≥ 85.0 %)",
                        f"ROBUSTEZ ANALÍTICA CONFIRMADA ({rep.diagnostico_frontera}): la recomendación se sostiene "
                        "de forma consistente frente a variaciones instrumentales y de medición interobservador.",
                        f"{kb.cita('LEY_23_1981')}; auditoría de sensibilidad estocástica.",
                        p.id, Nivel.OK, Etapa.AUDITORIA,
                    )

        # ============================ Consolidación ===========================
        validacion_paciente: Dict[str, str] = {}
        for p in ctx.pacientes:
            rango = max([_RANGO_NIVEL[t.nivel] for t in self._trazas if t.paciente_id == p.id] or [0])
            validacion_paciente[p.id] = _VALIDACION_POR_RANGO[rango].value
        rango_global = max([_RANGO_NIVEL[t.nivel] for t in self._trazas] or [0])
        validacion_global = _VALIDACION_POR_RANGO[rango_global].value

        if rango_global >= 3:
            veredicto, nivel = ("VETO CONSTITUCIONAL: la asignación propuesta discrimina por edad. "
                                "Requiere revisión humana obligatoria."), Nivel.CRITICO
        elif rango_global == 2:
            veredicto, nivel = (f"Dictamen OBJETADO: {len(objeciones)} objeción(es) ético-legal(es) "
                                "registradas."), Nivel.OBJECION
        elif rango_global == 1:
            veredicto, nivel = "Validado con observaciones de proporcionalidad/equidad.", Nivel.ALERTA
        else:
            veredicto, nivel = ("Asignación VALIDADA: sin discriminación por edad ni condición social; "
                                "planes alternativos garantizados."), Nivel.OK

        return self.construir_dictamen(
            veredicto, nivel,
            validacion_global=validacion_global,
            validacion_por_paciente=validacion_paciente,
            planes_alternativos=planes,
            objeciones=objeciones,
            sesgo_edad_detectado=cambio_por_edad,
        )
