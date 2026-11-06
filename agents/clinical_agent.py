"""
agents/clinical_agent.py
========================

Agente 1 — ``ClinicalTriageAgent`` (Especialista en Medicina Crítica).

Evalúa EXCLUSIVAMENTE la urgencia fisiológica y la probabilidad de sobrevida a
corto plazo. Por diseño, ninguna regla de este agente lee la edad, el nombre, la
hora de llegada ni la vulnerabilidad social del paciente (principio de
minimización de variables y no discriminación; verificado por REG-BIO-01).

Reglas de producción:
    REG-CLIN-00  Registro de hechos y triage de urgencias (Res. 5596/2015)
    REG-CLIN-01  Hipoxemia refractaria (C1)
    REG-CLIN-01B Clasificación de oxigenación SDRA (Berlín 2012)
    REG-CLIN-02  Evaluación multiorgánica SOFA (C2)
    REG-CLIN-03  Futilidad terapéutica / adecuación del esfuerzo (C3)
    REG-CLIN-04  Compromiso neurológico (Glasgow)
    REG-CLIN-05  Registro de comorbilidades (informativo, no excluyente)
    REG-CLIN-06  Clasificación final de prioridad UCI (1-4)
"""

from __future__ import annotations

from typing import Dict, List

from agents.base_agent import BaseAgent
from core import knowledge_base as kb
from core.models import (
    DictamenAgente, Etapa, EvaluacionClinica, Nivel, Paciente, Prioridad,
)


class ClinicalTriageAgent(BaseAgent):
    nombre = "ClinicalTriageAgent"
    rol = "Especialista en Medicina Crítica · Severidad y reversibilidad fisiológica"

    # Variables que este agente tiene permitido leer (auditable por REG-BIO-01).
    VARIABLES_UTILIZADAS = ("sat_o2", "pafi", "escala_sofa", "glasgow",
                            "requiere_ventilador", "enfermedad_terminal_irreversible",
                            "comorbilidades_graves (solo registro)")

    # ----------------------------------------------------------------- público
    def evaluar(self, pacientes: List[Paciente]) -> DictamenAgente:  # type: ignore[override]
        self.reiniciar()
        evaluaciones: Dict[str, EvaluacionClinica] = {}
        for paciente in pacientes:
            evaluaciones[paciente.id] = self._evaluar_paciente(paciente)

        conteo = {p: sum(1 for e in evaluaciones.values() if e.prioridad == p) for p in (1, 2, 3, 4)}
        veredicto = (
            f"{len(pacientes)} pacientes evaluados → P1: {conteo[1]} · P2: {conteo[2]} · "
            f"P3: {conteo[3]} · P4: {conteo[4]}."
        )
        if conteo[1] or conteo[4]:
            nivel = Nivel.CRITICO
        elif conteo[2]:
            nivel = Nivel.ALERTA
        else:
            nivel = Nivel.OK
        return self.construir_dictamen(veredicto, nivel, evaluaciones=evaluaciones, conteo=conteo)

    # ----------------------------------------------------------------- privado
    @staticmethod
    def calcular_puntaje_urgencia(p: Paciente) -> float:
        """Índice docente de severidad fisiológica inmediata (0-100)."""
        hip = min(1.0, max(0.0, (100.0 - p.sat_o2) / 30.0))
        oxi = min(1.0, max(0.0, (400.0 - p.pafi) / 350.0))
        neuro = (15 - p.glasgow) / 12.0
        sofa = p.escala_sofa / 24.0
        puntaje = 100.0 * (0.30 * hip + 0.30 * oxi + 0.20 * neuro + 0.20 * sofa)
        if p.requiere_ventilador:
            puntaje += 5.0
        return round(min(100.0, puntaje), 1)

    def _evaluar_paciente(self, p: Paciente) -> EvaluacionClinica:
        pid = p.id

        # REG-CLIN-00 · Registro de hechos + triage de urgencias -------------
        triage = kb.clasificar_triage_urgencias(p.sat_o2, p.glasgow, p.pafi, p.requiere_ventilador)
        self.registrar(
            "REG-CLIN-00",
            f"Hechos: SatO2 = {p.sat_o2:.1f} % · PaFi = {p.pafi} · SOFA = {p.escala_sofa} · "
            f"Glasgow = {p.glasgow} · Requiere ventilador = {'Sí' if p.requiere_ventilador else 'No'}",
            f"Clasificación inicial: {triage}.",
            f"{kb.cita('RES_5596_2015')}; Sistema de Triage Manchester.",
            pid, Nivel.INFO, Etapa.HECHO,
        )

        # REG-CLIN-01 · Hipoxemia refractaria (C1) ---------------------------
        sat_baja = p.sat_o2 < kb.UMBRAL_SATO2_HIPOXEMIA_REFRACTARIA
        pafi_baja = p.pafi < kb.UMBRAL_PAFI_HIPOXEMIA_REFRACTARIA
        hipoxemia = sat_baja or pafi_baja
        cond_c1 = (f"SatO2 ({p.sat_o2:.1f} %) < {kb.UMBRAL_SATO2_HIPOXEMIA_REFRACTARIA:.0f} % "
                   f"[{'VERDADERO' if sat_baja else 'FALSO'}] OR PaFi ({p.pafi}) < "
                   f"{kb.UMBRAL_PAFI_HIPOXEMIA_REFRACTARIA} [{'VERDADERO' if pafi_baja else 'FALSO'}]")
        if hipoxemia:
            self.registrar(
                "REG-CLIN-01", cond_c1,
                "HIPOXEMIA REFRACTARIA: se activa alerta de soporte ventilatorio invasivo inmediato "
                "(intubación orotraqueal y ventilación mecánica protectora).",
                "Guías de manejo de insuficiencia respiratoria aguda hipoxémica; ventilación protectora "
                "(ARDSNet). Umbrales docentes SatO2 < 85 % / PaFi < 150.",
                pid, Nivel.CRITICO, Etapa.REGLA,
            )
        else:
            self.registrar(
                "REG-CLIN-01", cond_c1,
                "No cumple criterios de hipoxemia refractaria; no se activa alerta de soporte invasivo.",
                "Umbrales docentes SatO2 < 85 % / PaFi < 150.",
                pid, Nivel.OK, Etapa.REGLA,
            )

        # REG-CLIN-01B · Berlín ---------------------------------------------
        sdra = kb.clasificar_sdra_berlin(p.pafi)
        self.registrar(
            "REG-CLIN-01B", f"PaFi = {p.pafi} mmHg",
            f"Oxigenación: {sdra}.",
            "ARDS Definition Task Force. Definición de Berlín del SDRA. JAMA 2012;307(23):2526-33.",
            pid, Nivel.ALERTA if p.pafi <= 200 else Nivel.INFO, Etapa.REGLA,
        )

        # REG-CLIN-04 · Glasgow ---------------------------------------------
        if p.glasgow <= kb.GLASGOW_PROTECCION_VIA_AEREA:
            concl_g, nivel_g = ("Compromiso neurológico severo: indicación de protección de vía aérea "
                                "(intubación).", Nivel.CRITICO)
        elif p.glasgow <= kb.GLASGOW_COMPROMISO_MODERADO:
            concl_g, nivel_g = "Compromiso neurológico moderado: vigilancia neurológica estrecha.", Nivel.ALERTA
        else:
            concl_g, nivel_g = "Estado de conciencia conservado o con compromiso leve.", Nivel.OK
        self.registrar(
            "REG-CLIN-04", f"Glasgow = {p.glasgow} (≤ 8 severo · 9-12 moderado · 13-15 leve)",
            concl_g, "Teasdale G, Jennett B. Escala de Coma de Glasgow. Lancet 1974.",
            pid, nivel_g, Etapa.REGLA,
        )

        # REG-CLIN-02 · SOFA (C2) -------------------------------------------
        sofa = p.escala_sofa
        reversibilidad = kb.reversibilidad_por_sofa(sofa)
        prob = kb.prob_sobrevida_por_sofa(sofa)
        if reversibilidad == "ALTA":
            cond_s, concl_s, nivel_s = (
                f"SOFA ({sofa}) ≤ {kb.SOFA_REVERSIBILIDAD_ALTA_MAX}",
                f"Alta probabilidad de reversibilidad (> 80 % de sobrevida estimada; tabla docente: "
                f"{prob:.0%}).", Nivel.OK)
        elif reversibilidad == "MEDIA":
            cond_s, concl_s, nivel_s = (
                f"{kb.SOFA_REVERSIBILIDAD_ALTA_MAX + 1} ≤ SOFA ({sofa}) ≤ {kb.SOFA_REVERSIBILIDAD_MEDIA_MAX}",
                f"Disfunción multiorgánica moderada-severa pero rescatable en UCI (sobrevida estimada "
                f"{prob:.0%}).", Nivel.ALERTA)
        else:
            cond_s, concl_s, nivel_s = (
                f"SOFA ({sofa}) ≥ {kb.SOFA_REVERSIBILIDAD_MEDIA_MAX + 1}",
                f"Disfunción crítica severa con alta mortalidad (> 50 %; sobrevida estimada {prob:.0%}).",
                Nivel.CRITICO)
        self.registrar(
            "REG-CLIN-02", cond_s, concl_s,
            "Vincent JL et al. SOFA score, Intensive Care Med 1996; Ferreira FL et al. JAMA 2001 "
            "(aproximación docente de sobrevida).",
            pid, nivel_s, Etapa.REGLA,
        )

        # REG-CLIN-05 · Comorbilidades (solo registro) ----------------------
        if p.comorbilidades_graves:
            self.registrar(
                "REG-CLIN-05", f"Comorbilidades graves = {', '.join(p.comorbilidades_graves)}",
                "Se registran como antecedente clínico. Su impacto pronóstico ya está expresado en la reserva "
                "funcional (SOFA); NO se usan como criterio autónomo de exclusión en este agente.",
                "Principio de no discriminación por diagnóstico (Const. Art. 13) y Sepsis-3 (Singer M et al., "
                "JAMA 2016): la disfunción orgánica se mide por SOFA.",
                pid, Nivel.INFO, Etapa.HECHO,
            )

        # REG-CLIN-03 · Futilidad (C3) --------------------------------------
        futil = p.enfermedad_terminal_irreversible and sofa > kb.SOFA_FUTILIDAD_MIN_EXCLUSIVO
        if p.enfermedad_terminal_irreversible:
            cond_f = (f"Enfermedad terminal irreversible == Verdadero AND SOFA ({sofa}) > "
                      f"{kb.SOFA_FUTILIDAD_MIN_EXCLUSIVO} [{'VERDADERO' if futil else 'FALSO'}]")
            if futil:
                self.registrar(
                    "REG-CLIN-03", cond_f,
                    "FUTILIDAD TERAPÉUTICA: el ingreso a UCI no modificaría el pronóstico y prolongaría el "
                    "sufrimiento (distanasia / encarnizamiento terapéutico). Se recomienda Prioridad 4: "
                    "Cuidado Paliativo Integral con control de síntomas, acompañamiento familiar y sedación "
                    "paliativa si es necesaria. Esto es adecuación del esfuerzo terapéutico, NO abandono.",
                    f"{kb.cita('LEY_1733_2014')}; {kb.cita('LEY_23_1981')} Art. 13; Res. 2665 de 2018 "
                    "(Documento de Voluntad Anticipada).",
                    pid, Nivel.CRITICO, Etapa.REGLA,
                )
            else:
                self.registrar(
                    "REG-CLIN-03", cond_f,
                    "Existe enfermedad terminal, pero la reserva funcional NO alcanza el umbral de futilidad. "
                    "El paciente conserva su derecho a soporte avanzado; se sugiere explorar directivas "
                    "anticipadas y objetivos de cuidado con paciente y familia.",
                    f"{kb.cita('LEY_1733_2014')}; Res. 2665 de 2018 (Documento de Voluntad Anticipada).",
                    pid, Nivel.ALERTA, Etapa.REGLA,
                )

        # REG-CLIN-06 · Clasificación final ---------------------------------
        necesidad_critica = hipoxemia or p.requiere_ventilador or p.glasgow <= kb.GLASGOW_PROTECCION_VIA_AEREA
        vigilancia = (p.sat_o2 < kb.UMBRAL_SATO2_VIGILANCIA or p.pafi < kb.UMBRAL_PAFI_VIGILANCIA
                      or p.glasgow <= kb.GLASGOW_COMPROMISO_MODERADO or sofa >= kb.SOFA_VIGILANCIA_MIN)

        if futil:
            prioridad = Prioridad.P4_PALIATIVO
            reversibilidad = "NULA"
            cond_p = "Futilidad terapéutica (REG-CLIN-03) == Verdadero"
            concl_p = "PRIORIDAD 4 · Condición irreversible extrema: derivación a cuidados paliativos dignos."
            nivel_p = Nivel.CRITICO
        elif necesidad_critica:
            prioridad = Prioridad.P1_CRITICO_INESTABLE
            cond_p = (f"Necesidad crítica (hipoxemia refractaria = {hipoxemia}, ventilador = "
                      f"{p.requiere_ventilador}, Glasgow ≤ 8 = {p.glasgow <= 8}) AND NOT futilidad")
            extra = ("" if reversibilidad != "BAJA" else
                     " Reversibilidad BAJA documentada: ingresa a P1 porque no cumple causal de futilidad "
                     "(REG-CLIN-03) y el principio de no abandono impide excluirlo sin causa clínica.")
            concl_p = (f"PRIORIDAD 1 · Crítico inestable: requiere UCI inmediata. Reversibilidad "
                       f"{reversibilidad}.{extra}")
            nivel_p = Nivel.CRITICO
        elif vigilancia:
            prioridad = Prioridad.P2_MONITOREO_INTENSIVO
            cond_p = ("SatO2 < 92 % OR PaFi < 300 OR Glasgow ≤ 12 OR SOFA ≥ 2 (Sepsis-3), sin necesidad de "
                      "intubación inmediata")
            concl_p = ("PRIORIDAD 2 · Requiere monitoreo intensivo continuo por riesgo de deterioro, sin "
                       "intubación inmediata.")
            nivel_p = Nivel.ALERTA
        else:
            prioridad = Prioridad.P3_ESTABLE
            cond_p = "Ninguna regla de severidad disparada"
            concl_p = "PRIORIDAD 3 · Estable o con patología leve: manejo en hospitalización general."
            nivel_p = Nivel.OK

        self.registrar(
            "REG-CLIN-06", cond_p, concl_p,
            "Protocolo de priorización de ingreso a UCI (Society of Critical Care Medicine, guías de "
            "admisión y triage UCI 2016; adaptación docente de 4 prioridades).",
            pid, nivel_p, Etapa.REGLA,
        )

        return EvaluacionClinica(
            paciente_id=pid,
            prioridad=int(prioridad),
            reversibilidad=reversibilidad,
            prob_sobrevida=prob,
            puntaje_urgencia=self.calcular_puntaje_urgencia(p),
            hipoxemia_refractaria=hipoxemia,
            requiere_soporte_invasivo=necesidad_critica,
            clasificacion_sdra=sdra,
            triage_urgencias=triage,
            futilidad_terapeutica=futil,
        )
