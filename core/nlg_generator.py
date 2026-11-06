"""
core/nlg_generator.py
=====================

Módulo de Generación de Lenguaje Natural Simbólico (NLG Clínico-Forense).

Produce de manera determinista la Nota Médica y Bioética Oficial para la Historia
Clínica Electrónica conforme a los requerimientos de la Resolución 1995 de 1999
del Ministerio de Salud y Protección Social de Colombia, la Ley 23 de 1981 (Código
de Ética Médica) y la Ley Estatutaria 1751 de 2015 (Ley Estatutaria de Salud).

Diseño 100 % simbólico:
- Basado en plantillas gramaticales clínicas formales y reglas de composición sintáctica.
- Sin llamadas a LLMs externos ni dependencias de red (privacidad de datos sensibles,
  Ley 1581 de 2012).
- Reproducible, auditable y con solemnidad médico-legal.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from core import knowledge_base as kb
from core.models import (
    DecisionFinal,
    DictamenAgente,
    EvaluacionClinica,
    Paciente,
    ReporteRobustez,
    ResultadoComite,
)


class ClinicalNarrativeGenerator:
    """Generador simbólico de narrativa clínica y bioética oficial para historia clínica."""

    def generar_nota_historia_clinica(
        self,
        paciente_id: str,
        resultado: ResultadoComite,
        reporte_robustez: Optional[ReporteRobustez] = None,
        fecha_hora: Optional[str] = None,
        institucion: str = "HOSPITAL UNIVERSITARIO DE ALTA COMPLEJIDAD",
        servicio: str = "Servicio de Urgencias y Cuidado Crítico / UCI",
    ) -> str:
        """Redacta la Nota Médica y Bioética Oficial estructurada en 7 secciones solemnes.

        Cumple estrictamente los requisitos de la Resolución 1995 de 1999, Ley 23 de 1981,
        Ley Estatutaria 1751 de 2015 y Ley 1733 de 2014.
        """
        paciente: Paciente = resultado.paciente(paciente_id)
        evaluacion: EvaluacionClinica = resultado.evaluaciones[paciente_id]
        decision: DecisionFinal = resultado.decisiones[paciente_id]

        dictamen_clinico: DictamenAgente = resultado.dictamenes.get(
            "ClinicalTriageAgent",
            DictamenAgente("ClinicalTriageAgent", "Medicina Crítica", "Sin dictamen registrado"),
        )
        dictamen_recursos: DictamenAgente = resultado.dictamenes.get(
            "ResourceManagerAgent",
            DictamenAgente("ResourceManagerAgent", "Gestión de Recursos", "Sin dictamen registrado"),
        )
        dictamen_bioetico: DictamenAgente = resultado.dictamenes.get(
            "BioethicsAuditAgent",
            DictamenAgente("BioethicsAuditAgent", "Auditor Bioético", "Sin dictamen registrado"),
        )

        rep_rob: Optional[ReporteRobustez] = reporte_robustez
        if rep_rob is None and resultado.reportes_robustez:
            rep_rob = resultado.reportes_robustez.get(paciente_id)

        timestamp_nota = fecha_hora or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        marco_info = kb.MARCOS_ETICOS.get(resultado.modo, {
            "nombre": resultado.modo,
            "escuela": "Marco Filosófico Institucional",
        })

        # Composición sintáctica de las 7 secciones formales
        seccion1 = self._seccion_encabezado(paciente, institucion, servicio, timestamp_nota)
        seccion2 = self._seccion_motivo(resultado, marco_info)
        seccion3 = self._seccion_hechos_clinicos(paciente, evaluacion)
        seccion4 = self._seccion_dictamen_agentes(evaluacion, decision, dictamen_clinico,
                                                  dictamen_recursos, dictamen_bioetico, resultado.modo)
        seccion5 = self._seccion_resolucion_conducta(paciente, evaluacion, decision)
        seccion6 = self._seccion_robustez_analitica(rep_rob)
        seccion7 = self._seccion_clausula_hitl()

        separador = "=" * 84

        nota_completa = (
            f"{separador}\n"
            f"NOTA MÉDICA Y BIOÉTICA OFICIAL DE HISTORIA CLÍNICA ELECTRÓNICA\n"
            f"COMITÉ EXTRAORDINARIO DE TRIAGE ÉTICO Y ASIGNACIÓN DE RECURSOS CRÍTICOS\n"
            f"Soporte Documental conforme a la Resolución 1995 de 1999 (MinSalud Colombia)\n"
            f"{separador}\n\n"
            f"{seccion1}\n\n"
            f"{seccion2}\n\n"
            f"{seccion3}\n\n"
            f"{seccion4}\n\n"
            f"{seccion5}\n\n"
            f"{seccion6}\n\n"
            f"{seccion7}\n\n"
            f"{separador}\n"
            f"REGISTRO CLÍNICO ELECTRÓNICO AUDITABLE — COPIA FIEL AL EXPEDIENTE MÉDICO\n"
            f"{separador}"
        )
        return nota_completa

    # -------------------------------------------------------------------------
    # Reglas de composición sintáctica por sección
    # -------------------------------------------------------------------------
    def _seccion_encabezado(
        self,
        p: Paciente,
        institucion: str,
        servicio: str,
        timestamp_nota: str,
    ) -> str:
        proteccion_especial = (
            "Sujeto de Especial Protección Constitucional (Art. 13 CP / Art. 11 Ley 1751 de 2015)"
            if p.vulnerabilidad_social
            else "Sin registro de vulnerabilidad social diferencial específica"
        )
        return (
            "1. ENCABEZADO HOSPITALARIO E IDENTIFICACIÓN DEL PACIENTE\n"
            "--------------------------------------------------------------------\n"
            f"Institución Prestadora de Salud: {institucion}\n"
            f"Servicio / Unidad Asistencial:   {servicio}\n"
            f"Fecha y Hora de Deliberación:    {timestamp_nota}\n"
            f"Marco de Expedición:             Resolución 1995 de 1999 (Historia Clínica) y Ley 23 de 1981\n"
            f"Paciente:                        {p.nombre}\n"
            f"Identificador Único (Historia):  {p.id}\n"
            f"Edad Cronológica:                {p.edad} años\n"
            f"Ubicación Física Actual:         Urgencias — Cubículo de Monitoreo / Reanimación\n"
            f"Condición Constitucional:        {proteccion_especial}"
        )

    def _seccion_motivo(self, r: ResultadoComite, marco_info: dict) -> str:
        es_escasez = r.estado_hospital == "ESCASEZ_CRITICA"
        d_rec = r.dictamenes.get("ResourceManagerAgent")
        camas_disp = d_rec.datos.get("camas_disponibles", 0) if d_rec else "N/D"
        camas_tot = d_rec.datos.get("camas_totales", 0) if d_rec else "N/D"
        vents_disp = d_rec.datos.get("ventiladores_disponibles", 0) if d_rec else "N/D"
        deficit = d_rec.datos.get("deficit_camas", 0) if d_rec else 0
        cand_p1 = len(r.orden_prioridad)

        if es_escasez:
            declaratoria = (
                "DECLARATORIA DE ESCASEZ CRÍTICA DE RECURSOS (Triaje de Catástrofe / Emergencia Hospitalaria\n"
                "conforme a la Ley 1523 de 2012 y Resolución 5596 de 2015). Se constata un desbalance crítico\n"
                "entre el número de pacientes en estado crítico inestable con indicación de soporte invasivo\n"
                "y la disponibilidad inmediata de camas y ventiladores en la Unidad de Cuidados Intensivos (UCI)."
            )
        else:
            declaratoria = (
                "CAPACIDAD ASISTENCIAL NORMAL (OPERACIÓN PROTOCOLIZADA). La oferta de camas críticas cubre la\n"
                "demanda emergente de pacientes Prioridad 1; la deliberación colegiada opera como garantía de\n"
                "proporcionalidad y asignación técnica protocolizada."
            )

        return (
            "2. MOTIVO DE DELIBERACIÓN EXTRAORDINARIA Y ESTADO HOSPITALARIO\n"
            "--------------------------------------------------------------------\n"
            f"Situación Operativa: {declaratoria}\n\n"
            "Inventario Crítico Institucional en el Turno:\n"
            f"  • Camas UCI libres en la institución:      {camas_disp} de {camas_tot} totales\n"
            f"  • Ventiladores mecánicos invasivos libres: {vents_disp}\n"
            f"  • Pacientes en Prioridad 1 en disputa:     {cand_p1} paciente(s)\n"
            f"  • Déficit inmediato de camas UCI:          {deficit} cama(s)\n"
            f"  • Marco Ético-Filosófico Aplicado:         {marco_info.get('nombre', r.modo)} "
            f"({marco_info.get('escuela', '')})"
        )

    def _seccion_hechos_clinicos(self, p: Paciente, e: EvaluacionClinica) -> str:
        comorb_txt = ", ".join(p.comorbilidades_graves) if p.comorbilidades_graves else "Sin comorbilidades mayores documentadas"
        terminal_txt = (
            "PRESENTE — Condición médica terminal crónica e irreversible (Ley 1733 de 2014)"
            if p.enfermedad_terminal_irreversible
            else "AUSENTE — Sin patología de base terminal irreversible"
        )
        vent_txt = (
            "INDICACIÓN INMEDIATA DE VENTILACIÓN MECÁNICA INVASIVA (Soporte Vital Avanzado)"
            if p.requiere_ventilador
            else "Ventilación espontánea conservada / soporte no invasivo al momento del triaje"
        )

        if p.glasgow <= kb.GLASGOW_PROTECCION_VIA_AEREA:
            glasgow_desc = f"Severo ({p.glasgow}/15) — Requiere protección urgente de vía aérea (intubación)"
        elif p.glasgow <= kb.GLASGOW_COMPROMISO_MODERADO:
            glasgow_desc = f"Moderado ({p.glasgow}/15) — Compromiso de conciencia bajo vigilancia neurológica"
        else:
            glasgow_desc = f"Conservado ({p.glasgow}/15) — Estado de alerta neurológica adecuado"

        return (
            "3. RESUMEN DE HECHOS FISIOLÓGICOS Y ESCALAS CLÍNICAS OBJETIVAS\n"
            "--------------------------------------------------------------------\n"
            f"  • Oximetría de Pulso (SatO2):            {p.sat_o2:.1f} %\n"
            f"  • Gasometría Arterial (PaFi PaO2/FiO2):   {p.pafi} mmHg ({e.clasificacion_sdra})\n"
            f"  • Disfunción Multiorgánica (SOFA):        {p.escala_sofa} / 24 puntos "
            f"(Sobrevida a corto plazo estimada: {e.prob_sobrevida:.0%}, Reversibilidad: {e.reversibilidad})\n"
            f"  • Estado Neurológico (Escala Glasgow):    {glasgow_desc}\n"
            f"  • Requerimiento de Soporte Ventilatorio:  {vent_txt}\n"
            f"  • Comorbilidades y Antecedentes Crónicos: {comorb_txt}\n"
            f"  • Condición de Futilidad / Terminalidad:  {terminal_txt}"
        )

    def _seccion_dictamen_agentes(
        self,
        e: EvaluacionClinica,
        d: DecisionFinal,
        d_clin: DictamenAgente,
        d_rec: DictamenAgente,
        d_bio: DictamenAgente,
        modo: str,
    ) -> str:
        es_paliativo = e.prioridad == 4
        juicio_clin = (
            "Paciente en condición de futilidad terapéutica médica (REG-CLIN-03). La admisión a UCI no modifica "
            "el pronóstico vital y derivaría en distanasia."
            if es_paliativo
            else f"Paciente con urgencia fisiológica {e.puntaje_urgencia:.1f}/100, reversibilidad {e.reversibilidad} "
                 f"y necesidad de soporte intensivo ({e.etiqueta_prioridad})."
        )
        asign_uci_txt = "ASIGNACIÓN FÍSICA AUTORIZADA" if d.asignado_uci else "NO ASIGNADO POR DÉFICIT DE CAMAS"
        auditoria_txt = (
            f"Veredicto del Auditor Bioético: {d.validacion_etica}.\n"
            f"    Análisis: Se aplicó control de garantías fundamentales (Arts. 1, 13 y 46 CP; Sentencia T-760/2008).\n"
            f"    La edad cronológica {'NO interviene como criterio de exclusión' if modo != 'UTILITARISTA_CLINICO' else 'ha sido auditada bajo prueba contrafáctica de edadismo'}."
        )

        return (
            "4. DICTAMEN INDIVIDUAL DE CADA AGENTE DEL COMITÉ MULTIAGENTE\n"
            "--------------------------------------------------------------------\n"
            "4.1 Agente de Medicina Crítica (ClinicalTriageAgent):\n"
            f"    Prioridad Determinada: {e.etiqueta_prioridad} · Triage de Urgencias: {e.triage_urgencias}\n"
            f"    Juicio Fisiológico:    {juicio_clin}\n"
            f"    Conclusión Clínica:    {d_clin.veredicto}\n\n"
            "4.2 Agente de Gestión de Recursos (ResourceManagerAgent):\n"
            f"    Balance Oferta/Demanda: {d_rec.veredicto}\n"
            f"    Resolución Logística:   {asign_uci_txt} "
            f"({'Con asignación de ventilador mecánico' if d.asignado_ventilador else 'Sin asignación de ventilador'})\n\n"
            "4.3 Agente Auditor Bioético y Legal (BioethicsAuditAgent):\n"
            f"    {auditoria_txt}\n"
            f"    Conclusión Bioética:    {d_bio.veredicto}"
        )

    def _seccion_resolucion_conducta(
        self,
        p: Paciente,
        e: EvaluacionClinica,
        d: DecisionFinal,
    ) -> str:
        if d.asignado_uci:
            conducta = (
                f"DESTINO ADJUDICADO: {d.destino}\n"
                "JUSTIFICACIÓN FORMAL DE PROPORCIONALIDAD:\n"
                f"Se justifica el ingreso prioritario a la Unidad de Cuidados Intensivos (Posición #{d.posicion_ranking})\n"
                f"al cumplirse los criterios de proporcionalidad terapéutica, alta probabilidad de rescate vital\n"
                f"y reversibilidad fisiológica documentada (SOFA {p.escala_sofa}, reversibilidad {e.reversibilidad}).\n"
                "Se ordena el traslado inmediato y la apertura del protocolo intensivo."
            )
        elif e.prioridad == 1:
            conducta = (
                f"DESTINO DETERMINADO: {d.destino}\n"
                "ACTIVACIÓN DE RUTA ALTERNATIVA OBLIGATORIA (Ley Estatutaria 1751 de 2015, Arts. 10 y 14):\n"
                f"Al no existir disponibilidad física inmediata de cama UCI institucional (Posición #{d.posicion_ranking}\n"
                "en la lista del comité), se adopta de forma imperativa la siguiente conducta:\n"
                "  1. Activación inmediata del Centro Regulador de Urgencias y Emergencias (CRUE - Resolución 1220 de 2010)\n"
                "     para remisión interinstitucional urgente con solicitud de cupo UCI en la red hospitalaria.\n"
                "  2. Soporte vital avanzado continuo (área de expansión/reanimación con monitoreo hemodinámico estrecho).\n"
                "  3. Reevaluación seriada de gases arteriales y SOFA cada 6 horas, conservando la primera opción de ingreso\n"
                "     ante cualquier egreso o liberación de cupo en la UCI local.\n"
                "  4. Comunicación continua, clara y empática con el paciente y su núcleo familiar.\n"
                "PROHIBICIÓN EXPRESA DE ABANDONO: Conforme al Art. 14 de la Ley 1751 de 2015, la limitación de camas\n"
                "no autoriza la suspensión de la atención médica. El paciente permanece bajo cuidado activo ininterrumpido."
            )
        elif e.prioridad == 4:
            conducta = (
                f"DESTINO DETERMINADO: {d.destino}\n"
                "ACTIVACIÓN DE PLAN DE CUIDADO PALIATIVO INTEGRAL (Ley 1733 de 2014 — Ley Consuelo Devis Saavedra):\n"
                "Se determina la no admisión a UCI por criterio de futilidad terapéutica objetiva (enfermedad terminal\n"
                f"irreversible con SOFA {p.escala_sofa} > 14). Se certifica que esta decisión es adecuación del esfuerzo\n"
                "terapéutico y jamás denegación o abandono asistencial.\n"
                "Plan Terapéutico Integral Garantizado:\n"
                "  1. Control estricto de síntomas refractarios, dolor y disnea mediante analgesia multimodal y sedación paliativa.\n"
                "  2. Acompañamiento familiar irrestricto, confort bio-psico-social y espiritual.\n"
                "  3. Respeto absoluto a la voluntad anticipada (Res. 2665 de 2018) y a la dignidad humana (Art. 1 CP)."
            )
        else:
            conducta = (
                f"DESTINO DETERMINADO: {d.destino}\n"
                f"CONDUCTA CLÍNICA: Paciente en {e.etiqueta_prioridad}. Continuar manejo médico integral y monitoreo continuo\n"
                "en la unidad asignada con revaloración clínica periódica."
            )

        plan_alt = f"\nPlan Específico Registrado: {d.plan_alternativo}" if d.plan_alternativo else ""

        return (
            "5. RESOLUCIÓN DE CONSENSO Y CONDUCTA ASISTENCIAL\n"
            "--------------------------------------------------------------------\n"
            f"{conducta}{plan_alt}"
        )

    def _seccion_robustez_analitica(self, r: Optional[ReporteRobustez]) -> str:
        if r is not None:
            pct = r.porcentaje_estabilidad_decision
            return (
                "6. ÍNDICE DE ROBUSTEZ ANALÍTICA Y SENSIBILIDAD ESTOCÁSTICA (MONTE CARLO)\n"
                "--------------------------------------------------------------------\n"
                f"Estabilidad de la Recomendación:  {pct:.1f} % en {r.iteraciones_totales} iteraciones estocásticas\n"
                f"Diagnóstico de Frontera:          {r.diagnostico_frontera}\n"
                "Parámetros de Variabilidad Simulados:\n"
                "  • Tolerancia instrumental de pulsioxímetro:     SatO2 normal truncada (sigma = 1.5 %)\n"
                "  • Variabilidad analítica de gases arteriales:    PaFi normal truncada (sigma = 10.0 mmHg)\n"
                "  • Discrepancia interobservador en disfunción:   Escala SOFA +- 1 punto (p = 0.15)\n"
                f"Directriz de Seguridad Clínica (XAI):\n"
                f"  {r.advertencia_clinica}"
            )
        return (
            "6. ÍNDICE DE ROBUSTEZ ANALÍTICA Y SENSIBILIDAD ESTOCÁSTICA (MONTE CARLO)\n"
            "--------------------------------------------------------------------\n"
            "Estado: Simulación estocástica de Monte Carlo no requerida o pendiente de ejecución.\n"
            "Recomendación: Se sugiere ejecutar el análisis estocástico si el paciente presenta valores fisiológicos\n"
            "cercanos a los umbrales de decisión (SatO2 85%, PaFi 150 mmHg o |ΔSOFA| <= 1)."
        )

    def _seccion_clausula_hitl(self) -> str:
        return (
            "7. CLÁUSULA HUMAN-IN-THE-LOOP (HITL) Y FIRMAS DE RESPONSABILIDAD PROFESIONAL\n"
            "--------------------------------------------------------------------\n"
            "CONFORMIDAD ÉTICO-LEGAL (Ley 23 de 1981 - Código de Ética Médica, Arts. 1, 10, 13 y 15;\n"
            "Ley Estatutaria 1751 de 2015, Art. 17 - Autonomía Médica; Recomendación UNESCO 2021 sobre Ética de la IA):\n\n"
            "El presente documento es una RECOMENDACIÓN TÉCNICA AUDITABLE generada por un sistema multiagente de\n"
            "inteligencia artificial simbólica explicable (XAI). EL ACTO MÉDICO ES PERSONAL, AUTÓNOMO E INDELEGABLE.\n"
            "La decisión asistencial final, la orden de ingreso a UCI, la adecuación terapéutica y la información\n"
            "suministrada al paciente o a sus deudos recaen de manera privativa en el Médico Tratante y en el Comité\n"
            "de Ética Hospitalaria, quienes pueden apartarse motivadamente de este concepto dejando constancia escrita.\n\n\n"
            "___________________________________________        ___________________________________________\n"
            "FIRMA DEL MÉDICO TRATANTE / INTENSIVISTA           FIRMA AUDITOR / COMITÉ DE ÉTICA HOSPITALARIA\n"
            "Nombre: ___________________________________        Nombre: ___________________________________\n"
            "Registro Médico N°: _______________________        Registro Médico N°: _______________________\n"
            "Especialidad: Medicina Crítica y C.I.              Institución: Hospital Universitario"
        )
