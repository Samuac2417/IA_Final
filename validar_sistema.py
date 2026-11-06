"""
validar_sistema.py
==================

Prueba de validación por consola (sin dependencias de test externas).

    python validar_sistema.py

1. Compila todos los módulos (.py) del proyecto.
2. Verifica que no existan importaciones circulares (importa cada módulo aislado).
3. Ejecuta los escenarios precargados bajo los tres marcos éticos y comprueba el
   comportamiento esperado de la demostración.
"""

from __future__ import annotations

import compileall
import importlib
import os
import sys
from dataclasses import replace

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

FALLOS: list = []


def check(condicion: bool, descripcion: str) -> None:
    print(f"  [{'OK ' if condicion else 'FALLO'}] {descripcion}")
    if not condicion:
        FALLOS.append(descripcion)


def main() -> int:
    print("=" * 78)
    print("1) COMPILACIÓN DE MÓDULOS")
    print("=" * 78)
    ok = compileall.compile_dir(RAIZ, quiet=1, force=True,
                                rx=__import__("re").compile(r"[\\/](\.venv|venv|__pycache__)[\\/]"))
    check(bool(ok), "Todos los archivos .py compilan sin errores de sintaxis")

    print("\n" + "=" * 78)
    print("2) IMPORTACIONES (detección de ciclos)")
    print("=" * 78)
    modulos = ["core.models", "core.knowledge_base", "agents.base_agent", "agents.clinical_agent",
               "agents.resource_agent", "agents.bioethics_agent", "agents", "core.coordinator",
               "core.nlg_generator", "core"]
    for nombre in modulos:
        for m in [k for k in list(sys.modules) if k == "core" or k.startswith(("core.", "agents"))]:
            del sys.modules[m]
        try:
            importlib.import_module(nombre)
            check(True, f"import {nombre}")
        except Exception as exc:  # pragma: no cover
            check(False, f"import {nombre} -> {type(exc).__name__}: {exc}")

    from core import knowledge_base as kb
    from core.coordinator import ConsensusCoordinator
    from core.models import ModoEtico, Paciente, RecursoHospitalario

    U, D, C = (ModoEtico.UTILITARISTA_CLINICO.value, ModoEtico.DEONTOLOGICO_IGUALITARIO.value,
               ModoEtico.CONSENSO_CONSTITUCIONAL_COLOMBIA.value)

    def estado(camas: int, vents: int, modo: str = C) -> RecursoHospitalario:
        return RecursoHospitalario(camas_uci_totales=12, camas_uci_disponibles=camas,
                                   ventiladores_disponibles=vents, modo_etico_activo=modo)

    print("\n" + "=" * 78)
    print("3) ESCENARIO DILEMA CRÍTICO (3 pacientes vs 1 cama UCI)")
    print("=" * 78)
    pacientes = kb.escenario_dilema_critico()
    coord = ConsensusCoordinator()
    res = coord.comparar_marcos(pacientes, estado(1, 1))
    for modo, r in res.items():
        nombres = [r.paciente(x).nombre for x in r.asignados_uci]
        print(f"  · {modo:<34} → UCI: {nombres} · validación: {r.validacion_global}")

    r_c = res[C]
    check(r_c.evaluaciones["PAC-003"].prioridad == 4, "Jorge Valencia → Prioridad 4 (futilidad terapéutica)")
    check(r_c.evaluaciones["PAC-001"].prioridad == 1 and r_c.evaluaciones["PAC-002"].prioridad == 1,
          "Carlos y Elena → Prioridad 1")
    check(r_c.estado_hospital == "ESCASEZ_CRITICA", "ResourceManagerAgent declara ESCASEZ CRÍTICA (R2)")
    check(res[U].asignados_uci == ["PAC-001"], "Utilitarismo → cama para Carlos Mendoza")
    check(res[D].asignados_uci == ["PAC-002"], "Deontología (FIFO) → cama para Elena Restrepo (llegó primero)")
    check(res[C].asignados_uci == ["PAC-002"],
          "Consenso Constitucional (umbral ±1) → Elena por empate técnico + protección reforzada")
    check(res[U].validacion_global in ("OBJETADO", "VETO CONSTITUCIONAL"),
          "BioethicsAuditAgent objeta el método utilitarista (uso de edad)")
    check(any(t.regla_id == "REG-BIO-04" for t in r_c.dictamenes["BioethicsAuditAgent"].trazas),
          "Cláusula Human-in-the-Loop (REG-BIO-04) presente")
    check("PAC-003" in r_c.dictamenes["BioethicsAuditAgent"].datos["planes_alternativos"],
          "Plan paliativo obligatorio para Jorge (REG-BIO-03)")
    check("PAC-001" in r_c.dictamenes["BioethicsAuditAgent"].datos["planes_alternativos"],
          "Plan CRUE obligatorio para el P1 no asignado (REG-BIO-03)")

    r_c0 = ConsensusCoordinator(umbral_empate_sofa=0).deliberar(pacientes, estado(1, 1))
    check(r_c0.asignados_uci == ["PAC-001"], "Consenso con umbral 0 (sin empate técnico) → Carlos (SOFA 6 < 7)")

    r_n = coord.deliberar(pacientes, estado(2, 2))
    check(r_n.estado_hospital == "NORMAL" and set(r_n.asignados_uci) == {"PAC-001", "PAC-002"},
          "Con 2 camas → capacidad normal (R1), ambos P1 asignados")

    r_v = coord.deliberar(pacientes, estado(2, 1))
    check(r_v.estado_hospital == "ESCASEZ_CRITICA" and len(r_v.asignados_uci) == 1,
          "2 camas pero 1 ventilador → escasez por ventiladores (REG-REC-03)")

    print("\n" + "=" * 78)
    print("4) PRUEBA DE EDADISMO (auditoría E1 con contrafáctico)")
    print("=" * 78)
    res_e = coord.comparar_marcos(kb.escenario_prueba_edadismo(), estado(1, 1))
    for modo, r in res_e.items():
        print(f"  · {modo:<34} → UCI: {[r.paciente(x).nombre for x in r.asignados_uci]} · {r.validacion_global}")
    check(res_e[U].asignados_uci == ["PAC-102"], "Utilitarismo prefiere al joven con peor SOFA")
    check(res_e[U].validacion_global == "VETO CONSTITUCIONAL", "REG-BIO-01 emite VETO CONSTITUCIONAL")
    check(res_e[U].asignados_contrafactual_sin_edad == ["PAC-101"], "Contrafáctico sin edad → Rosa Gómez")
    check(res_e[C].asignados_uci == ["PAC-101"] and res_e[C].validacion_global == "VALIDADO",
          "Consenso Constitucional → Rosa Gómez, VALIDADO")

    print("\n" + "=" * 78)
    print("5) ESCENARIOS ADICIONALES Y ROBUSTEZ")
    print("=" * 78)
    for clave in kb.ESCENARIOS:
        info = kb.ESCENARIOS[clave]
        for modo, r in coord.comparar_marcos(kb.cargar_escenario(clave),
                                             estado(int(info["camas"]), int(info["ventiladores"]))).items():
            trazas = r.todas_las_trazas()
            sin_texto = [t for t in trazas if not (t.condicion_evaluada and t.conclusion and t.fundamento_normativo_o_medico)]
            check(not sin_texto, f"{clave:<9}/{modo[:12]:<12}: {len(trazas)} trazas completas · UCI "
                                 f"{[r.paciente(x).nombre for x in r.asignados_uci]}")
    r0 = coord.deliberar(kb.escenario_dilema_critico(), estado(0, 0))
    check(r0.asignados_uci == [], "0 camas → nadie asignado, todos con plan alternativo")
    check(coord.deliberar([], estado(1, 1)).asignados_uci == [], "Lista vacía de pacientes no produce errores")
    try:
        Paciente("X", "Inválido", 40, 0, 120.0, 100, 5, 15, True)
        check(False, "Validación de rango SatO2")
    except ValueError:
        check(True, "Validación de rango de hechos clínicos (SatO2 > 100 rechazada)")
    a = coord.deliberar(pacientes, estado(1, 1, U))
    b = coord.deliberar(pacientes, replace(estado(1, 1), modo_etico_activo=U))
    check(a.asignados_uci == b.asignados_uci and a.puntajes == b.puntajes, "Determinismo: misma entrada → misma salida")

    print("\n" + "=" * 78)
    print("6) ANÁLISIS DE SENSIBILIDAD ESTOCÁSTICA DE MONTE CARLO")
    print("=" * 78)
    from core.models import ReporteRobustez

    # 6.1 Determinismo: misma semilla produce exactamente idénticos porcentajes
    rep_mc1 = coord.ejecutar_analisis_monte_carlo(pacientes, estado(1, 1), n_iteraciones=300, semilla=42)
    rep_mc2 = coord.ejecutar_analisis_monte_carlo(pacientes, estado(1, 1), n_iteraciones=300, semilla=42)
    pcts1 = {pid: r.porcentaje_estabilidad_decision for pid, r in rep_mc1.items()}
    pcts2 = {pid: r.porcentaje_estabilidad_decision for pid, r in rep_mc2.items()}
    check(pcts1 == pcts2, f"Determinismo Monte Carlo: corrida 1 == corrida 2 ({pcts1})")

    # 6.2 Estructura y campos del ReporteRobustez
    r_carlos = rep_mc1["PAC-001"]
    check(isinstance(r_carlos, ReporteRobustez) and r_carlos.iteraciones_totales == 300,
          "ReporteRobustez generado con tipo y campos correctos")
    check(bool(r_carlos.diagnostico_frontera) and bool(r_carlos.advertencia_clinica),
          "Diagnóstico de frontera y advertencia clínica presentes")

    # 6.3 Disparo de REG-BIO-SENSIBILIDAD ante caso borde con estabilidad < 85%
    p_borde = [Paciente("PAC-BORDE", "Caso Borde Instrumental", 45, 10, 85.0, 160, 2, 15, False)]
    r_borde_mc = coord.ejecutar_analisis_monte_carlo(p_borde, estado(1, 1), n_iteraciones=300, semilla=42)
    pct_borde = r_borde_mc["PAC-BORDE"].porcentaje_estabilidad_decision
    check(pct_borde < 85.0, f"Caso borde de oximetría produce estabilidad < 85% ({pct_borde}%)")

    # Deliberación con los reportes de Monte Carlo
    res_borde = coord.deliberar(p_borde, estado(1, 1), reportes_robustez=r_borde_mc)
    trazas_bio = res_borde.dictamenes["BioethicsAuditAgent"].trazas
    t_sens = [t for t in trazas_bio if t.regla_id == "REG-BIO-SENSIBILIDAD"]
    check(len(t_sens) > 0 and t_sens[0].nivel == "ALERTA",
          "Regla REG-BIO-SENSIBILIDAD dispara ALERTA especial de Incertidumbre Instrumental (< 85%)")

    print("\n" + "=" * 78)
    print("7) GENERADOR DE LENGUAJE NATURAL SIMBÓLICO (NLG CLÍNICO-FORENSE)")
    print("=" * 78)
    from core.nlg_generator import ClinicalNarrativeGenerator
    gen_nlg = ClinicalNarrativeGenerator()
    res_dilema = coord.deliberar(pacientes, estado(1, 1), reportes_robustez=rep_mc1)

    for pid in ["PAC-001", "PAC-002", "PAC-003"]:
        nota = gen_nlg.generar_nota_historia_clinica(pid, res_dilema)
        nombre = res_dilema.paciente(pid).nombre

        # Verificar que no contenga llaves de formato sin sustituir
        check("{" not in nota and "}" not in nota, f"NLG [{pid} - {nombre}]: sin variables sin sustituir (llaves)")
        # Verificar que no contenga literales None
        check("None" not in nota, f"NLG [{pid} - {nombre}]: sin valores 'None' en el texto")
        # Verificar presencia de las normas exigidas
        check("1995" in nota, f"NLG [{pid} - {nombre}]: cita obligatoria de Resolución 1995 de 1999")
        check("1751" in nota, f"NLG [{pid} - {nombre}]: cita obligatoria de Ley Estatutaria 1751 de 2015")
        check("23" in nota, f"NLG [{pid} - {nombre}]: cita obligatoria de Ley 23 de 1981 (Código Ética Médica)")
        check("HUMAN-IN-THE-LOOP" in nota.upper() or "ACTO MÉDICO ES PERSONAL" in nota.upper(),
              f"NLG [{pid} - {nombre}]: cláusula Human-in-the-Loop y firmas profesionales presentes")
        check("MONTE CARLO" in nota.upper(), f"NLG [{pid} - {nombre}]: reporte de robustez analítica incluido")

    # Verificación de contenido específico de futilidad en PAC-003 (Jorge Valencia)
    nota_jorge = gen_nlg.generar_nota_historia_clinica("PAC-003", res_dilema)
    check("1733" in nota_jorge and "PALIATIVO" in nota_jorge.upper(),
          "NLG [PAC-003]: activación de Ley 1733 de 2014 y Plan de Cuidado Paliativo sin abandono")

    # Verificación de contenido específico de CRUE en PAC-001 (Carlos Mendoza, P1 no asignado)
    nota_carlos = gen_nlg.generar_nota_historia_clinica("PAC-001", res_dilema)
    check("CRUE" in nota_carlos and "PROHIBICIÓN EXPRESA DE ABANDONO" in nota_carlos,
          "NLG [PAC-001]: activación de remisión CRUE y cláusula de no abandono (Art. 14 Ley 1751)")

    print("\n" + "=" * 78)
    print("8) TAMIZAJE ULTRARRÁPIDO DINÁMICO: SCORE NEWS2 (RCP 2017)")
    print("=" * 78)
    # 8.1 Caso Score 0: Parámetros normales
    p_news_0 = Paciente("PAC-N0", "Normal Basal", 30, 0, 98.0, 400, 0, 15, False,
                        frecuencia_respiratoria=16, presion_sistolica=120, frecuencia_cardiaca=75,
                        temperatura=36.8, oxigeno_suplementario=False, escala_acvpu="A")
    s0, niv0, resp0 = kb.calcular_news2(p_news_0)
    check(s0 == 0 and niv0 == "BAJO", f"NEWS2 Score 0 exacto: {s0} ptos ({niv0})")

    # 8.2 Caso Score 5: Riesgo Medio
    p_news_5 = Paciente("PAC-N5", "Riesgo Medio", 45, 0, 96.0, 320, 2, 15, False,
                        frecuencia_respiratoria=22, presion_sistolica=120, frecuencia_cardiaca=95,
                        temperatura=36.8, oxigeno_suplementario=True, escala_acvpu="A")
    s5, niv5, resp5 = kb.calcular_news2(p_news_5)
    check(s5 == 5 and niv5 == "MEDIO", f"NEWS2 Score 5 exacto: {s5} ptos ({niv5})")

    # 8.3 Caso Score 9: Riesgo Alto / Emergencia
    p_news_9 = Paciente("PAC-N9", "Riesgo Alto", 60, 0, 90.0, 180, 5, 14, True,
                        frecuencia_respiratoria=26, presion_sistolica=120, frecuencia_cardiaca=95,
                        temperatura=36.8, oxigeno_suplementario=True, escala_acvpu="A")
    s9, niv9, resp9 = kb.calcular_news2(p_news_9)
    check(s9 == 9 and niv9 == "ALTO", f"NEWS2 Score 9 exacto: {s9} ptos ({niv9})")

    # 8.4 Disparo de REG-CLIN-NEWS2 en ClinicalTriageAgent
    d_clin_n = coord.agente_clinico.evaluar([p_news_9])
    t_news = [t for t in d_clin_n.trazas if t.regla_id == "REG-CLIN-NEWS2"]
    check(len(t_news) > 0 and t_news[0].nivel == "CRITICO",
          "Regla REG-CLIN-NEWS2 dispara alerta crítica para paciente con NEWS2 >= 7")

    print("\n" + "=" * 78)
    print("9) SIMULACIÓN TEMPORAL MULTI-RONDA (0h -> 6h -> 12h -> 24h)")
    print("=" * 78)
    evos_dilema = kb.escenario_evolucion_temporal_dilema()
    rondas_const = coord.simular_linea_tiempo(pacientes, estado(1, 1, C), evos_dilema)

    check(len(rondas_const) == 4, f"Simulación multi-ronda produjo 4 cortes temporales (0h, 6h, 12h, 24h)")
    r0, r6, r12, r24 = rondas_const

    # Corte 0h
    check(r0.resultado_comite.asignados_uci == ["PAC-002"],
          "t = 0h: Elena Restrepo asignada a UCI en el marco constitucional")

    # Corte 6h
    check(r6.resultado_comite.decisiones["PAC-002"].asignado_uci is True,
          "t = 6h: Elena Restrepo continúa en soporte invasivo UCI")
    check(r6.resultado_comite.decisiones["PAC-001"].asignado_uci is False,
          "t = 6h: Carlos Mendoza continúa en área de expansión (espera priorizada)")

    # Corte 12h: Liberación de cama por mejoría clínica y reasignación auditada al paciente en espera
    check(r12.resultado_comite.decisiones["PAC-002"].destino == "CUIDADOS INTERMEDIOS / MONITOREO CONTINUO",
          "t = 12h: Elena Restrepo extubada y trasladada a CUIDADOS INTERMEDIOS (libera cama UCI)")
    check(r12.resultado_comite.decisiones["PAC-001"].asignado_uci is True,
          "t = 12h: Cama UCI liberada reasignada de forma determinista y auditada a Carlos Mendoza")
    check(any("LIBERACIÓN DINÁMICA" in e for e in r12.eventos),
          "t = 12h: Evento de liberación dinámica registrado en el acta de la ronda")
    check(any("REASIGNACIÓN AUDITADA" in e for e in r12.eventos),
          "t = 12h: Evento de reasignación auditada registrado en el acta de la ronda")

    # Corte 24h
    check(r24.resultado_comite.decisiones["PAC-001"].asignado_uci is True,
          "t = 24h: Carlos Mendoza continúa tratamiento en UCI")
    check(r24.resultado_comite.decisiones["PAC-002"].destino == "CUIDADOS INTERMEDIOS / MONITOREO CONTINUO",
          "t = 24h: Elena Restrepo estable en Cuidados Intermedios")

    # Comprobación de reasignación en modo Utilitarista (Carlos inicia en UCI -> a 12h libera para Elena)
    rondas_util = coord.simular_linea_tiempo(pacientes, estado(1, 1, U), evos_dilema)
    check(rondas_util[0].resultado_comite.asignados_uci == ["PAC-001"],
          "Utilitarismo t = 0h: Carlos Mendoza inicia en UCI")
    check(rondas_util[2].resultado_comite.decisiones["PAC-002"].asignado_uci is True,
          "Utilitarismo t = 12h: Carlos libera cama y Elena Restrepo es admitida de inmediato")

    print("\n" + "=" * 78)
    if FALLOS:
        print(f"RESULTADO: {len(FALLOS)} prueba(s) fallida(s)")
        for f in FALLOS:
            print(f"   - {f}")
        return 1
    print("RESULTADO: TODAS LAS PRUEBAS SUPERADAS ✔  — el sistema está listo para `streamlit run app.py`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
