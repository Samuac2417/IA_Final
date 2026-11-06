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
               "agents.resource_agent", "agents.bioethics_agent", "agents", "core.coordinator", "core"]
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
    if FALLOS:
        print(f"RESULTADO: {len(FALLOS)} prueba(s) fallida(s)")
        for f in FALLOS:
            print(f"   - {f}")
        return 1
    print("RESULTADO: TODAS LAS PRUEBAS SUPERADAS ✔  — el sistema está listo para `streamlit run app.py`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
