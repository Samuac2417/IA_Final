"""
core/knowledge_base.py
======================

Base de Conocimiento del sistema experto (conocimiento declarativo).

Contiene, separados de la lógica de inferencia:

1. Umbrales de los protocolos clínicos (SOFA, PaFi / Berlín, Glasgow, SatO2,
   triage de urgencias tipo Manchester / Resolución 5596 de 2015).
2. Tabla docente de probabilidad de sobrevida estimada por SOFA.
3. Parámetros de los marcos éticos (pesos utilitaristas, umbral de empate).
4. Corpus normativo colombiano e internacional citado por las reglas.
5. Escenarios clínicos precargados para la demostración en vivo.

NOTA ACADÉMICA: las cifras de sobrevida son una aproximación pedagógica inspirada
en Ferreira et al. (JAMA, 2001) y Vincent et al. (Intensive Care Med, 1996). No
constituyen una calibración clínica validada y no deben usarse con pacientes reales.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from core.models import ModoEtico, Paciente, RecursoHospitalario

# ---------------------------------------------------------------------------
# 1. Umbrales clínicos
# ---------------------------------------------------------------------------
UMBRAL_SATO2_HIPOXEMIA_REFRACTARIA: float = 85.0     # Regla C1
UMBRAL_PAFI_HIPOXEMIA_REFRACTARIA: int = 150          # Regla C1
UMBRAL_SATO2_VIGILANCIA: float = 92.0                 # P2 (riesgo de deterioro)
UMBRAL_PAFI_VIGILANCIA: int = 300                     # P2 (SDRA leve o mayor)

SOFA_REVERSIBILIDAD_ALTA_MAX: int = 6                 # Regla C2: SOFA <= 6
SOFA_REVERSIBILIDAD_MEDIA_MAX: int = 11               # Regla C2: 7 <= SOFA <= 11
SOFA_FUTILIDAD_MIN_EXCLUSIVO: int = 14                # Regla C3: SOFA > 14 + terminal
SOFA_VIGILANCIA_MIN: int = 2                          # P2 si hay disfunción orgánica

GLASGOW_PROTECCION_VIA_AEREA: int = 8                 # Glasgow <= 8 -> intubación
GLASGOW_COMPROMISO_MODERADO: int = 12                 # 9..12 -> compromiso moderado

# Reloj del turno (minutos) usado para expresar "llegó hace X minutos".
MINUTO_ACTUAL_TURNO: int = 120


def clasificar_sdra_berlin(pafi: int) -> str:
    """Clasificación de oxigenación según la Definición de Berlín del SDRA (2012)."""
    if pafi <= 100:
        return "SDRA severo (PaFi ≤ 100)"
    if pafi <= 200:
        return "SDRA moderado (100 < PaFi ≤ 200)"
    if pafi <= 300:
        return "SDRA leve (200 < PaFi ≤ 300)"
    return "Sin criterio de SDRA por oxigenación (PaFi > 300)"


def clasificar_triage_urgencias(sat_o2: float, glasgow: int, pafi: int, requiere_ventilador: bool) -> str:
    """Triage de urgencias de 5 niveles (Manchester / Resolución 5596 de 2015, MinSalud)."""
    if sat_o2 < UMBRAL_SATO2_HIPOXEMIA_REFRACTARIA or glasgow <= GLASGOW_PROTECCION_VIA_AEREA or requiere_ventilador:
        return "Triage I · ROJO · Atención inmediata (riesgo vital)"
    if sat_o2 < UMBRAL_SATO2_VIGILANCIA or pafi < UMBRAL_PAFI_VIGILANCIA or glasgow <= GLASGOW_COMPROMISO_MODERADO:
        return "Triage II · NARANJA · Muy urgente (≤ 30 min)"
    if sat_o2 < 95:
        return "Triage III · AMARILLO · Urgente"
    return "Triage IV · VERDE · Menos urgente"


# ---------------------------------------------------------------------------
# 2. Tabla docente de sobrevida estimada por SOFA
# ---------------------------------------------------------------------------
_TABLA_SOBREVIDA_SOFA: Dict[int, float] = {
    0: 0.95, 1: 0.95, 2: 0.92, 3: 0.92, 4: 0.88, 5: 0.85, 6: 0.82,
    7: 0.75, 8: 0.70, 9: 0.65, 10: 0.58, 11: 0.52,
    12: 0.45, 13: 0.40, 14: 0.35, 15: 0.20, 16: 0.15, 17: 0.12, 18: 0.10,
}


def prob_sobrevida_por_sofa(sofa: int) -> float:
    """Probabilidad de sobrevida a corto plazo estimada (aproximación docente)."""
    if sofa >= 19:
        return 0.05
    return _TABLA_SOBREVIDA_SOFA[max(0, sofa)]


def reversibilidad_por_sofa(sofa: int) -> str:
    """Banda de reversibilidad fisiológica de la Regla C2."""
    if sofa <= SOFA_REVERSIBILIDAD_ALTA_MAX:
        return "ALTA"
    if sofa <= SOFA_REVERSIBILIDAD_MEDIA_MAX:
        return "MEDIA"
    return "BAJA"


# Comorbilidades fuertemente asociadas a determinantes sociales de la salud
# (pobreza, ruralidad, exposición a humo de biomasa, barreras de acceso).
COMORBILIDADES_SENSIBLES_A_DETERMINANTES_SOCIALES: List[str] = [
    "hipertens", "diabet", "epoc", "renal", "desnutri", "tuberculosis",
    "vih", "obesidad", "asma", "chagas", "cardiopat", "biomasa",
]


def comorbilidades_sensibles(comorbilidades: List[str]) -> List[str]:
    """Devuelve las comorbilidades que se asocian a brechas socioeconómicas."""
    resultado = []
    for c in comorbilidades:
        texto = c.lower()
        if any(clave in texto for clave in COMORBILIDADES_SENSIBLES_A_DETERMINANTES_SOCIALES):
            resultado.append(c)
    return resultado


# ---------------------------------------------------------------------------
# 3. Parámetros de los marcos éticos
# ---------------------------------------------------------------------------
EXPECTATIVA_VIDA_COLOMBIA: int = 77          # Aprox. proyección DANE (ambos sexos)
ANOS_VIDA_NORMALIZACION: int = 50            # Normaliza años de vida potenciales a [0, 1]
PESOS_UTILITARISTAS: Dict[str, float] = {"sobrevida": 0.50, "pafi": 0.15, "anos_vida": 0.35}
PENALIZACION_POR_COMORBILIDAD: float = 5.0   # puntos por comorbilidad grave
MAX_COMORBILIDADES_PENALIZADAS: int = 3
EDAD_REFERENCIA_CONTRAFACTUAL: int = 50      # Edad neutra usada en la prueba de sesgo E1

UMBRAL_EMPATE_TECNICO_SOFA: int = 1          # |ΔSOFA| <= 1 => empate técnico (variabilidad interobservador)
SEMILLA_SORTEO_AUDITABLE: int = 2026         # Semilla pública del sorteo (reproducible)
EDAD_UMBRAL_EDADISMO: int = 70               # Regla E1
DIFERENCIA_PROPORCIONALIDAD: float = 0.30    # Regla E5: Δ sobrevida que activa alerta

MARCOS_ETICOS: Dict[str, Dict[str, str]] = {
    ModoEtico.UTILITARISTA_CLINICO.value: {
        "nombre": "Utilitarismo Clínico Estricto",
        "escuela": "Consecuencialismo (J. Bentham, J. S. Mill, P. Singer)",
        "principio": "Maximizar el número de vidas y de años de vida salvados con el recurso escaso.",
        "criterio": "Puntaje de sobrevida neta = 50 % sobrevida estimada por SOFA + 15 % oxigenación (PaFi) "
                    "+ 35 % años de vida potencialmente ganados − penalización por comorbilidades.",
        "critica": "Maximiza la eficiencia biológica del recurso escaso, pero tiende a relegar a adultos "
                   "mayores o personas con enfermedades crónicas: el componente de años de vida introduce "
                   "la edad cronológica y la penalización por comorbilidad castiga doblemente a quienes "
                   "enfermaron por brechas socioeconómicas.",
        "color": "#d97706",
    },
    ModoEtico.DEONTOLOGICO_IGUALITARIO.value: {
        "nombre": "Deontología e Igualitarismo (Imperativo Kantiano)",
        "escuela": "Deontología (I. Kant) · Igualitarismo formal",
        "principio": "Toda persona es un fin en sí misma; está prohibido comparar el 'valor' de dos vidas.",
        "criterio": "Entre pacientes P1 con urgencia equivalente se desempata por un criterio objetivo neutral: "
                    "orden de llegada (FIFO) y, si persiste el empate, sorteo auditable con semilla pública.",
        "critica": "Respeta la igualdad formal absoluta, pero puede asignar el último ventilador a un paciente "
                   "con 15 % de probabilidad de sobrevivir mientras fallece en espera uno con 90 % de "
                   "probabilidad de rescate.",
        "color": "#2563eb",
    },
    ModoEtico.CONSENSO_CONSTITUCIONAL_COLOMBIA.value: {
        "nombre": "Consenso Bioético Constitucional Colombiano",
        "escuela": "Principialismo (Beauchamp & Childress) + Estado Social de Derecho (Const. 1991)",
        "principio": "Priorizar la reversibilidad fisiológica a corto plazo sin discriminar por edad ni "
                     "condición socioeconómica, con protección reforzada a sujetos de especial protección.",
        "criterio": "Orden por SOFA (reserva funcional). Si |ΔSOFA| ≤ umbral de empate técnico, se prefiere "
                    "al sujeto de especial protección constitucional; luego orden de llegada; luego sorteo. "
                    "Activación obligatoria de CRUE para los no asignados.",
        "critica": "Es el marco jurídicamente sostenible en Colombia, pero exige un umbral de empate técnico "
                   "definido ex ante y revisión humana: la frontera entre 'empate' y 'diferencia pronóstica' "
                   "es una decisión de política clínica que debe ser transparente.",
        "color": "#059669",
    },
}

# ---------------------------------------------------------------------------
# 4. Corpus normativo
# ---------------------------------------------------------------------------
NORMATIVA: Dict[str, Dict[str, str]] = {
    "CONST_ART_1": {
        "norma": "Constitución Política de Colombia, Art. 1",
        "contenido": "Colombia es un Estado Social de Derecho fundado en el respeto de la dignidad humana.",
        "aplicacion": "Fundamento del marco deontológico y de la cláusula de no abandono.",
    },
    "CONST_ART_13": {
        "norma": "Constitución Política de Colombia, Art. 13",
        "contenido": "Igualdad ante la ley y prohibición de discriminación; el Estado protegerá especialmente "
                     "a quienes se encuentren en circunstancia de debilidad manifiesta.",
        "aplicacion": "Reglas REG-BIO-01 (antiedadismo) y REG-BIO-02 (equidad); protección reforzada en empates.",
    },
    "CONST_ART_43_46": {
        "norma": "Constitución Política de Colombia, Arts. 43 y 46",
        "contenido": "Apoyo especial a la mujer cabeza de familia (Art. 43) y protección a las personas de la "
                     "tercera edad (Art. 46).",
        "aplicacion": "Identificación de sujetos de especial protección constitucional.",
    },
    "CONST_ART_49": {
        "norma": "Constitución Política de Colombia, Art. 49",
        "contenido": "La atención de la salud es un servicio público a cargo del Estado bajo principios de "
                     "eficiencia, universalidad y solidaridad.",
        "aplicacion": "Fundamento de la gestión eficiente del recurso escaso (ResourceManagerAgent).",
    },
    "T_760_2008": {
        "norma": "Corte Constitucional, Sentencia T-760 de 2008",
        "contenido": "Reconoce la salud como derecho fundamental autónomo y ordena eliminar barreras de acceso.",
        "aplicacion": "REG-BIO-01: la edad cronológica no puede ser, por sí sola, barrera de acceso a UCI.",
    },
    "LEY_1751_2015": {
        "norma": "Ley Estatutaria 1751 de 2015 (Ley Estatutaria de Salud)",
        "contenido": "Art. 6 (principios: equidad, oportunidad, continuidad, pro homine); Art. 10 (derechos de "
                     "las personas en la prestación del servicio); Art. 11 (sujetos de especial protección); "
                     "Art. 14 (prohibición de negación de servicios; la urgencia no requiere autorización "
                     "administrativa); Art. 17 (autonomía profesional).",
        "aplicacion": "REG-BIO-03: todo paciente sin cama UCI recibe plan alternativo obligatorio (CRUE o "
                      "cuidado paliativo sin abandono).",
    },
    "LEY_23_1981": {
        "norma": "Ley 23 de 1981 (Código de Ética Médica)",
        "contenido": "El acto médico es personal e indelegable; Art. 13: el médico usará los medios a su alcance "
                     "mientras subsista la esperanza de aliviar o curar la enfermedad.",
        "aplicacion": "REG-BIO-04 (Human-in-the-Loop) y fundamento de la adecuación del esfuerzo terapéutico.",
    },
    "LEY_1733_2014": {
        "norma": "Ley 1733 de 2014 (Ley Consuelo Devis Saavedra)",
        "contenido": "Regula los servicios de cuidados paliativos para el manejo integral de pacientes con "
                     "enfermedades terminales, crónicas, degenerativas e irreversibles.",
        "aplicacion": "REG-CLIN-03 y REG-BIO-06: la Prioridad 4 activa cuidado paliativo integral, no abandono.",
    },
    "LEY_1251_2008": {
        "norma": "Ley 1251 de 2008",
        "contenido": "Normas para la protección, promoción y defensa de los derechos de los adultos mayores.",
        "aplicacion": "REG-BIO-01: refuerzo legal de la prohibición de edadismo.",
    },
    "LEY_1581_2012": {
        "norma": "Ley Estatutaria 1581 de 2012 (Habeas Data)",
        "contenido": "Protección de datos personales; los datos de salud son datos sensibles (Arts. 5 y 6).",
        "aplicacion": "El nombre y la condición social no intervienen en el puntaje clínico; el sistema es local "
                      "y no transmite datos a terceros.",
    },
    "LEY_1523_2012": {
        "norma": "Ley 1523 de 2012 (Gestión del Riesgo de Desastres)",
        "contenido": "Política nacional de gestión del riesgo y manejo de situaciones de desastre.",
        "aplicacion": "REG-REC-02: declaratoria de escasez crítica / triaje de catástrofe.",
    },
    "RES_1220_2010": {
        "norma": "Resolución 1220 de 2010 (MinSalud) — CRUE",
        "contenido": "Condiciones de operación de los Centros Reguladores de Urgencias, Emergencias y Desastres.",
        "aplicacion": "REG-BIO-03: remisión interinstitucional urgente de pacientes P1 no asignados.",
    },
    "RES_5596_2015": {
        "norma": "Resolución 5596 de 2015 (MinSalud) — Triage",
        "contenido": "Criterios técnicos del sistema de selección y clasificación de pacientes en urgencias.",
        "aplicacion": "REG-CLIN-00: clasificación inicial de triage de urgencias.",
    },
    "RES_1995_1999": {
        "norma": "Resolución 1995 de 1999 (MinSalud) — Historia Clínica",
        "contenido": "Normas para el manejo de la historia clínica: integralidad, secuencialidad, disponibilidad.",
        "aplicacion": "Las trazas XAI exportables sirven como soporte documental auditable de la decisión.",
    },
    "UNESCO_2021": {
        "norma": "UNESCO — Recomendación sobre la Ética de la Inteligencia Artificial (2021)",
        "contenido": "Supervisión y decisión humanas, transparencia y explicabilidad, equidad y no "
                     "discriminación, responsabilidad y rendición de cuentas.",
        "aplicacion": "Arquitectura XAI simbólica, auditoría de sesgos (E1, E2) y cláusula HITL (E4).",
    },
    "MARCO_ETICO_IA_CO": {
        "norma": "Marco Ético para la Inteligencia Artificial en Colombia (2021) y CONPES 3975 de 2019",
        "contenido": "Principios de transparencia, explicación, privacidad, control humano, seguridad, "
                     "responsabilidad, no discriminación e inclusión.",
        "aplicacion": "Diseño 100 % determinista, local y auditable.",
    },
}


def cita(clave: str) -> str:
    """Devuelve la referencia normativa abreviada para incrustar en una traza."""
    ref = NORMATIVA.get(clave)
    return ref["norma"] if ref else clave


MATRIZ_RESPONSABILIDAD: List[Dict[str, str]] = [
    {
        "actor": "Equipo desarrollador (Universidad / proveedor)",
        "responsabilidad": "Corrección técnica de las reglas, ausencia de sesgos de diseño, documentación, "
                           "trazabilidad y pruebas de validación.",
        "tipo": "Responsabilidad por producto defectuoso / diligencia profesional (Ley 1480 de 2011, Estatuto "
                "del Consumidor, por analogía).",
        "no_responde_por": "El acto médico final ni la decisión de ingreso.",
    },
    {
        "actor": "Institución Prestadora de Salud (Hospital)",
        "responsabilidad": "Validación local del sistema, capacitación, gobierno de datos (Ley 1581), "
                           "activación de CRUE y garantía de plan alternativo.",
        "tipo": "Responsabilidad institucional / falla del servicio.",
        "no_responde_por": "Errores de diseño no informados por el desarrollador.",
    },
    {
        "actor": "Médico tratante / Comité de Ética Hospitalaria",
        "responsabilidad": "Decisión final de ingreso a UCI, valoración clínica integral, comunicación con la "
                           "familia y registro en historia clínica.",
        "tipo": "Responsabilidad médica personal (Ley 23 de 1981; Ley 1164 de 2007).",
        "no_responde_por": "Fallas técnicas ocultas del software que no pudo advertir con diligencia.",
    },
    {
        "actor": "Sistema Experto (este software)",
        "responsabilidad": "Ninguna responsabilidad jurídica: es una herramienta de soporte (CDSS) que emite "
                           "recomendaciones explicables y auditables.",
        "tipo": "Sin personalidad jurídica.",
        "no_responde_por": "—",
    },
]


# ---------------------------------------------------------------------------
# 5. Escenarios precargados
# ---------------------------------------------------------------------------
def escenario_dilema_critico() -> List[Paciente]:
    """Caso central de la demostración: 3 pacientes compiten por 1 cama UCI."""
    return [
        Paciente(
            id="PAC-001", nombre="Carlos Mendoza", edad=34,
            hora_llegada=MINUTO_ACTUAL_TURNO - 10,
            sat_o2=79.0, pafi=110, escala_sofa=6, glasgow=14, requiere_ventilador=True,
            comorbilidades_graves=[], enfermedad_terminal_irreversible=False, vulnerabilidad_social=False,
            descripcion="Joven sin comorbilidades. Neumonía grave con SDRA moderado. Excelente pronóstico "
                        "de recuperación si recibe UCI. Llegó hace 10 minutos.",
        ),
        Paciente(
            id="PAC-002", nombre="Elena Restrepo", edad=76,
            hora_llegada=MINUTO_ACTUAL_TURNO - 45,
            sat_o2=81.0, pafi=125, escala_sofa=7, glasgow=14, requiere_ventilador=True,
            comorbilidades_graves=["Hipertensión arterial crónica"],
            enfermedad_terminal_irreversible=False, vulnerabilidad_social=True,
            descripcion="Adulta mayor hipertensa pero autónoma, madre cabeza de familia rural (sujeto de "
                        "especial protección constitucional). Primera en llegar (hace 45 minutos).",
        ),
        Paciente(
            id="PAC-003", nombre="Jorge Valencia", edad=58,
            hora_llegada=MINUTO_ACTUAL_TURNO - 30,
            sat_o2=74.0, pafi=85, escala_sofa=16, glasgow=9, requiere_ventilador=True,
            comorbilidades_graves=["Adenocarcinoma de páncreas metastásico (estadio IV)"],
            enfermedad_terminal_irreversible=True, vulnerabilidad_social=False,
            descripcion="Falla multiorgánica severa sobre enfermedad oncológica terminal avanzada. "
                        "Llegó hace 30 minutos.",
        ),
    ]


def escenario_prueba_edadismo() -> List[Paciente]:
    """Caso de auditoría E1: la edad, por sí sola, cambia el resultado utilitarista."""
    return [
        Paciente(
            id="PAC-101", nombre="Rosa Gómez", edad=82,
            hora_llegada=MINUTO_ACTUAL_TURNO - 60,
            sat_o2=82.0, pafi=130, escala_sofa=5, glasgow=15, requiere_ventilador=True,
            comorbilidades_graves=[], enfermedad_terminal_irreversible=False, vulnerabilidad_social=False,
            descripcion="Adulta mayor funcional, sin comorbilidades. Mejor reserva funcional (SOFA 5).",
        ),
        Paciente(
            id="PAC-102", nombre="Andrés Pérez", edad=29,
            hora_llegada=MINUTO_ACTUAL_TURNO - 20,
            sat_o2=80.0, pafi=120, escala_sofa=7, glasgow=14, requiere_ventilador=True,
            comorbilidades_graves=[], enfermedad_terminal_irreversible=False, vulnerabilidad_social=False,
            descripcion="Adulto joven con peor reserva funcional (SOFA 7).",
        ),
    ]


def escenario_saturacion_masiva() -> List[Paciente]:
    """Desastre masivo: 6 pacientes con prioridades mixtas para 2 camas UCI."""
    return [
        Paciente("PAC-201", "Luisa Fernanda Ortiz", 45, 30, 78.0, 95, 9, 12, True,
                 ["Diabetes mellitus tipo 2"], False, True,
                 "Trabajadora informal, diabética sin control por barreras de acceso."),
        Paciente("PAC-202", "Miguel Ángel Rojas", 67, 55, 83.0, 140, 4, 15, True,
                 [], False, False, "SDRA moderado, buena reserva funcional."),
        Paciente("PAC-203", "Ana Sofía Cárdenas", 19, 70, 84.0, 145, 12, 7, True,
                 [], False, False, "Politraumatismo con TCE severo; reversibilidad baja pero sin futilidad."),
        Paciente("PAC-204", "Pedro Nel Gutiérrez", 88, 40, 72.0, 70, 17, 6, True,
                 ["Insuficiencia cardiaca terminal"], True, True,
                 "Cardiopatía terminal con falla multiorgánica."),
        Paciente("PAC-205", "Camila Herrera", 31, 90, 90.0, 240, 3, 15, False,
                 [], False, False, "Neumonía con SDRA leve; riesgo de deterioro."),
        Paciente("PAC-206", "Julio César Mosquera", 52, 100, 96.0, 380, 1, 15, False,
                 [], False, True, "Desplazado por el conflicto armado; cuadro leve y estable."),
    ]


def escenario_capacidad_normal() -> List[Paciente]:
    """Escenario de control: la oferta UCI cubre la demanda (sin dilema ético)."""
    return [
        Paciente("PAC-301", "Beatriz Salazar", 63, 40, 83.0, 135, 6, 14, True,
                 ["EPOC severo"], False, False, "Exacerbación de EPOC con hipoxemia refractaria."),
        Paciente("PAC-302", "Hernán Castaño", 41, 80, 80.0, 115, 8, 13, True,
                 [], False, False, "Neumonía grave bilateral."),
        Paciente("PAC-303", "Daniela Muñoz", 27, 100, 91.0, 260, 2, 15, False,
                 [], False, False, "Neumonía con SDRA leve en vigilancia."),
    ]


ESCENARIOS: Dict[str, Dict[str, object]] = {
    "dilema": {
        "titulo": "Dilema Crítico (3 Pacientes vs 1 Cama UCI)",
        "constructor": escenario_dilema_critico,
        "camas": 1, "ventiladores": 1,
    },
    "edadismo": {
        "titulo": "Prueba de Edadismo — Auditoría E1 (2 Pacientes vs 1 Cama)",
        "constructor": escenario_prueba_edadismo,
        "camas": 1, "ventiladores": 1,
    },
    "masivo": {
        "titulo": "Saturación Masiva (6 Pacientes vs 2 Camas)",
        "constructor": escenario_saturacion_masiva,
        "camas": 2, "ventiladores": 2,
    },
    "normal": {
        "titulo": "Capacidad Normal — Control sin Dilema (3 Camas)",
        "constructor": escenario_capacidad_normal,
        "camas": 3, "ventiladores": 3,
    },
}


def cargar_escenario(clave: str) -> List[Paciente]:
    constructor: Callable[[], List[Paciente]] = ESCENARIOS[clave]["constructor"]  # type: ignore[assignment]
    return constructor()


def estado_hospital_por_defecto(modo: ModoEtico = ModoEtico.CONSENSO_CONSTITUCIONAL_COLOMBIA) -> RecursoHospitalario:
    return RecursoHospitalario(camas_uci_totales=12, camas_uci_disponibles=1,
                               ventiladores_disponibles=1, modo_etico_activo=modo.value)
