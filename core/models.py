"""
core/models.py
==============

Modelo de datos del Sistema Experto Multiagente para Triage Médico y Asignación
Ética de Recursos Críticos (UCI).

Todas las estructuras son ``dataclasses`` con tipado estricto. No contienen
lógica de decisión (esa vive en los agentes y en el coordinador); únicamente
validan la integridad de los hechos clínicos que ingresan a la base de hechos.

Nombres canónicos (español) y alias (inglés) solicitados en la especificación:

    Paciente             <-> Patient
    TrazaInferencia      <-> RuleTrace
    RecursoHospitalario  <-> HospitalState
    DictamenAgente       <-> AgentVerdict
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumeraciones del dominio
# ---------------------------------------------------------------------------
class ModoEtico(str, Enum):
    """Marcos filosóficos de resolución de dilemas disponibles en el comité."""

    UTILITARISTA_CLINICO = "UTILITARISTA_CLINICO"
    DEONTOLOGICO_IGUALITARIO = "DEONTOLOGICO_IGUALITARIO"
    CONSENSO_CONSTITUCIONAL_COLOMBIA = "CONSENSO_CONSTITUCIONAL_COLOMBIA"


class Prioridad(IntEnum):
    """Prioridades de ingreso a UCI (protocolo de triaje de emergencia en UCI)."""

    P1_CRITICO_INESTABLE = 1
    P2_MONITOREO_INTENSIVO = 2
    P3_ESTABLE = 3
    P4_PALIATIVO = 4


class Nivel(str, Enum):
    """Severidad semántica de una traza (se usa para colorear la interfaz)."""

    INFO = "INFO"
    OK = "OK"
    ALERTA = "ALERTA"
    OBJECION = "OBJECION"
    CRITICO = "CRITICO"


class Etapa(str, Enum):
    """Etapa de la cadena explicativa XAI a la que pertenece una traza."""

    HECHO = "HECHO CLÍNICO"
    REGLA = "REGLA DISPARADA"
    RECURSO = "GESTIÓN DE RECURSOS"
    AUDITORIA = "AUDITORÍA ÉTICA"
    DECISION = "RECOMENDACIÓN FINAL"


class EstadoHospital(str, Enum):
    NORMAL = "NORMAL"
    ESCASEZ_CRITICA = "ESCASEZ_CRITICA"


class Destino(str, Enum):
    """Recomendación final de destino asistencial para cada paciente."""

    UCI = "INGRESO A UCI"
    CRUE_EXPANSION = "SOPORTE VITAL EN ÁREA DE EXPANSIÓN + REMISIÓN URGENTE CRUE"
    INTERMEDIOS = "CUIDADOS INTERMEDIOS / MONITOREO CONTINUO"
    HOSPITALIZACION = "HOSPITALIZACIÓN GENERAL"
    PALIATIVO = "CUIDADO PALIATIVO INTEGRAL (SIN ABANDONO)"


class ValidacionEtica(str, Enum):
    VALIDADO = "VALIDADO"
    CON_OBSERVACIONES = "VALIDADO CON OBSERVACIONES"
    OBJETADO = "OBJETADO"
    VETO = "VETO CONSTITUCIONAL"


# ---------------------------------------------------------------------------
# Hechos: Paciente
# ---------------------------------------------------------------------------
@dataclass
class Paciente:
    """Hecho clínico atómico que ingresa a la base de hechos del sistema experto.

    ``hora_llegada`` es el minuto del turno en el que el paciente ingresó a
    urgencias (un valor menor significa que llegó antes). Se usa únicamente como
    criterio de justicia igualitaria (FIFO) en el marco deontológico y como
    desempate final neutral en el marco constitucional.
    """

    id: str
    nombre: str
    edad: int
    hora_llegada: int
    sat_o2: float
    pafi: int
    escala_sofa: int
    glasgow: int
    requiere_ventilador: bool
    comorbilidades_graves: List[str] = field(default_factory=list)
    enfermedad_terminal_irreversible: bool = False
    vulnerabilidad_social: bool = False
    descripcion: str = ""
    # Parámetros fisiológicos basales para tamizaje NEWS2 (RCP 2017)
    frecuencia_respiratoria: int = 16
    presion_sistolica: int = 120
    frecuencia_cardiaca: int = 75
    temperatura: float = 36.8
    oxigeno_suplementario: bool = False
    escala_acvpu: str = "A"

    def __post_init__(self) -> None:
        errores: List[str] = []
        if not self.id or not str(self.id).strip():
            errores.append("El identificador del paciente es obligatorio.")
        if not (0 <= self.edad <= 120):
            errores.append(f"Edad fuera de rango fisiológico (0-120): {self.edad}")
        if self.hora_llegada < 0:
            errores.append(f"La hora de llegada no puede ser negativa: {self.hora_llegada}")
        if not (0.0 <= float(self.sat_o2) <= 100.0):
            errores.append(f"SatO2 fuera de rango (0-100 %): {self.sat_o2}")
        if not (20 <= self.pafi <= 700):
            errores.append(f"PaFi fuera de rango plausible (20-700): {self.pafi}")
        if not (0 <= self.escala_sofa <= 24):
            errores.append(f"SOFA fuera de rango (0-24): {self.escala_sofa}")
        if not (3 <= self.glasgow <= 15):
            errores.append(f"Glasgow fuera de rango (3-15): {self.glasgow}")
        if not (0 <= self.frecuencia_respiratoria <= 80):
            errores.append(f"Frecuencia respiratoria fuera de rango plausible (0-80 rpm): {self.frecuencia_respiratoria}")
        if not (30 <= self.presion_sistolica <= 300):
            errores.append(f"Presión sistólica fuera de rango plausible (30-300 mmHg): {self.presion_sistolica}")
        if not (20 <= self.frecuencia_cardiaca <= 250):
            errores.append(f"Frecuencia cardíaca fuera de rango plausible (20-250 lpm): {self.frecuencia_cardiaca}")
        if not (25.0 <= float(self.temperatura) <= 45.0):
            errores.append(f"Temperatura fuera de rango plausible (25-45 °C): {self.temperatura}")
        self.escala_acvpu = str(self.escala_acvpu).strip().upper()
        if self.escala_acvpu not in {"A", "C", "V", "P", "U"}:
            errores.append(f"Escala ACVPU inválida ('{self.escala_acvpu}'). Opciones: A, C, V, P, U")
        if errores:
            raise ValueError(f"Paciente {self.id}: " + " | ".join(errores))
        self.sat_o2 = float(self.sat_o2)
        self.temperatura = float(self.temperatura)
        self.oxigeno_suplementario = bool(self.oxigeno_suplementario)
        self.comorbilidades_graves = list(self.comorbilidades_graves or [])

    def minutos_espera(self, minuto_actual: int) -> int:
        """Minutos transcurridos desde la llegada respecto al reloj del turno."""
        return max(0, minuto_actual - self.hora_llegada)


# ---------------------------------------------------------------------------
# Estado del entorno: recursos hospitalarios
# ---------------------------------------------------------------------------
@dataclass
class RecursoHospitalario:
    """Inventario de recursos críticos y marco ético activo del comité."""

    camas_uci_totales: int
    camas_uci_disponibles: int
    ventiladores_disponibles: int
    modo_etico_activo: str = ModoEtico.CONSENSO_CONSTITUCIONAL_COLOMBIA.value

    def __post_init__(self) -> None:
        if isinstance(self.modo_etico_activo, ModoEtico):
            self.modo_etico_activo = self.modo_etico_activo.value
        validos = {m.value for m in ModoEtico}
        if self.modo_etico_activo not in validos:
            raise ValueError(f"Modo ético inválido: {self.modo_etico_activo}. Opciones: {sorted(validos)}")
        if self.camas_uci_totales < 0 or self.camas_uci_disponibles < 0 or self.ventiladores_disponibles < 0:
            raise ValueError("Los recursos hospitalarios no pueden ser negativos.")
        if self.camas_uci_disponibles > self.camas_uci_totales:
            raise ValueError("Las camas disponibles no pueden superar las camas totales.")

    @property
    def modo(self) -> ModoEtico:
        return ModoEtico(self.modo_etico_activo)

    @property
    def ocupacion_pct(self) -> float:
        if self.camas_uci_totales == 0:
            return 100.0
        ocupadas = self.camas_uci_totales - self.camas_uci_disponibles
        return round(100.0 * ocupadas / self.camas_uci_totales, 1)


# ---------------------------------------------------------------------------
# Explicabilidad: traza de inferencia
# ---------------------------------------------------------------------------
@dataclass
class TrazaInferencia:
    """Paso atómico de inferencia (XAI). Cada regla disparada produce una traza."""

    agente: str
    regla_id: str
    condicion_evaluada: str
    conclusion: str
    fundamento_normativo_o_medico: str
    paciente_id: Optional[str] = None
    nivel: str = Nivel.INFO.value
    etapa: str = Etapa.REGLA.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Resultados intermedios
# ---------------------------------------------------------------------------
@dataclass
class EvaluacionClinica:
    """Conclusiones del ClinicalTriageAgent para un paciente."""

    paciente_id: str
    prioridad: int
    reversibilidad: str            # ALTA | MEDIA | BAJA | NULA
    prob_sobrevida: float          # 0..1 (aproximación docente por SOFA)
    puntaje_urgencia: float        # 0..100 (severidad fisiológica inmediata)
    hipoxemia_refractaria: bool
    requiere_soporte_invasivo: bool
    clasificacion_sdra: str
    triage_urgencias: str
    futilidad_terapeutica: bool
    news2_score: int = 0
    news2_riesgo: str = "BAJO"
    news2_respuesta: str = ""

    @property
    def etiqueta_prioridad(self) -> str:
        return ETIQUETAS_PRIORIDAD.get(self.prioridad, f"P{self.prioridad}")


ETIQUETAS_PRIORIDAD: Dict[int, str] = {
    1: "P1 · Crítico inestable",
    2: "P2 · Monitoreo intensivo",
    3: "P3 · Estable",
    4: "P4 · Paliativo",
}


@dataclass
class DictamenAgente:
    """Veredicto de un agente del comité, con todas sus trazas explicativas."""

    agente: str
    rol: str
    veredicto: str
    nivel: str = Nivel.INFO.value
    trazas: List[TrazaInferencia] = field(default_factory=list)
    datos: Dict[str, Any] = field(default_factory=dict)

    def trazas_de(self, paciente_id: Optional[str]) -> List[TrazaInferencia]:
        return [t for t in self.trazas if t.paciente_id == paciente_id]

    def trazas_globales(self) -> List[TrazaInferencia]:
        return [t for t in self.trazas if t.paciente_id is None]


@dataclass
class ReporteRobustez:
    """Informe de sensibilidad estocástica (Monte Carlo) ante variabilidad instrumental.

    Mide la estabilidad y robustez de las recomendaciones del comité frente al error
    instrumental de pulsioxímetros (SatO2), variabilidad analítica de gases arteriales (PaFi)
    y discrepancia interobservador en escalas de disfunción orgánica (SOFA).
    """

    paciente_id: str
    porcentaje_estabilidad_decision: float
    iteraciones_totales: int = 1000
    desviaciones_simuladas: Dict[str, float] = field(default_factory=dict)
    diagnostico_frontera: str = ""
    advertencia_clinica: str = ""

    def __post_init__(self) -> None:
        self.porcentaje_estabilidad_decision = round(float(self.porcentaje_estabilidad_decision), 2)
        if not self.diagnostico_frontera:
            self.diagnostico_frontera = (
                "ROBUSTA / SEGURA" if self.porcentaje_estabilidad_decision >= 90.0
                else "ZONA FRONTERA / CASO BORDE"
            )
        if not self.advertencia_clinica:
            if self.porcentaje_estabilidad_decision >= 90.0:
                self.advertencia_clinica = (
                    "Recomendación clínicamente robusta y segura ante variabilidad instrumental y "
                    "de observador. La asignación se mantiene consistente en simulaciones estocásticas."
                )
            else:
                self.advertencia_clinica = (
                    "ALERTA CLÍNICA: Recomendación en zona frontera / caso borde. La decisión es "
                    f"sensible a la tolerancia de los instrumentos de medición (inestabilidad del "
                    f"{100.0 - self.porcentaje_estabilidad_decision:.1f}%). Se sugiere repetición inmediata de gases "
                    "arteriales y reevaluación SOFA antes de ejecutar el acto médico definitivo."
                )


@dataclass
class ContextoAuditoria:
    """Expediente que el coordinador entrega al BioethicsAuditAgent."""

    pacientes: List[Paciente]
    evaluaciones: Dict[str, EvaluacionClinica]
    estado: RecursoHospitalario
    estado_hospital: str
    candidatos_p1: List[str]
    orden_prioridad: List[str]
    puntajes: Dict[str, float]
    asignados: List[str]
    no_asignados: List[str]
    asignados_contrafactual_sin_edad: Optional[List[str]] = None
    proteccion_reforzada_aplicada: List[str] = field(default_factory=list)
    empates_detectados: List[List[str]] = field(default_factory=list)
    reportes_robustez: Optional[Dict[str, ReporteRobustez]] = None


@dataclass
class DecisionFinal:
    """Recomendación consolidada del comité para un paciente."""

    paciente_id: str
    nombre: str
    prioridad: int
    destino: str
    asignado_uci: bool
    asignado_ventilador: bool
    posicion_ranking: Optional[int]
    puntaje_desempate: Optional[float]
    validacion_etica: str
    justificacion: str
    plan_alternativo: str = ""


@dataclass
class ResultadoComite:
    """Acta completa de una sesión de deliberación del comité multiagente."""

    modo: str
    estado_hospital: str
    pacientes: List[Paciente]
    evaluaciones: Dict[str, EvaluacionClinica]
    dictamenes: Dict[str, DictamenAgente]
    decisiones: Dict[str, DecisionFinal]
    orden_prioridad: List[str]
    puntajes: Dict[str, float]
    trazas_coordinador: List[TrazaInferencia]
    acta_consenso: str
    validacion_global: str
    asignados_contrafactual_sin_edad: Optional[List[str]] = None
    etiqueta_puntaje: str = ""
    reportes_robustez: Optional[Dict[str, ReporteRobustez]] = None

    # -- utilidades de consulta ------------------------------------------------
    def paciente(self, paciente_id: str) -> Paciente:
        for p in self.pacientes:
            if p.id == paciente_id:
                return p
        raise KeyError(paciente_id)

    @property
    def asignados_uci(self) -> List[str]:
        return [pid for pid, d in self.decisiones.items() if d.asignado_uci]

    def todas_las_trazas(self) -> List[TrazaInferencia]:
        trazas: List[TrazaInferencia] = []
        for dictamen in self.dictamenes.values():
            trazas.extend(dictamen.trazas)
        trazas.extend(self.trazas_coordinador)
        return trazas

    def cadena_xai(self, paciente_id: str) -> List[TrazaInferencia]:
        """Cadena explicativa ordenada por etapa: Hecho → Regla → Recurso → Ética → Decisión."""
        orden_etapas = [e.value for e in Etapa]
        trazas = [t for t in self.todas_las_trazas() if t.paciente_id == paciente_id]
        return sorted(trazas, key=lambda t: orden_etapas.index(t.etapa) if t.etapa in orden_etapas else 99)


# ---------------------------------------------------------------------------
# Simulación Temporal Multi-Ronda (0h -> 6h -> 12h -> 24h)
# ---------------------------------------------------------------------------
@dataclass
class EvolucionPaciente:
    """Registro de evolución fisiológica y de cuidado de un paciente en un corte temporal.

    Permite simular la trayectoria clínica (respuesta a terapia prono/ventilatoria,
    falla orgánica progresiva o transición paliativa).
    """

    paciente_id: str
    hora: int                      # 0, 6, 12, 24
    delta_sofa: int                # ej. -3 indica mejoría, +4 deterioro severo
    delta_sat_o2: float
    delta_pafi: int
    evento_clinico: str            # ej. "Extubación exitosa por mejoría", "Falla hemodinámica refractaria", "Estable"
    libera_cama: bool              # True si pasa a cuidados intermedios o fallece/pasa a paliativos
    nuevo_estado_cuidado: str      # "UCI", "INTERMEDIOS", "PALIATIVOS", "ALTA"


@dataclass
class EstadoRonda:
    """Fotografía del estado del hospital y deliberación del comité en un corte temporal."""

    hora: int
    recursos: RecursoHospitalario
    resultado_comite: ResultadoComite
    eventos: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Alias en inglés exigidos por la especificación
# ---------------------------------------------------------------------------
Patient = Paciente
RuleTrace = TrazaInferencia
HospitalState = RecursoHospitalario
AgentVerdict = DictamenAgente

__all__ = [
    "ModoEtico", "Prioridad", "Nivel", "Etapa", "EstadoHospital", "Destino", "ValidacionEtica",
    "Paciente", "RecursoHospitalario", "TrazaInferencia", "EvaluacionClinica", "DictamenAgente",
    "ContextoAuditoria", "DecisionFinal", "ResultadoComite", "ETIQUETAS_PRIORIDAD",
    "ReporteRobustez", "EvolucionPaciente", "EstadoRonda",
    "Patient", "RuleTrace", "HospitalState", "AgentVerdict",
]
