"""
app.py — Consola de Triage Ético UCI (Comité Clínico-Bioético Multiagente)
=============================================================================

Interfaz Streamlit del Sistema Experto Multiagente Simbólico para el triage en
Cuidados Intensivos (UCI) y la asignación ética de recursos críticos.

Principios de diseño de esta interfaz:
    * Primero la respuesta: la recomendación del comité se ve sin hacer scroll.
    * Revelación progresiva: el detalle técnico (reglas, trazas, normas) está a
      un clic, dentro de pestañas y desplegables, nunca encima del resultado.
    * Paleta clara y de bajo contraste (ver .streamlit/config.toml) con color
      usado solo para comunicar estado clínico.

Ejecución:
    cd triage_multiagent_system
    streamlit run app.py

Universidad Católica Luis Amigó · Curso de Análisis de Inteligencia Artificial.
"""

from __future__ import annotations

import html
import json
import textwrap
from dataclasses import asdict
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import knowledge_base as kb
from core.coordinator import ConsensusCoordinator
from core.models import (
    ETIQUETAS_PRIORIDAD, DecisionFinal, Etapa, ModoEtico, Paciente, RecursoHospitalario, ReporteRobustez,
    ResultadoComite, TrazaInferencia,
)
from core.nlg_generator import ClinicalNarrativeGenerator

# =============================================================================
# Configuración de página
# =============================================================================
st.set_page_config(
    page_title="Triage Ético UCI · Comité Clínico-Bioético",
    page_icon=":material/local_hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Sistema de diseño (tokens de color y etiquetas legibles)
# =============================================================================
# Colores suaves para gráficos y acentos (no saturados).
HEX = {
    "teal": "#0E7C7B", "verde": "#3FA37C", "ambar": "#E3A23B", "rojo": "#D9605A",
    "azul": "#5B8FD1", "violeta": "#8C7BC4", "gris": "#94A3B8", "texto": "#1E293B", "muted": "#64748B",
}

PRIORIDAD_UI = {
    1: {"corto": "P1 · Crítico", "badge": "red", "hex": HEX["rojo"]},
    2: {"corto": "P2 · Vigilancia intensiva", "badge": "orange", "hex": HEX["ambar"]},
    3: {"corto": "P3 · Estable", "badge": "blue", "hex": HEX["azul"]},
    4: {"corto": "P4 · Paliativo", "badge": "violet", "hex": HEX["violeta"]},
}

VALIDACION_UI = {
    "VALIDADO": ("green", ":material/verified:", "Validado"),
    "VALIDADO CON OBSERVACIONES": ("orange", ":material/info:", "Validado con observaciones"),
    "OBJETADO": ("orange", ":material/report:", "Objetado"),
    "VETO CONSTITUCIONAL": ("red", ":material/gpp_bad:", "Veto constitucional"),
}

NIVEL_HEX = {"INFO": "#A3B1C2", "OK": HEX["verde"], "ALERTA": HEX["ambar"], "OBJECION": "#E07B3C",
             "CRITICO": HEX["rojo"]}

AGENTE_UI = {
    "ClinicalTriageAgent": {
        "icono": ":material/stethoscope:", "titulo": "Evaluación clínica", "color": "blue",
        "sub": "Gravedad fisiológica, reversibilidad (SOFA) y prioridad P1–P4.",
    },
    "ResourceManagerAgent": {
        "icono": ":material/bed:", "titulo": "Gestión de camas", "color": "green",
        "sub": "Contrasta la demanda crítica con camas UCI y ventiladores.",
    },
    "BioethicsAuditAgent": {
        "icono": ":material/balance:", "titulo": "Auditoría bioética y legal", "color": "violet",
        "sub": "Bloquea edadismo, exige plan de no abandono y valida la legalidad.",
    },
    "ConsensusCoordinator": {
        "icono": ":material/hub:", "titulo": "Coordinador de consenso", "color": "gray",
        "sub": "Integra los dictámenes y aplica el marco ético activo.",
    },
}

MARCO_CORTO = {
    ModoEtico.UTILITARISTA_CLINICO.value: "Utilitarista",
    ModoEtico.DEONTOLOGICO_IGUALITARIO.value: "Deontológico",
    ModoEtico.CONSENSO_CONSTITUCIONAL_COLOMBIA.value: "Constitucional",
}

ETAPA_UI = {
    Etapa.HECHO.value: "Datos clínicos del paciente",
    Etapa.REGLA.value: "Reglas clínicas aplicadas",
    Etapa.RECURSO.value: "Disponibilidad de recursos",
    Etapa.AUDITORIA.value: "Auditoría ética y legal",
    Etapa.DECISION.value: "Recomendación final",
}

# Estilos mínimos: solo lo que el tema nativo de Streamlit no cubre.
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      [data-testid="stMetricLabel"] p {color: #64748B;}
      .stTabs [data-baseweb="tab-list"] {gap: 0.35rem;}
      .stTabs [data-baseweb="tab"] {padding-left: 0.6rem; padding-right: 0.6rem;}

      .eyebrow {font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
                color: #0E7C7B; margin-bottom: -0.4rem;}

      .callout {border-radius: 10px; padding: 12px 16px; margin: 4px 0 14px; font-size: .95rem; line-height: 1.5;}
      .callout.warn {background: #FFF8EC; border: 1px solid #F5E1B8; color: #6B4A0F;}
      .callout.ok   {background: #EFF8F4; border: 1px solid #D2EBDF; color: #1F5A43;}

      .pname {font-size: 1.08rem; font-weight: 600; color: #1E293B; line-height: 1.3;}
      .pmeta {font-size: .82rem; color: #64748B; margin-bottom: 2px;}

      .vitals {display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin: 2px 0 6px;}
      .vital {background: #F6F8FA; border-radius: 8px; padding: 6px 8px;}
      .vital .lbl {font-size: .68rem; color: #64748B; text-transform: uppercase; letter-spacing: .04em;}
      .vital .val {font-size: 1rem; font-weight: 600; color: #1E293B;}
      .vital.alerta {background: #FFF6E6;} .vital.alerta .val {color: #A15C07;}
      .vital.critico {background: #FDEEEE;} .vital.critico .val {color: #B93A35;}

      .names {font-size: 1.12rem; font-weight: 600; color: #1E293B; margin: 2px 0 4px; line-height: 1.45;}
      .verdict {background: #F6F8FA; border-radius: 8px; padding: 10px 12px; font-size: .9rem; color: #334155;
                margin: 4px 0 8px; line-height: 1.45;}

      .step {display: flex; align-items: center; gap: 10px; margin: 18px 0 4px;}
      .step .num {width: 26px; height: 26px; border-radius: 50%; background: #E3F1F0; color: #0E7C7B;
                  font-weight: 700; font-size: .85rem; display: flex; align-items: center; justify-content: center;}
      .step .ttl {font-weight: 600; color: #1E293B;}

      .trace {border-left: 3px solid #CBD5E1; background: #FAFBFC; border-radius: 0 8px 8px 0;
              padding: 8px 12px; margin: 6px 0 6px 13px; font-size: .88rem; line-height: 1.5; color: #334155;}
      .trace .code {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .74rem;
                    color: #475569; background: #EEF2F6; padding: 1px 6px; border-radius: 4px; margin-right: 6px;}
      .trace .ag {font-size: .76rem; color: #64748B;}
      .trace .cond {color: #64748B; font-size: .82rem; margin-top: 2px;}
      .trace .fund {color: #94A3B8; font-size: .78rem; margin-top: 2px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Estado de sesión y callbacks
# =============================================================================
def _init_estado() -> None:
    if "pacientes" not in st.session_state:
        st.session_state.pacientes = kb.escenario_dilema_critico()
    st.session_state.setdefault("camas", 1)
    st.session_state.setdefault("ventiladores", 1)
    st.session_state.setdefault("camas_totales", 12)
    st.session_state.setdefault("modo", ModoEtico.CONSENSO_CONSTITUCIONAL_COLOMBIA.value)
    st.session_state.setdefault("umbral", kb.UMBRAL_EMPATE_TECNICO_SOFA)
    st.session_state.setdefault("aviso", "")


def cb_cargar_escenario(clave: str) -> None:
    info = kb.ESCENARIOS[clave]
    st.session_state.pacientes = kb.cargar_escenario(clave)
    st.session_state.camas = int(info["camas"])
    st.session_state.ventiladores = int(info["ventiladores"])
    st.session_state.aviso = f"Escenario cargado: {info['titulo']}"


def cb_vaciar() -> None:
    st.session_state.pacientes = []
    st.session_state.aviso = "Sala de urgencias vaciada."


def cb_retirar() -> None:
    retirar = set(st.session_state.get("retirar_ids", []))
    st.session_state.pacientes = [p for p in st.session_state.pacientes if p.id not in retirar]
    st.session_state.retirar_ids = []
    if retirar:
        st.session_state.aviso = f"{len(retirar)} paciente(s) retirado(s)."


def _siguiente_id() -> str:
    usados = {p.id for p in st.session_state.pacientes}
    n = 401
    while f"PAC-{n:03d}" in usados:
        n += 1
    return f"PAC-{n:03d}"


_init_estado()


# =============================================================================
# Cálculo (con caché para que la interfaz responda al instante)
# =============================================================================
def _firma_pacientes(pacientes: List[Paciente]) -> str:
    return json.dumps([asdict(p) for p in pacientes], sort_keys=True, ensure_ascii=False, default=str)


@st.cache_data(show_spinner=False, max_entries=64)
def analisis_monte_carlo(_pacientes: List[Paciente], firma: str, camas: int, ventiladores: int,
                         camas_totales: int, modo: str, umbral: int) -> Dict[str, ReporteRobustez]:
    """Análisis de sensibilidad Monte Carlo (1000 iteraciones, semilla 42), cacheado por escenario."""
    estado_mc = RecursoHospitalario(camas_uci_totales=camas_totales, camas_uci_disponibles=camas,
                                    ventiladores_disponibles=ventiladores, modo_etico_activo=modo)
    return ConsensusCoordinator(umbral_empate_sofa=umbral).ejecutar_analisis_monte_carlo(
        _pacientes, estado_mc, n_iteraciones=1000, semilla=42)


# =============================================================================
# Utilidades de presentación
# =============================================================================
def esc(texto: object) -> str:
    return html.escape(str(texto))


def badge_prioridad(p: int) -> str:
    ui = PRIORIDAD_UI.get(p, {"corto": f"P{p}", "badge": "gray"})
    return f":{ui['badge']}-badge[{ui['corto']}]"


def badge_validacion(v: str) -> str:
    color, icono, etiqueta = VALIDACION_UI.get(v, ("gray", ":material/info:", v.capitalize()))
    return f":{color}-badge[{icono} {etiqueta}]"


def badge_news2(p: Paciente) -> str:
    puntaje, nivel, _ = kb.calcular_news2(p)
    color = {"BAJO": "green", "MEDIO": "orange"}.get(nivel, "red")
    return f":{color}-badge[NEWS2 {puntaje} · riesgo {nivel.lower()}]"


def destino_ui(d: DecisionFinal) -> Dict[str, str]:
    """Traduce la decisión técnica a una etiqueta corta y amigable."""
    destino = (d.destino or "").upper()
    if d.asignado_uci:
        extra = " + ventilador" if d.asignado_ventilador else ""
        return {"label": f"Cama UCI asignada{extra}", "corto": "UCI", "icono": ":material/bed:",
                "color": "green", "cat": "uci"}
    if "PALIATIVO" in destino or d.prioridad == 4:
        return {"label": "Cuidado paliativo integral", "corto": "Paliativo", "icono": ":material/spa:",
                "color": "violet", "cat": "otros"}
    if "CRUE" in destino or d.prioridad == 1:
        pos = f" (puesto #{d.posicion_ranking})" if d.posicion_ranking else ""
        return {"label": f"En espera · remisión CRUE{pos}", "corto": "Espera · CRUE",
                "icono": ":material/ambulance:", "color": "orange", "cat": "crue"}
    if "INTERMEDIOS" in destino or d.prioridad == 2:
        return {"label": "Cuidados intermedios", "corto": "Intermedios", "icono": ":material/monitor_heart:",
                "color": "blue", "cat": "otros"}
    if "HOSPITALIZACI" in destino or d.prioridad == 3:
        return {"label": "Hospitalización general", "corto": "Hospitalización", "icono": ":material/home_health:",
                "color": "gray", "cat": "otros"}
    return {"label": d.destino.capitalize(), "corto": d.destino.capitalize(), "icono": ":material/info:",
            "color": "gray", "cat": "otros"}


def _clase_vital(valor: float, critico: bool, alerta: bool) -> str:
    return "critico" if critico else ("alerta" if alerta else "")


def html_vitales(p: Paciente) -> str:
    items = [
        ("SOFA", p.escala_sofa, _clase_vital(p.escala_sofa, p.escala_sofa >= 12, p.escala_sofa >= 7)),
        ("PaFi", p.pafi, _clase_vital(p.pafi, p.pafi <= 100, p.pafi <= 200)),
        ("SatO₂", f"{p.sat_o2:g}%", _clase_vital(p.sat_o2, p.sat_o2 < 85, p.sat_o2 < 92)),
        ("Glasgow", p.glasgow, _clase_vital(p.glasgow, p.glasgow <= 8, p.glasgow <= 12)),
    ]
    celdas = "".join(
        f'<div class="vital {c}"><div class="lbl">{esc(lbl)}</div><div class="val">{esc(val)}</div></div>'
        for lbl, val, c in items
    )
    return f'<div class="vitals">{celdas}</div>'


def render_traza(t: TrazaInferencia, mostrar_agente: bool = True) -> None:
    color = NIVEL_HEX.get(t.nivel, "#CBD5E1")
    agente = AGENTE_UI.get(t.agente, {}).get("titulo", t.agente)
    st.markdown(
        f'<div class="trace" style="border-left-color:{color}">'
        f'<div><span class="code">{esc(t.regla_id)}</span>'
        f'{f"<span class=ag>{esc(agente)}</span>" if mostrar_agente else ""}</div>'
        f'<div>{esc(t.conclusion)}</div>'
        f'<div class="cond">Condición evaluada: {esc(t.condicion_evaluada)}</div>'
        f'<div class="fund">Fundamento: {esc(t.fundamento_normativo_o_medico)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def estilo_figura(fig: go.Figure, titulo: str, alto: int) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=14, color=HEX["texto"]), x=0, xanchor="left"),
        height=alto, template="plotly_white", margin=dict(t=46, b=8, l=8, r=8),
        font=dict(size=12, color="#475569"), legend=dict(orientation="h", y=-0.2, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="#EEF2F6", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF2F6", zeroline=False)
    return fig


PLOTLY_CONFIG = {"displayModeBar": False}


def _dot_escape(texto: str, ancho: int = 38) -> str:
    texto = texto.replace("\\", "/").replace('"', "'")
    return "\\n".join(textwrap.wrap(texto, ancho)) or " "


def arbol_dot(r: ResultadoComite, pid: str) -> str:
    """Árbol de inferencia XAI en lenguaje DOT (Graphviz renderizado por Streamlit)."""
    p = r.paciente(pid)
    cadena = r.cadena_xai(pid)
    lineas = [
        "digraph G {",
        'rankdir=TB; nodesep=0.25; ranksep=0.35; bgcolor="transparent";',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, fillcolor="#FFFFFF", '
        'color="#D5DDE6", fontcolor="#334155"];',
        'edge [color="#B8C4D2", arrowsize=0.6];',
        f'root [label="{_dot_escape(p.nombre + " · " + p.id, 26)}", shape=ellipse, fillcolor="{HEX["teal"]}", '
        f'fontcolor="white", fontsize=11, color="{HEX["teal"]}"];',
    ]
    anterior = "root"
    for i, etapa in enumerate(e.value for e in Etapa):
        trazas = [t for t in cadena if t.etapa == etapa]
        if not trazas:
            continue
        hub = f"etapa{i}"
        lineas.append(f'{hub} [label="{ETAPA_UI.get(etapa, etapa)}", fillcolor="#EEF6F6", color="#BFDCDB", '
                      f'fontsize=10, fontcolor="#0E5C5B"];')
        lineas.append(f'{anterior} -> {hub} [penwidth=1.6, color="{HEX["teal"]}"];')
        for j, t in enumerate(trazas):
            nodo = f"t{i}_{j}"
            color = NIVEL_HEX.get(t.nivel, "#CBD5E1")
            concl = t.conclusion if len(t.conclusion) < 95 else t.conclusion[:92] + "..."
            lineas.append(f'{nodo} [label="{_dot_escape(t.regla_id)}\\n{_dot_escape(concl)}", color="{color}", '
                          f'penwidth=1.4, fillcolor="#FAFBFC"];')
            lineas.append(f"{hub} -> {nodo};")
        anterior = hub
    lineas.append("}")
    return "\n".join(lineas)


def paso(numero: int, titulo: str) -> None:
    st.markdown(f'<div class="step"><div class="num">{numero}</div><div class="ttl">{esc(titulo)}</div></div>',
                unsafe_allow_html=True)


# =============================================================================
# Diálogo: agregar paciente (antes era un formulario largo en la barra lateral)
# =============================================================================
@st.dialog("Agregar paciente a urgencias", width="large")
def dialogo_agregar_paciente() -> None:
    st.caption("Complete los datos clínicos básicos. Los signos vitales para NEWS2 son opcionales.")
    with st.form("form_paciente", border=False):
        nombre = st.text_input("Nombre completo", placeholder="Ej. Mariana Duque")
        c1, c2, c3 = st.columns(3)
        edad = c1.number_input("Edad (años)", 0, 120, 48)
        espera = c2.number_input("Llegó hace (min)", 0, kb.MINUTO_ACTUAL_TURNO, 15)
        glasgow = c3.number_input("Glasgow (3-15)", 3, 15, 14)
        c4, c5, c6 = st.columns(3)
        sat = c4.number_input("SatO₂ (%)", 50.0, 100.0, 86.0, step=0.5)
        pafi = c5.number_input("PaFi", 40, 600, 160)
        sofa = c6.number_input("SOFA (0-24)", 0, 24, 6)
        comorb = st.text_input("Comorbilidades graves", placeholder="Separadas por coma. Ej. Hipertensión, Asma")

        st.markdown("**Condiciones del paciente**")
        vent = st.checkbox("Requiere ventilador mecánico", value=True)
        terminal = st.checkbox("Enfermedad terminal irreversible")
        vulnerable = st.checkbox("Sujeto de especial protección constitucional")

        with st.expander("Signos vitales para NEWS2 (opcional)"):
            n1, n2, n3 = st.columns(3)
            fr = n1.number_input("Frecuencia respiratoria (rpm)", 0, 80, 16)
            pas = n2.number_input("Presión sistólica (mmHg)", 30, 300, 120)
            fc = n3.number_input("Frecuencia cardíaca (lpm)", 20, 250, 75)
            n4, n5 = st.columns(2)
            temp = n4.number_input("Temperatura (°C)", 25.0, 45.0, 36.8, step=0.1)
            acvpu = n5.selectbox(
                "Nivel de conciencia (ACVPU)", ["A", "C", "V", "P", "U"],
                format_func=lambda x: {"A": "A · Alerta", "C": "C · Confusión nueva", "V": "V · Responde a la voz",
                                       "P": "P · Responde al dolor", "U": "U · Inconsciente"}[x],
            )
            o2_sup = st.checkbox("Oxígeno suplementario al ingreso")

        enviado = st.form_submit_button("Ingresar a urgencias", type="primary", icon=":material/person_add:",
                                        width="stretch")

    if enviado:
        try:
            nuevo = Paciente(
                id=_siguiente_id(),
                nombre=nombre.strip() or "Paciente sin nombre",
                edad=int(edad),
                hora_llegada=kb.MINUTO_ACTUAL_TURNO - int(espera),
                sat_o2=float(sat),
                pafi=int(pafi),
                escala_sofa=int(sofa),
                glasgow=int(glasgow),
                requiere_ventilador=vent,
                comorbilidades_graves=[c.strip() for c in comorb.split(",") if c.strip()],
                enfermedad_terminal_irreversible=terminal,
                vulnerabilidad_social=vulnerable,
                descripcion="Paciente registrado manualmente en el censo actual.",
                frecuencia_respiratoria=int(fr),
                presion_sistolica=int(pas),
                frecuencia_cardiaca=int(fc),
                temperatura=float(temp),
                oxigeno_suplementario=bool(o2_sup),
                escala_acvpu=str(acvpu),
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state.pacientes = st.session_state.pacientes + [nuevo]
        st.session_state.aviso = f"{nuevo.nombre} ({nuevo.id}) ingresó a urgencias."
        st.rerun()


# =============================================================================
# Barra lateral: configuración del caso (4 pasos numerados)
# =============================================================================
with st.sidebar:
    st.markdown("### :material/tune: Configuración del caso")
    st.caption("Los resultados se actualizan al instante con cada cambio.")

    st.markdown("**1 · Escenario clínico**")
    escenario_sel = st.selectbox(
        "Escenario clínico", list(kb.ESCENARIOS), format_func=lambda k: kb.ESCENARIOS[k]["titulo"],
        key="escenario_sel", label_visibility="collapsed",
    )
    st.button("Cargar escenario", type="primary", icon=":material/play_circle:", width="stretch",
              on_click=cb_cargar_escenario, args=(escenario_sel,),
              help="Reemplaza los pacientes actuales por los del escenario seleccionado.")

    st.divider()
    st.markdown("**2 · Recursos disponibles**")
    st.slider("Camas UCI libres", 0, 5, key="camas",
              help="Camas de cuidados intensivos libres en este turno.")
    st.slider("Ventiladores libres", 0, 5, key="ventiladores",
              help="Ventiladores mecánicos invasivos disponibles de inmediato.")
    with st.expander("Capacidad instalada"):
        st.slider("Camas UCI totales de la institución", 5, 40, key="camas_totales")

    st.divider()
    st.markdown("**3 · Marco ético para desempatar**")
    modo_sel = st.radio(
        "Marco ético", [m.value for m in ModoEtico], key="modo",
        format_func=lambda v: kb.MARCOS_ETICOS[v]["nombre"], label_visibility="collapsed",
    )
    st.caption(kb.MARCOS_ETICOS[modo_sel]["principio"])

    st.divider()
    st.markdown("**4 · Opciones avanzadas**")
    simulacion_activa = st.toggle(
        "Simulación temporal (0 h → 24 h)", key="modo_simulacion_temporal",
        help="Simula la evolución fisiológica a 6, 12 y 24 horas, con liberación de camas UCI y "
             "reasignación auditada al paciente en espera.",
    )
    if simulacion_activa:
        st.select_slider(
            "Corte temporal", options=[0, 6, 12, 24], key="hora_simulacion",
            format_func=lambda h: f"{h} h" + (" · ingreso" if h == 0 else ""),
        )
    with st.expander("Parámetros del consenso"):
        st.slider(
            "Umbral de empate técnico SOFA (± puntos)", 0, 3, key="umbral",
            help="Diferencias de SOFA menores o iguales a este valor se consideran clínicamente "
                 "indistinguibles (variabilidad interobservador). Solo aplica al Consenso Constitucional.",
        )
        st.caption(f"Semilla pública del sorteo auditable: `{kb.SEMILLA_SORTEO_AUDITABLE}`")

    st.divider()
    st.caption(":material/info: Prototipo académico de apoyo a la decisión. La decisión final corresponde "
               "al médico tratante y al Comité de Ética (Ley 23 de 1981).")


# =============================================================================
# Encabezado
# =============================================================================
col_tit, col_estado = st.columns([4, 1.5], vertical_alignment="center")
with col_tit:
    st.markdown('<div class="eyebrow">Unidad de Cuidados Intensivos · Comité clínico-bioético</div>',
                unsafe_allow_html=True)
    st.title("Triage ético de recursos UCI", anchor=False)
    st.caption("Sistema experto multiagente y explicable (XAI) · Universidad Católica Luis Amigó · "
               "Análisis de Inteligencia Artificial")
with col_estado:
    st.markdown(":green-badge[:material/check_circle: Motor local activo]  \n"
                ":gray-badge[:material/person: Decisión final: equipo médico]")

if st.session_state.aviso:
    st.toast(st.session_state.aviso, icon=":material/check_circle:")
    st.session_state.aviso = ""

with st.expander(":material/help: ¿Cómo usar esta herramienta?"):
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**1 · Defina el caso**")
        st.caption("Cargue un escenario en el panel izquierdo o agregue pacientes con el botón "
                   "*Agregar paciente*.")
    with g2:
        st.markdown("**2 · Ajuste recursos y ética**")
        st.caption("Cambie camas, ventiladores y el marco ético. El comité vuelve a deliberar al instante.")
    with g3:
        st.markdown("**3 · Lea el resultado**")
        st.caption("*Resumen* muestra quién recibe la cama; *Explicación* muestra paso a paso el porqué.")
    st.markdown("**Prioridades clínicas:** " + " ".join(badge_prioridad(i) for i in (1, 2, 3, 4)))

pacientes: List[Paciente] = st.session_state.pacientes

# Estado vacío amigable
if not pacientes:
    with st.container(border=True):
        st.subheader("La sala de urgencias está vacía", anchor=False)
        st.caption("Cargue un caso de demostración o agregue un paciente para que el comité delibere.")
        e1, e2, _ = st.columns([1, 1, 2])
        e1.button("Cargar caso de demostración", type="primary", icon=":material/play_circle:", width="stretch",
                  on_click=cb_cargar_escenario, args=("dilema",))
        if e2.button("Agregar paciente", icon=":material/person_add:", width="stretch"):
            dialogo_agregar_paciente()
    st.stop()


# =============================================================================
# Deliberación del comité
# =============================================================================
estado = RecursoHospitalario(
    camas_uci_totales=max(st.session_state.camas_totales, st.session_state.camas),
    camas_uci_disponibles=st.session_state.camas,
    ventiladores_disponibles=st.session_state.ventiladores,
    modo_etico_activo=st.session_state.modo,
)
coordinador = ConsensusCoordinator(umbral_empate_sofa=st.session_state.umbral)

with st.spinner("El comité está deliberando…"):
    reportes_robustez = analisis_monte_carlo(
        pacientes, _firma_pacientes(pacientes), estado.camas_uci_disponibles, estado.ventiladores_disponibles,
        estado.camas_uci_totales, estado.modo_etico_activo, st.session_state.umbral,
    )

sim_activa = st.session_state.get("modo_simulacion_temporal", False)
rondas_sim = None
eventos_ronda_actual: List[str] = []
hora_actual = 0

if sim_activa:
    rondas_sim = coordinador.simular_linea_tiempo(pacientes, estado, kb.escenario_evolucion_temporal_dilema())
    hora_actual = st.session_state.get("hora_simulacion", 0)
    ronda_actual = next((r for r in rondas_sim if r.hora == hora_actual), rondas_sim[0])
    resultado = ronda_actual.resultado_comite
    pacientes = list(resultado.pacientes)
    estado = ronda_actual.recursos
    eventos_ronda_actual = ronda_actual.eventos
else:
    resultado = coordinador.deliberar(pacientes, estado, reportes_robustez=reportes_robustez)

comparacion: Dict[str, ResultadoComite] = coordinador.comparar_marcos(pacientes, estado)

por_id = {p.id: p for p in pacientes}
d_rec = resultado.dictamenes["ResourceManagerAgent"]
d_clin = resultado.dictamenes["ClinicalTriageAgent"]
d_bio = resultado.dictamenes["BioethicsAuditAgent"]
marco = kb.MARCOS_ETICOS[resultado.modo]

n_p1 = sum(1 for e in resultado.evaluaciones.values() if e.prioridad == 1)
deficit = d_rec.datos.get("deficit_camas", 0)
critico = resultado.estado_hospital == "ESCASEZ_CRITICA"
destinos = {pid: destino_ui(d) for pid, d in resultado.decisiones.items()}


# =============================================================================
# Situación del hospital (una frase) + indicadores
# =============================================================================
if critico:
    st.markdown(
        f'<div class="callout warn"><b>Escasez crítica.</b> {n_p1} paciente(s) crítico(s) compiten por '
        f'{estado.camas_uci_disponibles} cama(s) UCI y {estado.ventiladores_disponibles} ventilador(es). '
        f'El comité desempata con el marco <b>{esc(marco["nombre"])}</b> y garantiza remisión CRUE a quien '
        f'quede en espera (Ley 1751 de 2015).</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="callout ok"><b>Capacidad suficiente.</b> Hay recursos para todos los pacientes críticos '
        f'({n_p1}). No se requiere desempate ético.</div>',
        unsafe_allow_html=True,
    )

k1, k2, k3, k4 = st.columns(4)
k1.metric("Pacientes en urgencias", len(pacientes), border=True)
k2.metric("Pacientes críticos (P1)", n_p1, border=True,
          help=f"{d_rec.datos.get('demanda_p1_ventilador', 0)} de ellos requieren ventilador.")
k3.metric("Camas UCI libres", estado.camas_uci_disponibles, border=True,
          delta=(f"-{deficit} de déficit" if deficit else "Suficientes"),
          help=f"Capacidad instalada: {estado.camas_uci_totales} camas.")
k4.metric("Ventiladores libres", estado.ventiladores_disponibles, border=True)


# =============================================================================
# Pestañas
# =============================================================================
tab_res, tab_com, tab_mar, tab_xai, tab_nota, tab_leg = st.tabs([
    ":material/space_dashboard: Resumen",
    ":material/groups: Comité",
    ":material/balance: Marcos éticos",
    ":material/account_tree: Explicación",
    ":material/description: Nota clínica",
    ":material/gavel: Marco legal",
])


# -----------------------------------------------------------------------------
# PESTAÑA · Resumen
# -----------------------------------------------------------------------------
with tab_res:
    st.subheader("Recomendación del comité", anchor=False)
    st.markdown(f"{badge_validacion(resultado.validacion_global)} &nbsp; "
                f":gray-badge[:material/balance: {marco['nombre']}]")

    grupos = {
        "uci": [pid for pid in resultado.decisiones if destinos[pid]["cat"] == "uci"],
        "crue": [pid for pid in resultado.decisiones if destinos[pid]["cat"] == "crue"],
        "otros": [pid for pid in resultado.decisiones if destinos[pid]["cat"] == "otros"],
    }
    r1, r2, r3 = st.columns(3)
    with r1.container(border=True, height="stretch"):
        st.markdown(":green[:material/bed:] **Ingresan a UCI**")
        nombres = "<br>".join(esc(por_id[x].nombre) for x in grupos["uci"]) or "Ningún paciente"
        st.markdown(f'<div class="names">{nombres}</div>', unsafe_allow_html=True)
        st.caption("Cama y soporte ventilatorio asegurados en la institución.")
    with r2.container(border=True, height="stretch"):
        st.markdown(":orange[:material/ambulance:] **En espera · remisión CRUE**")
        nombres = "<br>".join(esc(por_id[x].nombre) for x in grupos["crue"]) or "Nadie en espera"
        st.markdown(f'<div class="names">{nombres}</div>', unsafe_allow_html=True)
        st.caption("Soporte vital en área de expansión y traslado urgente. Nadie queda sin atención.")
    with r3.container(border=True, height="stretch"):
        st.markdown(":violet[:material/spa:] **Otros destinos**")
        nombres = "<br>".join(f"{esc(por_id[x].nombre)} <span style='font-weight:400;color:#64748B;"
                              f"font-size:.9rem'>· {esc(destinos[x]['corto'])}</span>"
                              for x in grupos["otros"]) or "Ninguno"
        st.markdown(f'<div class="names">{nombres}</div>', unsafe_allow_html=True)
        st.caption("Cuidados intermedios, hospitalización o cuidado paliativo sin abandono (Ley 1733).")

    # ---- Simulación temporal (solo si está activa)
    if sim_activa and rondas_sim:
        st.write("")
        with st.container(border=True):
            st.markdown(f"**:material/schedule: Simulación temporal · corte de {hora_actual} h**")
            st.markdown(" → ".join(
                f":primary-badge[{h} h]" if h == hora_actual else f":gray-badge[{h} h]" for h in (0, 6, 12, 24)
            ))
            if eventos_ronda_actual:
                for ev in eventos_ronda_actual:
                    icono = ":green[:material/swap_horiz:]" if ("LIBERACIÓN" in ev or "REASIGNACIÓN" in ev) \
                        else ":gray[:material/info:]"
                    st.markdown(f"{icono} {ev}")
            else:
                st.caption("Sin eventos clínicos en este corte.")

            etiquetas_h = ["0 h", "6 h", "12 h", "24 h"]
            uci_ocupadas = [len([1 for d in r.resultado_comite.decisiones.values() if d.asignado_uci])
                            for r in rondas_sim]
            camas_libres_vals = [r.recursos.camas_uci_disponibles for r in rondas_sim]
            espera_vals = [len([1 for d in r.resultado_comite.decisiones.values()
                                if d.prioridad == 1 and not d.asignado_uci]) for r in rondas_sim]
            intermedios_vals = [len([1 for d in r.resultado_comite.decisiones.values()
                                     if d.destino == "CUIDADOS INTERMEDIOS / MONITOREO CONTINUO"])
                                for r in rondas_sim]
            fig_flujo = go.Figure()
            fig_flujo.add_trace(go.Bar(name="En UCI", x=etiquetas_h, y=uci_ocupadas, marker_color=HEX["verde"],
                                       text=uci_ocupadas, textposition="auto"))
            fig_flujo.add_trace(go.Bar(name="En espera CRUE", x=etiquetas_h, y=espera_vals,
                                       marker_color=HEX["ambar"], text=espera_vals, textposition="auto"))
            fig_flujo.add_trace(go.Bar(name="Pasaron a intermedios", x=etiquetas_h, y=intermedios_vals,
                                       marker_color=HEX["azul"], text=intermedios_vals, textposition="auto"))
            fig_flujo.add_trace(go.Scatter(name="Camas UCI libres", x=etiquetas_h, y=camas_libres_vals,
                                           mode="lines+markers", line=dict(color=HEX["teal"], width=2, dash="dot"),
                                           marker=dict(size=8)))
            fig_flujo.add_vline(x=etiquetas_h[[0, 6, 12, 24].index(hora_actual)], line_width=2, line_dash="dot",
                                line_color=HEX["gris"])
            fig_flujo.update_layout(barmode="group", yaxis_title="Pacientes / camas")
            st.plotly_chart(estilo_figura(fig_flujo, "Rotación de camas UCI en el tiempo", 300),
                            width="stretch", config=PLOTLY_CONFIG)

    # ---- Pacientes
    st.write("")
    ph1, ph2 = st.columns([3, 2], vertical_alignment="bottom")
    with ph1:
        st.subheader("Pacientes", anchor=False)
        st.caption("Ordenados según la recomendación: primero UCI, luego espera y otros destinos.")
    with ph2:
        b1, b2 = st.columns(2)
        if b1.button("Agregar paciente", icon=":material/person_add:", width="stretch",
                     disabled=sim_activa, help="Desactivado durante la simulación temporal." if sim_activa else None):
            dialogo_agregar_paciente()
        with b2.popover("Retirar", icon=":material/person_remove:", width="stretch", disabled=sim_activa):
            st.multiselect(
                "Pacientes a retirar", [p.id for p in st.session_state.pacientes], key="retirar_ids",
                format_func=lambda i: next(f"{p.nombre} ({p.id})" for p in st.session_state.pacientes if p.id == i),
            )
            st.button("Retirar seleccionados", on_click=cb_retirar, type="primary", width="stretch")
            st.button("Vaciar sala de urgencias", on_click=cb_vaciar, icon=":material/delete_sweep:",
                      width="stretch")

    orden_cat = {"uci": 0, "crue": 1, "otros": 2}
    ordenados = sorted(
        pacientes,
        key=lambda p: (orden_cat[destinos[p.id]["cat"]], resultado.decisiones[p.id].prioridad,
                       resultado.decisiones[p.id].posicion_ranking or 99),
    )
    n_cols = 3
    for inicio in range(0, len(ordenados), n_cols):
        cols = st.columns(n_cols)
        for col, p in zip(cols, ordenados[inicio:inicio + n_cols]):
            d = resultado.decisiones[p.id]
            e = resultado.evaluaciones[p.id]
            dui = destinos[p.id]
            with col.container(border=True):
                st.markdown(
                    f'<div class="pname">{esc(p.nombre)}</div>'
                    f'<div class="pmeta">{esc(p.id)} · {p.edad} años · llegó hace '
                    f'{p.minutos_espera(kb.MINUTO_ACTUAL_TURNO)} min</div>',
                    unsafe_allow_html=True,
                )
                etiquetas = [badge_prioridad(e.prioridad), badge_news2(p)]
                if p.vulnerabilidad_social:
                    etiquetas.append(":blue-badge[Especial protección]")
                if p.enfermedad_terminal_irreversible:
                    etiquetas.append(":gray-badge[Enfermedad terminal]")
                st.markdown(" ".join(etiquetas))
                st.markdown(html_vitales(p), unsafe_allow_html=True)
                st.markdown(f"**:{dui['color']}[{dui['icono']} {dui['label']}]**")

                with st.expander("Ver detalle clínico"):
                    news2_sc, _, news2_resp = kb.calcular_news2(p)
                    st.markdown(f"**Estado respiratorio:** {kb.clasificar_sdra_berlin(p.pafi)}")
                    st.markdown(f"**Reversibilidad:** {e.reversibilidad} · sobrevida estimada "
                                f"{e.prob_sobrevida:.0%}")
                    st.markdown(f"**Comorbilidades:** {', '.join(p.comorbilidades_graves) or 'Ninguna relevante'}")
                    st.markdown(f"**Ventilador:** {'requerido' if p.requiere_ventilador else 'no requerido'}")
                    st.markdown(
                        f"**Signos NEWS2 ({news2_sc} pts):** FR {p.frecuencia_respiratoria} rpm · "
                        f"PAS {p.presion_sistolica} mmHg · FC {p.frecuencia_cardiaca} lpm · "
                        f"T {p.temperatura:.1f} °C · O₂ {'sí' if p.oxigeno_suplementario else 'no'} · "
                        f"ACVPU {p.escala_acvpu}"
                    )
                    st.caption(news2_resp)
                    st.markdown(f"**Plan clínico:** {d.plan_alternativo or d.justificacion}")
                    st.markdown(f"**Auditoría ética:** {badge_validacion(d.validacion_etica)}")
                    rep = reportes_robustez.get(p.id) if reportes_robustez else None
                    if rep:
                        st.markdown(f"**Estabilidad de la recomendación:** {rep.porcentaje_estabilidad_decision:.1f} %")

    with st.expander(":material/table_view: Ver todos los datos en tabla"):
        st.dataframe(pd.DataFrame([
            {
                "ID": p.id, "Nombre": p.nombre, "Edad": p.edad,
                "Espera (min)": p.minutos_espera(kb.MINUTO_ACTUAL_TURNO),
                "SatO₂ (%)": p.sat_o2, "PaFi": p.pafi, "SOFA": p.escala_sofa, "Glasgow": p.glasgow,
                "NEWS2": kb.calcular_news2(p)[0],
                "Ventilador": "Sí" if p.requiere_ventilador else "No",
                "Comorbilidades": ", ".join(p.comorbilidades_graves) or "Ninguna",
                "Terminal": "Sí" if p.enfermedad_terminal_irreversible else "No",
                "Especial protección": "Sí" if p.vulnerabilidad_social else "No",
                "Prioridad": ETIQUETAS_PRIORIDAD[resultado.evaluaciones[p.id].prioridad],
                "Destino": resultado.decisiones[p.id].destino,
            }
            for p in pacientes
        ]), hide_index=True, width="stretch")


# -----------------------------------------------------------------------------
# PESTAÑA · Comité (deliberación multiagente)
# -----------------------------------------------------------------------------
with tab_com:
    st.subheader("Cómo deliberó el comité", anchor=False)
    st.caption("Tres agentes especialistas analizan el caso por separado y el coordinador integra sus dictámenes.")
    st.markdown(":blue-badge[1 · Evaluación clínica] → :green-badge[2 · Gestión de camas] → "
                ":violet-badge[3 · Auditoría bioética] → :gray-badge[4 · Consenso]")

    cols_ag = st.columns(3)
    for col, nombre_agente in zip(cols_ag, ["ClinicalTriageAgent", "ResourceManagerAgent", "BioethicsAuditAgent"]):
        d_ag = resultado.dictamenes[nombre_agente]
        ui = AGENTE_UI[nombre_agente]
        with col.container(border=True, height="stretch"):
            st.markdown(f"#### :{ui['color']}[{ui['icono']}] {ui['titulo']}")
            st.caption(ui["sub"])
            st.markdown(f'<div class="verdict">{esc(d_ag.veredicto)}</div>', unsafe_allow_html=True)
            if nombre_agente == "ClinicalTriageAgent":
                for pid, e in resultado.evaluaciones.items():
                    st.markdown(f"**{por_id[pid].nombre}** {badge_prioridad(e.prioridad)}  \n"
                                f":gray[Reversibilidad {e.reversibilidad.lower()} · sobrevida "
                                f"{e.prob_sobrevida:.0%}]")
            elif nombre_agente == "ResourceManagerAgent":
                if resultado.orden_prioridad:
                    for pid in resultado.orden_prioridad:
                        ok = pid in d_ag.datos.get("asignados", [])
                        st.markdown(f"**{por_id[pid].nombre}** "
                                    + (":green-badge[Cama asignada]" if ok else ":orange-badge[En espera · CRUE]"))
                else:
                    st.caption("No hay pacientes compitiendo por UCI.")
            else:
                for pid, v in d_ag.datos.get("validacion_por_paciente", {}).items():
                    st.markdown(f"**{por_id[pid].nombre}** {badge_validacion(v)}")
            with st.expander(f"Reglas aplicadas ({len(d_ag.trazas)})"):
                for t in d_ag.trazas:
                    render_traza(t, mostrar_agente=False)

    # ---- Acta de consenso
    st.write("")
    st.subheader("Acta de consenso", anchor=False)
    a1, a2 = st.columns([3, 2])
    with a1.container(border=True, height="stretch"):
        st.markdown(badge_validacion(resultado.validacion_global))
        st.markdown(resultado.acta_consenso)
    with a2.container(border=True, height="stretch"):
        st.markdown(f"**{marco['nombre']}**")
        st.caption(marco["escuela"])
        st.markdown(f"**Criterio:** {marco['criterio']}")
        st.markdown(f"**Tensión ética:** {marco['critica']}")

    # ---- Estabilidad (Monte Carlo)
    st.write("")
    st.subheader("¿Qué tan estable es la recomendación?", anchor=False)
    st.caption(
        "Se simularon 1.000 variaciones de los datos con errores de medición realistas (SatO₂ ±1,5 %, "
        "PaFi ±10 mmHg, SOFA ±1 punto). Si el destino casi no cambia (≥ 90 %), la recomendación es robusta; "
        "por debajo, es un caso borde y conviene repetir mediciones."
    )
    if reportes_robustez:
        with st.container(border=True):
            for p_mc in pacientes:
                rep_mc = reportes_robustez.get(p_mc.id)
                if not rep_mc:
                    continue
                pct = rep_mc.porcentaje_estabilidad_decision
                m1, m2, m3 = st.columns([2, 4, 2], vertical_alignment="center")
                m1.markdown(f"**{p_mc.nombre}**")
                m2.progress(min(max(pct / 100.0, 0.0), 1.0), text=f"{pct:.1f} %")
                m3.markdown(":green-badge[Robusta]" if pct >= 90.0 else
                            (":orange-badge[Caso borde]" if pct >= 85.0 else ":red-badge[Caso borde]"))
        with st.expander("Ver recomendaciones por paciente"):
            for p_mc in pacientes:
                rep_mc = reportes_robustez.get(p_mc.id)
                if rep_mc:
                    st.markdown(f"**{p_mc.nombre}:** {rep_mc.advertencia_clinica}")

    # ---- Transcripción
    with st.expander(":material/forum: Transcripción de la sesión del comité"):
        with st.chat_message("ClinicalTriageAgent", avatar=":material/stethoscope:"):
            st.markdown(f"**Evaluación clínica:** {d_clin.veredicto}")
            for pid, e in resultado.evaluaciones.items():
                futil = " — *futilidad terapéutica: cuidado paliativo digno*" if e.futilidad_terapeutica else ""
                st.markdown(f"- {por_id[pid].nombre}: **{e.etiqueta_prioridad}**, {e.clasificacion_sdra}, "
                            f"reversibilidad {e.reversibilidad}{futil}.")
        with st.chat_message("ResourceManagerAgent", avatar=":material/bed:"):
            st.markdown(f"**Gestión de camas:** {d_rec.veredicto}")
        with st.chat_message("ConsensusCoordinator", avatar=":material/hub:"):
            st.markdown(f"**Coordinador:** aplico el marco **{marco['nombre']}**.")
            for t in resultado.trazas_coordinador:
                if t.etapa == Etapa.REGLA.value and t.paciente_id and t.paciente_id in por_id:
                    st.markdown(f"- {por_id[t.paciente_id].nombre}: {t.conclusion}")
        with st.chat_message("ResourceManagerAgent", avatar=":material/bed:"):
            asign = ", ".join(por_id[x].nombre for x in d_rec.datos.get("asignados", []) if x in por_id) or "nadie"
            st.markdown(f"**Gestión de camas:** recursos asignados a **{asign}**.")
        with st.chat_message("BioethicsAuditAgent", avatar=":material/balance:"):
            st.markdown(f"**Auditoría bioética:** {d_bio.veredicto}")
            relevantes = [t for t in d_bio.trazas if t.nivel in ("ALERTA", "OBJECION", "CRITICO")
                          or "PROTECCIÓN REFORZADA" in t.conclusion or t.regla_id in ("REG-BIO-06", "REG-BIO-04")]
            for t in relevantes:
                quien = f" ({por_id[t.paciente_id].nombre})" if t.paciente_id in por_id else ""
                st.markdown(f"- `{t.regla_id}`{quien}: {t.conclusion}")
        with st.chat_message("ConsensusCoordinator", avatar=":material/hub:"):
            st.markdown(f"**Coordinador — consenso final:**\n\n{resultado.acta_consenso}")


# -----------------------------------------------------------------------------
# PESTAÑA · Marcos éticos
# -----------------------------------------------------------------------------
with tab_mar:
    st.subheader("El mismo caso, tres formas de decidir", anchor=False)
    st.caption("Compare a quién asignaría la cama cada enfoque ético. El marco activo se elige en el panel lateral.")

    filas = []
    for p in pacientes:
        fila = {"Paciente": f"{p.nombre} ({p.edad} años · SOFA {p.escala_sofa})"}
        for modo_k, r_k in comparacion.items():
            fila[MARCO_CORTO[modo_k]] = destino_ui(r_k.decisiones[p.id])["corto"]
        filas.append(fila)
    fila_val = {"Paciente": "Auditoría bioético-legal"}
    for modo_k, r_k in comparacion.items():
        fila_val[MARCO_CORTO[modo_k]] = VALIDACION_UI.get(r_k.validacion_global, ("", "", r_k.validacion_global))[2]
    filas.append(fila_val)

    estilos_celda = {
        "UCI": "background-color:#E8F5EE;color:#1F6B4A;font-weight:600",
        "Espera · CRUE": "background-color:#FFF6E5;color:#8A5A00",
        "Paliativo": "background-color:#F3EFFA;color:#5B4A8B",
        "Intermedios": "background-color:#EAF2FB;color:#2C5D8F",
        "Validado": "background-color:#E8F5EE;color:#1F6B4A",
        "Validado con observaciones": "background-color:#FFF6E5;color:#8A5A00",
        "Objetado": "background-color:#FFF1E8;color:#9A4A12",
        "Veto constitucional": "background-color:#FDEEEE;color:#B93A35;font-weight:600",
    }
    df_comp = pd.DataFrame(filas)
    styler = df_comp.style
    aplicar = getattr(styler, "map", None) or getattr(styler, "applymap")
    styler = aplicar(lambda v: estilos_celda.get(v, ""), subset=list(MARCO_CORTO.values()))
    st.dataframe(styler, hide_index=True, width="stretch")

    st.write("")
    cols_m = st.columns(3)
    for col_m, clave_modo in zip(cols_m, [m.value for m in ModoEtico]):
        info_m = kb.MARCOS_ETICOS[clave_modo]
        r_modo = comparacion[clave_modo]
        activo = clave_modo == resultado.modo
        ganadores = ", ".join(por_id[x].nombre for x in r_modo.asignados_uci if x in por_id) or "Nadie"
        with col_m.container(border=True, height="stretch"):
            st.markdown(f"<span style='color:{info_m['color']};font-size:1.1rem'>●</span> "
                        f"**{info_m['nombre']}**", unsafe_allow_html=True)
            st.caption(info_m["escuela"])
            if activo:
                st.markdown(":primary-badge[:material/check: Marco activo]")
            st.markdown(f"**Principio:** {info_m['principio']}")
            st.markdown(f"**Cama UCI para:** {ganadores}")
            st.markdown(badge_validacion(r_modo.validacion_global))
            with st.expander("Criterio y crítica"):
                st.markdown(f"**Criterio:** {info_m['criterio']}")
                st.markdown(f"**Crítica:** {info_m['critica']}")


# -----------------------------------------------------------------------------
# PESTAÑA · Explicación (XAI)
# -----------------------------------------------------------------------------
with tab_xai:
    st.subheader("¿Por qué el comité recomendó esto?", anchor=False)
    st.caption("Seleccione un paciente para ver, paso a paso, las reglas que llevaron a su recomendación.")

    ids_xai = [p.id for p in ordenados]
    pid_sel = st.pills("Paciente", ids_xai, selection_mode="single", default=ids_xai[0],
                       format_func=lambda i: por_id[i].nombre, key="xai_paciente",
                       label_visibility="collapsed") or ids_xai[0]
    if pid_sel not in por_id:
        pid_sel = ids_xai[0]
    p_sel = por_id[pid_sel]
    d_sel = resultado.decisiones[pid_sel]
    dui_sel = destinos[pid_sel]
    rep_sel = reportes_robustez.get(pid_sel) if reportes_robustez else None

    with st.container(border=True):
        x1, x2 = st.columns([3, 1], vertical_alignment="center")
        with x1:
            st.markdown(f"**{p_sel.nombre}** · {p_sel.id} · {p_sel.edad} años  \n"
                        f"**:{dui_sel['color']}[{dui_sel['icono']} {dui_sel['label']}]** &nbsp; "
                        f"{badge_validacion(d_sel.validacion_etica)}")
            if p_sel.descripcion:
                st.caption(p_sel.descripcion)
        with x2:
            if rep_sel:
                st.metric("Estabilidad", f"{rep_sel.porcentaje_estabilidad_decision:.1f} %",
                          help=rep_sel.advertencia_clinica)

    vista = st.segmented_control(
        "Vista", ["Paso a paso", "Árbol de inferencia", "Tabla de trazas"], default="Paso a paso",
        key="xai_vista", label_visibility="collapsed",
    ) or "Paso a paso"
    cadena = resultado.cadena_xai(pid_sel)

    if vista == "Paso a paso":
        n = 0
        for etapa in (x.value for x in Etapa):
            trazas = [t for t in cadena if t.etapa == etapa]
            if not trazas:
                continue
            n += 1
            paso(n, ETAPA_UI.get(etapa, etapa))
            for t in trazas:
                render_traza(t)
    elif vista == "Árbol de inferencia":
        st.graphviz_chart(arbol_dot(resultado, pid_sel), width="stretch")
    else:
        st.dataframe(pd.DataFrame([t.to_dict() for t in cadena]), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Vista general del triage", anchor=False)
    g1, g2 = st.columns([3, 2])
    with g1:
        fig = go.Figure()
        for prioridad in (1, 2, 3, 4):
            ids = [pid for pid, e in resultado.evaluaciones.items() if e.prioridad == prioridad]
            if not ids:
                continue
            fig.add_trace(go.Scatter(
                x=[resultado.evaluaciones[i].prob_sobrevida * 100 for i in ids],
                y=[resultado.evaluaciones[i].puntaje_urgencia for i in ids],
                mode="markers+text", name=PRIORIDAD_UI[prioridad]["corto"],
                text=[por_id[i].nombre for i in ids], textposition="top center",
                textfont=dict(size=11, color="#475569"),
                marker=dict(size=[24 if resultado.decisiones[i].asignado_uci else 14 for i in ids],
                            symbol=["star" if resultado.decisiones[i].asignado_uci else "circle" for i in ids],
                            color=PRIORIDAD_UI[prioridad]["hex"], line=dict(width=1.5, color="white")),
                customdata=[[por_id[i].edad, por_id[i].escala_sofa, por_id[i].pafi, destinos[i]["label"]]
                            for i in ids],
                hovertemplate="<b>%{text}</b><br>Sobrevida estimada: %{x:.0f} %<br>Urgencia: %{y:.1f}<br>"
                              "Edad %{customdata[0]} · SOFA %{customdata[1]} · PaFi %{customdata[2]}<br>"
                              "%{customdata[3]}<extra></extra>",
            ))
        fig.add_vline(x=50, line_dash="dot", line_color="#D5DDE6")
        fig.add_hline(y=50, line_dash="dot", line_color="#D5DDE6")
        fig.add_annotation(x=98, y=100, text="Urgente y reversible", showarrow=False, xanchor="right",
                           font=dict(size=10, color=HEX["muted"]))
        fig.add_annotation(x=2, y=100, text="Urgente, baja reversibilidad", showarrow=False, xanchor="left",
                           font=dict(size=10, color=HEX["muted"]))
        fig.update_xaxes(title="Sobrevida estimada por SOFA (%)", range=[0, 105])
        fig.update_yaxes(title="Urgencia fisiológica (0-100)", range=[0, 108])
        st.plotly_chart(estilo_figura(fig, "Urgencia vs. reversibilidad  (★ = recibe UCI)", 380),
                        width="stretch", config=PLOTLY_CONFIG)
    with g2:
        if resultado.orden_prioridad:
            orden = resultado.orden_prioridad
            figb = go.Figure(go.Bar(
                x=[resultado.puntajes[i] for i in orden],
                y=[f"#{k} {por_id[i].nombre}" for k, i in enumerate(orden, 1)],
                orientation="h", text=[f"{resultado.puntajes[i]:.1f}" for i in orden], textposition="auto",
                marker_color=[HEX["verde"] if resultado.decisiones[i].asignado_uci else HEX["ambar"] for i in orden],
            ))
            figb.update_yaxes(autorange="reversed")
            st.plotly_chart(estilo_figura(figb, "Orden del comité", 380), width="stretch", config=PLOTLY_CONFIG)
            st.caption(f"{resultado.etiqueta_puntaje}. Verde = recibe cama UCI · ámbar = en espera con plan CRUE.")
        else:
            st.info("No hay pacientes en Prioridad 1 compitiendo por recursos.", icon=":material/info:")

    with st.expander("Trazas globales del sistema (no asociadas a un paciente)"):
        for t in [t for t in resultado.todas_las_trazas() if t.paciente_id is None]:
            render_traza(t)

    st.markdown("**Exportar auditoría**")
    traza_json = json.dumps({
        "modo": resultado.modo, "estado_hospital": resultado.estado_hospital,
        "recursos": {"camas_disponibles": estado.camas_uci_disponibles,
                     "ventiladores_disponibles": estado.ventiladores_disponibles},
        "validacion_global": resultado.validacion_global,
        "decisiones": {pid: vars(d) for pid, d in resultado.decisiones.items()},
        "trazas": [t.to_dict() for t in resultado.todas_las_trazas()],
    }, ensure_ascii=False, indent=2)
    dl1, dl2, _ = st.columns([1, 1, 1])
    dl1.download_button("Auditoría completa (JSON)", traza_json, "auditoria_uci_xai.json", "application/json",
                        icon=":material/download:", width="stretch")
    dl2.download_button("Trazas de reglas (CSV)",
                        pd.DataFrame([t.to_dict() for t in resultado.todas_las_trazas()]).to_csv(index=False)
                        .encode("utf-8-sig"), "trazas_uci.csv", "text/csv",
                        icon=":material/download:", width="stretch")


# -----------------------------------------------------------------------------
# PESTAÑA · Nota clínica (NLG)
# -----------------------------------------------------------------------------
with tab_nota:
    st.subheader("Nota para la historia clínica", anchor=False)
    st.caption("Texto generado automáticamente a partir de las reglas del comité (Resolución 1995 de 1999, "
               "Ley 23 de 1981, Ley 1751 de 2015). El médico puede editarlo antes de descargarlo.")

    pid_nota = st.selectbox(
        "Paciente", [p.id for p in ordenados], key="sel_paciente_nlg",
        format_func=lambda pid: f"{por_id[pid].nombre} · {pid} · {resultado.evaluaciones[pid].etiqueta_prioridad}",
    )
    texto_nota = ClinicalNarrativeGenerator().generar_nota_historia_clinica(
        pid_nota, resultado, reporte_robustez=reportes_robustez.get(pid_nota) if reportes_robustez else None,
    )
    nota_editada = st.text_area(
        "Nota (editable)", value=texto_nota, height=420, key=f"txt_area_nlg_{pid_nota}",
        help="Documento médico-legal en 7 secciones. Puede complementarse antes de guardarlo.",
    )
    n1, n2, n3 = st.columns(3)
    n1.download_button("Descargar nota (.txt)", nota_editada.encode("utf-8"),
                       f"nota_oficial_historia_clinica_{pid_nota}.txt", "text/plain",
                       icon=":material/download:", width="stretch", type="primary")
    n2.download_button("Exportar expediente (.md)", nota_editada.encode("utf-8"),
                       f"expediente_comite_{pid_nota}.md", "text/markdown",
                       icon=":material/download:", width="stretch")
    with n3.popover("Copiar al portapapeles", icon=":material/content_copy:", width="stretch"):
        st.caption("Use el ícono de copiar en la esquina del recuadro.")
        st.code(nota_editada, language=None, wrap_lines=True, height=300)


# -----------------------------------------------------------------------------
# PESTAÑA · Marco legal
# -----------------------------------------------------------------------------
with tab_leg:
    st.subheader("Marco jurídico y responsabilidad", anchor=False)
    st.caption("Normas colombianas e internacionales que fundamentan las reglas del sistema.")

    l1, l2 = st.columns(2)
    with l1.container(border=True, height="stretch"):
        st.markdown("**:material/handshake: Supervisión humana (Ley 23 de 1981)**")
        st.markdown("El sistema es un **apoyo a la decisión clínica (CDSS)**. El acto médico es personal e "
                    "indelegable: el médico tratante y el Comité de Ética pueden apartarse motivadamente de la "
                    "recomendación, dejando constancia en la historia clínica (Resolución 1995 de 1999).")
    with l2.container(border=True, height="stretch"):
        st.markdown("**:material/public: Ética de la IA (UNESCO 2021 y Marco Colombia)**")
        st.markdown("Transparencia y explicabilidad (trazas sin cajas negras), equidad (bloqueo de edadismo con "
                    "prueba contrafáctica REG-BIO-01), supervisión humana y rendición de cuentas (auditoría "
                    "exportable).")
    l3, l4 = st.columns(2)
    with l3.container(border=True, height="stretch"):
        st.markdown("**:material/lock: Datos sensibles (Ley 1581 de 2012)**")
        st.markdown("Todo se procesa localmente, sin APIs externas. El nombre del paciente nunca interviene en "
                    "la evaluación clínica ni en los puntajes.")
    with l4.container(border=True, height="stretch"):
        st.markdown("**:material/warning: Alcance y limitaciones**")
        st.markdown("Prototipo académico. Las probabilidades de sobrevida por SOFA son aproximaciones pedagógicas "
                    "y no están calibradas para uso clínico real. No reemplaza el juicio médico.")

    st.write("")
    st.markdown("**Matriz de responsabilidad civil y profesional**")
    st.dataframe(pd.DataFrame([
        {"Actor": m["actor"], "Responsabilidad": m["responsabilidad"], "Régimen jurídico": m["tipo"],
         "No responde por": m["no_responde_por"]}
        for m in kb.MATRIZ_RESPONSABILIDAD
    ]), hide_index=True, width="stretch")

    with st.expander(":material/menu_book: Normas aplicadas por las reglas"):
        st.dataframe(pd.DataFrame([
            {"Norma": v["norma"], "Contenido": v["contenido"], "Aplicación en el sistema": v["aplicacion"]}
            for v in kb.NORMATIVA.values()
        ]), hide_index=True, width="stretch")

    with st.expander(":material/link: Trazabilidad norma ↔ reglas disparadas en esta sesión"):
        conteo: Dict[str, Dict[str, object]] = {}
        for t in resultado.todas_las_trazas():
            fila = conteo.setdefault(t.regla_id, {"Regla": t.regla_id, "Agente": t.agente, "Disparos": 0,
                                                  "Fundamento": t.fundamento_normativo_o_medico})
            fila["Disparos"] = int(fila["Disparos"]) + 1
        st.dataframe(pd.DataFrame(sorted(conteo.values(), key=lambda f: str(f["Regla"]))),
                     hide_index=True, width="stretch")
