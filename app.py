"""
app.py — Consola Hospitalaria de Decisión Clínico-Bioética (CDSS-XAI)
=============================================================================

Sistema Experto Multiagente Simbólico para el Triage en Cuidados Intensivos (UCI)
y Asignación Ética de Recursos Críticos.

Universidad Católica Luis Amigó · Curso de Análisis de Inteligencia Artificial.
"""

from __future__ import annotations

import html
import json
import textwrap
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import knowledge_base as kb
from core.coordinator import ConsensusCoordinator
from core.models import (
    ETIQUETAS_PRIORIDAD, Etapa, ModoEtico, Paciente, RecursoHospitalario, ReporteRobustez, ResultadoComite, TrazaInferencia,
)
from core.nlg_generator import ClinicalNarrativeGenerator

# =============================================================================
# Configuración de Página y Paleta Visual Hospitalaria
# =============================================================================
st.set_page_config(
    page_title="Hospital CDSS · Triage Ético UCI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta semántica médica suave (no saturada, accesible, profesional)
COLOR_P1 = "#dc2626"       # Rojo suave / prioridad inmediata
COLOR_P2 = "#d97706"       # Ámbar / monitoreo intensivo
COLOR_P3 = "#2563eb"       # Azul / hospitalización general
COLOR_P4 = "#7c3aed"       # Púrpura / cuidado paliativo
COLOR_OK = "#059669"       # Verde esmeralda clínico / validado o asignado
COLOR_WARN = "#d97706"     # Advertencia / observaciones
COLOR_DANGER = "#dc2626"   # Veto / objeción crítica
COLOR_NEUTRAL = "#475569"  # Gris pizarra neutro

PRIORIDAD_COLOR = {1: COLOR_P1, 2: COLOR_P2, 3: COLOR_P3, 4: COLOR_P4}
NIVEL_COLOR = {"INFO": "#0284c7", "OK": COLOR_OK, "ALERTA": COLOR_WARN, "OBJECION": "#ea580c", "CRITICO": COLOR_DANGER}
NIVEL_ICONO = {"INFO": "ℹ️", "OK": "✅", "ALERTA": "⚠️", "OBJECION": "🛑", "CRITICO": "🚨"}
VALIDACION_COLOR = {
    "VALIDADO": COLOR_OK,
    "VALIDADO CON OBSERVACIONES": COLOR_WARN,
    "OBJETADO": "#ea580c",
    "VETO CONSTITUCIONAL": COLOR_DANGER,
}
AGENTE_INFO = {
    "ClinicalTriageAgent": {
        "icono": "🩺",
        "titulo": "Medicina Crítica & Triage",
        "color": "#0284c7",
        "sub": "Evaluación fisiológica objetiva, gravedad SOFA y reversibilidad",
    },
    "ResourceManagerAgent": {
        "icono": "🏥",
        "titulo": "Gestión de Capacidad y Camas",
        "color": "#0f766e",
        "sub": "Monitoreo de inventario físico, camas UCI y soporte ventilatorio",
    },
    "BioethicsAuditAgent": {
        "icono": "⚖️",
        "titulo": "Auditoría Bioética y Legal",
        "color": "#6366f1",
        "sub": "Control constitucional, bloqueo de edadismo y garantía de no abandono",
    },
    "ConsensusCoordinator": {
        "icono": "🧭",
        "titulo": "Coordinador de Consenso",
        "color": "#0f172a",
        "sub": "Integración multidisciplinaria, desempate transparente y acta oficial",
    },
}

# Inyección de estilos CSS de nivel clínico profesional
st.markdown(
    """
    <style>
      /* Tipografía y contenedor general */
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
      
      html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      }
      
      .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
      }
      
      /* Encabezado Hospitalario Institucional */
      .hospital-navbar {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        border-left: 6px solid #0284c7;
      }
      .hospital-navbar .brand-dept {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #0284c7;
        margin-bottom: 2px;
      }
      .hospital-navbar h1 {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height: 1.25;
      }
      .hospital-navbar p {
        font-size: 0.86rem;
        color: #64748b;
        margin: 4px 0 0 0;
      }
      .hospital-navbar .status-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #166534;
      }

      /* Alertas de Estado Hospitalario */
      .status-alert {
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 16px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
      }
      .status-alert.critical {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 5px solid #dc2626;
        color: #991b1b;
      }
      .status-alert.normal {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        color: #166534;
      }
      .status-alert-title {
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 2px;
      }
      .status-alert-desc {
        font-size: 0.86rem;
        opacity: 0.92;
        line-height: 1.4;
      }

      /* Tarjeta de Resumen Ejecutivo de la Deliberación */
      .exec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
      }
      .exec-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 10px;
        margin-bottom: 12px;
      }
      .exec-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
      }
      
      /* Tarjetas de Paciente (Estilo Ficha Clínica de Urgencias) */
      .patient-board-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
      }
      .patient-board-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.06);
      }
      .patient-name-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
      }
      .patient-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0f172a;
      }
      .patient-meta {
        font-size: 0.82rem;
        color: #64748b;
      }
      
      /* Chips de signos vitales */
      .vital-chip {
        display: inline-flex;
        align-items: center;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.77rem;
        font-weight: 600;
        color: #334155;
        margin: 2px 4px 2px 0;
      }
      .vital-chip b {
        color: #0f172a;
        margin-left: 3px;
      }
      .vital-chip.alert {
        background: #fffbeb;
        border-color: #fde68a;
        color: #92400e;
      }
      .vital-chip.critical {
        background: #fef2f2;
        border-color: #fecaca;
        color: #991b1b;
      }

      /* Insignias (Pills) limpias */
      .pill {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
      }
      .pill-assigned {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
      }
      .pill-waiting {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
      }
      .pill-palliative {
        background: #faf5ff;
        color: #6b21a8;
        border: 1px solid #e9d5ff;
      }
      
      /* Tarjetas de Agentes Especialistas */
      .agent-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
      }
      .agent-panel-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 10px;
        margin-bottom: 12px;
        border-bottom: 1px solid #f1f5f9;
      }
      .agent-avatar {
        font-size: 1.5rem;
        background: #f1f5f9;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
      }
      .agent-role-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
      }
      .agent-role-sub {
        font-size: 0.77rem;
        color: #64748b;
        margin: 0;
      }
      .verdict-box {
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 10px 0;
        line-height: 1.35;
      }

      /* Trazas XAI limpias */
      .traza-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #64748b;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.84rem;
        line-height: 1.45;
      }
      .traza-item .head {
        font-weight: 700;
        font-size: 0.78rem;
        color: #475569;
        margin-bottom: 4px;
      }
      .traza-item .code-box {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 2px 6px;
        border-radius: 4px;
        color: #0f172a;
      }
      .traza-item .foundation {
        margin-top: 5px;
        font-size: 0.77rem;
        color: #64748b;
      }

      /* Tarjeta de Marco Filosófico */
      .framework-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
      }
      .framework-card.active {
        border: 2px solid #059669;
        background: #fbfdfc;
      }

      /* Ajustes para tablas */
      .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Gestión de Estado y Callbacks
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


def _siguiente_id() -> str:
    usados = {p.id for p in st.session_state.pacientes}
    n = 401
    while f"PAC-{n:03d}" in usados:
        n += 1
    return f"PAC-{n:03d}"


_init_estado()


# =============================================================================
# Funciones Auxiliares de Presentación Visual
# =============================================================================
def badge(texto: str, bg_color: str, text_color: str = "#ffffff") -> str:
    return (
        f'<span style="display:inline-block; padding:2px 8px; border-radius:999px; '
        f'background:{bg_color}; color:{text_color}; font-size:0.75rem; font-weight:700; '
        f'letter-spacing:0.02em; margin:2px 4px 2px 0;">{html.escape(texto)}</span>'
    )


def badge_prioridad(p: int) -> str:
    labels = {
        1: ("P1 · Crítico Inestable (UCI Inmediata)", "#fef2f2", "#991b1b", "#fecaca"),
        2: ("P2 · Monitoreo Intensivo / Intermedios", "#fffbeb", "#92400e", "#fde68a"),
        3: ("P3 · Hospitalización General", "#eff6ff", "#1e40af", "#bfdbfe"),
        4: ("P4 · Cuidado Paliativo Integral", "#faf5ff", "#6b21a8", "#e9d5ff"),
    }
    lbl, bg, fg, border = labels.get(p, (f"P{p}", "#f1f5f9", "#334155", "#cbd5e1"))
    return (
        f'<span style="display:inline-flex; align-items:center; padding:3px 10px; border-radius:999px; '
        f'background:{bg}; color:{fg}; border:1px solid {border}; font-size:0.75rem; font-weight:700;">'
        f'{html.escape(lbl)}</span>'
    )


def badge_validacion(v: str) -> str:
    estilos = {
        "VALIDADO": ("#ecfdf5", "#065f46", "#a7f3d0", "✅"),
        "VALIDADO CON OBSERVACIONES": ("#fffbeb", "#92400e", "#fde68a", "⚠️"),
        "OBJETADO": ("#fff7ed", "#9a3412", "#fed7aa", "🛑"),
        "VETO CONSTITUCIONAL": ("#fef2f2", "#991b1b", "#fecaca", "🚨"),
    }
    bg, fg, border, icon = estilos.get(v, ("#f8fafc", "#334155", "#cbd5e1", "ℹ️"))
    return (
        f'<span style="display:inline-flex; align-items:center; gap:4px; padding:3px 10px; border-radius:999px; '
        f'background:{bg}; color:{fg}; border:1px solid {border}; font-size:0.75rem; font-weight:700;">'
        f'{icon} {html.escape(v)}</span>'
    )


def destino_asistencial_badge(asignado_uci: bool, prioridad: int, ranking: int | None = None) -> str:
    if asignado_uci:
        return (
            '<span style="display:inline-flex; align-items:center; gap:5px; padding:4px 12px; '
            'border-radius:8px; background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0; '
            'font-weight:700; font-size:0.82rem;">🛏️ CAMA UCI ASIGNADA</span>'
        )
    if prioridad == 1:
        rank_txt = f" (Posición #{ranking})" if ranking else ""
        return (
            f'<span style="display:inline-flex; align-items:center; gap:5px; padding:4px 12px; '
            f'border-radius:8px; background:#fffbeb; color:#92400e; border:1px solid #fde68a; '
            f'font-weight:700; font-size:0.82rem;">🚑 EXPANSIÓN + TRASLADO CRUE{rank_txt}</span>'
        )
    if prioridad == 2:
        return (
            '<span style="display:inline-flex; align-items:center; gap:5px; padding:4px 12px; '
            'border-radius:8px; background:#eff6ff; color:#1e40af; border:1px solid #bfdbfe; '
            'font-weight:700; font-size:0.82rem;">🩺 CUIDADOS INTERMEDIOS</span>'
        )
    if prioridad == 3:
        return (
            '<span style="display:inline-flex; align-items:center; gap:5px; padding:4px 12px; '
            'border-radius:8px; background:#f8fafc; color:#334155; border:1px solid #cbd5e1; '
            'font-weight:700; font-size:0.82rem;">🏠 HOSPITALIZACIÓN GENERAL</span>'
        )
    return (
        '<span style="display:inline-flex; align-items:center; gap:5px; padding:4px 12px; '
        'border-radius:8px; background:#faf5ff; color:#6b21a8; border:1px solid #e9d5ff; '
        'font-weight:700; font-size:0.82rem;">🕊️ CUIDADO PALIATIVO INTEGRAL</span>'
    )


def render_traza_moderna(t: TrazaInferencia, mostrar_agente: bool = True) -> None:
    color = NIVEL_COLOR.get(t.nivel, "#64748b")
    info_agente = AGENTE_INFO.get(t.agente, {"icono": "🤖", "titulo": t.agente, "color": "#64748b"})
    icono = NIVEL_ICONO.get(t.nivel, "•")
    
    agente_header = f"{info_agente['icono']} {info_agente['titulo']} · " if mostrar_agente else ""
    st.markdown(
        f'<div class="traza-item" style="border-left-color:{color}">'
        f'<div class="head">{icono} {html.escape(agente_header)}<span class="code-box">{html.escape(t.regla_id)}</span> '
        f'· <b>{html.escape(t.etapa)}</b></div>'
        f'<div><b>Condición:</b> {html.escape(t.condicion_evaluada)}</div>'
        f'<div style="margin-top:3px;"><b>Conclusión:</b> {html.escape(t.conclusion)}</div>'
        f'<div class="foundation">📚 <i>Fundamento: {html.escape(t.fundamento_normativo_o_medico)}</i></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _dot_escape(texto: str, ancho: int = 38) -> str:
    texto = texto.replace("\\", "/").replace('"', "'")
    return "\\n".join(textwrap.wrap(texto, ancho)) or " "


def arbol_dot(r: ResultadoComite, pid: str) -> str:
    p = r.paciente(pid)
    cadena = r.cadena_xai(pid)
    lineas = [
        'digraph G {',
        'rankdir=TB; nodesep=0.25; ranksep=0.35;',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, fillcolor="#ffffff", color="#cbd5e1"];',
        'edge [color="#94a3b8", arrowsize=0.7];',
        f'root [label="{_dot_escape(p.nombre + " · " + p.id, 26)}", shape=ellipse, fillcolor="#0284c7", '
        f'fontcolor="white", fontsize=11, style="filled"];',
    ]
    anterior = "root"
    for i, etapa in enumerate(e.value for e in Etapa):
        trazas = [t for t in cadena if t.etapa == etapa]
        if not trazas:
            continue
        hub = f"etapa{i}"
        lineas.append(f'{hub} [label="{etapa}", shape=hexagon, fillcolor="#f1f5f9", fontsize=10, color="#94a3b8"];')
        lineas.append(f"{anterior} -> {hub} [penwidth=1.8, color=\"#0284c7\"];")
        for j, t in enumerate(trazas):
            nodo = f"t{i}_{j}"
            color = NIVEL_COLOR.get(t.nivel, "#64748b")
            concl = t.conclusion if len(t.conclusion) < 95 else t.conclusion[:92] + "..."
            lineas.append(
                f'{nodo} [label="{_dot_escape(t.regla_id)}\\n{_dot_escape(concl)}", color="{color}", '
                f'penwidth=1.5, fillcolor="#f8fafc"];'
            )
            lineas.append(f"{hub} -> {nodo};")
        anterior = hub
    lineas.append("}")
    return "\n".join(lineas)


# =============================================================================
# Panel Lateral (Sidebar de Control Asistencial)
# =============================================================================
with st.sidebar:
    st.markdown(
        '<div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">'
        '<span style="font-size:1.6rem;">🏥</span>'
        '<div><h3 style="margin:0; font-size:1.15rem; color:#0f172a;">Control Asistencial</h3>'
        '<span style="font-size:0.75rem; color:#64748b;">Consola de Triage Crítico</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 1. Escenarios clínicos
    st.markdown("#### ⚡ Casos de Demostración")
    st.button(
        "🚨 Cargar Caso de Dilema Crítico (3 vs 1)",
        type="primary",
        use_container_width=True,
        on_click=cb_cargar_escenario,
        args=("dilema",),
        help="Caso principal: Carlos (joven), Elena (adulta mayor vulnerable) y Jorge (terminal) compiten por 1 cama UCI.",
    )

    with st.expander("📂 Otros escenarios clínicos precargados", expanded=False):
        otro = st.selectbox(
            "Seleccionar escenario",
            [k for k in kb.ESCENARIOS if k != "dilema"],
            format_func=lambda k: kb.ESCENARIOS[k]["titulo"],
            label_visibility="collapsed",
        )
        st.button("Cargar escenario seleccionado", use_container_width=True, on_click=cb_cargar_escenario, args=(otro,))

    st.divider()

    # 2. Recursos físicos
    st.markdown("#### 🛏️ Disponibilidad de Recursos")
    st.slider("Camas UCI disponibles", 0, 5, key="camas", help="Número de camas físicas de cuidados intensivos libres en este turno.")
    st.slider("Ventiladores mecánicos libres", 0, 5, key="ventiladores", help="Ventiladores invasivos disponibles inmediatamente.")
    with st.expander("Configuración de capacidad instalada", expanded=False):
        st.slider("Camas UCI totales de la institución", 5, 40, key="camas_totales")

    st.divider()

    # 3. Marco ético
    st.markdown("#### ⚖️ Marco Ético de Decisión")
    modo_sel = st.radio(
        "Marco filosófico activo",
        [m.value for m in ModoEtico],
        key="modo",
        format_func=lambda v: kb.MARCOS_ETICOS[v]["nombre"],
        label_visibility="collapsed",
    )
    
    # Breve tarjeta descriptiva del marco activo en sidebar
    info_marco_sb = kb.MARCOS_ETICOS[modo_sel]
    st.caption(f"**Principio:** {info_marco_sb['principio']}")

    with st.expander("⚙️ Parámetros avanzados de consenso", expanded=False):
        st.slider(
            "Umbral de empate técnico SOFA (±puntos)",
            0, 3, key="umbral",
            help="Diferencias de SOFA menores o iguales a este valor se consideran clínicamente indistinguibles. "
                 "Permite activar criterios afirmativos de justicia constitucional.",
        )
        st.caption(f"Semilla pública de sorteo auditable: `{kb.SEMILLA_SORTEO_AUDITABLE}`")

    st.divider()

    # 4. Modo Simulación Dinámica Temporal
    st.markdown("#### ⏱️ Simulación Dinámica Temporal")
    simulacion_activa = st.toggle(
        "Activar Simulación Multi-Ronda (0h → 24h)",
        key="modo_simulacion_temporal",
        help="Simula la evolución fisiológica a 6h, 12h y 24h, con liberación dinámica de camas UCI y reasignación auditada al paciente en espera.",
    )
    if simulacion_activa:
        st.select_slider(
            "Paso temporal de evaluación",
            options=[0, 6, 12, 24],
            format_func=lambda h: f"{h} horas" + (" (Ingreso)" if h == 0 else ""),
            key="hora_simulacion",
            help="Permite inspeccionar el estado del hospital y del comité en cada corte temporal.",
        )

    st.divider()

    # 5. Ingreso de pacientes en vivo
    st.markdown("#### ➕ Ingreso de Paciente en Vivo")
    with st.form("form_paciente", clear_on_submit=True):
        nombre = st.text_input("Nombre completo", placeholder="Ej. Mariana Duque")
        col_f1, col_f2 = st.columns(2)
        edad = col_f1.number_input("Edad", 0, 120, 48)
        espera = col_f2.number_input("Llegó hace (min)", 0, kb.MINUTO_ACTUAL_TURNO, 15)
        
        sat = col_f1.number_input("SatO2 (%)", 50.0, 100.0, 86.0, step=0.5)
        pafi = col_f2.number_input("PaFi", 40, 600, 160)
        
        sofa = col_f1.number_input("SOFA (0-24)", 0, 24, 6)
        glasgow = col_f2.number_input("Glasgow (3-15)", 3, 15, 14)
        
        comorb = st.text_input("Comorbilidades (separadas por coma)", placeholder="Ej. Hipertensión, Asma")
        vent = st.checkbox("Requiere soporte ventilatorio invasivo", value=True)
        terminal = st.checkbox("Enfermedad terminal irreversible")
        vulnerable = st.checkbox("Sujeto de especial protección constitucional")

        with st.expander("🩺 Signos Vitales para Tamizaje NEWS2 (RCP 2017)", expanded=False):
            col_n1, col_n2 = st.columns(2)
            fr = col_n1.number_input("Frecuencia Respiratoria (rpm)", 0, 80, 16)
            pas = col_n2.number_input("Presión Sistólica (mmHg)", 30, 300, 120)
            fc = col_n1.number_input("Frecuencia Cardíaca (lpm)", 20, 250, 75)
            temp = col_n2.number_input("Temperatura (°C)", 25.0, 45.0, 36.8, step=0.1)
            o2_sup = st.checkbox("Oxígeno suplementario al ingreso")
            acvpu = st.selectbox(
                "Nivel de Conciencia (ACVPU)",
                ["A", "C", "V", "P", "U"],
                format_func=lambda x: {
                    "A": "A - Alerta",
                    "C": "C - Confusión nueva",
                    "V": "V - Responde a la voz",
                    "P": "P - Responde al dolor",
                    "U": "U - Inconsciente",
                }[x],
            )

        if st.form_submit_button("Ingresar a Urgencias", use_container_width=True):
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
                st.session_state.pacientes = st.session_state.pacientes + [nuevo]
                st.session_state.aviso = f"Paciente {nuevo.id} ({nuevo.nombre}) admitido a urgencias."
            except ValueError as exc:
                st.error(str(exc))

    if st.session_state.pacientes:
        with st.expander("🗑️ Gestión y Retiro de Pacientes", expanded=False):
            st.multiselect(
                "Seleccionar pacientes a dar de alta/retirar",
                [p.id for p in st.session_state.pacientes],
                key="retirar_ids",
                format_func=lambda i: next(f"{p.id} · {p.nombre}" for p in st.session_state.pacientes if p.id == i),
            )
            st.button("Retirar seleccionados", on_click=cb_retirar, use_container_width=True)
            st.button("Vaciar sala de urgencias", on_click=cb_vaciar, use_container_width=True)


# =============================================================================
# Encabezado Hospitalario Principal
# =============================================================================
st.markdown(
    """
    <div class="hospital-navbar">
      <div>
        <div class="brand-dept">🏥 Hospital Universitario · Unidad de Cuidados Intensivos</div>
        <h1>Sistema de Soporte a Decisiones Clínico-Bioéticas (CDSS-XAI)</h1>
        <p>Comité Autónomo Multiagente Simbólico para Triage UCI y Asignación Justa de Recursos Críticos · Universidad Católica Luis Amigó</p>
      </div>
      <div style="text-align:right;">
        <span class="status-tag">🟢 SISTEMA OPERATIVO | AUDITORÍA XAI EN VIVO</span>
        <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Supervisión Human-in-the-Loop · Ley 23 de 1981</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.aviso:
    st.toast(st.session_state.aviso)
    st.session_state.aviso = ""

pacientes: List[Paciente] = st.session_state.pacientes
if not pacientes:
    st.info("ℹ️ La sala de urgencias está vacía. Seleccione un escenario clínico en el panel lateral o ingrese pacientes manualmente.")
    st.stop()

# =============================================================================
# Ejecución del Comité Multiagente y Cálculo de Consenso
# =============================================================================
estado = RecursoHospitalario(
    camas_uci_totales=max(st.session_state.camas_totales, st.session_state.camas),
    camas_uci_disponibles=st.session_state.camas,
    ventiladores_disponibles=st.session_state.ventiladores,
    modo_etico_activo=st.session_state.modo,
)

coordinador = ConsensusCoordinator(umbral_empate_sofa=st.session_state.umbral)
reportes_robustez = coordinador.ejecutar_analisis_monte_carlo(pacientes, estado, n_iteraciones=1000, semilla=42)

sim_activa = st.session_state.get("modo_simulacion_temporal", False)
rondas_sim = None
eventos_ronda_actual = []

if sim_activa:
    evos_dilema = kb.escenario_evolucion_temporal_dilema()
    rondas_sim = coordinador.simular_linea_tiempo(pacientes, estado, evos_dilema)
    hora_sel = st.session_state.get("hora_simulacion", 0)
    ronda_actual = next((r for r in rondas_sim if r.hora == hora_sel), rondas_sim[0])
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

# =============================================================================
# Banner de Situación Asistencial y Tarjetas KPI
# =============================================================================
n_p1 = sum(1 for e in resultado.evaluaciones.values() if e.prioridad == 1)
deficit = d_rec.datos.get("deficit_camas", 0)
critico = resultado.estado_hospital == "ESCASEZ_CRITICA"

if critico:
    st.markdown(
        f"""
        <div class="status-alert critical">
          <span style="font-size:1.5rem; line-height:1;">🚨</span>
          <div>
            <div class="status-alert-title">ESCASEZ CRÍTICA DECLARADA (Triaje de Catástrofe / REG-REC-02)</div>
            <div class="status-alert-desc">
              La demanda de pacientes en Prioridad 1 ({n_p1} pacientes) supera la oferta inmediata de camas UCI ({estado.camas_uci_disponibles} disponibles). 
              El comité activa el protocolo de desempate bajo el <b>{marco['nombre']}</b>, garantizando activación del Centro Regulador de Urgencias (CRUE) 
              y protección de no abandono (Ley 1751 de 2015).
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="status-alert normal">
          <span style="font-size:1.5rem; line-height:1;">✅</span>
          <div>
            <div class="status-alert-title">CAPACIDAD ASISTENCIAL NORMAL (Operación Regular / REG-REC-01)</div>
            <div class="status-alert-desc">
              La institución dispone de {estado.camas_uci_disponibles} cama(s) libres y {estado.ventiladores_disponibles} ventilador(es). 
              Toda la demanda crítica en Prioridad 1 ({n_p1} paciente(s)) puede ser absorbida directamente sin conflicto ético distributivo.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# KPIs Médicos Principales
k1, k2, k3, k4 = st.columns(4)
k1.metric("Pacientes en Urgencias", len(pacientes), delta=f"Turno min: {kb.MINUTO_ACTUAL_TURNO}", delta_color="off")
k2.metric("Prioridad 1 (Críticos)", n_p1, delta=f"{d_rec.datos.get('demanda_p1_ventilador', 0)} requieren ventilador", delta_color="off")
k3.metric("Camas UCI Libres", f"{estado.camas_uci_disponibles} / {estado.camas_uci_totales}",
          delta=f"-{deficit} cama(s) de déficit" if deficit else "Oferta suficiente",
          delta_color="inverse" if deficit else "normal")
k4.metric("Validación Bioética", resultado.validacion_global, delta=marco["nombre"].split()[0], delta_color="off")


# =============================================================================
# Resumen Ejecutivo de la Deliberación del Comité (Inmediatamente Visible)
# =============================================================================
with st.container():
    asignados_nombres = [por_id[pid].nombre for pid in resultado.asignados_uci]
    espera_crue_nombres = [por_id[pid].nombre for pid in resultado.decisiones if resultado.decisiones[pid].prioridad == 1 and not resultado.decisiones[pid].asignado_uci]
    paliativos_nombres = [por_id[pid].nombre for pid in resultado.decisiones if resultado.decisiones[pid].prioridad == 4]
    
    st.markdown(
        f"""
        <div class="exec-card">
          <div class="exec-header">
            <div>
              <span style="font-size:0.75rem; font-weight:700; color:#0284c7; text-transform:uppercase; letter-spacing:0.05em;">Resolución Oficial de la Sesión</span>
              <div class="exec-title">📋 Dictamen Consolidado del Comité Clínico-Bioético</div>
            </div>
            <div>
              {badge_validacion(resultado.validacion_global)}
            </div>
          </div>
          <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:16px;">
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 14px;">
              <div style="font-size:0.78rem; font-weight:700; color:#065f46; text-transform:uppercase; margin-bottom:4px;">🛏️ Asignación Inmediata de Cama UCI</div>
              <div style="font-size:1.02rem; font-weight:800; color:#0f172a;">{', '.join(asignados_nombres) or 'Ningún paciente asignado'}</div>
              <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">Recursos físicos de UCI y soporte ventilatorio asegurados en la institución.</div>
            </div>
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 14px;">
              <div style="font-size:0.78rem; font-weight:700; color:#92400e; text-transform:uppercase; margin-bottom:4px;">🚑 Red Externa (Protocolo CRUE / Ley 1751)</div>
              <div style="font-size:1.02rem; font-weight:800; color:#0f172a;">{', '.join(espera_crue_nombres) or 'Sin remisiones pendientes'}</div>
              <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">Soporte vital transitorio en sala de expansión y solicitud urgente de traslado.</div>
            </div>
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 14px;">
              <div style="font-size:0.78rem; font-weight:700; color:#6b21a8; text-transform:uppercase; margin-bottom:4px;">🕊️ Adecuación del Esfuerzo (Ley 1733)</div>
              <div style="font-size:1.02rem; font-weight:800; color:#0f172a;">{', '.join(paliativos_nombres) or 'Sin pacientes en futilidad'}</div>
              <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">Cuidado paliativo integral sin encarnizamiento terapéutico (cero abandono).</div>
            </div>
          </div>
          <div style="margin-top:12px; padding-top:10px; border-top:1px solid #f1f5f9; display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; color:#64748b;">
            <div><b>Marco de Resolución:</b> {marco['nombre']} ({marco['escuela']})</div>
            <div><b>Principio Rector:</b> {marco['principio']}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Navegación por Pestañas (Estructura Limpia y Modular)
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Tablero de Pacientes y Asignación",
    "👥 Deliberación de Especialistas (Multiagente)",
    "⚖️ Dilemas Éticos Comparados",
    "🔍 Explicabilidad Fisiológica y XAI",
    "📜 Marco Normativo y Responsabilidad",
])


# =============================================================================
# PESTAÑA 1 · Tablero de Pacientes y Asignación Asistencial
# =============================================================================
with tab1:
    st.markdown("### 🛏️ Censo Clínico de Triage en Urgencias")
    st.caption("Fichas clínicas de evaluación individualizada, estratificación por gravedad y asignación de destino asistencial.")

    if sim_activa and rondas_sim:
        hora_actual = st.session_state.get("hora_simulacion", 0)
        st.markdown(
            f"""
            <div style="background: linear-gradient(90deg, #0f172a, #1e3a8a); color: white; padding: 14px 20px; border-radius: 12px; margin-bottom: 16px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <span style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#38bdf8;">
                    ⏱️ Modo Simulación Dinámica Temporal Multi-Ronda (NEWS2 + Rotación UCI)
                  </span>
                  <h3 style="margin:2px 0 0 0; color:white; font-size:1.25rem;">
                    Corte Clínico a las {hora_actual} Horas {'(Ingreso Inicial Urgencias)' if hora_actual == 0 else ''}
                  </h3>
                </div>
                <div style="text-align:right;">
                  <span style="background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25); padding:4px 12px; border-radius:999px; font-size:0.8rem; font-weight:700;">
                    4 Cortes: 0h | 6h | 12h | 24h
                  </span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Bitácora de eventos clínicos y de rotación
        if eventos_ronda_actual:
            with st.container():
                st.markdown(f"**📜 Bitácora de Eventos Clínicos y Rotación de Camas (t = {hora_actual}h):**")
                cols_ev = st.columns(len(eventos_ronda_actual) if len(eventos_ronda_actual) <= 2 else 2)
                for i_ev, ev in enumerate(eventos_ronda_actual):
                    with cols_ev[i_ev % len(cols_ev)]:
                        if "LIBERACIÓN" in ev or "REASIGNACIÓN" in ev:
                            st.success(ev)
                        else:
                            st.info(ev)

        # Gráfico Plotly de flujo temporal
        horas_label = ["0h (Ingreso)", "6h", "12h", "24h"]
        uci_ocupadas = [len([pid for pid, d in r.resultado_comite.decisiones.items() if d.asignado_uci]) for r in rondas_sim]
        camas_libres_vals = [r.recursos.camas_uci_disponibles for r in rondas_sim]
        lista_espera_vals = [len([pid for pid, d in r.resultado_comite.decisiones.items() if d.prioridad == 1 and not d.asignado_uci]) for r in rondas_sim]
        intermedios_vals = [len([pid for pid, d in r.resultado_comite.decisiones.items() if d.destino == "CUIDADOS INTERMEDIOS / MONITOREO CONTINUO"]) for r in rondas_sim]

        fig_flujo = go.Figure()
        fig_flujo.add_trace(go.Bar(
            name="🛏️ Pacientes en UCI",
            x=horas_label, y=uci_ocupadas,
            marker_color="#059669",
            text=uci_ocupadas, textposition="auto",
        ))
        fig_flujo.add_trace(go.Bar(
            name="🚑 Lista de Espera CRUE (P1 sin cama)",
            x=horas_label, y=lista_espera_vals,
            marker_color="#dc2626",
            text=lista_espera_vals, textposition="auto",
        ))
        fig_flujo.add_trace(go.Bar(
            name="🩺 Cuidados Intermedios (Liberaron UCI)",
            x=horas_label, y=intermedios_vals,
            marker_color="#2563eb",
            text=intermedios_vals, textposition="auto",
        ))
        fig_flujo.add_trace(go.Scatter(
            name="🟢 Camas UCI Disponibles",
            x=horas_label, y=camas_libres_vals,
            mode="lines+markers",
            line=dict(color="#0284c7", width=3, dash="dash"),
            marker=dict(size=10, symbol="diamond"),
        ))

        hora_map = {0: "0h (Ingreso)", 6: "6h", 12: "12h", 24: "24h"}
        fig_flujo.add_vline(
            x=hora_map[hora_actual],
            line_width=3, line_dash="dot", line_color="#f59e0b",
            annotation_text=f"Paso actual: {hora_map[hora_actual]}",
            annotation_position="top left",
        )

        fig_flujo.update_layout(
            title="📊 Dinámica Temporal del Flujo Hospitalario: Rotación de Camas UCI y Reducción de Lista de Espera",
            barmode="group",
            xaxis_title="Corte Temporal Clínico",
            yaxis_title="Pacientes / Recursos",
            height=340,
            margin=dict(t=40, b=10, l=40, r=40),
            legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center"),
            template="plotly_white",
        )
        st.plotly_chart(fig_flujo, use_container_width=True)
        st.divider()

    # Renderizado en tarjetas clínicas individuales
    cols_pacientes = st.columns(len(pacientes) if len(pacientes) <= 3 else 3)
    for idx, p in enumerate(pacientes):
        col = cols_pacientes[idx % len(cols_pacientes)]
        d = resultado.decisiones[p.id]
        e = resultado.evaluaciones[p.id]
        
        # Clasificación SDRA Berlín
        sdra_info = kb.clasificar_sdra_berlin(p.pafi)

        # Score NEWS2
        news2_sc, news2_niv, news2_resp = kb.calcular_news2(p)
        news2_bg = "#ecfdf5" if news2_niv == "BAJO" else ("#fffbeb" if news2_niv == "MEDIO" else "#fef2f2")
        news2_fg = "#065f46" if news2_niv == "BAJO" else ("#92400e" if news2_niv == "MEDIO" else "#991b1b")
        news2_border = "#a7f3d0" if news2_niv == "BAJO" else ("#fde68a" if news2_niv == "MEDIO" else "#fecaca")
        
        with col:
            comorb_str = ", ".join(p.comorbilidades_graves) if p.comorbilidades_graves else "Sin antecedentes relevantes"
            border_dest = "#059669" if d.asignado_uci else ("#d97706" if d.prioridad == 1 else "#cbd5e1")
            
            st.markdown(
                f"""
                <div class="patient-board-card" style="border-top: 5px solid {border_dest};">
                  <div class="patient-name-row">
                    <div>
                      <div class="patient-title">{html.escape(p.nombre)}</div>
                      <div class="patient-meta"><b>{p.id}</b> · {p.edad} años · Llegó hace {p.minutos_espera(kb.MINUTO_ACTUAL_TURNO)} min</div>
                    </div>
                    <div>
                      {badge_prioridad(e.prioridad)}
                    </div>
                  </div>
                  
                  <div style="margin: 10px 0;">
                    <span class="vital-chip {'critical' if p.escala_sofa >= 12 else ('alert' if p.escala_sofa >= 7 else '')}">
                      SOFA: <b>{p.escala_sofa}</b> (Rev. {e.reversibilidad})
                    </span>
                    <span class="vital-chip {'critical' if p.pafi <= 100 else ('alert' if p.pafi <= 200 else '')}">
                      PaFi: <b>{p.pafi}</b>
                    </span>
                    <span class="vital-chip {'critical' if p.sat_o2 < 85 else ('alert' if p.sat_o2 < 92 else '')}">
                      SatO₂: <b>{p.sat_o2}%</b>
                    </span>
                    <span class="vital-chip">
                      Glasgow: <b>{p.glasgow}/15</b>
                    </span>
                    <span class="vital-chip">
                      Ventilador: <b>{'Requerido' if p.requiere_ventilador else 'No'}</b>
                    </span>
                  </div>

                  <!-- Tarjeta NEWS2 (RCP 2017) -->
                  <div style="background:{news2_bg}; border:1px solid {news2_border}; border-radius:8px; padding:8px 10px; margin:8px 0; font-size:0.8rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                      <span style="font-weight:800; color:{news2_fg};">🩺 Score NEWS2: {news2_sc} pts</span>
                      <span style="font-weight:700; color:{news2_fg}; font-size:0.72rem; text-transform:uppercase; background:rgba(255,255,255,0.7); padding:2px 8px; border-radius:999px;">Riesgo {news2_niv}</span>
                    </div>
                    <div style="color:#475569; font-size:0.75rem; margin-top:4px;">
                      FR: <b>{p.frecuencia_respiratoria} rpm</b> · PAS: <b>{p.presion_sistolica} mmHg</b> · FC: <b>{p.frecuencia_cardiaca} lpm</b> · Temp: <b>{p.temperatura:.1f} °C</b> · O₂: <b>{'Sí' if p.oxigeno_suplementario else 'No'}</b> · ACVPU: <b>{p.escala_acvpu}</b>
                    </div>
                    <div style="color:{news2_fg}; font-size:0.73rem; margin-top:3px; font-weight:600;">
                      👉 {news2_resp}
                    </div>
                  </div>

                  <div style="font-size:0.8rem; color:#475569; margin-bottom:8px;">
                    <div><b>Estado Respiratorio:</b> {sdra_info}</div>
                    <div><b>Comorbilidades:</b> {html.escape(comorb_str)}</div>
                    {f"<div style='color:#0369a1; font-weight:600;'>⭐ Sujeto de especial protección constitucional</div>" if p.vulnerabilidad_social else ""}
                    {f"<div style='color:#7c3aed; font-weight:600;'>⚠️ Enfermedad terminal irreversible declarada</div>" if p.enfermedad_terminal_irreversible else ""}
                  </div>

                  <div style="border-top:1px solid #f1f5f9; padding-top:10px; margin-top:8px;">
                    <div style="font-size:0.75rem; text-transform:uppercase; font-weight:700; color:#64748b; margin-bottom:4px;">Destino Asistencial Asignado:</div>
                    <div>{destino_asistencial_badge(d.asignado_uci, d.prioridad, d.posicion_ranking)}</div>
                    <div style="font-size:0.8rem; color:#475569; margin-top:6px;">
                      <b>Plan Clínico:</b> {html.escape(d.plan_alternativo or d.justificacion)}
                    </div>
                    <div style="margin-top:6px;">
                      {badge_validacion(d.validacion_etica)}
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Vista tabular complementaria expandible para auditoría
    with st.expander("📊 Ver matriz clínica consolidada (Base de Hechos)", expanded=False):
        df_hechos = pd.DataFrame([
            {
                "ID": p.id,
                "Nombre": p.nombre,
                "Edad": p.edad,
                "Espera (min)": p.minutos_espera(kb.MINUTO_ACTUAL_TURNO),
                "SatO₂ (%)": p.sat_o2,
                "PaFi": p.pafi,
                "SOFA": p.escala_sofa,
                "Glasgow": p.glasgow,
                "Ventilador": "Sí" if p.requiere_ventilador else "No",
                "Comorbilidades": ", ".join(p.comorbilidades_graves) or "Ninguna",
                "Terminal": "Sí" if p.enfermedad_terminal_irreversible else "No",
                "Especial Protección": "Sí" if p.vulnerabilidad_social else "No",
                "Prioridad Asignada": ETIQUETAS_PRIORIDAD[resultado.evaluaciones[p.id].prioridad],
                "Destino": resultado.decisiones[p.id].destino,
            }
            for p in pacientes
        ])
        st.dataframe(df_hechos, hide_index=True, use_container_width=True)


# =============================================================================
# PESTAÑA 2 · Deliberación de Especialistas (Multiagente)
# =============================================================================
with tab2:
    st.markdown("### 👥 Deliberación del Comité Médico-Bioético")
    st.caption("Cada agente del sistema opera con un dominio funcional autónomo e inspectable, aportando su dictamen a la deliberación.")

    col_a1, col_a2, col_a3 = st.columns(3)
    agentes_cols = [
        (col_a1, "ClinicalTriageAgent"),
        (col_a2, "ResourceManagerAgent"),
        (col_a3, "BioethicsAuditAgent"),
    ]

    for col, nombre_agente in agentes_cols:
        d_agente = resultado.dictamenes[nombre_agente]
        info_ui = AGENTE_INFO[nombre_agente]
        color_agente = info_ui["color"]
        
        with col:
            # Cuerpo de la tarjeta según el agente
            resumen_html = ""
            if nombre_agente == "ClinicalTriageAgent":
                for pid, e in resultado.evaluaciones.items():
                    resumen_html += (
                        f'<div style="margin-bottom:6px; font-size:0.83rem;">'
                        f'<b>{html.escape(por_id[pid].nombre)}:</b> {badge_prioridad(e.prioridad)} '
                        f'· Rev. {e.reversibilidad} · Sobrevida {e.prob_sobrevida:.0%}'
                        f'</div>'
                    )
            elif nombre_agente == "ResourceManagerAgent":
                for pid in resultado.orden_prioridad:
                    ok = pid in d_agente.datos.get("asignados", [])
                    pill_badge = (
                        '<span class="pill pill-assigned">ASIGNADO</span>'
                        if ok else '<span class="pill pill-waiting">EN COLA CRUE</span>'
                    )
                    resumen_html += (
                        f'<div style="margin-bottom:6px; font-size:0.83rem;">'
                        f'{pill_badge} <b>{html.escape(por_id[pid].nombre)}</b>'
                        f'</div>'
                    )
                if not resultado.orden_prioridad:
                    resumen_html = '<div style="font-size:0.83rem; color:#64748b;">No hay pacientes compitiendo por UCI.</div>'
            else:  # BioethicsAuditAgent
                for pid, v in d_agente.datos.get("validacion_por_paciente", {}).items():
                    resumen_html += (
                        f'<div style="margin-bottom:6px; font-size:0.83rem;">'
                        f'{badge_validacion(v)} <b>{html.escape(por_id[pid].nombre)}</b>'
                        f'</div>'
                    )

            st.markdown(
                f"""
                <div class="agent-panel" style="border-top: 5px solid {color_agente};">
                  <div class="agent-panel-header">
                    <div class="agent-avatar">{info_ui['icono']}</div>
                    <div>
                      <div class="agent-role-title">{info_ui['titulo']}</div>
                      <div class="agent-role-sub">{info_ui['sub']}</div>
                    </div>
                  </div>
                  <div class="verdict-box" style="background:{color_agente}15; color:{color_agente}; border:1px solid {color_agente}30;">
                    {html.escape(d_agente.veredicto)}
                  </div>
                  <div style="margin-top:10px;">
                    {resumen_html}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            with st.expander(f"🔍 Ver reglas disparadas ({nombre_agente})", expanded=False):
                for t in d_agente.trazas:
                    render_traza_moderna(t, mostrar_agente=False)

    st.markdown("---")
    
    # Análisis de Sensibilidad Estocástica (Monte Carlo)
    st.markdown("#### 🎲 Análisis de Estabilidad y Casos Borde (Simulación de Monte Carlo)")
    st.caption(
        "Evaluación de sensibilidad estocástica (1000 iteraciones con semilla reproducible). "
        "Mide la estabilidad de las recomendaciones ante error instrumental en oximetría (sigma = 1.5%), "
        "variabilidad de gasometría arterial PaFi (sigma = 10 mmHg) y discrepancia interobservador SOFA (±1 punto con p=0.15)."
    )

    if reportes_robustez:
        cols_mc = st.columns(len(pacientes) if len(pacientes) <= 3 else 3)
        for idx_mc, p_mc in enumerate(pacientes):
            col_mc = cols_mc[idx_mc % len(cols_mc)]
            rep_mc = reportes_robustez.get(p_mc.id)
            if not rep_mc:
                continue
            with col_mc:
                pct_mc = rep_mc.porcentaje_estabilidad_decision
                color_mc = "#16a34a" if pct_mc >= 90.0 else ("#f59e0b" if pct_mc >= 85.0 else "#dc2626")
                badge_diag_mc = (
                    f'<span style="display:inline-block; padding:3px 10px; border-radius:999px; '
                    f'background:{"#ecfdf5" if pct_mc >= 90.0 else "#fef2f2"}; '
                    f'color:{"#065f46" if pct_mc >= 90.0 else "#991b1b"}; '
                    f'border:1px solid {"#a7f3d0" if pct_mc >= 90.0 else "#fecaca"}; '
                    f'font-size:0.75rem; font-weight:700;">'
                    f'{"ROBUSTA / SEGURA" if pct_mc >= 90.0 else "ZONA FRONTERA / CASO BORDE"}</span>'
                )

                fig_gauge_mc = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pct_mc,
                    number={"suffix": "%", "font": {"size": 26, "color": color_mc}},
                    title={"text": f"<b>{html.escape(p_mc.nombre)}</b><br><span style='font-size:10px;color:#64748b;'>{p_mc.id}</span>", "font": {"size": 13}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8", "tickfont": {"size": 9}},
                        "bar": {"color": color_mc, "thickness": 0.28},
                        "bgcolor": "#f8fafc",
                        "borderwidth": 1,
                        "bordercolor": "#e2e8f0",
                        "steps": [
                            {"range": [0, 85], "color": "#fee2e2"},
                            {"range": [85, 90], "color": "#fef3c7"},
                            {"range": [90, 100], "color": "#dcfce7"},
                        ],
                        "threshold": {
                            "line": {"color": "#dc2626", "width": 2},
                            "thickness": 0.75,
                            "value": 85,
                        },
                    },
                ))
                fig_gauge_mc.update_layout(height=175, margin=dict(t=35, b=5, l=15, r=15))

                st.markdown(
                    f"""
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:12px; margin-bottom:12px;">
                      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-size:0.8rem; font-weight:700; color:#64748b;">Estabilidad Decisión</span>
                        {badge_diag_mc}
                      </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig_gauge_mc, use_container_width=True, key=f"gauge_mc_{p_mc.id}")
                st.progress(pct_mc / 100.0)
                st.markdown(
                    f"""
                      <div style="font-size:0.77rem; color:#334155; margin-top:8px; line-height:1.4;">
                        <b>Directriz Clínica XAI:</b> {html.escape(rep_mc.advertencia_clinica)}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    
    # Acta oficial del Coordinador de Consenso
    st.markdown("#### 🧭 Acta Formal de Consenso del Comité")
    c_acta1, c_acta2 = st.columns([3, 2])
    with c_acta1:
        with st.container(border=True):
            st.markdown(badge_validacion(resultado.validacion_global), unsafe_allow_html=True)
            st.markdown(resultado.acta_consenso)
    with c_acta2:
        with st.container(border=True):
            st.markdown(f"**Marco Filosófico:** {marco['nombre']}")
            st.markdown(f"**Escuela:** *{marco['escuela']}*")
            st.markdown(f"**Criterio Operativo:** {marco['criterio']}")
            st.markdown(f"**Tensión Ética Identificada:** {marco['critica']}")

    # Transcripción estructurada de la sesión clínica
    with st.expander("💬 Transcripción de la Sesión Deliberativa del Comité", expanded=False):
        with st.chat_message("ClinicalTriageAgent", avatar="🩺"):
            st.markdown(f"**Especialista en Medicina Crítica (ClinicalTriageAgent):**\n\n{d_clin.veredicto}")
            for pid, e in resultado.evaluaciones.items():
                futil = " — *Futilidad confirmada (adecuación del esfuerzo)*" if e.futilidad_terapeutica else ""
                st.markdown(f"- **{por_id[pid].nombre}**: {e.etiqueta_prioridad}, {e.clasificacion_sdra}, reversibilidad {e.reversibilidad}{futil}.")

        with st.chat_message("ResourceManagerAgent", avatar="🏥"):
            st.markdown(f"**Gestor Hospitalario (ResourceManagerAgent):**\n\n{d_rec.veredicto}")

        with st.chat_message("ConsensusCoordinator", avatar="🧭"):
            st.markdown(f"**Coordinador del Comité (ConsensusCoordinator):**\n\nSe activa el marco de desempate **{marco['nombre']}**.")
            for t in resultado.trazas_coordinador:
                if t.etapa == Etapa.REGLA.value and t.paciente_id:
                    st.markdown(f"- {por_id[t.paciente_id].nombre}: {t.conclusion}")

        with st.chat_message("ResourceManagerAgent", avatar="🏥"):
            asignados_txt = ", ".join([por_id[x].nombre for x in d_rec.datos.get("asignados", [])]) or "ninguno"
            st.markdown(f"**Gestor Hospitalario (ResourceManagerAgent):** Recursos físicos vinculados a: **{asignados_txt}**.")

        with st.chat_message("BioethicsAuditAgent", avatar="⚖️"):
            st.markdown(f"**Auditor Bioético-Legal (BioethicsAuditAgent):**\n\n{d_bio.veredicto}")
            relevantes = [t for t in d_bio.trazas if t.nivel in ("ALERTA", "OBJECION", "CRITICO") or "PROTECCIÓN REFORZADA" in t.conclusion or t.regla_id in ("REG-BIO-06", "REG-BIO-04")]
            for t in relevantes:
                paciente_str = f" ({por_id[t.paciente_id].nombre})" if t.paciente_id else ""
                st.markdown(f"- `{t.regla_id}`{paciente_str}: {t.conclusion}")


# =============================================================================
# PESTAÑA 3 · Dilemas Éticos Comparados
# =============================================================================
with tab3:
    st.markdown("### ⚖️ Contraste de Marcos Filosóficos y Dilemas Éticos")
    st.caption("Análisis pedagógico de cómo diferentes escuelas de la ética resuelven el mismo dilema clínico de escasez de recursos.")

    # 3 Tarjetas de los marcos éticos
    col_m1, col_m2, col_m3 = st.columns(3)
    marcos_cols = [
        (col_m1, ModoEtico.UTILITARISTA_CLINICO.value),
        (col_m2, ModoEtico.DEONTOLOGICO_IGUALITARIO.value),
        (col_m3, ModoEtico.CONSENSO_CONSTITUCIONAL_COLOMBIA.value),
    ]

    for col_m, clave_modo in marcos_cols:
        info_m = kb.MARCOS_ETICOS[clave_modo]
        r_modo = comparacion[clave_modo]
        es_activo = clave_modo == resultado.modo
        ganadores_modo = ", ".join(por_id[x].nombre for x in r_modo.asignados_uci) or "Nadie asignado"
        
        with col_m:
            border_css = "border: 2px solid #059669;" if es_activo else "border: 1px solid #e2e8f0;"
            tag_activo = '<span class="pill pill-assigned" style="margin-left:8px;">ACTIVO EN SESIÓN</span>' if es_activo else ""
            
            st.markdown(
                f"""
                <div class="framework-card {'active' if es_activo else ''}" style="border-top: 6px solid {info_m['color']};">
                  <div style="font-size:0.75rem; text-transform:uppercase; font-weight:700; color:{info_m['color']};">
                    {info_m['escuela']}
                  </div>
                  <h4 style="margin:4px 0 8px 0; font-size:1.05rem; color:#0f172a;">
                    {info_m['nombre']}{tag_activo}
                  </h4>
                  
                  <div style="font-size:0.82rem; margin-bottom:8px;">
                    <b>Principio:</b> {info_m['principio']}
                  </div>
                  <div style="font-size:0.82rem; color:#475569; margin-bottom:8px;">
                    <b>Criterio de Asignación:</b> {info_m['criterio']}
                  </div>
                  
                  <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px; margin:10px 0;">
                    <div style="font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase;">Resultado en el Escenario Actual:</div>
                    <div style="font-size:0.95rem; font-weight:800; color:#0f172a; margin:2px 0;">🛏️ UCI: {ganadores_modo}</div>
                    <div style="margin-top:4px;">{badge_validacion(r_modo.validacion_global)}</div>
                  </div>

                  <div style="font-size:0.78rem; color:#991b1b; background:#fef2f2; border:1px solid #fecaca; border-radius:6px; padding:8px;">
                    <b>Tensión / Crítica Ética:</b> {info_m['critica']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Matriz comparativa detallada
    st.markdown("#### 🔀 Matriz de Asignación Comparativa: ¿Quién recibe UCI en cada marco?")
    filas_comp = []
    for p in pacientes:
        fila = {
            "Paciente": f"{p.nombre} ({p.edad} años, SOFA {p.escala_sofa})",
            "Condición Clave": "Sujeto protección reforzada" if p.vulnerabilidad_social else ("Enfermedad terminal" if p.enfermedad_terminal_irreversible else "Joven sin comorbilidad"),
        }
        for modo_k, r_k in comparacion.items():
            fila[kb.MARCOS_ETICOS[modo_k]["nombre"]] = "🛏️ Cama UCI Asignada" if p.id in r_k.asignados_uci else ("🕊️ Cuidado Paliativo" if r_k.evaluaciones[p.id].prioridad == 4 else "🚑 Cola CRUE")
        filas_comp.append(fila)

    fila_val = {
        "Paciente": "⚖️ Dictamen de Auditoría Bioética",
        "Condición Clave": "Constitucionalidad y No Discriminación",
    }
    for modo_k, r_k in comparacion.items():
        fila_val[kb.MARCOS_ETICOS[modo_k]["nombre"]] = r_k.validacion_global
    filas_comp.append(fila_val)

    st.dataframe(pd.DataFrame(filas_comp), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📝 Nota Médica y Bioética Oficial para Historia Clínica Electrónica (NLG Simbólico)")
    st.caption(
        "Motor determinista de Generación de Lenguaje Natural Simbólico estructurado bajo los estándares de la "
        "Resolución 1995 de 1999 (MinSalud Colombia), Ley 23 de 1981 (Código de Ética Médica) y Ley Estatutaria 1751 de 2015."
    )

    with st.expander("📋 Desplegar Nota Oficial para Historia Clínica (Resolución 1995 de 1999)", expanded=True):
        col_nlg_sel, col_nlg_btn = st.columns([3, 1])
        with col_nlg_sel:
            paciente_nlg_id = st.selectbox(
                "Seleccione paciente para expedir la Nota Oficial de Historia Clínica:",
                [p.id for p in pacientes],
                format_func=lambda pid: f"{pid} · {por_id[pid].nombre} ({por_id[pid].edad} años) — {resultado.evaluaciones[pid].etiqueta_prioridad}",
                key="sel_paciente_nlg_tab3",
            )

        nlg_gen = ClinicalNarrativeGenerator()
        rep_p_nlg = reportes_robustez.get(paciente_nlg_id) if reportes_robustez else None
        texto_nota_nlg = nlg_gen.generar_nota_historia_clinica(
            paciente_nlg_id,
            resultado,
            reporte_robustez=rep_p_nlg,
        )

        with col_nlg_btn:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("📋 Copiar Nota Oficial para Historia Clínica", key="btn_copiar_nota_tab3", use_container_width=True, type="primary"):
                st.toast("✅ ¡Nota Oficial lista para copiar al portapapeles / Historia Clínica!")
                st.success("Nota generada y validada conforme a la Res. 1995 de 1999. Use Ctrl+A / Ctrl+C en el área de texto inferior.")

        # Área de texto editable para el profesional médico
        nota_editada_nlg = st.text_area(
            f"Nota Oficial para Historia Clínica Electrónica — {por_id[paciente_nlg_id].nombre} ({paciente_nlg_id}) [Editable]:",
            value=texto_nota_nlg,
            height=450,
            key=f"txt_area_nlg_{paciente_nlg_id}",
            help="Documento médico-legal solemne en 7 secciones. El médico tratante puede complementar o ajustar la nota antes de guardarla.",
        )

        col_d1_nlg, col_d2_nlg = st.columns(2)
        with col_d1_nlg:
            st.download_button(
                label=f"💾 Descargar Nota Oficial de {por_id[paciente_nlg_id].nombre} (.txt)",
                data=nota_editada_nlg.encode("utf-8"),
                file_name=f"nota_oficial_historia_clinica_{paciente_nlg_id}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_d2_nlg:
            st.download_button(
                label="📄 Exportar Expediente Completo con Trazas XAI (.md)",
                data=nota_editada_nlg.encode("utf-8"),
                file_name=f"expediente_comite_{paciente_nlg_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )


# =============================================================================
# PESTAÑA 4 · Explicabilidad Fisiológica y XAI
# =============================================================================
with tab4:
    st.markdown("### 🔍 Explicabilidad Fisiológica y Trazabilidad XAI")
    st.caption("Trazas de inferencia auditable: dispersión clínica, ranking determinista y cadena paso a paso por paciente.")

    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        # Gráfico Plotly: Urgencia vs Reversibilidad
        fig = go.Figure()
        for prioridad_val in (1, 2, 3, 4):
            ids = [pid for pid, e in resultado.evaluaciones.items() if e.prioridad == prioridad_val]
            if not ids:
                continue
            fig.add_trace(go.Scatter(
                x=[resultado.evaluaciones[i].prob_sobrevida * 100 for i in ids],
                y=[resultado.evaluaciones[i].puntaje_urgencia for i in ids],
                mode="markers+text",
                name=ETIQUETAS_PRIORIDAD[prioridad_val],
                text=[por_id[i].nombre for i in ids],
                textposition="top center",
                marker=dict(
                    size=[28 if resultado.decisiones[i].asignado_uci else 16 for i in ids],
                    symbol=["star" if resultado.decisiones[i].asignado_uci else "circle" for i in ids],
                    color=PRIORIDAD_COLOR[prioridad_val],
                    line=dict(width=1.5, color="#ffffff"),
                ),
                customdata=[[por_id[i].edad, por_id[i].escala_sofa, por_id[i].pafi, resultado.decisiones[i].destino] for i in ids],
                hovertemplate="<b>%{text}</b><br>Sobrevida Estimada: %{x:.0f}%<br>Puntaje Urgencia: %{y:.1f}<br>"
                              "Edad: %{customdata[0]} años · SOFA: %{customdata[1]} · PaFi: %{customdata[2]}<br>"
                              "<b>Destino:</b> %{customdata[3]}<extra></extra>",
            ))
        
        fig.add_vline(x=50, line_dash="dot", line_color="#cbd5e1")
        fig.add_hline(y=50, line_dash="dot", line_color="#cbd5e1")
        fig.add_annotation(x=98, y=98, text="Alta Urgencia + Alta Reversibilidad", showarrow=False, xanchor="right", font=dict(size=10, color="#64748b"))
        fig.add_annotation(x=2, y=98, text="Alta Urgencia + Baja Reversibilidad", showarrow=False, xanchor="left", font=dict(size=10, color="#64748b"))
        
        fig.update_layout(
            title="Matriz de Triage: Reversibilidad (Sobrevida SOFA) vs. Urgencia Fisiológica",
            xaxis=dict(title="Reversibilidad Fisiológica Estimada (%)", range=[0, 105], gridcolor="#f1f5f9"),
            yaxis=dict(title="Puntaje de Urgencia Fisiológica (0-100)", range=[0, 105], gridcolor="#f1f5f9"),
            height=390,
            template="plotly_white",
            legend=dict(orientation="h", y=-0.22, x=0),
            margin=dict(t=45, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_g2:
        if resultado.orden_prioridad:
            orden_p = resultado.orden_prioridad
            colores_bar = ["#059669" if resultado.decisiones[i].asignado_uci else "#d97706" for i in orden_p]
            fig_bar = go.Figure(go.Bar(
                x=[resultado.puntajes[i] for i in orden_p],
                y=[f"#{k} {por_id[i].nombre}" for k, i in enumerate(orden_p, 1)],
                orientation="h",
                text=[f"{resultado.puntajes[i]:.1f}" for i in orden_p],
                textposition="auto",
                marker_color=colores_bar,
            ))
            fig_bar.update_layout(
                title=f"Orden de Prioridad ({resultado.etiqueta_puntaje})",
                height=390,
                yaxis=dict(autorange="reversed"),
                template="plotly_white",
                margin=dict(t=45, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No hay pacientes en Prioridad 1 en competencia distributiva.")

    st.markdown("---")

    # Cadena explicativa individualizada
    st.markdown("#### 🔎 Cadena Lógica Individualizada por Paciente")
    paciente_sel_id = st.selectbox(
        "Seleccione paciente para inspeccionar su trazabilidad:",
        [p.id for p in pacientes],
        format_func=lambda i: f"{por_id[i].id} · {por_id[i].nombre} ({por_id[i].edad} años) — {resultado.evaluaciones[i].etiqueta_prioridad}",
    )

    if paciente_sel_id:
        p_sel = por_id[paciente_sel_id]
        d_sel = resultado.decisiones[p_sel.id]
        e_sel = resultado.evaluaciones[p_sel.id]
        
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:14px; margin-bottom:12px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <h4 style="margin:0; font-size:1.05rem;">{p_sel.nombre} · {p_sel.id}</h4>
                  <div style="font-size:0.82rem; color:#64748b;">{p_sel.descripcion}</div>
                </div>
                <div>
                  {destino_asistencial_badge(d_sel.asignado_uci, d_sel.prioridad, d_sel.posicion_ranking)}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rep_sel = reportes_robustez.get(p_sel.id) if reportes_robustez else None
        if rep_sel:
            pct_s = rep_sel.porcentaje_estabilidad_decision
            col_rb1, col_rb2 = st.columns([1, 3])
            with col_rb1:
                st.metric("Estabilidad Monte Carlo", f"{pct_s:.1f}%", delta=rep_sel.diagnostico_frontera,
                          delta_color="normal" if pct_s >= 90.0 else "inverse")
            with col_rb2:
                st.info(f"**Directriz XAI:** {rep_sel.advertencia_clinica}")

        vista_sel = st.radio(
            "Formato de visualización:",
            ["Cadena paso a paso estructurada", "Árbol de inferencia (Graphviz)", "Tabla de trazas crudas"],
            horizontal=True,
            key=f"vista_{p_sel.id}",
        )

        cadena_sel = resultado.cadena_xai(p_sel.id)

        if vista_sel == "Cadena paso a paso estructurada":
            for i, etapa_val in enumerate(x.value for x in Etapa):
                trazas_etapa = [t for t in cadena_sel if t.etapa == etapa_val]
                if not trazas_etapa:
                    continue
                st.markdown(f"**Etapa {i+1}: {etapa_val}**")
                for t in trazas_etapa:
                    render_traza_moderna(t)
        elif vista_sel == "Árbol de inferencia (Graphviz)":
            st.graphviz_chart(arbol_dot(resultado, p_sel.id), use_container_width=True)
        else:
            st.dataframe(pd.DataFrame([t.to_dict() for t in cadena_sel]), hide_index=True, use_container_width=True)

    with st.expander("🌐 Ver trazas del sistema (Gestión global de recursos hospitalarios)", expanded=False):
        for t in [t for t in resultado.todas_las_trazas() if t.paciente_id is None]:
            render_traza_moderna(t)

    # Descarga de Auditoría Médica
    st.markdown("#### 💾 Exportación de Registro de Auditoría")
    traza_json = json.dumps({
        "modo": resultado.modo,
        "estado_hospital": resultado.estado_hospital,
        "recursos": {
            "camas_disponibles": estado.camas_uci_disponibles,
            "ventiladores_disponibles": estado.ventiladores_disponibles,
        },
        "validacion_global": resultado.validacion_global,
        "decisiones": {pid: vars(d) for pid, d in resultado.decisiones.items()},
        "trazas": [t.to_dict() for t in resultado.todas_las_trazas()],
    }, ensure_ascii=False, indent=2)

    c_d1, c_d2 = st.columns(2)
    c_d1.download_button("⬇️ Descargar Expediente de Auditoría (JSON)", traza_json, "auditoria_uci_xai.json", "application/json", use_container_width=True)
    c_d2.download_button("⬇️ Descargar Trazas de Reglas (CSV)",
                         pd.DataFrame([t.to_dict() for t in resultado.todas_las_trazas()]).to_csv(index=False).encode("utf-8-sig"),
                         "trazas_uci.csv", "text/csv", use_container_width=True)


# =============================================================================
# PESTAÑA 5 · Marco Normativo y Responsabilidad
# =============================================================================
with tab5:
    st.markdown("### 📜 Marco Jurídico Colombiano y Matriz de Responsabilidad")
    st.caption("Fundamentación legal, constitucional y bioética que rige el diseño de las reglas de inferencia y la auditoría del sistema.")

    col_leg1, col_leg2 = st.columns(2)
    with col_leg1:
        st.markdown(
            """
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:16px; margin-bottom:14px;">
              <h4 style="margin:0 0 8px 0; color:#0f172a; font-size:1rem;">🤝 Supervisión Humana Obligatoria (Ley 23 de 1981)</h4>
              <p style="font-size:0.84rem; color:#475569; margin:0; line-height:1.45;">
                El sistema actúa estrictamente como un <b>Sistema de Soporte a la Decisión Clínica (CDSS)</b>. 
                El acto médico es indelegable y personal: el médico tratante y el Comité de Ética Asistencial conservan la 
                autonomía profesional para acoger o apartarse motivadamente de la recomendación, con registro en la historia clínica 
                (Resolución 1995 de 1999).
              </p>
            </div>
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:16px;">
              <h4 style="margin:0 0 8px 0; color:#0f172a; font-size:1rem;">🔐 Privacidad y Datos Sensibles (Ley 1581 de 2012)</h4>
              <p style="font-size:0.84rem; color:#475569; margin:0; line-height:1.45;">
                Los datos de salud son sensibles. El motor de inferencia corre localmente sin transmisión de datos a APIs externas. 
                La identidad del paciente nunca interviene en las funciones de evaluación clínica ni cálculo de puntajes.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_leg2:
        st.markdown(
            """
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:16px; margin-bottom:14px;">
              <h4 style="margin:0 0 8px 0; color:#0f172a; font-size:1rem;">🌍 Ética de IA UNESCO (2021) y Marco Colombia</h4>
              <p style="font-size:0.84rem; color:#475569; margin:0; line-height:1.45;">
                Cumplimiento estricto de los 4 pilares: <b>Transparencia y Explicabilidad</b> (trazas simbólicas sin cajas negras), 
                <b>Equidad</b> (bloqueo activo de edadismo mediante prueba contrafáctica REG-BIO-01), <b>Supervisión Humana</b> 
                y <b>Rendición de Cuentas</b> con auditoría inmutable.
              </p>
            </div>
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:16px;">
              <h4 style="margin:0 0 8px 0; color:#0f172a; font-size:1rem;">⚠️ Declaración de Alcance y Limitaciones</h4>
              <p style="font-size:0.84rem; color:#475569; margin:0; line-height:1.45;">
                Prototipo con fines docentes y de investigación académica. Las probabilidades de sobrevida basadas en SOFA corresponden 
                a aproximaciones pedagógicas de la literatura médica y no deben emplearse para decisiones diagnósticas o terapéuticas reales.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("#### 🏛️ Matriz de Responsabilidad Civil y Profesional")
    st.dataframe(pd.DataFrame([
        {
            "Actor": m["actor"],
            "Responsabilidad": m["responsabilidad"],
            "Régimen Jurídico": m["tipo"],
            "Límite de Exclusión": m["no_responde_por"],
        }
        for m in kb.MATRIZ_RESPONSABILIDAD
    ]), hide_index=True, use_container_width=True)

    st.markdown("#### 📜 Corpus Normativo Aplicado por las Reglas de Inferencia")
    st.dataframe(pd.DataFrame([
        {
            "Norma": v["norma"],
            "Contenido": v["contenido"],
            "Aplicación Concreta en el Sistema": v["aplicacion"],
        }
        for v in kb.NORMATIVA.values()
    ]), hide_index=True, use_container_width=True)

    st.markdown("#### 🔗 Trazabilidad en Vivo: Norma ↔ Reglas Disparadas en la Sesión")
    conteo_norma: Dict[str, Dict[str, object]] = {}
    for t in resultado.todas_las_trazas():
        fila = conteo_norma.setdefault(t.regla_id, {
            "Regla": t.regla_id,
            "Agente": t.agente,
            "Disparos": 0,
            "Fundamento Normativo o Médico": t.fundamento_normativo_o_medico,
        })
        fila["Disparos"] = int(fila["Disparos"]) + 1
    st.dataframe(pd.DataFrame(sorted(conteo_norma.values(), key=lambda f: str(f["Regla"]))), hide_index=True, use_container_width=True)
