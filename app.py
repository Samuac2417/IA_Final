"""
app.py — Dashboard Hospitalario del Sistema Experto Multiagente de Triage UCI
=============================================================================

Interfaz Streamlit del Comité Clínico-Bioético Autónomo.

Ejecución:
    streamlit run app.py

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
    ETIQUETAS_PRIORIDAD, Etapa, ModoEtico, Paciente, RecursoHospitalario, ResultadoComite, TrazaInferencia,
)

# =============================================================================
# Configuración visual
# =============================================================================
st.set_page_config(page_title="Comité Multiagente UCI · Triage Ético", page_icon="🏥", layout="wide")

NIVEL_COLOR = {"INFO": "#64748b", "OK": "#16a34a", "ALERTA": "#d97706", "OBJECION": "#ea580c", "CRITICO": "#dc2626"}
NIVEL_ICONO = {"INFO": "ℹ️", "OK": "✅", "ALERTA": "⚠️", "OBJECION": "🛑", "CRITICO": "🚨"}
PRIORIDAD_COLOR = {1: "#dc2626", 2: "#f59e0b", 3: "#16a34a", 4: "#7c3aed"}
VALIDACION_COLOR = {"VALIDADO": "#16a34a", "VALIDADO CON OBSERVACIONES": "#d97706",
                    "OBJETADO": "#ea580c", "VETO CONSTITUCIONAL": "#dc2626"}
AGENTE_ICONO = {"ClinicalTriageAgent": "🩺", "ResourceManagerAgent": "🏥",
                "BioethicsAuditAgent": "⚖️", "ConsensusCoordinator": "🧭"}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      .hdr {background: linear-gradient(90deg,#0f172a,#1e3a8a); color:#fff; padding:18px 24px;
            border-radius:14px; margin-bottom:14px;}
      .hdr h1 {font-size:1.65rem; margin:0; color:#fff;}
      .hdr p {margin:4px 0 0 0; opacity:.85; font-size:.92rem;}
      .card {border-radius:12px; padding:14px 16px; background:rgba(148,163,184,.08);
             border:1px solid rgba(148,163,184,.30); height:100%;}
      .card h4 {margin:0 0 4px 0;}
      .pill {display:inline-block; padding:2px 10px; border-radius:999px; color:#fff;
             font-size:.75rem; font-weight:600; margin:2px 4px 2px 0;}
      .traza {border-left:5px solid #64748b; padding:8px 12px; margin:6px 0; border-radius:6px;
              background:rgba(148,163,184,.08); font-size:.86rem;}
      .traza .cab {font-weight:700; font-size:.78rem; opacity:.85;}
      .banner {padding:10px 16px; border-radius:10px; color:#fff; font-weight:700; margin:6px 0 12px 0;}
      .flecha {text-align:center; font-size:1.3rem; opacity:.6; margin:-2px 0;}
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


def _siguiente_id() -> str:
    usados = {p.id for p in st.session_state.pacientes}
    n = 401
    while f"PAC-{n:03d}" in usados:
        n += 1
    return f"PAC-{n:03d}"


_init_estado()


# =============================================================================
# Utilidades de presentación
# =============================================================================
def pill(texto: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}">{html.escape(texto)}</span>'


def render_traza(t: TrazaInferencia, mostrar_agente: bool = True) -> None:
    color = NIVEL_COLOR.get(t.nivel, "#64748b")
    agente = f"{AGENTE_ICONO.get(t.agente, '🤖')} {t.agente} · " if mostrar_agente else ""
    st.markdown(
        f'<div class="traza" style="border-left-color:{color}">'
        f'<div class="cab">{NIVEL_ICONO.get(t.nivel, "")} {html.escape(agente)}{html.escape(t.regla_id)} · '
        f'{html.escape(t.etapa)}</div>'
        f'<b>SI</b> {html.escape(t.condicion_evaluada)}<br>'
        f'<b>ENTONCES</b> {html.escape(t.conclusion)}<br>'
        f'<i>📚 Fundamento: {html.escape(t.fundamento_normativo_o_medico)}</i></div>',
        unsafe_allow_html=True,
    )


def icono_destino(r: ResultadoComite, pid: str) -> str:
    d = r.decisiones[pid]
    if d.asignado_uci:
        return "🛏️ UCI" + (" + ventilador" if d.asignado_ventilador else "")
    return {1: f"🚑 CRUE · espera #{d.posicion_ranking}", 2: "🩺 Intermedios",
            3: "🏠 Hosp. general", 4: "🕊️ Paliativo"}[d.prioridad]


def _dot_escape(texto: str, ancho: int = 42) -> str:
    texto = texto.replace("\\", "/").replace('"', "'")
    return "\\n".join(textwrap.wrap(texto, ancho)) or " "


def arbol_dot(r: ResultadoComite, pid: str) -> str:
    """Árbol de inferencia XAI en lenguaje DOT (Graphviz renderizado por Streamlit)."""
    p = r.paciente(pid)
    cadena = r.cadena_xai(pid)
    lineas = ['digraph G {', 'rankdir=TB; nodesep=0.25; ranksep=0.35;',
              'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, fillcolor="#f8fafc"];',
              'edge [color="#94a3b8"];',
              f'root [label="{_dot_escape(p.nombre + " · " + p.id, 30)}", shape=ellipse, fillcolor="#1e3a8a", '
              f'fontcolor="white", fontsize=12];']
    anterior = "root"
    for i, etapa in enumerate(e.value for e in Etapa):
        trazas = [t for t in cadena if t.etapa == etapa]
        if not trazas:
            continue
        hub = f"etapa{i}"
        lineas.append(f'{hub} [label="{etapa}", shape=hexagon, fillcolor="#e2e8f0", fontsize=11];')
        lineas.append(f"{anterior} -> {hub} [penwidth=2, color=\"#1e3a8a\"];")
        for j, t in enumerate(trazas):
            nodo = f"t{i}_{j}"
            color = NIVEL_COLOR.get(t.nivel, "#64748b")
            concl = t.conclusion if len(t.conclusion) < 110 else t.conclusion[:107] + "..."
            lineas.append(f'{nodo} [label="{_dot_escape(t.regla_id)}\\n{_dot_escape(concl)}", color="{color}", '
                          f'penwidth=2];')
            lineas.append(f"{hub} -> {nodo};")
        anterior = hub
    lineas.append("}")
    return "\n".join(lineas)


# =============================================================================
# Sidebar · Panel de control del escenario
# =============================================================================
with st.sidebar:
    st.header("🎛️ Panel de Control")
    st.button("🚨 Cargar Caso de Dilema Crítico (3 Pacientes vs 1 Cama UCI)", type="primary",
              width="stretch", on_click=cb_cargar_escenario, args=("dilema",))

    with st.expander("📂 Otros escenarios precargados"):
        otro = st.selectbox("Escenario", [k for k in kb.ESCENARIOS if k != "dilema"],
                            format_func=lambda k: kb.ESCENARIOS[k]["titulo"])
        st.button("Cargar escenario seleccionado", width="stretch", on_click=cb_cargar_escenario, args=(otro,))

    st.subheader("🏥 Recursos críticos")
    st.slider("Camas UCI disponibles", 0, 5, key="camas")
    st.slider("Ventiladores mecánicos disponibles", 0, 5, key="ventiladores")
    st.slider("Camas UCI totales de la institución", 5, 40, key="camas_totales")

    st.subheader("⚖️ Marco ético de decisión")
    st.radio("Marco filosófico de resolución", [m.value for m in ModoEtico], key="modo",
             format_func=lambda v: kb.MARCOS_ETICOS[v]["nombre"])
    with st.expander("⚙️ Parámetros avanzados del consenso"):
        st.slider("Umbral de empate técnico SOFA (±puntos)", 0, 3, key="umbral",
                  help="Diferencias de SOFA menores o iguales a este valor se consideran clínicamente "
                       "indistinguibles (variabilidad interobservador). Solo aplica al Consenso Constitucional.")
        st.caption(f"Semilla pública del sorteo auditable: `{kb.SEMILLA_SORTEO_AUDITABLE}`")

    st.subheader("➕ Ingresar paciente en vivo")
    with st.form("form_paciente", clear_on_submit=True):
        nombre = st.text_input("Nombre", placeholder="Ej. María Torres")
        c1, c2 = st.columns(2)
        edad = c1.number_input("Edad", 0, 120, 50)
        espera = c2.number_input("Llegó hace (min)", 0, kb.MINUTO_ACTUAL_TURNO, 5)
        sat = c1.number_input("SatO2 (%)", 50.0, 100.0, 88.0, step=0.5)
        pafi = c2.number_input("PaFi", 40, 600, 180)
        sofa = c1.number_input("SOFA (0-24)", 0, 24, 5)
        glasgow = c2.number_input("Glasgow (3-15)", 3, 15, 15)
        comorb = st.text_input("Comorbilidades graves (separadas por coma)")
        vent = st.checkbox("Requiere ventilador mecánico")
        terminal = st.checkbox("Enfermedad terminal irreversible")
        vulnerable = st.checkbox("Vulnerabilidad social (sujeto de especial protección)")
        if st.form_submit_button("Ingresar a urgencias", width="stretch"):
            try:
                nuevo = Paciente(
                    id=_siguiente_id(), nombre=nombre.strip() or "Paciente sin nombre", edad=int(edad),
                    hora_llegada=kb.MINUTO_ACTUAL_TURNO - int(espera), sat_o2=float(sat), pafi=int(pafi),
                    escala_sofa=int(sofa), glasgow=int(glasgow), requiere_ventilador=vent,
                    comorbilidades_graves=[c.strip() for c in comorb.split(",") if c.strip()],
                    enfermedad_terminal_irreversible=terminal, vulnerabilidad_social=vulnerable,
                    descripcion="Paciente ingresado manualmente durante la sesión.",
                )
                st.session_state.pacientes = st.session_state.pacientes + [nuevo]
                st.session_state.aviso = f"Paciente {nuevo.id} ingresado."
            except ValueError as exc:
                st.error(str(exc))

    if st.session_state.pacientes:
        with st.expander("🗑️ Retirar pacientes"):
            st.multiselect("Seleccione", [p.id for p in st.session_state.pacientes], key="retirar_ids",
                           format_func=lambda i: next(f"{p.id} · {p.nombre}" for p in st.session_state.pacientes
                                                      if p.id == i))
            st.button("Retirar seleccionados", on_click=cb_retirar, width="stretch")
            st.button("Vaciar sala de urgencias", on_click=cb_vaciar, width="stretch")

# =============================================================================
# Encabezado
# =============================================================================
st.markdown(
    '<div class="hdr"><h1>🏥 Comité Clínico-Bioético Multiagente · Triage UCI</h1>'
    '<p>Sistema Experto Simbólico (reglas de producción + agentes) para la asignación ética de recursos críticos · '
    'Universidad Católica Luis Amigó · Análisis de Inteligencia Artificial</p></div>',
    unsafe_allow_html=True,
)
if st.session_state.aviso:
    st.toast(st.session_state.aviso)
    st.session_state.aviso = ""

pacientes: List[Paciente] = st.session_state.pacientes
if not pacientes:
    st.info("No hay pacientes en urgencias. Cargue el caso de dilema crítico o ingrese pacientes desde el panel lateral.")
    st.stop()

estado = RecursoHospitalario(
    camas_uci_totales=max(st.session_state.camas_totales, st.session_state.camas),
    camas_uci_disponibles=st.session_state.camas,
    ventiladores_disponibles=st.session_state.ventiladores,
    modo_etico_activo=st.session_state.modo,
)
coordinador = ConsensusCoordinator(umbral_empate_sofa=st.session_state.umbral)
resultado = coordinador.deliberar(pacientes, estado)
comparacion: Dict[str, ResultadoComite] = coordinador.comparar_marcos(pacientes, estado)
por_id = {p.id: p for p in pacientes}
d_rec = resultado.dictamenes["ResourceManagerAgent"]
marco = kb.MARCOS_ETICOS[resultado.modo]

# =============================================================================
# KPIs
# =============================================================================
n_p1 = sum(1 for e in resultado.evaluaciones.values() if e.prioridad == 1)
deficit = d_rec.datos["deficit_camas"]
critico = resultado.estado_hospital == "ESCASEZ_CRITICA"
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total pacientes en urgencias", len(pacientes), border=True)
k2.metric("Pacientes en Prioridad 1", n_p1, border=True,
          delta=f"{d_rec.datos['demanda_p1_ventilador']} requieren ventilador", delta_color="off")
k3.metric("Camas UCI libres", estado.camas_uci_disponibles, border=True,
          delta=(f"-{deficit} cama(s) de déficit" if deficit else "Oferta suficiente"),
          delta_description=f"{estado.ventiladores_disponibles} ventilador(es) disponibles")
k4.metric("Estado del hospital", "SATURACIÓN CRÍTICA" if critico else "NORMAL", border=True,
          delta=("-Triaje de catástrofe activo" if critico else "Operación normal"))
st.markdown(
    f'<div class="banner" style="background:{"#dc2626" if critico else "#16a34a"}">'
    + ("🚨 ESCASEZ CRÍTICA DECLARADA (REG-REC-02/03): el comité aplica el marco "
       f"«{marco['nombre']}» para desempatar." if critico else
       "✅ CAPACIDAD NORMAL (REG-REC-01): todos los pacientes P1 reciben UCI sin conflicto ético distributivo.")
    + "</div>",
    unsafe_allow_html=True,
)

with st.expander("📋 Pacientes en sala de urgencias (base de hechos)", expanded=False):
    st.dataframe(pd.DataFrame([{
        "ID": p.id, "Nombre": p.nombre, "Edad": p.edad,
        "Espera (min)": p.minutos_espera(kb.MINUTO_ACTUAL_TURNO), "SatO2 %": p.sat_o2, "PaFi": p.pafi,
        "SOFA": p.escala_sofa, "Glasgow": p.glasgow, "Ventilador": "Sí" if p.requiere_ventilador else "No",
        "Comorbilidades": ", ".join(p.comorbilidades_graves) or "—",
        "Terminal": "Sí" if p.enfermedad_terminal_irreversible else "No",
        "Vulnerable": "Sí" if p.vulnerabilidad_social else "No",
    } for p in pacientes]), hide_index=True, width="stretch")

tab1, tab2, tab3 = st.tabs(["🧠 Deliberación del Comité (En Vivo)", "🌳 Árbol de Explicabilidad (XAI Trace)",
                            "⚖️ Marco Ético, Legal y Responsabilidad"])

# =============================================================================
# TAB 1 · Deliberación
# =============================================================================
with tab1:
    cols = st.columns(3)
    for col, nombre_agente in zip(cols, ["ClinicalTriageAgent", "ResourceManagerAgent", "BioethicsAuditAgent"]):
        d = resultado.dictamenes[nombre_agente]
        color = NIVEL_COLOR.get(d.nivel, "#64748b")
        with col:
            cuerpo = ""
            if nombre_agente == "ClinicalTriageAgent":
                for pid, e in resultado.evaluaciones.items():
                    cuerpo += (f"<div>{pill('P' + str(e.prioridad), PRIORIDAD_COLOR[e.prioridad])}"
                               f"<b>{html.escape(por_id[pid].nombre)}</b> · rev. {e.reversibilidad} · "
                               f"sobrevida {e.prob_sobrevida:.0%}</div>")
            elif nombre_agente == "ResourceManagerAgent":
                for pid in resultado.orden_prioridad:
                    ok = pid in d.datos.get("asignados", [])
                    cuerpo += (f"<div>{pill('ASIGNADO' if ok else 'SIN CAMA', '#16a34a' if ok else '#dc2626')}"
                               f"<b>{html.escape(por_id[pid].nombre)}</b></div>")
                if not resultado.orden_prioridad:
                    cuerpo += "<div>Sin candidatos P1.</div>"
            else:
                for pid, v in d.datos["validacion_por_paciente"].items():
                    cuerpo += (f"<div>{pill(v, VALIDACION_COLOR.get(v, '#64748b'))}"
                               f"<b>{html.escape(por_id[pid].nombre)}</b></div>")
            st.markdown(
                f'<div class="card" style="border-top:6px solid {color}">'
                f'<h4>{AGENTE_ICONO[nombre_agente]} {nombre_agente}</h4>'
                f'<div style="font-size:.8rem;opacity:.75">{html.escape(d.rol)}</div><hr style="margin:8px 0">'
                f'<div style="font-weight:600;color:{color}">{html.escape(d.veredicto)}</div>'
                f'<div style="margin-top:8px;font-size:.88rem">{cuerpo}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### 🧭 Acta de consenso del ConsensusCoordinator")
    a1, a2 = st.columns([3, 2])
    with a1:
        color_v = VALIDACION_COLOR.get(resultado.validacion_global, "#64748b")
        with st.container(border=True):
            st.markdown(pill(resultado.validacion_global, color_v), unsafe_allow_html=True)
            st.markdown(resultado.acta_consenso)
    with a2:
        caja = st.success if resultado.modo == ModoEtico.CONSENSO_CONSTITUCIONAL_COLOMBIA.value else st.warning
        caja(f"**{marco['nombre']}** — *{marco['escuela']}*\n\n**Principio:** {marco['principio']}\n\n"
             f"**Criterio:** {marco['criterio']}\n\n**Crítica:** {marco['critica']}")

    st.markdown("### 📑 Recomendación final por paciente")
    st.dataframe(pd.DataFrame([{
        "Paciente": f"{d.nombre} ({por_id[d.paciente_id].edad} a)", "Prioridad": ETIQUETAS_PRIORIDAD[d.prioridad],
        "SOFA": por_id[d.paciente_id].escala_sofa, "Posición": d.posicion_ranking or "—",
        resultado.etiqueta_puntaje or "Puntaje": "—" if d.puntaje_desempate is None else d.puntaje_desempate,
        "Recomendación": d.destino, "Auditoría ética": d.validacion_etica,
        "Plan alternativo": d.plan_alternativo or "—",
    } for d in resultado.decisiones.values()]), hide_index=True, width="stretch")

    st.markdown("### 🔀 Matriz comparativa: ¿quién recibe la cama en cada marco ético?")
    filas = []
    for p in pacientes:
        fila = {"Paciente": f"{p.nombre} ({p.edad} a, SOFA {p.escala_sofa})"}
        for modo, r in comparacion.items():
            fila[kb.MARCOS_ETICOS[modo]["nombre"]] = icono_destino(r, p.id)
        filas.append(fila)
    fila_v = {"Paciente": "⚖️ Validación bioético-legal"}
    for modo, r in comparacion.items():
        fila_v[kb.MARCOS_ETICOS[modo]["nombre"]] = r.validacion_global
    filas.append(fila_v)
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

    st.markdown("### 💬 Transcripción de la sesión del comité")
    d_clin = resultado.dictamenes["ClinicalTriageAgent"]
    d_bio = resultado.dictamenes["BioethicsAuditAgent"]
    with st.chat_message("ClinicalTriageAgent", avatar="🩺"):
        st.markdown(f"**ClinicalTriageAgent:** {d_clin.veredicto}")
        for pid, e in resultado.evaluaciones.items():
            extra = " — **futilidad terapéutica: cuidado paliativo digno**" if e.futilidad_terapeutica else ""
            st.markdown(f"- {por_id[pid].nombre}: **{e.etiqueta_prioridad}**, {e.clasificacion_sdra}, "
                        f"reversibilidad {e.reversibilidad}{extra}.")
    with st.chat_message("ResourceManagerAgent", avatar="🏥"):
        st.markdown(f"**ResourceManagerAgent:** {d_rec.veredicto}")
    with st.chat_message("ConsensusCoordinator", avatar="🧭"):
        st.markdown(f"**ConsensusCoordinator:** aplico el marco **{marco['nombre']}**.")
        for t in resultado.trazas_coordinador:
            if t.etapa == Etapa.REGLA.value and t.paciente_id:
                st.markdown(f"- {por_id[t.paciente_id].nombre}: {t.conclusion}")
    with st.chat_message("ResourceManagerAgent", avatar="🏥"):
        asign = [por_id[x].nombre for x in d_rec.datos.get("asignados", [])]
        st.markdown(f"**ResourceManagerAgent:** recursos asignados a **{', '.join(asign) or 'nadie'}**.")
    with st.chat_message("BioethicsAuditAgent", avatar="⚖️"):
        st.markdown(f"**BioethicsAuditAgent:** {d_bio.veredicto}")
        relevantes = [t for t in d_bio.trazas if t.nivel in ("ALERTA", "OBJECION", "CRITICO")
                      or "PROTECCIÓN REFORZADA" in t.conclusion or t.regla_id in ("REG-BIO-06", "REG-BIO-04")]
        for t in relevantes:
            quien = f" ({por_id[t.paciente_id].nombre})" if t.paciente_id else ""
            st.markdown(f"- {NIVEL_ICONO.get(t.nivel, '')} `{t.regla_id}`{quien}: {t.conclusion}")
    with st.chat_message("ConsensusCoordinator", avatar="🧭"):
        st.markdown(f"**ConsensusCoordinator — consenso final:**\n\n{resultado.acta_consenso}")

# =============================================================================
# TAB 2 · Explicabilidad
# =============================================================================
with tab2:
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
                mode="markers+text", name=ETIQUETAS_PRIORIDAD[prioridad],
                text=[por_id[i].nombre for i in ids], textposition="top center",
                marker=dict(size=[30 if resultado.decisiones[i].asignado_uci else 18 for i in ids],
                            symbol=["star" if resultado.decisiones[i].asignado_uci else "circle" for i in ids],
                            color=PRIORIDAD_COLOR[prioridad], line=dict(width=1.5, color="white")),
                customdata=[[por_id[i].edad, por_id[i].escala_sofa, por_id[i].pafi,
                             resultado.decisiones[i].destino] for i in ids],
                hovertemplate="<b>%{text}</b><br>Sobrevida estimada: %{x:.0f} %<br>Urgencia: %{y:.1f}<br>"
                              "Edad: %{customdata[0]} · SOFA: %{customdata[1]} · PaFi: %{customdata[2]}<br>"
                              "%{customdata[3]}<extra></extra>",
            ))
        fig.add_vline(x=50, line_dash="dot", line_color="#94a3b8")
        fig.add_hline(y=50, line_dash="dot", line_color="#94a3b8")
        fig.add_annotation(x=97, y=97, text="Urgente y reversible", showarrow=False, xanchor="right", opacity=.6)
        fig.add_annotation(x=3, y=97, text="Urgente, baja reversibilidad", showarrow=False, xanchor="left", opacity=.6)
        fig.update_layout(title="Urgencia fisiológica vs. reversibilidad (★ = recibe UCI)",
                          xaxis=dict(title="Reversibilidad fisiológica · sobrevida estimada por SOFA (%)", range=[0, 100]),
                          yaxis=dict(title="Puntaje de urgencia fisiológica (0-100)", range=[0, 100]),
                          height=430, legend=dict(orientation="h", y=-0.22), margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with g2:
        if resultado.orden_prioridad:
            orden = resultado.orden_prioridad
            figb = go.Figure(go.Bar(
                x=[resultado.puntajes[i] for i in orden], y=[f"#{k} {por_id[i].nombre}" for k, i in enumerate(orden, 1)],
                orientation="h", text=[f"{resultado.puntajes[i]:.1f}" for i in orden], textposition="auto",
                marker_color=["#16a34a" if resultado.decisiones[i].asignado_uci else "#dc2626" for i in orden],
            ))
            figb.update_layout(title=f"Orden del comité · {resultado.etiqueta_puntaje}", height=430,
                               yaxis=dict(autorange="reversed"), margin=dict(t=50, b=10))
            st.plotly_chart(figb, width="stretch")
            st.caption("Verde = recibe cama UCI · Rojo = lista de espera con plan CRUE obligatorio.")
        else:
            st.info("No hay pacientes en Prioridad 1 compitiendo por recursos.")

    st.markdown("### 🔎 Cadena lógica por paciente: [Hecho Clínico] → [Regla Disparada] → [Recursos] → "
                "[Auditoría Ética] → [Recomendación Final]")
    for p in pacientes:
        d = resultado.decisiones[p.id]
        e = resultado.evaluaciones[p.id]
        titulo = (f"{p.id} · {p.nombre} ({p.edad} años) — {e.etiqueta_prioridad} → {icono_destino(resultado, p.id)}"
                  f" · {d.validacion_etica}")
        with st.expander(titulo, expanded=False):
            if p.descripcion:
                st.caption(p.descripcion)
            vista = st.radio("Vista", ["Cadena paso a paso", "Árbol de inferencia (Graphviz)", "Tabla"],
                             horizontal=True, key=f"vista_{p.id}", label_visibility="collapsed")
            cadena = resultado.cadena_xai(p.id)
            if vista == "Cadena paso a paso":
                for i, etapa in enumerate(x.value for x in Etapa):
                    trazas = [t for t in cadena if t.etapa == etapa]
                    if not trazas:
                        continue
                    if i:
                        st.markdown('<div class="flecha">⬇</div>', unsafe_allow_html=True)
                    st.markdown(f"**{etapa}**")
                    for t in trazas:
                        render_traza(t)
            elif vista == "Árbol de inferencia (Graphviz)":
                st.graphviz_chart(arbol_dot(resultado, p.id), width="stretch")
            else:
                st.dataframe(pd.DataFrame([t.to_dict() for t in cadena]), hide_index=True, width="stretch")

    with st.expander("🌐 Trazas globales (no asociadas a un paciente)"):
        for t in [t for t in resultado.todas_las_trazas() if t.paciente_id is None]:
            render_traza(t)

    traza_json = json.dumps({
        "modo": resultado.modo, "estado_hospital": resultado.estado_hospital,
        "recursos": {"camas_disponibles": estado.camas_uci_disponibles,
                     "ventiladores_disponibles": estado.ventiladores_disponibles},
        "validacion_global": resultado.validacion_global,
        "decisiones": {pid: vars(d) for pid, d in resultado.decisiones.items()},
        "trazas": [t.to_dict() for t in resultado.todas_las_trazas()],
    }, ensure_ascii=False, indent=2)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Descargar auditoría completa (JSON)", traza_json, "auditoria_comite_uci.json",
                       "application/json", width="stretch")
    c2.download_button("⬇️ Descargar trazas (CSV)",
                       pd.DataFrame([t.to_dict() for t in resultado.todas_las_trazas()]).to_csv(index=False)
                       .encode("utf-8-sig"), "trazas_xai.csv", "text/csv", width="stretch")

# =============================================================================
# TAB 3 · Marco ético-legal
# =============================================================================
with tab3:
    st.markdown("### 🧩 Los tres marcos filosóficos implementados")
    cols = st.columns(3)
    for col, (modo, info) in zip(cols, kb.MARCOS_ETICOS.items()):
        r = comparacion[modo]
        ganadores = ", ".join(por_id[x].nombre for x in r.asignados_uci) or "nadie"
        activo = " · ACTIVO" if modo == resultado.modo else ""
        col.markdown(
            f'<div class="card" style="border-top:6px solid {info["color"]}">'
            f'<h4>{html.escape(info["nombre"])}{activo}</h4><div style="font-size:.8rem;opacity:.75">'
            f'{html.escape(info["escuela"])}</div><hr style="margin:8px 0">'
            f'<b>Principio:</b> {html.escape(info["principio"])}<br><br><b>Crítica:</b> {html.escape(info["critica"])}'
            f'<br><br><b>En el escenario actual:</b> UCI → {html.escape(ganadores)} '
            f'{pill(r.validacion_global, VALIDACION_COLOR.get(r.validacion_global, "#64748b"))}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📜 Normas aplicadas en el código")
    st.dataframe(pd.DataFrame([{"Norma": v["norma"], "Contenido": v["contenido"], "Aplicación en el sistema": v["aplicacion"]}
                               for v in kb.NORMATIVA.values()]), hide_index=True, width="stretch")

    st.markdown("### 🏛️ Matriz de responsabilidad civil y profesional")
    st.dataframe(pd.DataFrame([{"Actor": m["actor"], "Responsabilidad": m["responsabilidad"],
                                "Régimen": m["tipo"], "No responde por": m["no_responde_por"]}
                               for m in kb.MATRIZ_RESPONSABILIDAD]), hide_index=True, width="stretch")

    st.markdown("### 🔗 Trazabilidad norma ↔ regla disparada en esta sesión")
    conteo: Dict[str, Dict[str, object]] = {}
    for t in resultado.todas_las_trazas():
        fila = conteo.setdefault(t.regla_id, {"Regla": t.regla_id, "Agente": t.agente, "Disparos": 0,
                                              "Fundamento": t.fundamento_normativo_o_medico})
        fila["Disparos"] = int(fila["Disparos"]) + 1
    st.dataframe(pd.DataFrame(sorted(conteo.values(), key=lambda f: str(f["Regla"]))), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🤝 Human-in-the-Loop (Ley 23 de 1981)")
        st.info("El sistema emite **recomendaciones técnicas auditables**. El acto médico es personal e "
                "indelegable: el médico tratante y el Comité de Ética Hospitalaria toman la decisión final y pueden "
                "apartarse motivadamente de la recomendación, dejando constancia en la historia clínica "
                "(Resolución 1995 de 1999).")
        st.markdown("#### 🔐 Habeas Data (Ley 1581 de 2012)")
        st.info("Los datos de salud son **datos sensibles**. El prototipo procesa todo localmente, no usa APIs "
                "externas y el nombre del paciente nunca participa en la inferencia.")
    with c2:
        st.markdown("#### 🌍 UNESCO (2021) y Marco Ético de IA de Colombia")
        st.info("Transparencia y explicabilidad (trazas XAI), equidad y no discriminación (REG-BIO-01/02 y prueba "
                "contrafáctica), supervisión humana (REG-BIO-04), responsabilidad y rendición de cuentas "
                "(auditoría exportable).")
        st.markdown("#### ⚠️ Limitaciones declaradas")
        st.warning("Prototipo académico. Las probabilidades de sobrevida son una aproximación docente basada en la "
                   "literatura del SOFA y no están calibradas para uso clínico real. No reemplaza el juicio médico.")
