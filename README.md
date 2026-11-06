# Sistema Experto Multiagente para el Triage Médico y Asignación Ética de Recursos Críticos (UCI)

**Institución:** Universidad Católica Luis Amigó  
**Curso:** Análisis de Inteligencia Artificial  
**Proyecto:** Sistema de Soporte a la Decisión Clínica (CDSS) Simbólico y Explicable (XAI)  
**Stack:** Python 3.10+ (100% Simbólico / POO) · Streamlit · Plotly · Graphviz  

---

## 1. Resumen Ejecutivo y Justificación Académica

En situaciones de desastre masivo o emergencias sanitarias de gran escala (como una pandemia o contingencia epidemiológica), la demanda de camas en Unidades de Cuidados Intensivos (UCI) y de ventiladores mecánicos sobrepasa la oferta física instalada. En este escenario, la selección de pacientes no puede recaer en modelos de *Machine Learning* opacos ("cajas negras") debido a tres impedimentos críticos:

1. **Falta de Trazabilidad e Inexplicabilidad:** Un modelo neuronal o estadístico no puede emitir una fundamentación jurídica ni desglosar paso a paso qué norma constitucional o directriz clínica motivó el rechazo de un recurso vital.
2. **Amplificación de Sesgos Históricos:** Los datos de entrenamiento reflejan inequidades de acceso a la salud, castigando a poblaciones desfavorecidas.
3. **Incompatibilidad Normativa con la Ley Colombiana:** La Ley Estatutaria 1751 de 2015, la Ley 23 de 1981 (Código de Ética Médica) y la Sentencia T-760 de 2008 de la Corte Constitucional exigen que todo acto médico y criterio de asignación sea explícito, proporcional y sujeto a supervisión humana obligatoria (*Human-in-the-Loop*).

Este proyecto implementa una **Arquitectura Multiagente Simbólica Determinista**, donde el proceso decisorio reproduce la deliberación de un **Comité Clínico-Bioético Autónomo**. Cada agente evalúa los hechos clínicos con reglas de producción y registra una cadena explicativa (*RuleTrace*) que permite auditar cada deducción en lenguaje médico y legal colombiano.

---

## 2. Arquitectura de Archivos del Repositorio

```text
triage_multiagent_system/
│
├── app.py                     # Dashboard interactivo en Streamlit (UI hospitalaria, KPIs, XAI)
├── requirements.txt           # Dependencias mínimas (streamlit, pandas, plotly)
├── README.md                  # Documentación técnica, fundamentación y guía de sustentación
├── validar_sistema.py         # Script de validación integral y pruebas de regresión por consola
│
├── core/
│   ├── __init__.py            # Exportaciones de paquetes
│   ├── models.py              # Dataclasses tipadas (Paciente, RecursoHospitalario, TrazaInferencia, etc.)
│   ├── knowledge_base.py      # Umbrales clínicos (SOFA, PaFi, Glasgow), normativa y casos clínicos
│   └── coordinator.py         # Orquestador del comité, motor de desempate y síntesis de consenso
│
└── agents/
    ├── __init__.py            # Exportaciones de agentes
    ├── base_agent.py          # Clase abstracta BaseAgent con motor de registro de trazas
    ├── clinical_agent.py      # Agente 1: Severidad fisiológica, reversibilidad y futilidad terapéutica
    ├── resource_agent.py      # Agente 2: Gestión de inventario UCI y detección de colapso hospitalario
    └── bioethics_agent.py     # Agente 3: Auditoría bioética, bloqueo de edadismo y marco legal colombiano
```

---

## 3. Especificación de los Agentes y Reglas de Producción

### Agente 1: ClinicalTriageAgent (Medicina Crítica)
Evalúa únicamente parámetros fisiológicos objetivos, aislando deliberadamente factores demográficos y socioeconómicos:
- **REG-CLIN-00:** Clasificación inicial de ingreso mediante el sistema de triage de urgencias (Resolución 5596 de 2015 / Manchester).
- **REG-CLIN-NEWS2:** Tamizaje ultrarrápido dinámico mediante la escala NEWS2 (*National Early Warning Score 2*, Royal College of Physicians 2017). Evalúa 7 parámetros fisiológicos basales (FR, SpO2, O2 suplementario, PAS, FC, conciencia ACVPU y temperatura). Si $\text{NEWS2} \ge 7$, dispara alerta crítica de deterioro inminente previa a la toma de gases arteriales.
- **REG-CLIN-01 (C1):** Alerta de soporte ventilatorio invasivo inmediato ante hipoxemia refractaria ($\text{SatO}_2 < 85\%$ o $\text{PaFi} < 150$).
- **REG-CLIN-01B:** Estratificación del Síndrome de Dificultad Respiratoria Aguda (SDRA) según los criterios de Berlín (2012).
- **REG-CLIN-02 (C2):** Evaluación multiorgánica mediante la escala SOFA (*Sequential Organ Failure Assessment*):
  - $\text{SOFA} \le 6$: Alta reversibilidad ($>80\%$ sobrevida estimada).
  - $7 \le \text{SOFA} \le 11$: Disfunción moderada-severa pero rescatable en UCI.
  - $\text{SOFA} \ge 12$: Falla multiorgánica crítica con alta mortalidad ($>50\%$).
- **REG-CLIN-03 (C3):** Identificación de **Futilidad Terapéutica** ($\text{SOFA} > 14$ + enfermedad terminal irreversible). Dictamina **Prioridad 4 (Cuidado Paliativo Integral)** conforme a la Ley 1733 de 2014 para evitar el encarnizamiento terapéutico (*distanasia*).
- **REG-CLIN-04:** Evaluación del estado neurológico mediante la Escala de Coma de Glasgow ($\text{Glasgow} \le 8 \to$ intubación orotraqueal para protección de vía aérea).
- **REG-CLIN-06:** Categorización de ingreso en 4 prioridades (P1: Crítico inestable, P2: Monitoreo intensivo, P3: Estable en sala general, P4: Paliativo digno).

### Agente 2: ResourceManagerAgent (Capacidad Hospitalaria)
Modela el estado de saturación del hospital y ejecuta la asignación física:
- **REG-REC-00 / 01 (R1):** Si $\text{Demanda P1} \le \text{Camas UCI Disponibles}$, aprueba asignación directa sin conflicto distributivo.
- **REG-REC-02 (R2):** Si $\text{Demanda P1} > \text{Camas UCI Disponibles}$, emite **Declaratoria de Escasez Crítica (Triaje de Catástrofe)** y transfiere el caso al Coordinador y al Auditor Bioético.
- **REG-REC-03:** Detección de escasez específica de ventiladores mecánicos.
- **REG-REC-04:** Asignación física estricta vinculada a camas y ventiladores disponibles.
- **REG-REC-05:** Direccionamiento de pacientes no-P1 a pisos de hospitalización, intermedios o cuidados paliativos.

### Agente 3: BioethicsAuditAgent (Auditoría Bioética y Legal Colombiana)
Actúa como un tribunal de garantías para evitar discriminación y asegurar el cumplimiento normativo:
- **REG-BIO-01 (Bloqueo de Edadismo):** Audita que la edad cronológica no sea factor de exclusión. Aplica una **prueba contrafáctica** (recalculando la asignación con edad normalizada). Si un adulto mayor resulta excluido únicamente por su edad cronológica, emite un **Veto Constitucional** basado en el Art. 13 de la Constitución Política y la Sentencia T-760 de 2008.
- **REG-BIO-02 (Equidad y Determinantes Sociales):** Detecta si las comorbilidades provienen de brechas socioeconómicas y objeta si el algoritmo penaliza doblemente a sujetos en condición de vulnerabilidad social o debilidad manifiesta.
- **REG-BIO-03 (Garantía de No Abandono - Ley 1751 de 2015):** Exige que a todo paciente P1 que no obtenga cama física se le asigne un plan obligatorio: soporte transitorio en área de expansión y remisión interinstitucional prioritaria vía Centro Regulador de Urgencias y Emergencias (CRUE). A los pacientes P4 les garantiza cuidado paliativo sin abandono.
- **REG-BIO-04 (Cláusula Human-in-the-Loop):** Conforme a la Ley 23 de 1981 y las recomendaciones éticas de la UNESCO (2021), ratifica que el dictamen es un soporte analítico y la decisión final corresponde indelegablemente al personal médico humano.
- **REG-BIO-06 (Adecuación del Esfuerzo):** Valida que la no admisión a UCI por futilidad no sea discriminación, sino adecuación del esfuerzo terapéutico conforme a la Ley Consuelo Devis Saavedra (Ley 1733 de 2014).
- **REG-BIO-07 (Habeas Data):** Certifica el cumplimiento de la Ley 1581 de 2012 para datos sensibles de salud mediante ejecución 100% local.

---

## 4. Marcos Éticos de Resolución y Motor de Desempate

En situaciones de escasez crítica, el `ConsensusCoordinator` permite contrastar en vivo los resultados bajo tres paradigmas filosóficos:

| Marco Ético | Escuela Filosófica | Criterio de Decisión | Crítica Teórica / Tensión |
| :--- | :--- | :--- | :--- |
| **Utilitarismo Clínico Estricto** | Consecuencialismo (Bentham, Mill) | Maximizar vidas y años de vida salvados ($\text{Puntaje} = 0.50 \cdot \text{Sobrevida} + 0.15 \cdot \text{PaFi} + 0.35 \cdot \text{Años} - \text{Comorbilidades}$). | Maximiza la eficiencia biológica, pero discrimina indirectamente a adultos mayores y enfermos crónicos (edadismo). |
| **Deontología e Igualitarismo** | Imperativo Categórico Kantiano | Igual dignidad intrínseca de toda vida humana. Desempate neutral por orden de llegada (FIFO) o sorteo auditable. | Respeta la igualdad formal, pero puede asignar el recurso a un paciente con $15\%$ de sobrevida dejando morir a uno con $90\%$. |
| **Consenso Constitucional Colombiano** | Principialismo Bioético + Estado Social de Derecho | Prioriza por reversibilidad fisiológica a corto plazo ($\text{SOFA}$). Ante empate técnico ($|\Delta\text{SOFA}| \le 1$), aplica acción afirmativa (protección reforzada a población vulnerable) y activa traslado CRUE. | Marco armónico con el ordenamiento jurídico colombiano (MinSalud / AMCI). |

---

## 5. Escenario Central de Demostración (3 Pacientes vs. 1 Cama)

El sistema incluye el botón de carga del caso clínico de prueba:

1. **Carlos Mendoza (34 años):** $\text{SatO}_2 = 79\%$, $\text{PaFi} = 110$, $\text{SOFA} = 6$, requiere ventilador, llegó hace 10 min. (Excelente pronóstico).
2. **Elena Restrepo (76 años):** $\text{SatO}_2 = 81\%$, $\text{PaFi} = 125$, $\text{SOFA} = 7$, requiere ventilador, madre comunitaria rural vulnerable, llegó hace 45 min (primera en llegar).
3. **Jorge Valencia (58 años):** $\text{SatO}_2 = 74\%$, $\text{PaFi} = 85$, $\text{SOFA} = 16$, cáncer pancreático metastásico terminal, llegó hace 30 min.

### Resultados de la Deliberación Inicial (t = 0h):
- **Jorge Valencia:** Descartado para cama UCI en todos los marcos por **Futilidad Terapéutica (Prioridad 4)**. Se le activa protocolo de Cuidado Paliativo Digno para evitar distanasia (Ley 1733 de 2014).
- **Competencia directa Carlos vs. Elena (1 Cama):**
  - **Bajo Utilitarismo:** Se asigna a **Carlos Mendoza** (34 años) por mayor expectativa de años de vida. El `BioethicsAuditAgent` emite una **Objeción Bioética** por sesgo de edad.
  - **Bajo Deontología:** Se asigna a **Elena Restrepo** por criterio de justicia distributiva formal (primera en llegar, FIFO).
  - **Bajo Consenso Constitucional Colombiano:** Se identifica **empate técnico** ($|\text{SOFA}_A - \text{SOFA}_B| = |6 - 7| = 1 \le 1$). Se aplica **Protección Reforzada Constitucional** (Art. 13 y 43 de la Constitución) y se asigna la cama a **Elena Restrepo**, al tiempo que se tramita la remisión inmediata de Carlos Mendoza vía CRUE con soporte transitorio.

---

## 6. Motor de Simulación Dinámica Temporal Multi-Ronda (0h -> 6h -> 12h -> 24h)

El sistema trasciende la evaluación estática tradicional incorporando un motor dinámico determinista de seguimiento clínico:

1. **t = 0h (Ingreso):** Evaluación basal con tamizaje NEWS2, clasificación de SDRA y deliberación ética con asignación de recursos.
2. **t = 6h:** Monitoreo hemodinámico y ventilatorio en prono. El paciente en UCI continúa en tratamiento; el paciente en lista de espera recibe soporte transitorio en área de expansión.
3. **t = 12h (Liberación de Cama y Reasignación Auditada):**
   - El paciente en UCI muestra franca mejoría clínica post-terapia ($\text{SOFA} \le 3$, extubación exitosa) $\to$ se ordena traslado a **Cuidados Intermedios** y **SE LIBERA 1 CAMA UCI Y 1 VENTILADOR**.
   - El comité delibera automáticamente ante el recurso liberado y reasigna la cama de forma determinista y auditada al paciente que permanecía en lista de espera priorizada (reduciendo la cola de espera de 1 a 0).
4. **t = 24h:** Consolidación de la mejoría del nuevo paciente en UCI y estabilidad del paciente en cuidados intermedios.

---

## 7. Instalación y Ejecución

### Requisitos Previos
- Python 3.10 o superior instalado.

### Pasos de Instalación
1. Clonar o abrir la carpeta del proyecto en una terminal:
   ```bash
   cd triage_multiagent_system
   ```

2. Instalar las dependencias mínimas:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar la validación integral por consola (verifica sintaxis, ausencia de dependencias circulares y casos de prueba):
   ```bash
   python validar_sistema.py
   ```

4. Iniciar la aplicación web interactiva en Streamlit:
   ```bash
   streamlit run app.py
   ```

---

## 8. Guion Recomendado para el Video de Demostración (10 Minutos)

- **Minuto 0:00 - 1:30 | Introducción y Planteamiento del Problema:**
  - Presentar la institución (Universidad Católica Luis Amigó) y el objetivo del proyecto.
  - Explicar por qué los modelos de Machine Learning de caja negra no son éticamente viables en este contexto y por qué se optó por IA Simbólica Multiagente con trazabilidad total.
- **Minuto 1:30 - 3:00 | Arquitectura de los Agentes, NEWS2 y Panel Asistencial:**
  - Describir el rol del `ClinicalTriageAgent` (fisiología pura y tamizaje NEWS2), `ResourceManagerAgent` (capacidad e inventario) y `BioethicsAuditAgent` (auditoría constitucional y no discriminación).
  - Mostrar el nuevo diseño de nivel clínico hospitalario (CDSS-XAI) con fichas clínicas de urgencias y tarjetas NEWS2 (Royal College of Physicians 2017).
- **Minuto 3:00 - 6:00 | Demostración en Vivo del Caso Crítico y Dilemas Éticos:**
  - Presionar el botón *"🚨 Cargar Caso de Dilema Crítico"* (3 pacientes vs. 1 cama).
  - En la **Pestaña 1 (Tablero de Pacientes)**: Mostrar cómo Jorge Valencia es clasificado en Prioridad 4 (Paliativo) por futilidad, evitando la distanasia.
  - En la **Pestaña 3 (Dilemas Éticos Comparados)**: Demostrar el cambio en tiempo real al alternar entre Utilitarismo (gana Carlos Mendoza), Deontología (gana Elena Restrepo por llegada) y Consenso Constitucional (gana Elena por empate técnico + protección reforzada).
- **Minuto 6:00 - 8:00 | Simulación Temporal Multi-Ronda (0h -> 24h) y Rotación de Camas:**
  - Activar el toggle *"⏱️ Modo Simulación Dinámica Temporal"* en la barra lateral.
  - Mover el selector temporal a **12 Horas**: Observar cómo Elena Restrepo muestra franca mejoría ($\text{SOFA} \le 3$, extubada) y es trasladada a Cuidados Intermedios, liberando la cama UCI y el ventilador.
  - Mostrar cómo el comité reasigna inmediatamente la cama liberada a Carlos Mendoza (admitido desde la lista de espera CRUE).
  - Mostrar el gráfico interactivo de flujo asistencial en Plotly.
- **Minuto 8:00 - 10:00 | Cierre Normativo, Explicabilidad y Responsabilidad:**
  - Mostrar las trazas XAI en la **Pestaña 4** y el Marco Legal en la **Pestaña 5** con la matriz de responsabilidad civil (Desarrollador vs. Hospital vs. Médico).
  - Enfatizar el principio *Human-in-the-Loop* de la Ley 23 de 1981 y las directrices de la UNESCO para la ética en IA.

---

## 9. Marco Legal y Referencias Normativas

- **Constitución Política de Colombia (1991):** Artículos 1 (Dignidad Humana), 13 (Igualdad y Protección a Población Vulnerable), 43, 46 y 49.
- **Corte Constitucional de Colombia:** Sentencia T-760 de 2008 (Salud como derecho fundamental y prohibición de discriminación arbitraria).
- **Ley Estatutaria 1751 de 2015:** Artículos 6, 10, 11 y 14 (Prohibición de negación de urgencias y garantía de remisión).
- **Ley 23 de 1981:** Código de Ética Médica (El acto médico es indelegable).
- **Ley 1733 de 2014:** Cuidados Paliativos y adecuación del esfuerzo terapéutico.
- **Ley 1581 de 2012:** Protección de Datos Personales (Tratamiento de datos sensibles).
- **UNESCO (2021):** Recomendación sobre la Ética de la Inteligencia Artificial.
