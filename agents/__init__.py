"""
Paquete ``agents``: agentes simbólicos del Comité Clínico-Bioético.

    BaseAgent              → contrato abstracto con registro de trazas XAI
    ClinicalTriageAgent    → Agente 1: severidad y reversibilidad fisiológica
    ResourceManagerAgent   → Agente 2: inventario UCI y detección de colapso
    BioethicsAuditAgent    → Agente 3: auditoría ética y marco legal colombiano
"""

from agents.base_agent import BaseAgent
from agents.clinical_agent import ClinicalTriageAgent
from agents.resource_agent import ResourceManagerAgent
from agents.bioethics_agent import BioethicsAuditAgent

__all__ = ["BaseAgent", "ClinicalTriageAgent", "ResourceManagerAgent", "BioethicsAuditAgent"]
