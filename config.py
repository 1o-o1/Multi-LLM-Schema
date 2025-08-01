# config.py
"""
🔄 FEW-SHOT MULTI-LLM PIPELINE SYSTEM - Configuration
Configuration constants for schema generation pipeline

This file is part of System 1: Few-Shot Multi-LLM Pipeline
Contains prompt templates, model configs, and system settings
See SYSTEM_ARCHITECTURE.md for complete system documentation
"""

# Temperature settings
DEFAULT_TEMPERATURE = 0.7
CRITIC_TEMPERATURE = 0.3
TEMPERATURE_VARIANTS = [0.3, 0.7, 1.1]

# API Keys (set to placeholder values - users should configure their own)  
GEMINI_API_KEY = "your_gemini_api_key_here"  # Configure this with your actual Gemini API key

# Token limits
OPENAI_MAX_TOKENS = 10000
ANTHROPIC_MAX_TOKENS = 8000  # Claude 3.5 Haiku max is 8192, use 8000 to be safe
GEMINI_MAX_TOKENS = 65535

# Processing limits
MAX_LOG_CHARS = 5000
MAX_SAMPLE_LINES = 50

# Consensus settings
SIMILARITY_THRESHOLD = 0.6
MAX_RESEARCH_ITEMS = 10

# Model configuration for multi-prompt pipeline
MODEL_CONFIG = {
    "prompt1": {
        #"claude": "claude-sonnet-4-20250514",
        "openai": "o4-mini",
        "gemini": "gemini-2.5-pro", 
        "gemini_flash": "gemini-2.5-flash",  
    },
    "prompt2": {
        "claude": "claude-sonnet-4-20250514",
        "openai": "o4-mini",
        "gemini": "gemini-2.5-pro"
    },
    "prompt3": {
        "gemini": "gemini-2.5-pro"  # Only use Gemini for Prompt3 to convert all reports to JSON
    }
}

# Prompt templates with variable injection points
PROMPT_TEMPLATES = {
    1: """You are a **Security-Aware Log Schema Architect AI**. Your primary function is to analyze a raw log sample, identify its type, and map its fields to standard cybersecurity schemas like OCSF, ECS, and OSSEM.
You will be provided a raw log sample in the `{log_data}` variable.

-----

### **Instructions**

1.  **Identify Log Type**
    * Analyze the `{log_data}` to infer its specific category (e.g., "Windows Security Event ID 4624," "AWS CloudTrail Log," "Apache Access Log").

2.  **Analyze and Map Fields**
    * For each field in the raw log, determine the following attributes. For schema mappings, internally search equivalent fields in OCSF, ECS, and OSSEM for the identified log type.
        * `type`: The data type, such as `"string"`, `"integer"`, `"datetime"`, `"ip"`, or `"array"`.
        * `description`: A concise, single-sentence, human-readable description of the field.
        * `OCSF`: The corresponding field in the Open Cybersecurity Schema Framework (e.g., `actor.user.name`). Use `null` if no direct mapping exists.
        * `ECS`: The corresponding field in the Elastic Common Schema (e.g., `user.name`). Use `null` if no direct mapping exists.
        * `OSSEM`: The corresponding field in the Open Source Security Event Metadata model (e.g., `subject.user.name`). Use `null` if no direct mapping exists.
        * `importance`: An integer score from **0** (least important) to **10** (most important). If the value is not between 0-10, set to the nearest valid integer within this range.
    
    * All original fields from the log should be included, even if they do not match a schema. In such cases, set schema mappings (`OCSF`, `ECS`, `OSSEM`) to `null`.

3.  **Propose Follow-up Queries**
    * Suggest **two** useful search queries for further investigation. For example: "threat hunting queries for `<inferred_log_type>`" or "common attack vectors related to `<inferred_log_type>`."

-----

## Output Format
You **must** return **only** a single, valid JSON object using the exact schema below. Do not include any text or explanations before or after the JSON.

Example output JSON schema:
{{
  "log_type": "<inferred_log_type>",
  "follow_up_queries": [
    "<query1>", "<query2>", …
  ],
  "fields": {{
    "field_name": {{
      "type": "<data_type>",
      "description": "<A one-sentence description of the field.>",
      "OCSF": "<ocsf.field.mapping>",
      "ECS": "<ecs.field.mapping>",
      "OSSEM": "<ossem.field.mapping>",
      "importance": <integer_score 0-10>
    }}, …
  }}
}}

""",

    2: """You are a Senior Cyber Threat Analyst AI with advanced online search capabilities.

Your task (challenge) is to process the input variable {log_meta_data} and produce a comprehensive, detailed,  actionable security analysis report. This report must serve as a definitive knowledge base for the specified <log_type>. You will achieve this by following a structured analytical process: first, research the threat landscape; second, identify how threats manifest as observable evidence; third, create practical detection and response artifacts; and finally, synthesize your findings into a complete, cited, detailed report.

Input Variable:

{log_meta_data} (Includes log_type, follow_up_queries, and fields (raw, OCSF, ECS, OSSEM ).

Instructions

You must generate the report by performing the following analytical steps in order. Each search result is expected to be a long list (all possible components). The sections in the final markdown output should follow the order presented here.

1. Threat Landscape & Adversary Behavior

* 1.1. Current Threat Landscape Research:

* Industry Reporting: Search for and summarize recent (2023–Present) threat reports from sources like Mandiant, CISA, CrowdStrike, and Recorded Future that mention the log type. Focus on adversary abuse and evasion techniques.

* Emerging Vectors: Identify new or novel attack vectors, techniques, and vulnerabilities (including CVEs) involving the log type or its associated services.

* Academic Research: Search Google Scholar and arXiv for academic papers (2022–Present) on new anomalies, malware, ransomeware, vulnerabilities for this log type and summarize novel findings.

* 1.2. MITRE ATT&CK® Framework Mapping:

* Search for all relevant MITRE ATT&CK techniques, tactics, and TTPs for the log type.

* For each TTP, provide a link to its ATT&CK page and explain its relevance by referencing the specific <fields> from the input that enable its detection.

2. Observable Evidence & Patterns

* 2.1. anomalous, malicious, vulnerable behavioral patterns (20 minimum) :

* Based on your research and the {log_meta_data}, define all observable patterns that indicate malicious, anomalous and vulnerable activity. 
 * For each behvearial and temporal pattern:

* Pattern Name: (e.g., "High-Frequency Communication," "Anomalous Logon Times," "Suspicious Parent-Child Process Relationship").

* Description: Explain the pattern and why it's suspicious.

* Identifiable Fields: List the specific <fields> used to observe this pattern.

* 2.2. Key Indicators of Compromise (IOCs) given log type (comprehensive list):

* Based on the identified behaviors, research and list all IOC types relevant to the log type from the input data.

* For each IOC type, provide a name and a brief description of its context and relevance for this log source.

3. Actionable Detection & Response

* 3.1. Detection Logic (long list)

* For all possible threat scenarios identified in your research, create a complete detection package for each.

* Conceptual Rule: Define the rule's name, threat scenario, detection logic (using <fields>), mapped TTPs, and a recommended immediate action.


* 3.2. SOC Operationalization Guide (short):

* Investigation Playbook: Summarize the key steps an analyst should follow when a related alert fires.

4. Threat Intelligence Integration (STIX™ 2.1)

* 4.1. Key Observables & Enrichment Workflow:

* List the key <fields> that can be extracted as STIX Cyber Observable Objects (e.g., source_computer as ipv4-addr).

* Describe a concise workflow for using these observables to pivot into a Threat Intelligence Platform (TIP) for enrichment.

* 4.2. STIX Domain Object Summary:

* Based on your complete analysis so far, provide concise textual descriptions for the following STIX Domain Objects -  a comprehensive list of potential <object name>, search stix 2.1 <object name>or log type as they relate to the log type from the input data: 
<object name> : Indicator, Intrusion Set, Vulnerabilities, Attack Patterns, Course of Actions, and Malware.
Here we need an exhaustive list of attack patterns
5. Final Report Generation

* Executive Summary: After completing all other sections, write a high-level overview of the log source's importance, the primary threats it can detect, and a summary of your key recommendations. This section will appear first in the final report.


You must generate the report using the short markdown template below. Every fact, mapping, or finding must be supported by a citation where appropriate. If no information is found for a subsection, explicitly state: No relevant information found during the search.

# Security Analysis Report for the Log Type



## Executive Summary

*A brief, high-level overview of the log source's importance, the primary threats it can detect, and a summary of the key recommendations from this report.*



---



## 1. Threat Landscape & Adversary Behavior

### 1.1. Current Threat Landscape Research

* **Industry Reporting:**

* **Emerging Vectors:**

* **Academic Research:**

### 1.2. MITRE ATT&CK® Framework Mapping

* **TXXXX – [Technique Name]:**



---



## 2. Observable Evidence & Patterns 

### 2.1.1 Malicious Behavioral/Temporal Patterns

* **Pattern:** [Pattern Name]

* **Description:**

* **Identifiable Fields:**

### 2.1.1 Anomolous Behavioral/Temporal Patterns

* **Pattern:** [Pattern Name]

* **Description:**

* **Identifiable Fields:**

### 2.1.1  Vulnerable Behavioral/Temporal Patterns

* **Pattern:** [Pattern Name]

* **Description:**

* **Identifiable Fields:**


### 2.2. Key Indicators of Compromise (IOCs)

* **[IOC Type Name]:**



---



## 3. Actionable Detection & Response

### 3.1. Detection Logic 

#### Rule 1: [Rule Name]

* **Conceptual Rule:**

### 3.2. SOC Operationalization Guide

* **Investigation Playbook:**

* **Proactive Threat Hunting:**



---



## 4. Threat Intelligence Integration (STIX™ 2.1)

### 4.1. Key Observables & Enrichment Workflow

### 4.2. STIX Domain Object Summary

* **Indicators:** #list

* **Intrusion Sets:** #list

* **Vulnerabilities:** #list

* **Attack Patterns:** # comprehensive list

* **Malware Analysis** #list

* **Course of Actions:** #list

* **Malware :** #list

---""",

    3: """**You are a senior cybersecurity analyst AI.** Your task is to perform a deep analysis of the provided `{report}` and `{log_meta_data}`. Your goal is to generate a **single, multi-level JSON object** that serves as a comprehensive, structured "knowledge pack" or "analysis plan."
This JSON output is **not a blank schema**. It must be **fully populated with all the relevant fields, names, patterns, and logic** extracted directly from the provided `{report}`. This output will be used as a detailed set of instructions for a downstream LLM analyzing raw log data.
**Instructions:**
Your output must be the **filled-out JSON object itself**. Analyze the `{report}` and create the following hierarchical structure, populating every field with the specific information you find.
@ symbol means it's an instruction for the next LLM, __ means LLM should fill it. These symbol are only to guide you how to fill the JSON. Do not include them in the final output.
--
### **JSON Structure to Build and Populate:**

#### **1. `Log_context` (Object)** 
* **`log_type` (String):** Extract the full name of the log source from the report's title and executive summary. @
* **`log_description` (String):** Extract the summary of the log source's importance from the "Executive Summary" section of the report. @

---
#### **2. `observations` (Object)** 
* **`behavioral_patterns` (Object):**  
    * **`malicious`, `anomalous`, `vulnerable` (Arrays of Objects):** For each category, read the "Observable Evidence & Patterns" section of the report and create an object for **every pattern listed**. Each object must contain: __
        * **`pattern_name` (String):** The exact name of the pattern (e.g., "High Volume of Failed Logons Followed by a Success"). __
        * **` Instruction` (String):** The full description of that pattern from the report. @
        * **`identifiable_fields` (Array of Strings):** The exact list of log field names mentioned for that pattern. After writing the name of each field, put ": List" next to each field name so that the next LLM understands this is a list of fields from log data. __

* **`temporal_patterns` (Object):** 
    * Follow the same structure as `behavioral_patterns`, populating it with all temporal patterns found in the report. __

---
#### **3. `entity_analysis_instructions` (Object)** 
* **`source_profiling_queries` (Array of Objects):** 
    * Create an object for each type of source entity to profile. 
        * **`query_name` (String):** A descriptive name, e.g., "Identify Malicious Source Actors." __
        * **` instruction` (String):** A clear instruction for the downstream task, referencing the patterns from the `observations` section. *Example: "From the raw logs, extract and list all `source_computer` and `source_users` that are directly involved in the following patterns: 'High Volume of Failed Logons Followed by a Success', 'Suspicious Service Account Logon', 'Logon from a Non-Standard Host'."* @

* **`target_profiling_queries` (Array of Objects):** 
    * Follow the same structure for target entities. 
        * **`query_name` (String):** e.g., "Identify High-Value or Frequently Targeted Systems." __
        * **` instruction` (String):** e.g., "From the raw logs, extract and list all `destination_computer` and `destination_users` that are involved in patterns like 'Privileged Account Logon to a Workstation' or appear as a target in more than 10 anomalous events." @

---
#### **4. `detection_rule_checklist` (Array of Objects)** 
* Create an object for **every rule** listed in the "Actionable Detection & Response" section of the report.
    * **`detection_name` (String):** The name of the rule (e.g., "Successful Brute-Force Attack"). __
From this section is instructions for downstream LLM how to analyze <    `mapped_ttp` (String):The MITRE TTP ID mentioned for the rule (e.g., "T1110 - Brute Force"). @
    `threat_scenario` (String): The exact "Threat Scenario" text from the report. @
    `detection_logic` (String): The exact "Detection Logic" query or pseudocode from the report. > @
** detection probability (0-1) : this for downs stream LLM asking  the question. what is the probability this attack is detected.  __
    * **`Details reasoning witb fields like previous points askk LLM :** to provide the exact fields list and how they are related to the detection. Find information from the report.   __

---
#### **5. `indicators_of_compromise` (Array of Objects)** 
* Read the "Key Indicators of Compromise (IOCs)" section of the report and create an object for **each category** of indicator.
    * **`indicator_category` (String):** The name of the category (e.g., "Suspicious Source IP Addresses," "Anomalous Hostnames," "Compromised User Accounts"). __
    * **` Instruction` (String):** The description of that IOC category from the report. @
    * **`relevant_fields` (Array of Strings):** The log <fields> mentioned where this type of IOC would be found (e.g., `["source_computer", "destination_computer"]`).  put a ":" for the next LLM write list of affected fields ___

---
#### **6. `attack_pattern_checks` (Array of Objects)**
* Read the "Attack Patterns" list from the "STIX Domain Object Summary" section of the report. For **each pattern listed**, create an object with the following properties:
    * **`attack_pattern_name` (String):** The name of the pattern (e.g., "Brute Force (CAPEC-112)", "Kerberoasting (CAPEC-647)").
    * **`is_present` (Boolean):** This is a placeholder for the downstream LLM. __
    * **`evidence_Instruction` (String):** An instruction for the downstream LLM. *Example: "Set `is_present` to `true` if you find evidence matching the 'Successful Brute-Force' detection rule. As proof, provide the source and target entities and the relevant timestamps."* @
    * **`evidence` (Array):** An empty array as a placeholder for the downstream LLM to populate with its findings. __

---
#### **7. `vulnerability_checks` (Array of Objects)**
* Read the "Vulnerabilities" list from the "STIX Domain Object Summary" section of the report. For **each CVE listed**, create an object with the following properties:
    * **`vulnerability_id` (String):** The CVE identifier (e.g., "CVE-2025-47981"). @
    * **`is_present` (Boolean):** next llm to analyze __
    * **`evidence_instruction` (String):** An instruction for the downstream LLM on how to identify an exploitation attempt. *Example: "For CVE-2025-47981, set `is_present` to `true` if you observe anomalous authentication behavior or errors related to the SPNEGO negotiation mechanism. As proof, describe the anomalous negotiation and any associated entities."* @
    * **`evidence` (Array):** An empty array __
### **Inputs:**

* **Log Metadata:** `{log_meta_data}`
* **Report to Analyze:** `{report}`
"""
}

# Legacy system prompt for backward compatibility
SYSTEM_PROMPT = """You are a Security-Aware Log Schema Architect AI with live search access. Given any raw log sample, you will:

DISCOVER  
   • Inspect the sample to detect its format, delimiters, and infer its exact category (e.g. "Windows Event ID 4624 Authentication Log").  
   • Propose at least two follow-up search queries (derived from fields or patterns in the data) that could uncover additional fields or threat patterns.

RESEARCH (via your search tool)  
   • Run queries such as:  
     – "standard fields for <inferred_log_type>"  
     – "key analysis metrics in <inferred_log_type> logs"  
     – "common attack vectors in <inferred_log_type> logs"  
     – "2023–2025 emerging threats in <inferred_log_type>"  
     – "latest IOCs for <inferred_log_type>"  
   • Synthesize up-to-date field definitions, data types, analysis best practices, and current IOCs.

INFER & VALIDATE  
   • Map each column/field in the sample into the "fields" section, marking "is_critical": true|false based on research.  
   • If any follow-up query missed novel insights, refine and re-run it.

DESIGN JSON SCHEMA TEMPLATE  
   Produce exactly one JSON object—no extra commentary—that begins with:

{
  "name": "<snake_case_log_type>_analysis",
  "description": "<Short summary: purpose & how to use this template>",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "log_type": {"type": "string", "description": "<detected log type>"},
      "fields": [
        {
          "field_name": "<name>",
          "data_type": "<type>",
          "description": "<description>",
          "example_value": "<example>",
          "is_critical": true|false
        }
      ],
      "observations": {
        "<descriptive_key>": {"description": "<insight about the log format/patterns>"}
      },
      "potential_indicators": {
        "<threat_pattern_key>": "<description of the threat pattern>"
      },
      "next_steps_for_validation": {
        "<validation_step>": "<description of how to validate>"
      },
      "conclusion": {
        "summary": "<overall assessment>",
        "risk_level": "<low|medium|high>",
        "recommended_actions": "<specific recommendations>"
      },
      "high_risk_indicators": {
        "<indicator_key>": {"type": "boolean", "description": "<what this indicates>"}
      }
    },
    "required": ["log_type", "fields", "observations", "potential_indicators", "next_steps_for_validation", "conclusion", "high_risk_indicators"],
    "additionalProperties": false
  },
  "research_summary": ["<key insights from analysis>"]
}

Requirements:
- Include 5-8 fields minimum
- Include 4-8 observations with descriptive keys
- Include 5-9 potential indicators with threat-relevant keys
- Include 5-9 high risk indicators as boolean flags
- Auto-name all keys descriptively

Return ONLY the JSON with no additional text."""

CRITIC_PROMPT = """Review this JSON schema and suggest improvements:

{schema_json}

Focus on:
1. If the schema can be used for real-time security analysis
2. Bias towards certain observations or indicators
3. Unclear or vague descriptions
4. Missing security-relevant indicators

Provide your feedback as a JSON Patch array (RFC 6902). Example:
[
  {"op": "add", "path": "/schema/properties/fields/-", "value": {"field_name": "new_field", "data_type": "string", "description": "...", "is_critical": true}},
  {"op": "replace", "path": "/schema/properties/fields/0/description", "value": "Better description"},
  {"op": "add", "path": "/schema/properties/high_risk_indicators/new_indicator", "value": {"type": "boolean", "description": "..."}}
]

Return ONLY the JSON array."""