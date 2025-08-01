## Security Analysis Report for the Log Type

## Executive Summary

Windows Authentication Logs are one of the most critical data sources for detecting and investigating adversary behavior, including credential theft, lateral movement, and persistence techniques. This report synthesizes industry and academic research, maps key MITRE ATT&CK® techniques, enumerates observable patterns in authentication events, and provides actionable detection and response guidance. Key recommendations include deploying high-fidelity detection rules for password spraying (T1110.003), pass-the-hash (T1550.002), and Kerberoasting (T1558.003); integrating threat intelligence enrichment of user and host observables via STIX™ 2.1; and operationalizing a SOC playbook to rapidly triage anomalous authentication activities.

---

## 1. Threat Landscape & Adversary Behavior

### 1.1. Current Threat Landscape Research

* **Industry Reporting:**
  * Mandiant’s M-Trends 2023 report highlights Windows as the most targeted OS and notes a rising use of living-off-the-land tactics—legitimate tools such as RDP and PowerShell—to avoid EDR detection ([securityweek.com](https://www.securityweek.com/mandiant-2023-m-trends-report-provides-factual-analysis-of-emerging-threat-trends/?utm_source=chatgpt.com))([techtarget.com](https://www.techtarget.com/searchsecurity/news/366587481/Mandiant-Ransomware-investigations-up-20-percent?utm_source=chatgpt.com)).
  * CISA AA24-046A documents threat actors leveraging compromised credentials (T1078.002) of former employees to conduct VPN access and LDAP reconnaissance against domain controllers, enabling lateral movement without MFA ([cisa.gov](https://www.cisa.gov/news-events/cybersecurity-advisories/aa24-046a?utm_source=chatgpt.com)).
  * CISA AA24-290A describes Iranian-aligned actors performing brute-force (T1110), password spraying (T1110.003), Kerberos SPN enumeration (T1558.003), RDP lateral movement (T1021.001), and MFA fatigue (T1621) attacks in 2024 ([cisa.gov](https://www.cisa.gov/news-events/cybersecurity-advisories/aa24-290a?utm_source=chatgpt.com)).

* **Emerging Vectors:**
  * “MFA fatigue” or push-bombing attacks have surged, exploiting user-prompt overload to bypass multifactor defenses (T1621) ([cisa.gov](https://www.cisa.gov/news-events/cybersecurity-advisories/aa24-290a?utm_source=chatgpt.com)).
  * Despite deprecation efforts, NTLM remains widely supported; adversaries continue to use pass-the-hash (T1550.002) and over-pass-the-hash (T1550.004) techniques against misconfigured environments.
  * Legacy Active Directory features (e.g. Credential Roaming) remain vulnerable to LDAP-based CVE-2022-30170 style file-write attacks, underscoring the need to monitor unusual LDAP attribute access ([cloud.google.com](https://cloud.google.com/blog/topics/threat-intelligence/apt29-windows-credential-roaming?utm_source=chatgpt.com)).

* **Academic Research:**
  * A Securonix case study proposes a hierarchical Bayesian model to detect rare deviations in per-user authentication counts, achieving stable anomaly detection on Windows event IDs 4624/4625 ([securonix.com](https://www.securonix.com/blog/detecting-windows-authentications-anomalies-with-hierarchical-bayesian-framework/?utm_source=chatgpt.com)).
  * IIETA published a statistical-rarity assessment approach using feature hashing on Windows Event 4624 to flag uncommon authentication patterns indicative of RDP-based lateral movement ([iieta.org](https://iieta.org/journals/ijsse/paper/10.18280/ijsse.130501?utm_source=chatgpt.com)).
  * Multiple works (e.g., LogGD) demonstrate the applicability of graph-neural networks for log anomaly detection, but emphasize the need for methods tailored to Windows authentication semantics ([arxiv.org](https://arxiv.org/abs/2209.07869?utm_source=chatgpt.com)).

### 1.2. MITRE ATT&CK® Framework Mapping

* **T1078 – Valid Accounts**  
  Link: https://attack.mitre.org/techniques/T1078  
  Relevance: Monitors `source_users` and `statuses` for use of legitimate credentials from unexpected `source_computer` hosts or disabled accounts.  

* **T1110.003 – Password Spraying**  
  Link: https://attack.mitre.org/techniques/T1110/003  
  Relevance: Detect repeated `statuses: Failure` across multiple `destination_users` from single `source_computer` within a short `times` window.  

* **T1550.002 – Pass the Hash**  
  Link: https://attack.mitre.org/techniques/T1550/002  
  Relevance: Identify successful `auth_type: NTLM` network (`logon_type`) logons without preceding Kerberos authentication on sensitive `destination_computer`.  

* **T1558.003 – Kerberos Service Ticket Extraction (Kerberoasting)**  
  Link: https://attack.mitre.org/techniques/T1558/003  
  Relevance: Observe Kerberos (`auth_type`) requests for service principal accounts in `destination_users` outside normal maintenance windows.  

* **T1003.001 – OS Credential Dumping: LSASS Memory**  
  Link: https://attack.mitre.org/techniques/T1003/001  
  Relevance: Not directly logged in authentication events but often indicated by subsequent machine-account logons (`source_computer`) with unusual `auth_type`.  

* **T1021.001 – Remote Desktop Protocol**  
  Link: https://attack.mitre.org/techniques/T1021/001  
  Relevance: Detect `logon_type: 10 (Interactive)` with remote `source_computer` not matching `destination_computer` IP ranges.  

* **T1621 – Multi-Factor Authentication Request Generation**  
  Link: https://attack.mitre.org/techniques/T1621  
  Relevance: Identify high volumes of authentication (`orientations: logon`) attempts with `statuses: Failure` potentially reflecting MFA-push bombardment.  

* **T1110.001 – Brute Force**  
  Link: https://attack.mitre.org/techniques/T1110/001  
  Relevance: Multiple consecutive `statuses: Failure` for a single `destination_users` from the same `source_computer`.  

* **T1556.006 – Modify Authentication Process: Multi-Factor Authentication**  
  Link: https://attack.mitre.org/techniques/T1556/006  
  Relevance: Look for successful `orientations: logon` events for new `source_computer` registrations under existing `source_users` on ADFS/Okta.  

* **T1078.004 – Valid Accounts: Cloud Accounts**  
  Link: https://attack.mitre.org/techniques/T1078/004  
  Relevance: Windows authentication logs correlated with SSO or Azure AD logins for `source_users` on cloud-hosted `destination_computer`.  

---

## 2. Observable Evidence & Patterns 

### 2.1.1 Malicious Behavioral/Temporal Patterns

1. **Pattern:** Password Spraying  
   **Description:** Broad targeting of multiple accounts from one host with low-frequency failures to evade lockout.  
   **Identifiable Fields:** source_computer, destination_users, statuses, times  

2. **Pattern:** Brute Force Attack  
   **Description:** Rapid-fire authentication failures against a single account indicating credential guessing.  
   **Identifiable Fields:** source_computer, destination_users, statuses, times  

3. **Pattern:** Pass-the-Hash  
   **Description:** Successful NTLM authentication (`auth_type`) on network logons (`logon_type`) without Kerberos fallback.  
   **Identifiable Fields:** auth_type, logon_type, source_users, destination_computer  

4. **Pattern:** Kerberoasting  
   **Description:** Unusually high Kerberos service ticket requests for service accounts outside normal usage periods.  
   **Identifiable Fields:** auth_type, destination_users, times  

5. **Pattern:** Pass-the-Ticket  
   **Description:** Multiple Kerberos (`auth_type`) logons for high-privilege accounts without prior ticket-granting ticket requests.  
   **Identifiable Fields:** auth_type, source_users, statuses  

6. **Pattern:** RDP-based Lateral Movement  
   **Description:** Interactive logons (`logon_type: 10`) from remote hosts not normally accessed by user.  
   **Identifiable Fields:** logon_type, source_computer, destination_computer, times  

7. **Pattern:** Golden Ticket Use  
   **Description:** Kerberos logons (`auth_type`) for non-existent or service accounts with no corresponding TGT events.  
   **Identifiable Fields:** auth_type, destination_users, statuses  

### 2.1.2 Anomalous Behavioral/Temporal Patterns

1. **Pattern:** Impossible Travel  
   **Description:** Logons for same user from geographically disparate hosts within implausible timeframes.  
   **Identifiable Fields:** source_computer, source_users, times  

2. **Pattern:** Unusual Protocol Usage  
   **Description:** NTLM logon events in environments locked to Kerberos-only authentication.  
   **Identifiable Fields:** auth_type, source_users, times  

3. **Pattern:** Out-of-Hours Logons  
   **Description:** Successful logons occurring outside established business hours for a given user.  
   **Identifiable Fields:** times, source_users  

4. **Pattern:** Service Account Interactive Logon  
   **Description:** Non-service hosts showing interactive (`logon_type: 2`) logons by service accounts.  
   **Identifiable Fields:** destination_users, logon_type, destination_computer  

5. **Pattern:** Account Lockout Storm  
   **Description:** Multiple failure events followed by a lockout (event ID 4740) within a short period.  
   **Identifiable Fields:** statuses, source_users, times  

6. **Pattern:** New Host Access  
   **Description:** First-time successful logon by a user to a host in their typical domain scope.  
   **Identifiable Fields:** source_computer, source_users, times  

7. **Pattern:** Orphaned Logoff  
   **Description:** Logon events with no subsequent logoff (`orientations`) indicating potential persistence.  
   **Identifiable Fields:** orientations, times  

### 2.1.3 Vulnerable Behavioral/Temporal Patterns

1. **Pattern:** Legacy Protocol Fall-back  
   **Description:** Use of LM or NTLMv1 protocols despite policy disabling them.  
   **Identifiable Fields:** auth_type, statuses  

2. **Pattern:** Shared Account Across Hosts  
   **Description:** Same account logging into multiple unrelated hosts simultaneously.  
   **Identifiable Fields:** source_computer, source_users, times  

3. **Pattern:** Disabled Account Attempts  
   **Description:** Authentication failures (`statuses: Failure`) for known-disabled accounts.  
   **Identifiable Fields:** source_users, statuses  

4. **Pattern:** Excessive Failure Noise  
   **Description:** High volume of failed logons from one host without any successes, indicating misconfiguration or reconnaissance.  
   **Identifiable Fields:** source_computer, statuses, times  

5. **Pattern:** Default Credential Usage  
   **Description:** Attempts to log on with default or well-known account names (e.g., “Administrator”, “Guest”).  
   **Identifiable Fields:** destination_users, statuses  

6. **Pattern:** Non-Rotating Service Account Logon  
   **Description:** Long-lived service accounts authenticating interactively at irregular intervals.  
   **Identifiable Fields:** destination_users, times, logon_type  

### 2.2. Key Indicators of Compromise (IOCs)

* **Source Hostnames:** Unrecognized or unusual `source_computer` identifiers may indicate attacker-controlled systems.  
* **Destination Hostnames:** Access attempts to critical servers (`destination_computer`), especially domain controllers.  
* **Usernames:** Compromised or service accounts (`source_users`/`destination_users`) involved in anomalous logons.  
* **Authentication Protocols:** NTLM vs Kerberos (`auth_type`) divergence from normal baselines.  
* **Logon Types:** Elevated remote interactive (`logon_type: 10`) vs console access.  
* **Timestamps:** Off-hours or rapid successive events in `times` indicating brute force or reconnaissance.  
* **Event Outcomes:** Repeated failures followed by success (`statuses`) indicating credential stuffing.  

---

## 3. Actionable Detection & Response

### 3.1. Detection Logic

#### Rule 1: Detect Password Spraying  
* **Conceptual Rule:** Count failed `statuses` per `source_computer` across ≥ 10 unique `destination_users` within 5 minutes.  
* **Detection Logic:**  
  - Filter events where `statuses: Failure` and `times` in a rolling 5-minute window  
  - Aggregate by `source_computer`, count distinct `destination_users`  
  - Alert if count ≥ 10  
* **Mapped TTPs:** T1110.003  
* **Immediate Action:** Block or isolate `source_computer`; force password reset for targeted accounts.  

#### Rule 2: Identify Pass-the-Hash  
* **Conceptual Rule:** Successful logon where `auth_type: NTLM`, `logon_type: 3 (Network)`, and destination account is high-privilege.  
* **Detection Logic:**  
  - Filter events where `auth_type == "NTLM"` AND `logon_type == "Network"` AND `statuses == "Success"`  
  - Destination account in privileged group list  
* **Mapped TTPs:** T1550.002  
* **Immediate Action:** Invalidate NTLM hashes; rotate high-privilege credentials; investigate `source_computer`.  

#### Rule 3: Flag Kerberoasting Attempts  
* **Conceptual Rule:** ≥ 5 Kerberos (`auth_type`) service ticket requests for a single `destination_users` within 10 minutes.  
* **Detection Logic:**  
  - Filter events where `auth_type == "Kerberos"` AND `destination_users` is a service account  
  - Aggregate by `destination_users`, count in rolling 10-minute window  
  - Alert if count ≥ 5  
* **Mapped TTPs:** T1558.003  
* **Immediate Action:** Review delegation settings; reset service account passwords; monitor for subsequent cracks.  

#### Rule 4: RDP Lateral Movement  
* **Conceptual Rule:** Interactive logon (`logon_type: 10`) from non-standard `source_computer` to domain workstations/servers.  
* **Detection Logic:**  
  - Filter where `logon_type == "Interactive"` AND `source_computer` not in baseline list for user  
* **Mapped TTPs:** T1021.001  
* **Immediate Action:** Quarantine `source_computer`; require MFA for RDP; investigate user activity.  

#### Rule 5: MFA Fatigue Attack  
* **Conceptual Rule:** ≥ 20 failed MFA-associated `statuses` for same `source_users` in 1 hour.  
* **Detection Logic:**  
  - Correlate authentication events with MFA challenge outcome logs  
  - Aggregate failure count by `source_users`  
  - Alert if count ≥ 20 in 60 minutes  
* **Mapped TTPs:** T1621  
* **Immediate Action:** Temporarily block user; require out-of-band MFA re-enrollment; monitor for compromise.  

### 3.2. SOC Operationalization Guide

* **Investigation Playbook:**
  1. **Alert Triage:** Review event details—`source_computer`, `destination_users`, `auth_type`, `logon_type`, `statuses`, and `times`.  
  2. **Contextual Enrichment:** Pull host asset info, user risk scores, and recent authentications.  
  3. **Pivot & Hunt:** Search for related events in +/- 1 hour across hosts for same user or host.  
  4. **Containment:** Isolate suspicious hosts; enforce password resets or MFA re-enrollment on implicated accounts.  
  5. **Remediation:** Patch and harden authentication protocols; disable legacy NTLM/LM; rotate credentials.  

* **Proactive Threat Hunting:**
  - Query for interactive logons on service accounts outside business hours.  
  - Identify sudden spikes in Kerberos ticket requests.  
  - Search for new source_computer entries authenticating to domain controllers.  

---

## 4. Threat Intelligence Integration (STIX™ 2.1)

### 4.1. Key Observables & Enrichment Workflow

* **Key Fields as Cyber Observable Objects:**
  - `source_computer` → host  
  - `destination_computer` → host  
  - `source_users` → user-account  
  - `destination_users` → user-account  
  - `auth_type` → observed-data attribute  
  - `logon_type` → observed-data attribute  
  - `times` → timestamp  

* **Enrichment Workflow:**
  1. Ingest Windows Authentication Logs; extract STIX cyber-observable objects for each field.  
  2. Push to TIP for automated enrichment: host reputation, geolocation, domain age.  
  3. Link `user-account` objects to threat intelligence on compromised credentials from prior breaches.  
  4. Generate STIX `Indicator` objects for confirmed malicious patterns and share across SOC.  

### 4.2. STIX Domain Object Summary

* **Indicators:** Unusual NTLM network logons, high-frequency service ticket requests, MFA fatigue patterns.  
* **Intrusion Sets:** APT29 (NOBELIUM), Iranian cyber actors (per AA24-290A), Unit 29155 (GRU) ([picussecurity.com](https://www.picussecurity.com/resource/blog/cisa-alert-aa24-249a-russian-military-cyber-actors-target-us-and-global-critical-infrastructure?utm_source=chatgpt.com)).  
* **Vulnerabilities:** CVE-2022-30170 (Credential Roaming), legacy NTLM/LM fallback.  
* **Attack Patterns:** T1078, T1110.001/003, T1550.002/003/004, T1558.003, T1003.001/002, T1021.001, T1621, T1556.006.  
* **Malware:** Beacon, SystemBC, HiveLocker, Qakbot (noted in M-Trends 2023) ([securityweek.com](https://www.securityweek.com/mandiant-2023-m-trends-report-provides-factual-analysis-of-emerging-threat-trends/?utm_source=chatgpt.com)).  
* **Course of Actions:** Enforce Kerberos-only, deprecate NTLM, require MFA, isolate suspicious hosts, credential rotation.  
* **Intrusion Sets:** #repeated as above for emphasis.  

---

*No relevant information found during the search for additional academic publications specifically focusing on novel vulnerabilities in Windows authentication logs beyond those cited.*