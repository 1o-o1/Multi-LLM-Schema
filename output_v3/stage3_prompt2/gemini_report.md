# Security Analysis Report for the Log Type

## Executive Summary

This report provides a comprehensive security analysis of **Windows Authentication Logs**, a critical data source for detecting and responding to a wide array of cyber threats. These logs are fundamental to security monitoring as they capture the core of user and system activity within a Windows environment: every logon, logoff, and authentication attempt.

The primary threats detectable through these logs include **lateral movement**, where adversaries use compromised credentials to move between systems, and **credential dumping**, the act of stealing usernames and passwords from memory. Analysis of Windows authentication events is crucial for identifying techniques such as Pass-the-Hash, brute-force attacks, and the abuse of legitimate administrative tools.

Key recommendations from this report include implementing robust audit policies to ensure necessary events are logged, centralizing log collection for correlation, and developing specific detection rules to identify anomalous patterns. This report provides detailed behavioral patterns to hunt for, actionable detection logic for Security Operations Centers (SOCs), and a guide for integrating findings with threat intelligence platforms using the STIX 2.1 framework. By leveraging the insights within this report, organizations can significantly enhance their ability to detect attacker activity, mitigate risks associated with compromised credentials, and improve their overall security posture.

---

## 1. Threat Landscape & Adversary Behavior

### 1.1. Current Threat Landscape Research

*   **Industry Reporting:**
    Recent industry analysis from 2023-2024 highlights a persistent focus by adversaries on exploiting Windows authentication mechanisms. A notable trend is the abuse of NTLM, a legacy authentication protocol that remains a critical fallback in many Windows environments. Attackers actively seek to trigger NTLM authentication leaks, particularly from applications like Microsoft Outlook, to capture hashes for use in Pass-the-Hash attacks, enabling lateral movement and privilege escalation without needing the plaintext password. Furthermore, credential dumping, especially from the Local Security Authority Subsystem Service (LSASS) process, remains a primary technique for attackers once they gain administrative privileges on a host. This allows them to harvest credentials for widespread network compromise. Reports also emphasize that adversaries leverage legitimate tools like PsExec, WMI, and scheduled tasks to move laterally, making detection difficult as this activity can blend with normal administrative behavior.

*   **Emerging Vectors:**
    A significant emerging vector involves vulnerabilities that bypass existing security features in authentication protocols. For instance, **CVE-2024-20674** is a critical vulnerability in Windows Kerberos that allows an attacker to perform a machine-in-the-middle (MITM) attack to impersonate a Kerberos authentication server, thereby bypassing security features. Another area of concern is the exploitation of certificate-based authentication. Vulnerabilities like **CVE-2022-26931** and **CVE-2022-26923** allow for privilege escalation because the Key Distribution Center (KDC) improperly handled certificate-based authentication requests, enabling attackers to spoof identities. Microsoft is actively working to reduce reliance on NTLM and strengthen Kerberos with features like IAKerb and a local KDC to close these security gaps.

*   **Academic Research:**
    Academic research from 2023 has explored novel methods for enhancing Windows authentication security. One study proposes a location-based authentication method using WiFi signal classification to create a non-intrusive, multi-factor authentication layer for the Windows OS. This approach aims to add contextual security by verifying the user's physical location. Other research focuses on applying machine learning, specifically Convolutional Neural Networks (CNNs), to Intrusion Detection Systems (IDS) for analyzing network traffic related to authentication, achieving high accuracy in detecting threats. These studies indicate a move towards more dynamic and intelligent authentication and threat detection mechanisms beyond traditional log analysis.

### 1.2. MITRE ATT&CK® Framework Mapping

Windows Authentication Logs are a primary data source for detecting a wide range of adversary tactics and techniques. The fields provided directly map to observable actions within the ATT&CK framework.

*   **T1078 – Valid Accounts:**
    *   **Relevance:** This is the most fundamental technique related to authentication logs. Adversaries use legitimate credentials to gain access and move laterally. Differentiating malicious use from normal activity is a key challenge.
    *   **Identifiable Fields:** `source_users`, `destination_users`, `source_computer`, `destination_computer`, `logon_type`, `statuses`. A successful logon (`statuses`: "Success") from an unusual `source_computer` or at an anomalous time (`times`) for a specific `source_users` can indicate abuse.

*   **T1110 – Brute Force:**
    *   **Relevance:** Adversaries attempt to guess passwords for accounts. This technique includes methods like password spraying (one password against many accounts) and credential stuffing (many passwords against one account).
    *   **Identifiable Fields:** `destination_users`, `source_computer`, `statuses`. A high volume of failed logon events (`statuses`: "Failure") from a single `source_computer` targeting one or multiple `destination_users` is a strong indicator.

*   **T1550 – Use Alternate Authentication Material:**
    *   **Relevance:** This technique involves using stolen authentication materials like hashes or tickets instead of plaintext passwords. Pass-the-Hash and Pass-the-Ticket are common sub-techniques.
    *   **Identifiable Fields:** `source_users`, `destination_computer`, `logon_type`, `auth_type`. A successful network logon (`logon_type`: "3") using NTLM (`auth_type`: "NTLM") where the user has not recently logged on interactively to the `source_computer` can indicate a Pass-the-Hash attack.

*   **T1558 – Steal or Forge Kerberos Tickets:**
    *   **Relevance:** Adversaries can steal Kerberos tickets to impersonate users. This includes techniques like Kerberoasting, where an attacker requests service tickets for accounts with weak passwords to crack them offline.
    *   **Identifiable Fields:** `source_users`, `destination_users`, `auth_type`. A spike in Kerberos service ticket requests (`auth_type`: "Kerberos") from a single `source_computer` or for unusual `destination_users` (service accounts) can be an indicator.

*   **T1003 – OS Credential Dumping:**
    *   **Relevance:** While the dumping action itself is often observed through process creation logs, the *use* of the dumped credentials manifests in authentication logs. After dumping credentials from LSASS, an attacker will use them to authenticate to other systems.
    *   **Identifiable Fields:** `source_users`, `destination_users`, `source_computer`, `destination_computer`. A successful logon from a user to a new `destination_computer` shortly after suspicious process activity (like accessing lsass.exe) on the `source_computer` is a key correlation.

*   **T1021 – Remote Services:**
    *   **Relevance:** Adversaries use remote services like RDP, WinRM, and SMB/Windows Admin Shares to execute code and move laterally. These actions generate authentication events.
    *   **Identifiable Fields:** `destination_computer`, `source_computer`, `source_users`, `logon_type`. A logon with `logon_type` "10" (RemoteInteractive/RDP) or "3" (Network) from a workstation (`source_computer`) to another workstation or server (`destination_computer`) by a user who does not typically perform such actions is suspicious.

---

## 2. Observable Evidence & Patterns

### 2.1.1 Malicious Behavioral/Temporal Patterns

*   **Pattern:** High Volume of Failed Logons (Brute Force)
    *   **Description:** A large number of failed logon attempts from a single source computer targeting one or more destination users in a short time frame. This is a classic indicator of a brute-force or password spraying attack.
    *   **Identifiable Fields:** `source_computer`, `destination_users`, `statuses`, `times`.

*   **Pattern:** Successful Logon After Multiple Failures
    *   **Description:** A sequence of failed logon events (`statuses`: "Failure") followed immediately by a successful logon (`statuses`: "Success") for the same user. This can indicate a successful password guessing attempt.
    *   **Identifiable Fields:** `destination_users`, `source_computer`, `statuses`, `times`.

*   **Pattern:** NTLM Logon for Remote Administrative Access (Pass-the-Hash)
    *   **Description:** A successful network logon (`logon_type`: 3) using NTLM authentication (`auth_type`: "NTLM") by an administrative account to a remote system. Kerberos is expected for legitimate domain admin activity; NTLM usage can suggest a Pass-the-Hash attack.
    *   **Identifiable Fields:** `source_users`, `destination_computer`, `logon_type`, `auth_type`, `statuses`.

*   **Pattern:** Lateral Movement via Remote Interactive Logon
    *   **Description:** A user account logging on to multiple workstations or servers via Remote Desktop (`logon_type`: 10) in rapid succession. This is highly indicative of an attacker exploring the network.
    *   **Identifiable Fields:** `source_users`, `destination_computer`, `logon_type`, `times`.

*   **Pattern:** Logon with Explicit Credentials by Unexpected Account
    *   **Description:** An event (ID 4648) showing that a user or process used the credentials of a *different* user to access a resource. This is common in "runas" scenarios but is suspicious when the source or target user is unexpected, often indicating lateral movement.
    *   **Identifiable Fields:** `source_users`, `destination_users`, `source_computer`, `destination_computer`.

*   **Pattern:** First-time Logon from a User to a Critical Server
    *   **Description:** A user account that has never logged into a specific critical asset (like a domain controller or database server) before suddenly authenticates successfully.
    *   **Identifiable Fields:** `source_users`, `destination_computer`, `statuses`.

*   **Pattern:** Service Account Interactive Logon
    *   **Description:** A service account, which should only perform automated tasks and network logons (`logon_type`: 5 or 3), performs an interactive logon (`logon_type`: 2) or remote interactive logon (`logon_type`: 10). This is highly anomalous and suggests account compromise.
    *   **Identifiable Fields:** `source_users`, `logon_type`.

### 2.1.2 Anomalous Behavioral/Temporal Patterns

*   **Pattern:** Impossible Travel
    *   **Description:** A single user account successfully authenticates from two geographically distant `source_computer` locations in a time frame that would be impossible to travel between.
    *   **Identifiable Fields:** `source_users`, `source_computer` (requires IP/geolocation mapping), `times`.

*   **Pattern:** Off-Hours Authentication
    *   **Description:** A user account authenticates at unusual times, such as late at night or on weekends, contrary to their established baseline of activity.
    *   **Identifiable Fields:** `source_users`, `times`.

*   **Pattern:** Multiple Users Logged into a Single Workstation
    *   **Description:** Multiple distinct user accounts are concurrently logged onto a standard end-user workstation within a short time window. This is abnormal for typical user behavior.
    *   **Identifiable Fields:** `destination_computer`, `source_users`, `times`, `orientations`.

*   **Pattern:** Same User Logged into Multiple Hosts Simultaneously
    *   **Description:** A single user account shows active logon sessions on a large number of different computers simultaneously. While some roaming is normal, a widespread pattern can indicate credential abuse.
    *   **Identifiable Fields:** `source_users`, `destination_computer`, `times`, `orientations`.

*   **Pattern:** Excessive Logoff/Logon Activity
    *   **Description:** An abnormally high frequency of logon and logoff events (`orientations`) for a single user or on a single machine, which could indicate scripted activity or an unstable connection being exploited.
    *   **Identifiable Fields:** `source_users`, `destination_computer`, `orientations`, `times`.

*   **Pattern:** Authentication from Non-Standard Hostname
    *   **Description:** Authentication attempts originating from a `source_computer` with a hostname that does not conform to the organization's standard naming convention.
    *   **Identifiable Fields:** `source_computer`.

*   **Pattern:** Sudden Spike in Kerberos Ticket Requests
    *   **Description:** A sudden, anomalous increase in Kerberos ticket requests (Event ID 4768/4769), especially from a single host, which could indicate reconnaissance or a Kerberoasting attack.
    *   **Identifiable Fields:** `source_computer`, `auth_type`, `times`.

### 2.1.3 Vulnerable Behavioral/Temporal Patterns

*   **Pattern:** Use of Weak Kerberos Encryption (RC4/DES)
    *   **Description:** Authentication events show the use of weak or deprecated Kerberos encryption types like RC4 or DES. This makes credentials susceptible to offline cracking and is a significant hygiene issue.
    *   **Identifiable Fields:** `auth_type`, (Requires deeper event data for encryption type).

*   **Pattern:** Excessive NTLMv1 Authentication
    *   **Description:** The continued use of the outdated and insecure NTLMv1 protocol. NTLMv1 is highly vulnerable to relay and cracking attacks and should be disabled.
    *   **Identifiable Fields:** `auth_type`.

*   **Pattern:** Accounts with No Pre-Authentication Required
    *   **Description:** Kerberos pre-authentication is a security feature that helps prevent offline password cracking. An authentication attempt for an account where this is disabled is a vulnerability. This is often detected via Kerberos event logs (4768).
    *   **Identifiable Fields:** (Requires specific Kerberos event data not in the provided fields, but is a key related pattern).

*   **Pattern:** Cleartext Password Submission over Network
    *   **Description:** Some configurations might allow passwords to be sent in cleartext, which would be logged in certain events (e.g., if WDigest is enabled). This is a critical vulnerability.
    *   **Identifiable Fields:** (Requires specific event data, such as Event ID 4624 with detailed logon process info).

*   **Pattern:** Privileged Account Logon to a Workstation
    *   **Description:** A domain administrator or other highly privileged account (`source_users`) logs on to a standard user workstation (`destination_computer`). This increases the risk of credential theft via LSASS dumping.
    *   **Identifiable Fields:** `source_users`, `destination_computer`, `logon_type`.

*   **Pattern:** Authentication from a Device with Known Vulnerabilities
    *   **Description:** A successful authentication from a `source_computer` that is known to be unpatched or have critical vulnerabilities. This indicates a high-risk entry point.
    *   **Identifiable Fields:** `source_computer`.

### 2.2. Key Indicators of Compromise (IOCs)

*   **Suspicious Source Hostnames:**
    *   **Description:** Hostnames that do not conform to internal naming conventions, are randomly generated, or are known to be associated with malicious tools. This can indicate an unauthorized device on the network.

*   **Anomalous User Agent Strings (in web-based authentication):**
    *   **Description:** While not in the provided fields, web authentication logs often contain User-Agent strings. Those associated with scripting tools (PowerShell, Python requests) instead of web browsers can indicate automated attacks.

*   **Known Malicious Source IP Addresses:**
    *   **Description:** Authentication attempts originating from IP addresses listed on threat intelligence feeds as being malicious, command-and-control (C2) servers, or Tor exit nodes.
    *   **Identifiable Fields:** `source_computer` (when it's an IP).

*   **Use of Common Hacking/Admin Tool Names as Usernames/Hostnames:**
    *   **Description:** Usernames or hostnames that match the names of common penetration testing or administration tools (e.g., "mimikatz", "psexec", "kali").

*   **Suspicious Service Names:**
    *   **Description:** When credential dumping or persistence is attempted via service creation (Event ID 4697), the service name may be random or attempt to masquerade as a legitimate service.

*   **Suspicious Process Names Accessing LSASS:**
    *   **Description:** While not a direct authentication log IOC, it's a critical correlated indicator. Processes like `procdump.exe` or `rundll32.exe` accessing the LSASS process are a hallmark of credential dumping. The credentials stolen are then used in attacks visible in authentication logs.

---

## 3. Actionable Detection & Response

### 3.1. Detection Logic

#### Rule 1: Potential Brute-Force or Password Spraying Attack
*   **Conceptual Rule:**
    *   **Threat Scenario:** An attacker is attempting to guess user credentials through brute force (many passwords for one user) or password spraying (one password for many users).
    *   **Detection Logic:** Alert when a single `source_computer` generates more than 20 failed logon events (`statuses`: "Failure") within a 5-minute window. Group by `source_computer`. A lower threshold (e.g., 5 failures) targeting more than 50 distinct `destination_users` can specifically detect password spraying.
    *   **Mapped TTPs:** T1110 (Brute Force)
    *   **Immediate Action:** Temporarily block the `source_computer` IP at the firewall. Investigate the source for signs of compromise. Notify the targeted users if a specific account is being attacked.

#### Rule 2: Successful Logon Immediately Following Failed Attempts
*   **Conceptual Rule:**
    *   **Threat Scenario:** An attacker successfully guesses a password after multiple attempts.
    *   **Detection Logic:** Alert when a successful logon event (`statuses`: "Success") for a `destination_users` from a specific `source_computer` is preceded by 5 or more failed logon events for the same user from the same source within the last 15 minutes.
    *   **Mapped TTPs:** T1110 (Brute Force), T1078 (Valid Accounts)
    *   **Immediate Action:** Isolate the `destination_computer`. Force a password reset for the compromised `destination_users` and expire all active sessions.

#### Rule 3: Potential Pass-the-Hash (PtH) Activity
*   **Conceptual Rule:**
    *   **Threat Scenario:** An attacker is using a stolen NTLM hash to authenticate to a remote system, bypassing the need for a password.
    *   **Detection Logic:** Alert on a successful network logon (`logon_type`: 3) that uses NTLM authentication (`auth_type`: "NTLM") for a privileged user account (`source_users`) to a `destination_computer` where that user has no prior interactive logon history.
    *   **Mapped TTPs:** T1550.002 (Use Alternate Authentication Material: Pass the Hash)
    *   **Immediate Action:** Isolate the `source_computer` and `destination_computer`. Investigate the `source_computer` for signs of credential dumping. Expire all sessions for the user account involved.

#### Rule 4: Anomalous Remote Desktop (RDP) Logon
*   **Conceptual Rule:**
    *   **Threat Scenario:** An attacker is moving laterally between hosts using RDP.
    *   **Detection Logic:** Alert when a user account (`source_users`) initiates a remote interactive logon (`logon_type`: 10) to a `destination_computer` that is not on a pre-approved list of administrative jump boxes or servers for that user. Also, alert if a user logs into more than 5 distinct hosts via RDP within one hour.
    *   **Mapped TTPs:** T1021.001 (Remote Services: Remote Desktop Protocol), T1078 (Valid Accounts)
    *   **Immediate Action:** Validate the activity with the user. If suspicious, isolate the `destination_computer` and investigate the `source_computer` for compromise.

#### Rule 5: Service Account Interactive Logon
*   **Conceptual Rule:**
    *   **Threat Scenario:** A service account has been compromised and is being used by an attacker for interactive access.
    *   **Detection Logic:** Alert whenever an account known to be a service account (based on naming convention or group membership) generates a logon event with `logon_type` 2 (Interactive) or 10 (Remote Interactive).
    *   **Mapped TTPs:** T1078 (Valid Accounts)
    *   **Immediate Action:** Immediately disable the service account. Isolate the `destination_computer` where the logon occurred. Investigate for persistence and further malicious activity. Initiate a password rotation for all service accounts.

### 3.2. SOC Operationalization Guide

*   **Investigation Playbook:**
    1.  **Triage Alert:** Immediately assess the alert's context. Identify the `source_users`, `source_computer`, and `destination_computer`. Note the `logon_type`, `auth_type`, and `statuses`.
    2.  **Correlate Activity:** Pivot on the key fields (`source_users`, `source_computer`, `times`). Look for other related security events immediately preceding or following the alert. Did a process creation alert for `procdump.exe` fire on the `source_computer`? Is there subsequent network traffic to a known malicious IP?
    3.  **Establish Baseline:** Quickly determine if this is normal behavior. Has this `source_users` logged in from this `source_computer` before? Is the `logon_type` typical for this user? Check against historical log data.
    4.  **Contain the Threat:** If the activity is deemed malicious, take immediate containment actions. Isolate the involved hosts from the network. Disable the compromised user account.
    5.  **Investigate and Eradicate:** Perform forensic analysis on the affected systems to understand the full scope of the compromise. Identify the initial access vector, check for persistence mechanisms, and remove all attacker artifacts.
    6.  **Recover and Report:** Restore systems from clean backups, reset all potentially compromised credentials, and document the incident, including the timeline, impact, and remediation steps.

*   **Proactive Threat Hunting:**
    *   **Hypothesis 1 (Lateral Movement):** Search for users (`source_users`) with successful network (`logon_type`: 3) or remote interactive (`logon_type`: 10) logons to an unusually high number of distinct hosts (`destination_computer`) within a 24-hour period. Filter out administrative and service accounts that are expected to do this.
    *   **Hypothesis 2 (Anomalous Credentials):** Hunt for successful logons (`statuses`: "Success") that use explicit credentials (`EventID 4648`) where the `source_users` is different from the `destination_users`. Investigate any instances that aren't related to legitimate administrative tasks.
    *   **Hypothesis 3 (NTLM Abuse):** Periodically review all successful NTLM authentication (`auth_type`: "NTLM"). Is it being used where Kerberos should be? Are privileged accounts using it to access workstations? This can uncover both hygiene issues and active attacks.

---

## 4. Threat Intelligence Integration (STIX™ 2.1)

### 4.1. Key Observables & Enrichment Workflow

The following fields from the Windows Authentication Log can be extracted as STIX Cyber Observable Objects for threat intelligence enrichment:

*   **source_computer:** Can be mapped to `ipv4-addr`, `ipv6-addr`, or `domain-name` objects.
*   **destination_computer:** Can be mapped to `ipv4-addr`, `ipv6-addr`, or `domain-name` objects.
*   **source_users:** Can be mapped to a `user-account` object, with the `account_login` property populated.
*   **destination_users:** Can be mapped to a `user-account` object.

**Enrichment Workflow:**

1.  **Extraction:** When a suspicious event is detected, an automated workflow or analyst extracts the key observables (e.g., the `source_computer` IP address and the `destination_users` account name).
2.  **Pivot to TIP:** The extracted observables are used to query a Threat Intelligence Platform (TIP). For example, the `source_computer` IP address is checked against IP reputation feeds.
3.  **Contextual Enrichment:** The TIP returns any related intelligence, such as:
    *   Is the IP address a known C2 server, part of a botnet, or a Tor node? (STIX `Indicator` object).
    *   Has this IP address been associated with a specific threat actor? (STIX `Intrusion Set` object).
    *   Has this `user-account` been seen in credential dumps?
4.  **Decision & Action:** The enriched data provides context to the analyst, helping them determine the alert's severity and priority. A logon from an IP associated with an APT group (`Intrusion Set`) is far more critical than one from an unknown source. This intelligence informs the response actions outlined in the SOC playbook.

### 4.2. STIX Domain Object Summary

*   **Indicators:**
    *   An IP address (`ipv4-addr`) consistently failing to authenticate against multiple accounts indicates a brute-force attack.
    *   A logon (`logon_type`: 3) using NTLM (`auth_type`: "NTLM") from a non-standard workstation to a domain controller indicates potential Pass-the-Hash.
    *   A user account (`user-account`) authenticating from two geographically impossible locations in a short time (`pattern: "impossible-travel"`) indicates a compromised account.

*   **Intrusion Sets:**
    *   Groups like **FIN6** and **APT29** are known to leverage stolen credentials for lateral movement. An authenticated session originating from an IP address associated with these groups would be a high-confidence indicator of a targeted attack.

*   **Vulnerabilities:**
    *   **CVE-2024-20674:** A security feature bypass vulnerability in Windows Kerberos.
    *   **CVE-2022-26931:** An elevation of privilege vulnerability related to certificate-based authentication.
    *   **NTLM Relay Vulnerabilities:** A class of vulnerabilities where an attacker can relay NTLM authentication challenges to gain unauthorized access.

*   **Attack Patterns:**
    *   **Brute Force (T1110):** Repeatedly attempting to guess credentials.
    *   **Password Spraying (T1110.003):** Using a small list of common passwords against many accounts.
    *   **Pass the Hash (T1550.002):** Using a stolen NTLM hash to authenticate.
    *   **Pass the Ticket (T1550.003):** Using a stolen Kerberos ticket to authenticate.
    *   **Kerberoasting (T1558.003):** Requesting service tickets for accounts to crack their passwords offline.
    *   **Remote Services: Remote Desktop Protocol (T1021.001):** Using RDP to move between systems.
    *   **Valid Accounts (T1078):** Abusing legitimate credentials to blend in with normal network traffic.
    *   **OS Credential Dumping: LSASS Memory (T1003.001):** Stealing credentials from the LSASS process memory.

*   **Malware:**
    *   **Mimikatz:** A post-exploitation tool widely used to dump credentials from memory (LSASS), which are then used in attacks visible in authentication logs.
    *   **PsExec:** A legitimate system administration tool that is frequently abused by attackers to execute commands and move laterally using authenticated sessions.

*   **Course of Actions:**
    *   **Enforce Multi-Factor Authentication (MFA):** Require MFA for all user accounts, especially privileged ones, to mitigate the impact of stolen credentials.
    *   **Disable Legacy Protocols:** Disable NTLMv1 and, where possible, restrict or disable NTLM entirely in favor of Kerberos.
    *   **Implement Strong Password Policies:** Enforce password complexity, length, and history requirements to make brute-force attacks more difficult.
    *   **Restrict Privileged Account Access:** Limit administrative accounts to specific, hardened administrative workstations or jump servers to reduce the risk of credential theft.
    *   **Centralize Log Collection:** Aggregate Windows Authentication logs from all endpoints and domain controllers into a central SIEM for effective correlation and analysis.

---